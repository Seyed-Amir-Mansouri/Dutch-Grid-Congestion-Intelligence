"""Estimate the day-ahead price, net position and the wind and solar forecasts of a delivery day, before they are published.

Per quarter-hour, from what is known before the day-ahead results on D-1: the national means of the weather forecast,
the load forecast, the calendar, and each target on D-1 and D-7 with its D-1 mean. One boosted-tree regressor per
target, trained on every earlier day each time it is used.

The net position depends mostly on Germany, Belgium and France, so it also gets their weather and load forecast, a
longer Dutch history of its own and of the price, totals and gaps built from them, and counts recent days more.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import schemas
from .features import clock_features, public_holidays

TARGETS = ["price_eur_mwh", "net_position_mw", "wind_forecast_mw", "solar_forecast_mw"]
WEATHER = ["temp", "rad", "rad_max", "cloud", "wind100", "wind3", "wind100_coast", "wind10"]
NET_POSITION = "net_position_mw"
# weather over the neighbours, per country, and the load forecast of each (both known before the day-ahead results)
NEIGHBOUR_WEATHER = ["temperature_c", "radiation_wm2", "cloud_cover_pct", "wind_100m_kmh", "wind3"]
NEIGHBOUR_LOAD = ["DE_LU_load", "BE_load", "FR_load"]
# the Dutch net position and price of earlier days, from the net position's point of view
HISTORY = ["np_d2", "np_d3", "np_wd4", "np_mean7", "np_mean3", "np_std7", "np_d1_min", "np_d1_max",
           "price_d1", "price_d7", "price_d1_mean"]
# totals and gaps built from the neighbour inputs, which a tree finds hard to build from single columns
DERIVED = ["x_wind3_sum", "x_rad_mean", "x_load_sum", "x_nl_share", "x_wind3_gap", "x_rad_gap", "x_resid_proxy"]
# a day this many days older counts half as much when training the net position
HALF_LIFE_DAYS = 180.0
CALENDAR = ["quarter", "hour", "weekday", "month", "doy_sin", "doy_cos", "off_day", "winter_time"]


def weather_means(weather: pd.DataFrame) -> pd.DataFrame:
    """National weather per delivery day and hour: means over the cells, wind also for the coast and as wind power."""
    lat = weather["cell"].str.extract(r"([\d.]+)N")[0].astype(float)
    lon = weather["cell"].str.extract(r"([\d.]+)E")[0].astype(float)
    weather = weather.assign(coast=(lon <= 4.75) | (lat >= 53.25),
                             wind3=(weather["wind_100m_kmh"].clip(upper=90) / 3.6) ** 3)
    g = weather.groupby("time")
    out = pd.DataFrame({"temp": g["temperature_c"].mean(), "rad": g["radiation_wm2"].mean(),
                        "rad_max": g["radiation_wm2"].max(), "cloud": g["cloud_cover_pct"].mean(),
                        "wind100": g["wind_100m_kmh"].mean(), "wind3": g["wind3"].mean(),
                        "wind100_coast": weather[weather["coast"]].groupby("time")["wind_100m_kmh"].mean(),
                        "wind10": g["wind_10m_kmh"].mean()}).reset_index()
    out["delivery_date"], out["hour"] = out["time"].dt.normalize(), out["time"].dt.hour
    # the repeated autumn hour is averaged, as in the model's weather inputs
    return out.drop(columns="time").groupby(["delivery_date", "hour"], as_index=False).mean()


def neighbour_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Means over the points of each neighbour per delivery day and hour, and the wind power of northern Germany."""
    weather = weather.assign(wind3=(weather["wind_100m_kmh"].clip(upper=90) / 3.6) ** 3,
                             delivery_date=weather["time"].dt.normalize(), hour=weather["time"].dt.hour)
    north = weather["cell"].str.extract(r"([\d.]+)N")[0].astype(float) >= 53.0
    g = weather.groupby(["delivery_date", "hour", "country"])[NEIGHBOUR_WEATHER].mean().unstack("country")
    g.columns = [f"nb_{country}_{name}" for name, country in g.columns]
    german_north = weather[(weather["country"] == "DE") & north].groupby(["delivery_date", "hour"])["wind3"].mean()
    return g.join(german_north.rename("nb_DE_north_wind3")).reset_index()


def neighbour_columns() -> list[str]:
    return [f"nb_{c}_{n}" for n in NEIGHBOUR_WEATHER for c in ("DE", "BE", "FR")] + ["nb_DE_north_wind3"]


def net_position_history(t: pd.DataFrame) -> pd.DataFrame:
    """The net position and price of earlier days: lags, the weeks before and the recent mean, spread and range."""
    wide = t.pivot_table(index="delivery_date", columns="quarter", values=NET_POSITION)
    daily = wide.mean(axis=1)
    out = t[schemas.MARKET_KEY].copy()
    for lag in (2, 3, 14, 21, 28):
        shifted = wide.shift(lag, freq="D").stack().rename(f"np_d{lag}").reset_index()
        out = out.merge(shifted, on=schemas.MARKET_KEY, how="left")
    out["np_wd4"] = out[["np_d14", "np_d21", "np_d28"]].mean(axis=1)
    roll = pd.DataFrame({"np_mean7": daily.rolling(7).mean().shift(1, freq="D"),
                         "np_mean3": daily.rolling(3).mean().shift(1, freq="D"),
                         "np_std7": daily.rolling(7).std().shift(1, freq="D"),
                         "np_d1_min": wide.min(axis=1).shift(1, freq="D"),
                         "np_d1_max": wide.max(axis=1).shift(1, freq="D")}).reset_index()
    out = out.merge(roll, on="delivery_date", how="left")
    price = t.pivot_table(index="delivery_date", columns="quarter", values="price_eur_mwh")
    for lag in (1, 7):
        shifted = price.shift(lag, freq="D").stack().rename(f"price_d{lag}").reset_index()
        out = out.merge(shifted, on=schemas.MARKET_KEY, how="left")
    mean = price.mean(axis=1).shift(1, freq="D").rename("price_d1_mean").reset_index()
    return out.merge(mean, on="delivery_date", how="left")


def derived_inputs(t: pd.DataFrame) -> pd.DataFrame:
    """Neighbour wind and sun totals, their load total, the Dutch share of it, the gaps to the Dutch weather and a
    rough residual load, each scaled by its spread over the first year of the table."""
    out = pd.DataFrame(index=t.index)
    wind = t[[f"nb_{c}_wind3" for c in ("DE", "BE", "FR")]]
    rad = t[[f"nb_{c}_radiation_wm2" for c in ("DE", "BE", "FR")]]
    out["x_wind3_sum"] = wind.sum(axis=1, min_count=1)
    out["x_rad_mean"] = rad.mean(axis=1)
    out["x_load_sum"] = t[NEIGHBOUR_LOAD].sum(axis=1, min_count=1)
    out["x_nl_share"] = t["load_forecast_mw"] / out["x_load_sum"]
    out["x_wind3_gap"] = t["wind3"] - out["x_wind3_sum"] / 3
    out["x_rad_gap"] = t["rad"] - out["x_rad_mean"]
    first = out[t["delivery_date"] < t["delivery_date"].min() + pd.Timedelta(days=365)]
    z = lambda col: (out[col] - first[col].mean()) / first[col].std()
    out["x_resid_proxy"] = z("x_load_sum") - z("x_wind3_sum") - z("x_rad_mean")
    return out[DERIVED]


def table(sdac: pd.DataFrame, forecasts: pd.DataFrame, weather: pd.DataFrame, day: pd.Timestamp,
          neighbours: dict | None = None) -> pd.DataFrame:
    """Every quarter-hour up to day with the targets where known and the SDAC Forecaster's inputs.

    neighbours holds the 'weather' and 'load' tables of Germany, Belgium and France; without it the net position
    is forecast from the Dutch inputs only.
    """
    t = sdac.merge(forecasts, on=schemas.MARKET_KEY, how="outer")
    t = t[t["delivery_date"] < day]
    grid = pd.DataFrame({"delivery_date": day, "quarter": range(96)})
    # the day's load forecast is out before the day-ahead results
    grid = grid.merge(forecasts[["delivery_date", "quarter", "load_forecast_mw"]], on=schemas.MARKET_KEY, how="left")
    t = pd.concat([t, grid], ignore_index=True).sort_values(schemas.MARKET_KEY)
    t["hour"] = t["quarter"] // 4
    t = t.merge(weather_means(weather), on=["delivery_date", "hour"], how="left")
    days = pd.DatetimeIndex(t["delivery_date"])
    off = (days.dayofweek >= 5) | days.isin(public_holidays(range(days.year.min(), days.year.max() + 1)))
    t = t.assign(weekday=days.dayofweek, month=days.month, doy_sin=np.sin(2 * np.pi * days.dayofyear / 365.25),
                 doy_cos=np.cos(2 * np.pi * days.dayofyear / 365.25), off_day=off.astype(float))
    t = t.merge(clock_features(pd.date_range(days.min(), days.max()))[["delivery_date", "winter_time"]],
                on="delivery_date", how="left")
    for target in TARGETS:
        wide = t.pivot_table(index="delivery_date", columns="quarter", values=target)
        for lag in (1, 7):
            shifted = wide.shift(lag, freq="D").stack().rename(f"{target}_d{lag}").reset_index()
            t = t.merge(shifted, on=schemas.MARKET_KEY, how="left")
        mean = wide.mean(axis=1).shift(1, freq="D").rename(f"{target}_d1_mean").reset_index()
        t = t.merge(mean, on="delivery_date", how="left")
    if neighbours is not None:
        t = t.merge(net_position_history(t), on=schemas.MARKET_KEY, how="left")
        t = t.merge(neighbour_weather(neighbours["weather"]), on=["delivery_date", "hour"], how="left")
        t = t.merge(neighbours["load"], on=schemas.MARKET_KEY, how="left")
        t = t.join(derived_inputs(t))
    return t.reset_index(drop=True)


def inputs_for(target: str, neighbours: bool = False) -> list[str]:
    """The SDAC Forecaster's inputs for one target; the net position also gets the neighbours' when they are known."""
    columns = CALENDAR + ["load_forecast_mw"] + WEATHER + [f"{target}_d1", f"{target}_d7", f"{target}_d1_mean"]
    if neighbours and target == NET_POSITION:
        columns += HISTORY + neighbour_columns() + NEIGHBOUR_LOAD + DERIVED
    return columns


def recency(train: pd.DataFrame, day: pd.Timestamp) -> np.ndarray:
    """Training weights that halve for every HALF_LIFE_DAYS a day lies before the one forecast."""
    return 0.5 ** ((day - train["delivery_date"]).dt.days.to_numpy() / HALF_LIFE_DAYS)


def load_neighbours(config) -> dict | None:
    """The weather and load forecast of the neighbours, or None while either file has not been downloaded."""
    weather, load = config.resolve(config.paths.neighbour_weather), config.resolve(config.paths.neighbour_load)
    if not (weather.exists() and load.exists()):
        return None
    return {"weather": pd.read_csv(weather, parse_dates=["time"]),
            "load": pd.read_csv(load, parse_dates=["delivery_date"])[schemas.MARKET_KEY + NEIGHBOUR_LOAD]}


def _regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=42)


def estimate(sdac: pd.DataFrame, forecasts: pd.DataFrame, weather: pd.DataFrame, day: pd.Timestamp,
             neighbours: dict | None = None) -> pd.DataFrame:
    """The targets for each quarter-hour of day, from regressors trained on every earlier day."""
    t = table(sdac, forecasts, weather, day, neighbours)
    train, rows = t[t["delivery_date"] < day], t[t["delivery_date"] == day]
    if rows[WEATHER].isna().all().all():
        raise ValueError(f"No weather forecast for {day:%Y-%m-%d}: the SDAC Forecaster needs it")
    out = rows[schemas.MARKET_KEY].reset_index(drop=True)
    for target in TARGETS:
        columns = inputs_for(target, neighbours is not None)
        known = train[train[target].notna()]
        weight = recency(known, day) if neighbours is not None and target == NET_POSITION else None
        out[target] = _regressor().fit(known[columns], known[target], sample_weight=weight).predict(rows[columns])
    return out


def run(config, days, sdac: pd.DataFrame, forecasts: pd.DataFrame, weather: pd.DataFrame, report=print) -> pd.DataFrame:
    """Estimate each day and save it, replacing earlier estimates of the same days."""
    found = []
    neighbours = load_neighbours(config)
    if neighbours is None:
        report("Neighbour weather and load are missing: the net position uses Dutch inputs only", flush=True)
    for day in days:
        found.append(estimate(sdac, forecasts, weather, day, neighbours))
        report(f"{day:%Y-%m-%d}: price {found[-1]['price_eur_mwh'].mean():.0f} EUR/MWh on average, wind up to "
               f"{found[-1]['wind_forecast_mw'].max():.0f} MW, solar up to {found[-1]['solar_forecast_mw'].max():.0f} MW",
               flush=True)
    found = pd.concat(found, ignore_index=True)
    path = config.resolve(config.paths.sdac_forecasts)
    old = saved(config)
    old = old[~old["delivery_date"].isin(found["delivery_date"])]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([old, found], ignore_index=True).sort_values(schemas.MARKET_KEY).to_csv(path, index=False)
    return found


def saved(config) -> pd.DataFrame:
    """The estimates saved so far."""
    path = config.resolve(config.paths.sdac_forecasts)
    if not path.exists():
        return pd.DataFrame(columns=schemas.MARKET_KEY + TARGETS)
    return pd.read_csv(path, parse_dates=["delivery_date"])


def for_days(config, days, sdac: pd.DataFrame, forecasts: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """The saved estimates of these days; days not saved yet are estimated now."""
    found = saved(config)
    found = found[found["delivery_date"].isin(days)]
    missing = [day for day in days if day not in set(found["delivery_date"])]
    if missing:
        found = pd.concat([found, run(config, missing, sdac, forecasts, weather)], ignore_index=True)
    return found


def fill(dataset: pd.DataFrame, found: pd.DataFrame) -> pd.DataFrame:
    """The dataset, changed in place, with the market inputs of the estimated days replaced and the residual load redone."""
    rows = dataset["delivery_date"].isin(found["delivery_date"]).to_numpy()
    values = dataset.loc[rows, schemas.MARKET_KEY].merge(found, on=schemas.MARKET_KEY, how="left")
    for target in TARGETS:
        dataset.loc[rows, target] = values[target].to_numpy(np.float32)
    dataset.loc[rows, "residual_load_mw"] = (dataset.loc[rows, "load_forecast_mw"] - dataset.loc[rows, "wind_forecast_mw"]
                                             - dataset.loc[rows, "solar_forecast_mw"])
    return dataset
