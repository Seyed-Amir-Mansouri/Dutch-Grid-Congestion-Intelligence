"""Estimate the day-ahead price, net position and the wind and solar forecasts of a delivery day, before they are published.

Per quarter-hour, from what is known before the day-ahead results on D-1: the national means of the weather forecast,
the load forecast, the calendar, and each target on D-1 and D-7 with its D-1 mean. One boosted-tree regressor per
target, trained on every earlier day each time it is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import schemas
from .features import clock_features, public_holidays

TARGETS = ["price_eur_mwh", "net_position_mw", "wind_forecast_mw", "solar_forecast_mw"]
WEATHER = ["temp", "rad", "rad_max", "cloud", "wind100", "wind3", "wind100_coast", "wind10"]
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


def table(sdac: pd.DataFrame, forecasts: pd.DataFrame, weather: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Every quarter-hour up to day with the targets where known and the SDAC Forecaster's inputs."""
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
    return t.reset_index(drop=True)


def inputs_for(target: str) -> list[str]:
    """The SDAC Forecaster's inputs for one target."""
    return CALENDAR + ["load_forecast_mw"] + WEATHER + [f"{target}_d1", f"{target}_d7", f"{target}_d1_mean"]


def _regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=42)


def estimate(sdac: pd.DataFrame, forecasts: pd.DataFrame, weather: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """The targets for each quarter-hour of day, from regressors trained on every earlier day."""
    t = table(sdac, forecasts, weather, day)
    train, rows = t[t["delivery_date"] < day], t[t["delivery_date"] == day]
    if rows[WEATHER].isna().all().all():
        raise ValueError(f"No weather forecast for {day:%Y-%m-%d}: the SDAC Forecaster needs it")
    out = rows[schemas.MARKET_KEY].reset_index(drop=True)
    for target in TARGETS:
        columns = inputs_for(target)
        known = train[train[target].notna()]
        out[target] = _regressor().fit(known[columns], known[target]).predict(rows[columns])
    return out


def run(config, days, sdac: pd.DataFrame, forecasts: pd.DataFrame, weather: pd.DataFrame, report=print) -> pd.DataFrame:
    """Estimate each day and save it, replacing earlier estimates of the same days."""
    found = []
    for day in days:
        found.append(estimate(sdac, forecasts, weather, day))
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
