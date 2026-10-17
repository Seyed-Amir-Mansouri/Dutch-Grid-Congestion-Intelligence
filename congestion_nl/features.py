from __future__ import annotations

import numpy as np
import pandas as pd

from . import schemas
from .config import Config
from .labels import expand_hours, label_grid
from .rolling import lagged_rolling

FEATURE_COLUMNS = [
    "cell_code",
    "cell_lat",
    "cell_lon",
    "hour",
    "weekday",
    "month",
    "is_weekend",
    "price_eur_mwh",
    "net_position_mw",
    "load_forecast_mw",
    "wind_forecast_mw",
    "solar_forecast_mw",
    "residual_load_mw",
    "price_hist_mean",
    "congested_hours_hist",
    "congested_hours_long",
    "hour_congested_hist",
    "hour_announced",
    "announced_hours",
    "known_hours_yesterday",
    "run_days",
    "days_since_request",
    "mw_hist",
]


def cell_table(locations: pd.DataFrame, size: float) -> pd.DataFrame:
    """Return one row per cell with its centre, numbered from north-west to south-east."""
    cells = pd.DataFrame({"cell": sorted(locations["cell"].unique())})
    corner = cells["cell"].str.extract(r"^([\d.]+)N ([\d.]+)E$").astype(float)
    cells["cell_lat"] = corner[0] + size / 2
    cells["cell_lon"] = corner[1] + size / 2
    cells = cells.sort_values(["cell_lat", "cell_lon"], ascending=[False, True]).reset_index(drop=True)
    cells["cell_code"] = np.arange(1, len(cells) + 1)
    return cells


def _streaks(daily: pd.DataFrame) -> pd.DataFrame:
    """Add, per cell, the run of request days and the days since the last request, both up to the day before."""
    parts = []
    for cell, group in daily.groupby("cell", sort=False):
        active = group.set_index("delivery_date")["known_hours"].gt(0).astype(int)
        active = active.reindex(pd.date_range(active.index.min(), active.index.max(), freq="D"), fill_value=0)
        run = active.groupby((active == 0).cumsum()).cumsum()
        last = pd.Series(active.index.where(active > 0), index=active.index).ffill()
        part = pd.DataFrame(
            {
                "delivery_date": active.index,
                "cell": cell,
                "run_days": run.shift(1).to_numpy(),
                "days_since_request": (active.index - last.shift(1)).dt.days.to_numpy(),
            }
        )
        parts.append(part)
    return daily.merge(pd.concat(parts), on=["delivery_date", "cell"], how="left")


def known_features(frame: pd.DataFrame, events: pd.DataFrame, settings) -> pd.DataFrame:
    """Add features from requests announced before the forecast is issued on the day before delivery."""
    issue = pd.Timedelta(days=-1, hours=settings.issue_hour)
    hours = expand_hours(events, ["event_id", "announced_at", "mw"])

    ahead = hours[hours["announced_at"] < hours["delivery_date"] + issue]
    ahead = ahead.drop_duplicates(subset=schemas.KEY)[schemas.KEY].assign(hour_announced=1)
    frame = frame.merge(ahead, on=schemas.KEY, how="left")
    frame["hour_announced"] = frame["hour_announced"].fillna(0).astype(int)

    # a past day counts what was announced by the issue time of the forecast for the next day
    seen = hours[hours["announced_at"] < hours["delivery_date"] + issue + pd.Timedelta(days=1)]
    key = ["delivery_date", "cell"]
    known = seen.drop_duplicates(subset=schemas.KEY).groupby(key).size().rename("known_hours")
    mw = seen.drop_duplicates(subset=key + ["event_id"]).groupby(key)["mw"].sum().rename("known_mw")

    daily = frame.groupby(key, as_index=False)["hour_announced"].sum()
    daily = daily.rename(columns={"hour_announced": "announced_hours"})
    daily = daily.merge(known, on=key, how="left").merge(mw, on=key, how="left")
    daily[["known_hours", "known_mw"]] = daily[["known_hours", "known_mw"]].fillna(0)
    daily = daily.sort_values(key).reset_index(drop=True)
    daily["known_hours_yesterday"] = lagged_rolling(daily, "known_hours", ["cell"], 1, 1, "sum")
    daily["mw_hist"] = lagged_rolling(daily, "known_mw", ["cell"], settings.history_days, 1, "sum")
    daily = _streaks(daily).drop(columns=["known_hours", "known_mw"])
    return frame.merge(daily, on=key, how="left")


def build_dataset(
    config: Config,
    sdac: pd.DataFrame,
    forecasts: pd.DataFrame,
    events: pd.DataFrame,
    cells: pd.DataFrame,
    label_end: pd.Timestamp | None,
) -> pd.DataFrame:
    """Join inputs and build features, with the target set for dates up to label_end."""
    settings = config.features

    market = sdac.merge(forecasts, on=schemas.MARKET_KEY, how="inner")
    market = market.sort_values(schemas.MARKET_KEY).reset_index(drop=True)
    market["residual_load_mw"] = (
        market["load_forecast_mw"] - market["wind_forecast_mw"] - market["solar_forecast_mw"]
    )
    market["price_hist_mean"] = lagged_rolling(
        market, "price_eur_mwh", ["hour"], settings.history_days, settings.price_lag_days, "mean"
    )
    market = market[market["delivery_date"] >= config.grid.start_date]

    frame = market.merge(cells, how="cross")
    frame["weekday"] = frame["delivery_date"].dt.weekday
    frame["month"] = frame["delivery_date"].dt.month
    frame["is_weekend"] = (frame["weekday"] >= 5).astype(int)
    frame = frame.sort_values(schemas.KEY).reset_index(drop=True)

    labelled = label_grid(frame[schemas.KEY], events)
    frame = frame.merge(labelled, on=schemas.KEY, how="left")

    frame["hour_congested_hist"] = lagged_rolling(
        frame, "congested", ["cell", "hour"], settings.history_days, settings.event_lag_days, "sum"
    )
    daily = frame.groupby(["delivery_date", "cell"], as_index=False)["congested"].sum()
    daily = daily.rename(columns={"congested": "congested_hours"})
    daily["congested_hours_hist"] = lagged_rolling(
        daily, "congested_hours", ["cell"], settings.history_days, settings.event_lag_days, "sum"
    )
    daily["congested_hours_long"] = lagged_rolling(
        daily, "congested_hours", ["cell"], settings.long_history_days, settings.event_lag_days, "sum"
    )
    frame = frame.merge(daily.drop(columns=["congested_hours"]), on=["delivery_date", "cell"], how="left")
    frame = known_features(frame, events, settings)

    if label_end is None:
        frame["target"] = np.nan
    else:
        frame["target"] = frame["congested"].where(frame["delivery_date"] <= label_end)
    return frame.drop(columns=["congested"])
