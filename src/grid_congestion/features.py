"""Feature and target construction for next-day congestion risk."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schemas
from .config import Config
from .labels import label_grid

FEATURE_COLUMNS = [
    "area_code",
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
    "outage_mw",
    "price_hist_mean",
    "congested_hours_hist",
    "hour_congested_hist",
]


def lagged_rolling(
    frame: pd.DataFrame,
    value: str,
    by: list[str],
    window: int,
    lag: int,
    how: str,
) -> np.ndarray:
    """Aggregate a value over trailing days per group, shifted by a lag in days."""
    result = np.full(len(frame), np.nan)
    positions = frame.reset_index(drop=True)
    for _, group in positions.groupby(by, sort=False):
        series = group.set_index("delivery_date")[value].sort_index()
        calendar = pd.date_range(series.index.min(), series.index.max(), freq="D")
        rolled = getattr(series.reindex(calendar).rolling(window, min_periods=1), how)()
        rolled = rolled.shift(lag)
        result[group.index.to_numpy()] = rolled.reindex(group["delivery_date"]).to_numpy()
    return result


def build_dataset(
    config: Config,
    sdac: pd.DataFrame,
    forecasts: pd.DataFrame,
    outages: pd.DataFrame | None,
    events: pd.DataFrame,
    label_end: pd.Timestamp | None,
) -> pd.DataFrame:
    """Join inputs and build features, with the target set for dates up to label_end."""
    days = config.features.history_days

    market = sdac.merge(forecasts, on=schemas.MARKET_KEY, how="inner")
    market = market.sort_values(schemas.MARKET_KEY).reset_index(drop=True)
    market["residual_load_mw"] = (
        market["load_forecast_mw"] - market["wind_forecast_mw"] - market["solar_forecast_mw"]
    )
    market["price_hist_mean"] = lagged_rolling(
        market, "price_eur_mwh", ["hour"], days, config.features.price_lag_days, "mean"
    )

    areas = pd.DataFrame({"area": list(config.areas)})
    frame = market.merge(areas, how="cross")
    if outages is not None:
        frame = frame.merge(outages, on=schemas.KEY, how="left")
    else:
        frame["outage_mw"] = np.nan

    frame["area_code"] = frame["area"].map({area: i for i, area in enumerate(config.areas)})
    frame["weekday"] = frame["delivery_date"].dt.weekday
    frame["month"] = frame["delivery_date"].dt.month
    frame["is_weekend"] = (frame["weekday"] >= 5).astype(int)
    frame = frame.sort_values(schemas.KEY).reset_index(drop=True)

    labelled = label_grid(frame[schemas.KEY], events)
    frame = frame.merge(labelled, on=schemas.KEY, how="left")

    frame["hour_congested_hist"] = lagged_rolling(
        frame, "congested", ["area", "hour"], days, config.features.event_lag_days, "sum"
    )
    daily = frame.groupby(["delivery_date", "area"], as_index=False)["congested"].sum()
    daily = daily.rename(columns={"congested": "congested_hours"})
    daily["congested_hours_hist"] = lagged_rolling(
        daily, "congested_hours", ["area"], days, config.features.event_lag_days, "sum"
    )
    frame = frame.merge(daily, on=["delivery_date", "area"], how="left")
    frame = frame.drop(columns=["congested_hours"])

    if label_end is None:
        frame["target"] = np.nan
    else:
        frame["target"] = frame["congested"].where(frame["delivery_date"] <= label_end)
    return frame.drop(columns=["congested"])
