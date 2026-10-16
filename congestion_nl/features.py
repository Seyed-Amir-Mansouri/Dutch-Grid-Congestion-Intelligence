from __future__ import annotations

import numpy as np
import pandas as pd

from . import schemas
from .config import Config
from .labels import label_grid
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

    if label_end is None:
        frame["target"] = np.nan
    else:
        frame["target"] = frame["congested"].where(frame["delivery_date"] <= label_end)
    return frame.drop(columns=["congested"])
