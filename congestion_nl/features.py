from __future__ import annotations

import numpy as np
import pandas as pd

from . import schemas
from .config import Config
from .labels import expand_quarters, label_grid
from .rolling import lagged_rolling

FEATURE_COLUMNS = [
    "area_code",
    "area_lat",
    "area_lon",
    "operator_code",
    "feed_in_area",
    "quarter",
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
    "area_yesterday",
    "area_week",
    "area_month",
    "area_slot_week",
    "run_days",
    "days_since_request",
    "area_age_days",
    "mw_week",
    "cell_yesterday",
    "cell_week",
    "cell_active_areas",
    "operator_yesterday",
    "operator_active_areas",
]

# the area's own history is also counted per request direction
DIRECTIONS = {"feed-in": "feed", "offtake": "off"}
DIRECTION_HISTORY = [
    "area_yesterday",
    "area_week",
    "area_month",
    "mw_week",
    "run_days",
    "days_since_request",
    "area_age_days",
    "area_slot_week",
]
FEATURE_COLUMNS += [f"{column}_{suffix}" for suffix in DIRECTIONS.values() for column in DIRECTION_HISTORY]


def area_table(locations: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Return one row per located area with requests, numbered from north-west to south-east."""
    main = events.groupby("area").agg(
        operator=("operator", lambda s: s.mode().iloc[0]),
        feed_in_area=("direction", lambda s: int((s == "feed-in").mean() >= 0.5)),
    )
    areas = locations.set_index("area")[["lat", "lon", "cell"]].join(main, how="inner").reset_index()
    areas = areas.rename(columns={"lat": "area_lat", "lon": "area_lon"})
    areas = areas.sort_values(["area_lat", "area_lon"], ascending=[False, True]).reset_index(drop=True)
    areas["area_code"] = np.arange(1, len(areas) + 1)
    areas["operator_code"] = areas["operator"].astype("category").cat.codes
    return areas


def _trailing(daily: pd.DataFrame, value: str, by: str, days: int) -> pd.Series:
    """Sum a daily value per group over the given number of days before each day."""
    return daily.groupby(by)[value].transform(lambda s: s.shift(1).rolling(days, min_periods=1).sum())


def known_quarters(events: pd.DataFrame, issue_hour: int) -> pd.DataFrame:
    """Return request quarter-hours that were announced by the issue time of the next day's forecast."""
    quarters = expand_quarters(events, ["event_id", "announced_at", "mw"])
    return quarters[quarters["announced_at"] < quarters["delivery_date"] + pd.Timedelta(hours=issue_hour)]


def daily_history(config: Config, events: pd.DataFrame, areas: pd.DataFrame, last_day: pd.Timestamp) -> pd.DataFrame:
    """Build per-area daily history features for every delivery day, using only earlier days."""
    settings = config.features
    seen = known_quarters(events, settings.issue_hour)
    key = ["delivery_date", "area"]
    counts = seen.drop_duplicates(subset=schemas.KEY).groupby(key).size().rename("known")
    mw = seen.drop_duplicates(subset=key + ["event_id"]).groupby(key)["mw"].sum().rename("known_mw")

    first = seen["delivery_date"].min() if len(seen) else config.grid.start_date
    dates = pd.date_range(min(first, config.grid.start_date), last_day, freq="D")
    daily = pd.MultiIndex.from_product([dates, areas["area"]], names=key).to_frame(index=False)
    daily = daily.merge(counts, on=key, how="left").merge(mw, on=key, how="left")
    daily[["known", "known_mw"]] = daily[["known", "known_mw"]].fillna(0)
    daily = daily.merge(areas[["area", "cell", "operator"]], on="area")
    daily = daily.sort_values(["area", "delivery_date"]).reset_index(drop=True)

    daily["area_yesterday"] = _trailing(daily, "known", "area", 1)
    daily["area_week"] = _trailing(daily, "known", "area", settings.history_days)
    daily["area_month"] = _trailing(daily, "known", "area", settings.long_history_days)
    daily["mw_week"] = _trailing(daily, "known_mw", "area", settings.history_days)

    daily["any"] = daily["known"].gt(0).astype(int)
    breaks = (daily["any"] == 0).astype(int).groupby(daily["area"]).cumsum()
    daily["run_days"] = daily.groupby(["area", breaks])["any"].cumsum().groupby(daily["area"]).shift(1)
    last = daily["delivery_date"].where(daily["any"] == 1)
    last = last.groupby(daily["area"]).ffill().groupby(daily["area"]).shift(1)
    daily["days_since_request"] = (daily["delivery_date"] - last).dt.days
    first_seen = daily["delivery_date"].where(daily["any"] == 1).groupby(daily["area"]).transform("min")
    daily["area_age_days"] = (daily["delivery_date"] - first_seen).dt.days.where(daily["delivery_date"] > first_seen)

    for group in ("cell", "operator"):
        totals = daily.groupby(["delivery_date", group], as_index=False)[["known", "any"]].sum()
        totals = totals.sort_values([group, "delivery_date"]).reset_index(drop=True)
        totals[f"{group}_yesterday"] = _trailing(totals, "known", group, 1)
        totals[f"{group}_active_areas"] = _trailing(totals, "any", group, 1)
        if group == "cell":
            totals["cell_week"] = _trailing(totals, "known", group, settings.history_days)
        daily = daily.merge(totals.drop(columns=["known", "any"]), on=["delivery_date", group], how="left")
    return daily.drop(columns=["known", "known_mw", "any", "cell", "operator"])


def slot_history(events: pd.DataFrame, settings) -> pd.DataFrame:
    """Count, per area and quarter-hour, the earlier days in the short window with a known request in that slot."""
    seen = known_quarters(events, settings.issue_hour).drop_duplicates(subset=schemas.KEY)[schemas.KEY]
    shifted = [
        seen.assign(delivery_date=seen["delivery_date"] + pd.Timedelta(days=d))
        for d in range(1, settings.history_days + 1)
    ]
    return pd.concat(shifted).groupby(schemas.KEY).size().rename("area_slot_week").reset_index()


def add_direction_history(
    config: Config, frame: pd.DataFrame, events: pd.DataFrame, areas: pd.DataFrame, last_day: pd.Timestamp
) -> pd.DataFrame:
    """Add the area's own history features counted separately for feed-in and offtake requests."""
    for direction, suffix in DIRECTIONS.items():
        chosen = events[events["direction"] == direction]
        if chosen.empty:
            # an all-missing column cannot be binned by the classifier, so a direction without requests is constant
            frame[[f"{column}_{suffix}" for column in DIRECTION_HISTORY]] = 0.0
            continue
        history = [column for column in DIRECTION_HISTORY if column != "area_slot_week"]
        daily = daily_history(config, chosen, areas, last_day)[["delivery_date", "area"] + history]
        frame = frame.merge(daily.rename(columns={c: f"{c}_{suffix}" for c in history}), on=["delivery_date", "area"], how="left")
        slots = slot_history(chosen, config.features).rename(columns={"area_slot_week": f"area_slot_week_{suffix}"})
        frame = frame.merge(slots, on=schemas.KEY, how="left")
        frame[f"area_slot_week_{suffix}"] = frame[f"area_slot_week_{suffix}"].fillna(0)
    return frame


def build_dataset(
    config: Config,
    sdac: pd.DataFrame,
    forecasts: pd.DataFrame,
    events: pd.DataFrame,
    areas: pd.DataFrame,
    label_end: pd.Timestamp | None,
) -> pd.DataFrame:
    """Join inputs and build features for active areas, with the target set for dates up to label_end."""
    settings = config.features

    market = sdac.merge(forecasts, on=schemas.MARKET_KEY, how="inner")
    market = market[market["delivery_date"] >= config.grid.start_date]
    market = market.sort_values(schemas.MARKET_KEY).reset_index(drop=True)
    market["residual_load_mw"] = (
        market["load_forecast_mw"] - market["wind_forecast_mw"] - market["solar_forecast_mw"]
    )
    market["price_hist_mean"] = lagged_rolling(
        market, "price_eur_mwh", ["quarter"], settings.history_days, settings.price_lag_days, "mean"
    )
    market["hour"] = market["quarter"] // 4

    # an area is forecast while it has had a known request within the active window
    daily = daily_history(config, events, areas, market["delivery_date"].max())
    daily = daily[daily["days_since_request"] <= settings.active_days]
    daily = daily.merge(areas.drop(columns=["operator"]), on="area")

    frame = market.merge(daily, on="delivery_date", how="inner")
    frame["weekday"] = frame["delivery_date"].dt.weekday
    frame["month"] = frame["delivery_date"].dt.month
    frame["is_weekend"] = (frame["weekday"] >= 5).astype(int)
    frame = frame.merge(slot_history(events, settings), on=schemas.KEY, how="left")
    frame["area_slot_week"] = frame["area_slot_week"].fillna(0)
    frame = add_direction_history(config, frame, events, areas, market["delivery_date"].max())
    frame = frame.sort_values(schemas.KEY).reset_index(drop=True)

    labelled = label_grid(frame[schemas.KEY], events)
    frame = frame.merge(labelled, on=schemas.KEY, how="left")
    if label_end is None:
        frame["target"] = np.nan
    else:
        frame["target"] = frame["congested"].where(frame["delivery_date"] <= label_end)
    numeric = [column for column in FEATURE_COLUMNS if frame[column].dtype.kind == "f"]
    frame[numeric] = frame[numeric].astype("float32")
    return frame.drop(columns=["congested"])
