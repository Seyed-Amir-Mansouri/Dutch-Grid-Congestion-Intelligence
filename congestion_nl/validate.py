from __future__ import annotations

import pandas as pd

from . import metrics, schemas
from .config import Config
from .labels import area_day_labels, expand_quarters


def latest_predictions(ledger: pd.DataFrame, label_end: pd.Timestamp) -> pd.DataFrame:
    """Keep the most recently issued prediction per area and day for settled delivery dates."""
    settled = ledger[ledger["delivery_date"] <= label_end]
    ordered = settled.sort_values("issued_at")
    return ordered.drop_duplicates(subset=schemas.AREA_KEY, keep="last").reset_index(drop=True)


def untracked(latest: pd.DataFrame, events: pd.DataFrame) -> int:
    """Count area-days with a request on a forecast day in an area that had no forecast."""
    days = expand_quarters(events, [])[schemas.AREA_KEY].drop_duplicates()
    days = days[days["delivery_date"].isin(latest["delivery_date"].unique())]
    seen = days.merge(latest[schemas.AREA_KEY].drop_duplicates(), on=schemas.AREA_KEY, how="left", indicator=True)
    return int((seen["_merge"] == "left_only").sum())


def evaluate(
    config: Config,
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    label_end: pd.Timestamp,
    as_of: pd.Timestamp,
) -> dict:
    """Score settled area-day predictions overall and over a recent window, and check the skill alert."""
    latest = latest_predictions(predictions, label_end)
    labelled = latest.merge(area_day_labels(latest, events), on=schemas.AREA_KEY, how="left")

    window_start = label_end - pd.Timedelta(days=config.validation.window_days - 1)
    recent = labelled[labelled["delivery_date"] >= window_start]
    overall = metrics.score(labelled["congested"], labelled["peak_probability"], flag=labelled["day_flag"])
    window = metrics.score(recent["congested"], recent["peak_probability"], flag=recent["day_flag"])
    overall["untracked"] = untracked(latest, events)
    window["untracked"] = untracked(latest[latest["delivery_date"] >= window_start], events)

    skill = window["skill"]
    alert = bool(pd.notna(skill) and skill < config.validation.min_skill)
    return {
        "as_of": as_of.isoformat(),
        "label_end": label_end.date().isoformat(),
        "window_start": window_start.date().isoformat(),
        "model_versions": ";".join(sorted(labelled["model_version"].unique())),
        **{f"window_{key}": value for key, value in window.items()},
        **{f"overall_{key}": value for key, value in overall.items()},
        "alert": alert,
    }
