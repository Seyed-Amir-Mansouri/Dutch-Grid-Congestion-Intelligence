from __future__ import annotations

import pandas as pd

from . import metrics, schemas
from .config import Config
from .labels import label_grid


def latest_predictions(ledger: pd.DataFrame, label_end: pd.Timestamp) -> pd.DataFrame:
    """Keep the most recently issued prediction per key for settled delivery dates."""
    settled = ledger[ledger["delivery_date"] <= label_end]
    ordered = settled.sort_values("issued_at")
    return ordered.drop_duplicates(subset=schemas.KEY, keep="last").reset_index(drop=True)


def evaluate(
    config: Config,
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    label_end: pd.Timestamp,
    as_of: pd.Timestamp,
) -> dict:
    """Score settled predictions overall and over a recent window, and check the skill alert."""
    latest = latest_predictions(predictions, label_end)
    labelled = latest.merge(label_grid(latest[schemas.KEY], events), on=schemas.KEY, how="left")

    window_start = label_end - pd.Timedelta(days=config.validation.window_days - 1)
    recent = labelled[labelled["delivery_date"] >= window_start]
    overall = metrics.score(labelled["congested"], labelled["risk_probability"], flag=labelled["risk_flag"])
    window = metrics.score(recent["congested"], recent["risk_probability"], flag=recent["risk_flag"])

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
