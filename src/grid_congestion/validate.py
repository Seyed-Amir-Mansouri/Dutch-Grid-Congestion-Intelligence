"""Score logged predictions against GOPACS congestion events as days settle."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import metrics, schemas
from .config import Config
from .labels import label_grid


def load_ledger(path: Path) -> pd.DataFrame:
    """Read the prediction ledger with parsed dates and issue times."""
    if not path.exists():
        raise FileNotFoundError(f"No prediction ledger at {path}; run the predict command first")
    ledger = pd.read_csv(path)
    ledger["delivery_date"] = pd.to_datetime(ledger["delivery_date"]).dt.normalize()
    ledger["issued_at"] = pd.to_datetime(ledger["issued_at"], utc=True)
    return ledger


def latest_predictions(ledger: pd.DataFrame, label_end: pd.Timestamp) -> pd.DataFrame:
    """Keep the most recently issued prediction per key for settled delivery dates."""
    settled = ledger[ledger["delivery_date"] <= label_end]
    ordered = settled.sort_values("issued_at")
    return ordered.drop_duplicates(subset=schemas.KEY, keep="last").reset_index(drop=True)


def evaluate(
    config: Config,
    ledger: pd.DataFrame,
    events: pd.DataFrame,
    label_end: pd.Timestamp,
    as_of: pd.Timestamp,
) -> dict:
    """Compute overall and recent-window scores and check the skill alert."""
    latest = latest_predictions(ledger, label_end)
    labelled = latest.merge(label_grid(latest[schemas.KEY], events), on=schemas.KEY, how="left")

    window_start = label_end - pd.Timedelta(days=config.validation.window_days - 1)
    recent = labelled[labelled["delivery_date"] >= window_start]
    threshold = config.model.threshold

    overall = metrics.score(labelled["congested"], labelled["risk_probability"], threshold)
    window = metrics.score(recent["congested"], recent["risk_probability"], threshold)

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


def append_history(result: dict, path: Path) -> pd.DataFrame:
    """Append one validation row to the history CSV and return the full history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([result])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)
    return pd.read_csv(path)
