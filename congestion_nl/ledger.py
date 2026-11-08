from __future__ import annotations

from pathlib import Path

import pandas as pd


def kind_of(model_version) -> str:
    """'before' for a forecast made before the day-ahead results, else 'after'."""
    return "before" if str(model_version).endswith(("+sdac_forecaster", "+before", "+market_estimator", "+forecaster")) else "after"


def append(predictions: pd.DataFrame, path: Path) -> None:
    """Append predictions to the ledger CSV, writing the header only once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, mode="a", header=not path.exists(), index=False)


def load(path: Path) -> pd.DataFrame:
    """Read the prediction ledger with parsed dates and issue times."""
    if not path.exists():
        raise FileNotFoundError(f"No prediction ledger at {path}; run the predict command first")
    ledger = pd.read_csv(path)
    ledger["delivery_date"] = pd.to_datetime(ledger["delivery_date"]).dt.normalize()
    ledger["area"] = ledger["area"].astype(str)
    ledger["issued_at"] = pd.to_datetime(ledger["issued_at"], utc=True)
    ledger["kind"] = ledger["model_version"].map(kind_of)
    return ledger


def append_history(result: dict, path: Path) -> pd.DataFrame:
    """Append one validation row to the history CSV and return the full history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    history = pd.concat([pd.read_csv(path), pd.DataFrame([result])], ignore_index=True) if path.exists() else pd.DataFrame([result])
    history.to_csv(path, index=False)
    return history
