from __future__ import annotations

from pathlib import Path

import pandas as pd


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
    ledger["cell"] = ledger["cell"].astype(str)
    ledger["issued_at"] = pd.to_datetime(ledger["issued_at"], utc=True)
    return ledger


def append_history(result: dict, path: Path) -> pd.DataFrame:
    """Append one validation row to the history CSV and return the full history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(path, mode="a", header=not path.exists(), index=False)
    return pd.read_csv(path)
