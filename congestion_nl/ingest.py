from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import schemas


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a CSV file and check that the required columns are present."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    return frame[columns].copy()


def _normalize_market(frame: pd.DataFrame) -> pd.DataFrame:
    """Set key dtypes on a national, area-less frame."""
    frame["delivery_date"] = pd.to_datetime(frame["delivery_date"]).dt.normalize()
    frame["hour"] = frame["hour"].astype(int)
    return frame.sort_values(schemas.MARKET_KEY).reset_index(drop=True)


def _normalize_keys(frame: pd.DataFrame, areas: tuple[str, ...]) -> pd.DataFrame:
    """Set key dtypes and keep only the configured areas."""
    frame["delivery_date"] = pd.to_datetime(frame["delivery_date"]).dt.normalize()
    frame["hour"] = frame["hour"].astype(int)
    frame["area"] = frame["area"].astype(str)
    frame = frame[frame["area"].isin(areas)]
    return frame.sort_values(schemas.KEY).reset_index(drop=True)


def load_sdac(path: Path) -> pd.DataFrame:
    """Load national day-ahead clearing prices and net positions per hour."""
    frame = _read_csv(path, schemas.SDAC_COLUMNS)
    return _normalize_market(frame)


def load_forecasts(path: Path) -> pd.DataFrame:
    """Load national load, wind and solar forecasts per hour."""
    frame = _read_csv(path, schemas.FORECAST_COLUMNS)
    return _normalize_market(frame)


def load_outages(path: Path | None, areas: tuple[str, ...]) -> pd.DataFrame | None:
    """Load planned grid outages per area and hour, or return None if not configured."""
    if path is None:
        return None
    frame = _read_csv(path, schemas.OUTAGE_COLUMNS)
    return _normalize_keys(frame, areas)


def load_gopacs(path: Path, areas: tuple[str, ...]) -> pd.DataFrame:
    """Load GOPACS congestion-management events with parsed start and end times."""
    frame = _read_csv(path, schemas.GOPACS_COLUMNS)
    frame["start_time"] = pd.to_datetime(frame["start_time"])
    frame["end_time"] = pd.to_datetime(frame["end_time"])
    frame["area"] = frame["area"].astype(str)
    frame = frame[frame["area"].isin(areas)]
    frame = frame[frame["end_time"] > frame["start_time"]]
    return frame.sort_values("start_time").reset_index(drop=True)
