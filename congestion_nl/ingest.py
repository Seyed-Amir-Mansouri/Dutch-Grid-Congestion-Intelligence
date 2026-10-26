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
    """Set key dtypes on a national frame."""
    frame["delivery_date"] = pd.to_datetime(frame["delivery_date"]).dt.normalize()
    frame["quarter"] = frame["quarter"].astype(int)
    return frame.groupby(schemas.MARKET_KEY, as_index=False).mean()


def load_sdac(path: Path) -> pd.DataFrame:
    """Load national day-ahead clearing prices and net positions per quarter-hour."""
    return _normalize_market(_read_csv(path, schemas.SDAC_COLUMNS))


def load_forecasts(path: Path) -> pd.DataFrame:
    """Load national load, wind and solar forecasts per quarter-hour."""
    return _normalize_market(_read_csv(path, schemas.FORECAST_COLUMNS))


def load_weather(path: Path) -> pd.DataFrame:
    """Load hourly local-time weather forecasts per grid cell."""
    frame = _read_csv(path, schemas.WEATHER_COLUMNS)
    frame["time"] = pd.to_datetime(frame["time"])
    return frame


def load_gopacs(path: Path) -> pd.DataFrame:
    """Load GOPACS congestion-management events with parsed start and end times."""
    frame = _read_csv(path, schemas.GOPACS_COLUMNS)
    frame["start_time"] = pd.to_datetime(frame["start_time"])
    frame["end_time"] = pd.to_datetime(frame["end_time"])
    frame["announced_at"] = pd.to_datetime(frame["announced_at"])
    frame["area"] = frame["area"].astype(str)
    frame = frame[frame["end_time"] > frame["start_time"]]
    return frame.sort_values("start_time").reset_index(drop=True)


def load_exclusions(path: Path) -> pd.DataFrame:
    """Load the hand-checked list of erroneous requests, by operator and announcement time window."""
    if not path.exists():
        return pd.DataFrame(columns=schemas.EXCLUSION_COLUMNS)
    frame = _read_csv(path, schemas.EXCLUSION_COLUMNS)
    frame["announced_from"] = pd.to_datetime(frame["announced_from"])
    frame["announced_to"] = pd.to_datetime(frame["announced_to"])
    return frame


def load_locations(path: Path) -> pd.DataFrame:
    """Load the located areas written by the locate command."""
    if not path.exists():
        raise FileNotFoundError(f"No area locations at {path}; run the locate command first")
    return _read_csv(path, schemas.LOCATION_COLUMNS)
