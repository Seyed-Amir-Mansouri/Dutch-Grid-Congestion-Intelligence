from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .estimator import Settings

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.toml"


@dataclass(frozen=True)
class Paths:
    sdac: Path
    forecasts: Path
    weather: Path
    gopacs: Path
    notices: Path
    postcodes: Path
    places: Path
    csc_report: Path
    manual_locations: Path
    excluded_requests: Path
    locations: Path
    model_dir: Path
    predictions: Path
    area_predictions: Path
    validation: Path


@dataclass(frozen=True)
class Grid:
    cell_degrees: float
    start_date: pd.Timestamp
    directions: tuple[str, ...]


@dataclass(frozen=True)
class Features:
    history_days: int
    long_history_days: int
    price_lag_days: int
    issue_hour: int
    active_days: int


@dataclass(frozen=True)
class Labels:
    settle_days: int


@dataclass(frozen=True)
class Inspector:
    start_date: pd.Timestamp
    veto_cut: float


@dataclass(frozen=True)
class Validation:
    window_days: int
    min_skill: float


@dataclass(frozen=True)
class Config:
    root: Path
    paths: Paths
    grid: Grid
    features: Features
    labels: Labels
    model: Settings
    inspector: Inspector
    validation: Validation

    def resolve(self, path: Path) -> Path:
        """Return an absolute path relative to the project root."""
        return path if path.is_absolute() else self.root / path


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    """Parse a TOML config file into a Config object."""
    path = Path(path).resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    grid = raw["grid"]
    return Config(
        root=path.parent,
        paths=Paths(**{name: Path(value) for name, value in raw["paths"].items()}),
        grid=Grid(
            cell_degrees=float(grid["cell_degrees"]),
            start_date=pd.Timestamp(grid["start_date"]),
            directions=tuple(grid["directions"]),
        ),
        features=Features(**raw["features"]),
        labels=Labels(**raw["labels"]),
        model=Settings(**raw["model"]),
        inspector=Inspector(
            start_date=pd.Timestamp(raw["inspector"]["start_date"]),
            veto_cut=float(raw["inspector"]["veto_cut"]),
        ),
        validation=Validation(**raw["validation"]),
    )
