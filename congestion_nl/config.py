from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
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
    school_holidays: Path
    locations: Path
    model_dir: Path
    predictions: Path
    area_predictions: Path
    validation: Path
    market_estimates: Path = Path("outputs/market_estimates.csv")


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
    issue_hours: tuple[int, ...] = ()
    market_hour: int = 0
    # before the day-ahead results, the main model with the market inputs forecast for the delivery day
    estimate_market: bool = False


@dataclass(frozen=True)
class Labels:
    settle_days: int


@dataclass(frozen=True)
class Inspector:
    start_date: pd.Timestamp
    veto_cut: float
    keep_above: float = 1.0
    learn_from: float = 1.0


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
    # Drive share links or IDs of the shared data and model files, by file name
    drive: dict[str, str] = field(default_factory=dict)

    def resolve(self, path: Path) -> Path:
        """Return an absolute path relative to the project root."""
        return path if path.is_absolute() else self.root / path

    @property
    def hours(self) -> tuple[int, ...]:
        """The issue hours a forecast can be made at, the configured one first."""
        return tuple(dict.fromkeys((self.features.issue_hour, *self.features.issue_hours)))

    def at_hour(self, hour: int | None, market_estimator: bool = False) -> Config:
        """This config for a forecast issued at another hour, with its own model folder.

        With the market estimator, an hour before the day-ahead results keeps the main model and its folder.
        """
        if market_estimator:
            if hour not in self.hours or hour >= self.features.market_hour:
                raise ValueError(f"the market estimator is for issue hours before {self.features.market_hour}:00")
            if self.features.issue_hour < self.features.market_hour:
                raise ValueError("the market estimator needs a main model issued after the day-ahead results")
            return replace(self, features=replace(self.features, issue_hour=hour, estimate_market=True))
        if hour is None or hour == self.features.issue_hour:
            return self
        if hour not in self.hours:
            raise ValueError(f"issue hour {hour} is not in issue_hours {list(self.hours)}")
        return replace(self, features=replace(self.features, issue_hour=hour),
                       paths=replace(self.paths, model_dir=self.paths.model_dir / f"issue_{hour}"))


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
        features=Features(**{**raw["features"], "issue_hours": tuple(raw["features"].get("issue_hours", ()))}),
        labels=Labels(**raw["labels"]),
        model=Settings(**raw["model"]),
        inspector=Inspector(
            start_date=pd.Timestamp(raw["inspector"]["start_date"]),
            veto_cut=float(raw["inspector"]["veto_cut"]),
            keep_above=float(raw["inspector"].get("keep_above", 1.0)),
            learn_from=float(raw["inspector"].get("learn_from", 1.0)),
        ),
        validation=Validation(**raw["validation"]),
        drive={name: link for name, link in raw.get("drive", {}).items() if link},
    )
