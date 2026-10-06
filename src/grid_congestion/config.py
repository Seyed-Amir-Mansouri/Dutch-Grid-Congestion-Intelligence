"""Typed project configuration read from config.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    sdac: Path
    forecasts: Path
    outages: Path | None
    gopacs: Path
    model_dir: Path
    predictions: Path
    validation: Path


@dataclass(frozen=True)
class Features:
    history_days: int
    event_lag_days: int
    price_lag_days: int


@dataclass(frozen=True)
class Labels:
    settle_days: int


@dataclass(frozen=True)
class Model:
    test_days: int
    threshold: float
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    random_state: int


@dataclass(frozen=True)
class Validation:
    window_days: int
    min_skill: float


@dataclass(frozen=True)
class Config:
    root: Path
    paths: Paths
    areas: tuple[str, ...]
    features: Features
    labels: Labels
    model: Model
    validation: Validation

    def resolve(self, path: Path) -> Path:
        """Return an absolute path relative to the project root."""
        return path if path.is_absolute() else self.root / path


def load_config(path: str | Path = "config.toml") -> Config:
    """Parse a TOML config file into a Config object."""
    path = Path(path).resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    root = path.parent
    paths = raw["paths"]
    outages = paths.get("outages") or None

    return Config(
        root=root,
        paths=Paths(
            sdac=Path(paths["sdac"]),
            forecasts=Path(paths["forecasts"]),
            outages=Path(outages) if outages else None,
            gopacs=Path(paths["gopacs"]),
            model_dir=Path(paths["model_dir"]),
            predictions=Path(paths["predictions"]),
            validation=Path(paths["validation"]),
        ),
        areas=tuple(raw["areas"]["codes"]),
        features=Features(**raw["features"]),
        labels=Labels(**raw["labels"]),
        model=Model(**raw["model"]),
        validation=Validation(**raw["validation"]),
    )
