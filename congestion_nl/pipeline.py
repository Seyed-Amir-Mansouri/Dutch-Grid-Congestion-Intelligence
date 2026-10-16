from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from . import estimator, ingest, schemas
from .config import Config
from .features import FEATURE_COLUMNS, build_dataset, cell_table


@dataclass(frozen=True)
class Inputs:
    sdac: pd.DataFrame
    forecasts: pd.DataFrame
    events: pd.DataFrame
    cells: pd.DataFrame


def located_events(config: Config, events: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    """Attach a cell to each event and keep the configured directions."""
    events = events[events["direction"].isin(config.grid.directions)]
    return events.merge(locations[["area", "cell"]], on="area", how="inner")


def load_inputs(config: Config) -> Inputs:
    """Read every configured input file."""
    locations = ingest.load_locations(config.resolve(config.paths.locations))
    events = ingest.load_gopacs(config.resolve(config.paths.gopacs))
    return Inputs(
        sdac=ingest.load_sdac(config.resolve(config.paths.sdac)),
        forecasts=ingest.load_forecasts(config.resolve(config.paths.forecasts)),
        events=located_events(config, events, locations),
        cells=cell_table(locations, config.grid.cell_degrees),
    )


def label_cutoff(config: Config, today: pd.Timestamp | None = None) -> pd.Timestamp:
    """Return the last delivery date whose GOPACS record is considered complete."""
    if today is None:
        today = pd.Timestamp(datetime.now(timezone.utc).date())
    return today.normalize() - pd.Timedelta(days=config.labels.settle_days)


def build(config: Config, inputs: Inputs, label_end: pd.Timestamp | None) -> pd.DataFrame:
    """Build the feature table for every cell and hour present in the inputs."""
    return build_dataset(config, inputs.sdac, inputs.forecasts, inputs.events, inputs.cells, label_end)


def train(config: Config, dataset: pd.DataFrame) -> tuple[dict, dict]:
    """Train the classifier on the labelled rows of a dataset."""
    return estimator.train(dataset, FEATURE_COLUMNS, config.model)


def predict_day(
    config: Config,
    bundle: dict,
    dataset: pd.DataFrame,
    day: pd.Timestamp,
    issued_at: datetime | None = None,
) -> pd.DataFrame:
    """Score one delivery day and return rows ready for the prediction ledger."""
    rows = dataset[dataset["delivery_date"] == day].copy()
    if rows.empty:
        raise ValueError(f"No inputs found for delivery date {day:%Y-%m-%d}")

    issued_at = issued_at or datetime.now(timezone.utc)
    rows["risk_probability"] = estimator.predict(bundle, rows)
    rows["risk_flag"] = (rows["risk_probability"] >= bundle["threshold"]).astype(int)
    rows["issued_at"] = issued_at.isoformat()
    rows["model_version"] = bundle["model_version"]
    return rows[schemas.PREDICTION_COLUMNS].reset_index(drop=True)
