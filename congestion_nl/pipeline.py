from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from . import estimator, ingest, schemas
from .config import Config
from .features import FEATURE_COLUMNS, area_table, build_dataset


@dataclass(frozen=True)
class Inputs:
    sdac: pd.DataFrame
    forecasts: pd.DataFrame
    weather: pd.DataFrame
    notices: pd.DataFrame
    events: pd.DataFrame
    areas: pd.DataFrame


def drop_excluded(events: pd.DataFrame, exclusions: pd.DataFrame) -> pd.DataFrame:
    """Remove requests that an operator announced in error, as listed in the exclusions."""
    wrong = pd.Series(False, index=events.index)
    for row in exclusions.itertuples():
        wrong |= (
            (events["operator"] == row.operator)
            & (events["announced_at"] >= row.announced_from)
            & (events["announced_at"] <= row.announced_to)
        )
    return events[~wrong]


def located_events(config: Config, events: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    """Attach a cell to each event and keep the configured directions."""
    events = events[events["direction"].isin(config.grid.directions)]
    return events.merge(locations[["area", "cell"]], on="area", how="inner")


def load_inputs(config: Config) -> Inputs:
    """Read every configured input file."""
    locations = ingest.load_locations(config.resolve(config.paths.locations))
    events = ingest.load_gopacs(config.resolve(config.paths.gopacs))
    events = drop_excluded(events, ingest.load_exclusions(config.resolve(config.paths.excluded_requests)))
    events = located_events(config, events, locations)
    return Inputs(
        sdac=ingest.load_sdac(config.resolve(config.paths.sdac)),
        forecasts=ingest.load_forecasts(config.resolve(config.paths.forecasts)),
        weather=ingest.load_weather(config.resolve(config.paths.weather)),
        notices=ingest.load_notices(config.resolve(config.paths.notices)),
        events=events,
        areas=area_table(locations, events),
    )


def label_cutoff(config: Config, events: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.Timestamp:
    """Return the last delivery date whose GOPACS record is considered complete."""
    if today is None:
        today = pd.Timestamp(datetime.now(timezone.utc).date())
    settle = pd.Timedelta(days=config.labels.settle_days)
    end = today.normalize() - settle
    # days after the GOPACS download have no requests yet, which must not count as quiet days
    if len(events):
        end = min(end, events["announced_at"].max().normalize() - settle)
    return end


def build(config: Config, inputs: Inputs, label_end: pd.Timestamp | None) -> pd.DataFrame:
    """Build the feature table for every active area and quarter-hour present in the inputs."""
    return build_dataset(
        config, inputs.sdac, inputs.forecasts, inputs.weather, inputs.events, inputs.areas, label_end, inputs.notices
    )


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
    rows["issued_at"] = issued_at.isoformat()
    rows["model_version"] = bundle["model_version"]
    return rows[schemas.PREDICTION_COLUMNS].reset_index(drop=True)


def area_forecast(bundle: dict, predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarise quarter-hour predictions per area for the day, most likely areas first."""
    days = estimator.area_days(predictions, bundle["day_threshold"])
    days["issued_at"] = predictions["issued_at"].iloc[0]
    days["model_version"] = bundle["model_version"]
    days = days.sort_values("peak_probability", ascending=False)
    return days[schemas.AREA_DAY_COLUMNS].reset_index(drop=True)
