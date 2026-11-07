from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import estimator, ingest, inspector, schemas
from .config import Config
from .features import area_table, build_dataset, feature_columns


@dataclass(frozen=True)
class Inputs:
    sdac: pd.DataFrame
    forecasts: pd.DataFrame
    weather: pd.DataFrame
    notices: pd.DataFrame
    holidays: pd.DataFrame
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
        holidays=ingest.load_school_holidays(config.resolve(config.paths.school_holidays)),
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
        config, inputs.sdac, inputs.forecasts, inputs.weather, inputs.events, inputs.areas, label_end, inputs.notices,
        inputs.holidays,
    )


def train(config: Config, dataset: pd.DataFrame) -> tuple[dict, dict]:
    """Train the classifier on the labelled rows of a dataset, then the inspector that vetoes its flags."""
    bundle, report = estimator.train(dataset, feature_columns(config), config.model)
    cache = inspector.history_path(config)
    past = inspector.walk_forward(config, dataset, feature_columns(config), cache)
    bundle["inspector"] = inspector.fit(config, past, dataset, bundle["day_threshold"])
    veto = bundle["inspector"]
    report["inspector"] = (
        {key: veto[key] for key in ("trained_flags", "trained_correct", "cut")} if veto else "too few past flags; not trained"
    )
    return bundle, report


def rolling_test(config: Config, dataset: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Forecast each settled day from start to end with a predictor and inspector trained only on labels up to D-2."""
    labelled = dataset[dataset["target"].notna()]
    threshold, window = config.model.day_threshold, config.model.window_threshold
    cache = inspector.history_path(config)
    past = inspector.walk_forward(config, dataset, feature_columns(config), cache)
    parts = []
    for day in pd.date_range(start, end):
        rows = labelled[labelled["delivery_date"] == day]
        if rows.empty:
            continue
        cutoff = day - pd.Timedelta(days=estimator.LABEL_GAP_DAYS)
        bundle = estimator.fit_bundle(labelled[labelled["delivery_date"] <= cutoff], feature_columns(config), config.model)
        known = past.assign(target=past["target"].where(past["delivery_date"] <= cutoff))
        bundle["inspector"] = inspector.fit(config, known, dataset, threshold)
        scored = rows.assign(risk_probability=estimator.predict(bundle, rows))
        days = inspector.judge(config, bundle, dataset, estimator.area_days(scored, threshold, window))
        outcome = rows.groupby(schemas.AREA_KEY, as_index=False)["target"].max().rename(columns={"target": "had_request"})
        parts.append(days.merge(outcome, on=schemas.AREA_KEY, how="left").merge(_timing(scored, window), on=schemas.AREA_KEY))
        print(f"{day:%Y-%m-%d}: {int(days['day_flag'].sum())} flagged, {int(days['shown'].sum())} kept", flush=True)
    if not parts:
        raise ValueError("No settled days in the test window")
    columns = schemas.AREA_KEY + ["peak_probability", "day_flag", "inspector_confidence", "shown", "had_request",
                                  "window_start", "window_end", "request_quarters", "window_quarters", "inside_quarters"]
    return pd.concat(parts, ignore_index=True)[columns]


def _timing(rows: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per area-day: requested quarters, quarters in the risky window, and requested quarters inside that window."""
    bounds = rows[rows["risk_probability"] >= threshold].groupby(schemas.AREA_KEY)["quarter"].agg(first="min", last="max")
    rows = rows.join(bounds, on=schemas.AREA_KEY)
    window = rows["quarter"].between(rows["first"], rows["last"])
    rows = rows.assign(window=window, inside=window & (rows["target"] == 1))
    return rows.groupby(schemas.AREA_KEY, as_index=False).agg(
        request_quarters=("target", "sum"), window_quarters=("window", "sum"), inside_quarters=("inside", "sum"))


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


def area_forecast(
    bundle: dict,
    predictions: pd.DataFrame,
    config: Config | None = None,
    dataset: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarise quarter-hour predictions per area for the day, most likely areas first, with the inspector's verdict."""
    days = estimator.area_days(predictions, bundle["day_threshold"], config.model.window_threshold if config else None)
    if config is not None and dataset is not None:
        days = inspector.judge(config, bundle, dataset, days)
    else:
        days = days.assign(inspector_confidence=np.nan, shown=days["day_flag"])
    days["issued_at"] = predictions["issued_at"].iloc[0]
    days["model_version"] = bundle["model_version"]
    days = days.sort_values("peak_probability", ascending=False)
    return days[schemas.AREA_DAY_COLUMNS].reset_index(drop=True)
