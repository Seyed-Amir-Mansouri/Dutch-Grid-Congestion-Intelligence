import dataclasses
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from congestion_nl import estimator, pipeline, schemas, validate
from congestion_nl.labels import area_day_labels
from congestion_nl.features import FEATURE_COLUMNS
from congestion_nl.synthetic import generate


@pytest.fixture
def trained(config):
    """Generate inputs, build the dataset and train a model."""
    generate(config, start="2025-01-01", days=230, seed=11)
    inputs = pipeline.load_inputs(config)
    label_end = pd.Timestamp("2025-07-19")
    dataset = pipeline.build(config, inputs, label_end)
    bundle, report = pipeline.train(config, dataset)
    return inputs, dataset, bundle, report


def test_training_reports_holdout_scores(trained):
    """Holdout report contains valid scores and a model version."""
    _, _, bundle, report = trained
    holdout = report["holdout"]
    assert holdout["n"] > 0
    assert 0.0 <= holdout["brier"] <= 1.0
    assert holdout["roc_auc"] > 0.5
    assert bundle["model_version"].startswith("hgb-")


def test_model_round_trip_keeps_predictions(trained, config, tmp_path):
    """A saved and reloaded model gives identical probabilities."""
    _, dataset, bundle, _ = trained
    path = estimator.save(bundle, tmp_path / "models")
    assert path.exists()
    reloaded = estimator.load(tmp_path / "models")
    rows = dataset.head(50)
    assert np.allclose(estimator.predict(bundle, rows), estimator.predict(reloaded, rows))


def test_predict_day_matches_schema_and_threshold(trained, config):
    """Prediction rows follow the ledger schemas and area-days follow the day threshold."""
    inputs, _, bundle, _ = trained
    day = pd.Timestamp("2025-07-25")
    dataset = pipeline.build(config, inputs, None)
    issued = datetime(2025, 7, 24, 11, 0, tzinfo=timezone.utc)
    predictions = pipeline.predict_day(config, bundle, dataset, day, issued)

    assert list(predictions.columns) == schemas.PREDICTION_COLUMNS
    assert len(predictions) == (dataset["delivery_date"] == day).sum()
    assert predictions["risk_probability"].between(0, 1).all()
    areas = pipeline.area_forecast(bundle, predictions)
    assert list(areas.columns) == schemas.AREA_DAY_COLUMNS
    assert len(areas) == predictions["area"].nunique()
    assert (areas["day_flag"] == (areas["peak_probability"] >= bundle["day_threshold"]).astype(int)).all()


def test_predict_unknown_day_raises(trained, config):
    """Predicting a day without inputs raises an error."""
    inputs, _, bundle, _ = trained
    dataset = pipeline.build(config, inputs, None)
    with pytest.raises(ValueError):
        pipeline.predict_day(config, bundle, dataset, pd.Timestamp("1999-01-01"))


def test_validation_uses_latest_prediction_and_settled_days(trained, config):
    """Validation keeps the latest issue per key and only settled days."""
    inputs, _, bundle, _ = trained
    dataset = pipeline.build(config, inputs, None)
    day = pd.Timestamp("2025-07-25")
    early = pipeline.area_forecast(bundle, pipeline.predict_day(
        config, bundle, dataset, day, datetime(2025, 7, 24, 9, tzinfo=timezone.utc)
    ))
    late = pipeline.area_forecast(bundle, pipeline.predict_day(
        config, bundle, dataset, day, datetime(2025, 7, 24, 12, tzinfo=timezone.utc)
    ))
    ledger = pd.concat([early, late], ignore_index=True)
    ledger["delivery_date"] = pd.to_datetime(ledger["delivery_date"])
    ledger["issued_at"] = pd.to_datetime(ledger["issued_at"], utc=True)

    settled = validate.latest_predictions(ledger, pd.Timestamp("2025-07-25"))
    assert len(settled) == len(early)
    assert (settled["issued_at"] == pd.Timestamp("2025-07-24 12:00", tz="UTC")).all()

    unsettled = validate.latest_predictions(ledger, pd.Timestamp("2025-07-24"))
    assert unsettled.empty


def test_evaluate_flags_low_skill(trained, config):
    """The skill alert fires only when skill is below the minimum."""
    inputs, _, bundle, _ = trained
    dataset = pipeline.build(config, inputs, None)
    days = pd.date_range("2025-07-25", "2025-08-05")
    issued = datetime(2025, 7, 24, 11, tzinfo=timezone.utc)
    ledger = pd.concat(
        [pipeline.area_forecast(bundle, pipeline.predict_day(config, bundle, dataset, day, issued)) for day in days],
        ignore_index=True,
    )
    ledger["delivery_date"] = pd.to_datetime(ledger["delivery_date"])
    ledger["issued_at"] = pd.to_datetime(ledger["issued_at"], utc=True)

    lenient = dataclasses.replace(config.validation, min_skill=-10.0)
    result = validate.evaluate(
        dataclasses.replace(config, validation=lenient),
        ledger,
        inputs.events,
        pd.Timestamp("2025-08-05"),
        pd.Timestamp("2025-08-06"),
    )
    assert result["alert"] is False

    strict = dataclasses.replace(config.validation, min_skill=1.5)
    result = validate.evaluate(
        dataclasses.replace(config, validation=strict),
        ledger,
        inputs.events,
        pd.Timestamp("2025-08-05"),
        pd.Timestamp("2025-08-06"),
    )
    assert result["alert"] is True
    assert result["window_n"] == len(ledger)


def test_backtest_covers_the_last_days(trained, config):
    """The backtest scores area-days for each of the last test days."""
    _, dataset, _, report = trained
    labelled = dataset[dataset["target"].notna()]
    last = sorted(labelled["delivery_date"].unique())[-config.model.test_days:]
    assert [pd.Timestamp(d["date"]) for d in report["holdout_days"]] == list(last)
    assert all(d["correct"] + d["wrong"] == d["flagged"] for d in report["holdout_days"])
    assert report["holdout"]["n"] == sum(d["n"] for d in report["holdout_days"])


def test_backtest_ignores_labels_not_known_at_issue_time(trained, config):
    """Changing the labels of the delivery day and the day before leaves that day's forecast unchanged."""
    _, dataset, _, _ = trained
    labelled = dataset[dataset["target"].notna()].reset_index(drop=True)
    last = labelled["delivery_date"].max()
    base = estimator.backtest(labelled, FEATURE_COLUMNS, config.model, 1)
    changed = labelled.copy()
    recent = changed["delivery_date"] >= last - pd.Timedelta(days=1)
    changed.loc[recent, "target"] = 1 - changed.loc[recent, "target"]
    after = estimator.backtest(changed, FEATURE_COLUMNS, config.model, 1)
    assert np.allclose(base["risk_probability"], after["risk_probability"])


def test_label_cutoff_stops_at_the_gopacs_download(config):
    """Days after the last known announcement are not labelled."""
    events = pd.DataFrame({"announced_at": pd.to_datetime(["2026-10-01 09:00", "2026-10-16 08:00"])})
    assert pipeline.label_cutoff(config, events, pd.Timestamp("2026-10-22")) == pd.Timestamp("2026-10-15")
    assert pipeline.label_cutoff(config, events, pd.Timestamp("2026-10-10")) == pd.Timestamp("2026-10-09")


def test_area_days_summarise_the_risky_window():
    """An area-day takes its peak probability and the span of quarters at or above the day threshold."""
    rows = pd.DataFrame({
        "delivery_date": pd.Timestamp("2026-10-20"),
        "area": ["A"] * 4 + ["B"] * 4,
        "quarter": [40, 41, 42, 43] * 2,
        "risk_probability": [0.1, 0.35, 0.5, 0.2, 0.05, 0.1, 0.2, 0.1],
        "target": [0, 1, 1, 0, 0, 0, 0, 0],
    })
    days = estimator.area_days(rows, 0.3).set_index("area")
    assert days.loc["A", "peak_probability"] == 0.5
    assert days.loc["A", ["window_start", "window_end", "day_flag", "target"]].tolist() == ["10:15", "10:45", 1, 1]
    assert days.loc["B", "day_flag"] == 0 and pd.isna(days.loc["B", "window_start"])
    assert days.loc["B", "expected_quarters"] == pytest.approx(0.45)


def test_excluded_requests_are_dropped():
    """Requests of the listed operator announced inside the window are removed, others are kept."""
    events = pd.DataFrame({
        "operator": ["Liander", "Liander", "Enexis"],
        "announced_at": pd.to_datetime(["2026-10-15 08:02", "2026-10-15 12:00", "2026-10-15 08:02"]),
    })
    exclusions = pd.DataFrame({
        "operator": ["Liander"],
        "announced_from": pd.to_datetime(["2026-10-15 00:00"]),
        "announced_to": pd.to_datetime(["2026-10-15 11:15"]),
        "reason": ["incident"],
    })
    kept = pipeline.drop_excluded(events, exclusions)
    assert kept.index.tolist() == [1, 2]


def test_area_day_labels_and_untracked_areas():
    """An area-day is positive when any request touches it; requests in areas without a forecast count as untracked."""
    events = pd.DataFrame({
        "area": ["A", "C"],
        "start_time": pd.to_datetime(["2026-10-20 23:30", "2026-10-21 10:00"]),
        "end_time": pd.to_datetime(["2026-10-21 00:30", "2026-10-21 11:00"]),
    })
    keys = pd.DataFrame({"delivery_date": pd.to_datetime(["2026-10-20", "2026-10-21", "2026-10-21"]), "area": ["A", "A", "B"]})
    assert area_day_labels(keys, events)["congested"].tolist() == [1, 1, 0]
    assert validate.untracked(keys, events) == 1


def test_direction_history_splits_the_area_history(config):
    """Feed-in and offtake history add up to the area's history for any direction."""
    generate(config, start="2025-01-01", days=60, seed=5)
    inputs = pipeline.load_inputs(config)
    events = inputs.events.copy()
    events.loc[events.index[::2], "direction"] = "offtake"
    dataset = pipeline.build(config, dataclasses.replace(inputs, events=events), None)
    for column in ("area_yesterday", "area_week", "area_slot_week"):
        split = dataset[f"{column}_feed"].fillna(0) + dataset[f"{column}_off"].fillna(0)
        assert np.allclose(split, dataset[column].fillna(0))
        assert dataset[f"{column}_off"].fillna(0).sum() > 0
