import dataclasses
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from congestion_nl import estimator, metrics, pipeline, schemas, validate
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
    """Prediction rows follow the ledger schema and the flag threshold."""
    inputs, _, bundle, _ = trained
    day = pd.Timestamp("2025-07-25")
    dataset = pipeline.build(config, inputs, None)
    issued = datetime(2025, 7, 24, 11, 0, tzinfo=timezone.utc)
    predictions = pipeline.predict_day(config, bundle, dataset, day, issued)

    assert list(predictions.columns) == schemas.PREDICTION_COLUMNS
    assert len(predictions) == len(inputs.cells) * 24
    assert predictions["risk_probability"].between(0, 1).all()
    expected_flag = (predictions["risk_probability"] >= bundle["threshold"]).astype(int)
    assert (predictions["risk_flag"] == expected_flag).all()


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
    early = pipeline.predict_day(
        config, bundle, dataset, day, datetime(2025, 7, 24, 9, tzinfo=timezone.utc)
    )
    late = pipeline.predict_day(
        config, bundle, dataset, day, datetime(2025, 7, 24, 12, tzinfo=timezone.utc)
    )
    ledger = pd.concat([early, late], ignore_index=True)
    ledger["delivery_date"] = pd.to_datetime(ledger["delivery_date"])
    ledger["issued_at"] = pd.to_datetime(ledger["issued_at"], utc=True)

    settled = validate.latest_predictions(ledger, pd.Timestamp("2025-07-25"))
    assert len(settled) == len(inputs.cells) * 24
    assert (settled["issued_at"] == pd.Timestamp("2025-07-24 12:00", tz="UTC")).all()

    unsettled = validate.latest_predictions(ledger, pd.Timestamp("2025-07-24"))
    assert unsettled.empty


def test_evaluate_flags_low_skill(trained, config):
    """The skill alert fires only when skill is below the minimum."""
    inputs, _, bundle, _ = trained
    dataset = pipeline.build(config, inputs, None)
    day = pd.Timestamp("2025-07-25")
    issued = datetime(2025, 7, 24, 11, tzinfo=timezone.utc)
    ledger = pipeline.predict_day(config, bundle, dataset, day, issued)
    ledger["delivery_date"] = pd.to_datetime(ledger["delivery_date"])
    ledger["issued_at"] = pd.to_datetime(ledger["issued_at"], utc=True)

    lenient = dataclasses.replace(config.validation, min_skill=-10.0)
    result = validate.evaluate(
        dataclasses.replace(config, validation=lenient),
        ledger,
        inputs.events,
        pd.Timestamp("2025-07-25"),
        pd.Timestamp("2025-07-26"),
    )
    assert result["alert"] is False

    strict = dataclasses.replace(config.validation, min_skill=1.5)
    result = validate.evaluate(
        dataclasses.replace(config, validation=strict),
        ledger,
        inputs.events,
        pd.Timestamp("2025-07-25"),
        pd.Timestamp("2025-07-26"),
    )
    assert result["alert"] is True
    assert result["window_n"] == len(inputs.cells) * 24


def test_choose_threshold_reaches_target_precision():
    """The chosen threshold is the lowest one that reaches the target precision."""
    y = np.array([0, 0, 1, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8])
    assert metrics.choose_threshold(y, p, 0.6, 0.5) == 0.2
    assert metrics.choose_threshold(y, p, 1.0, 0.5) == 0.6
    assert metrics.choose_threshold(np.zeros(6), p, 0.6, 0.5) == 0.5


def test_training_reports_chosen_threshold(trained):
    """The bundle keeps the threshold picked on the most recent window."""
    _, _, bundle, report = trained
    assert report["threshold"] == bundle["threshold"]
    assert 0.0 < bundle["threshold"] <= 1.0


def test_calibrator_keeps_the_ranking(trained):
    """Calibration changes probabilities but not their order."""
    _, dataset, bundle, _ = trained
    assert bundle["calibrator"] is not None
    rows = dataset.head(500)
    raw = bundle["estimator"].predict_proba(rows[bundle["features"]])[:, 1]
    calibrated = estimator.predict(bundle, rows)
    order = np.argsort(raw, kind="stable")
    assert np.all(np.diff(calibrated[order]) >= -1e-12)
