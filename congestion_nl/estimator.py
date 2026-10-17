from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from . import metrics

MODEL_FILE = "congestion_model.joblib"


@dataclass(frozen=True)
class Settings:
    test_days: int
    threshold: float
    target_precision: float
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    random_state: int


def _estimator(settings: Settings) -> HistGradientBoostingClassifier:
    """Build an untrained classifier from the model settings."""
    return HistGradientBoostingClassifier(
        learning_rate=settings.learning_rate,
        max_iter=settings.max_iter,
        max_leaf_nodes=settings.max_leaf_nodes,
        min_samples_leaf=settings.min_samples_leaf,
        l2_regularization=settings.l2_regularization,
        random_state=settings.random_state,
    )


def _split(labelled: pd.DataFrame, days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into those before and within the last given number of days."""
    cutoff = labelled["delivery_date"].max() - pd.Timedelta(days=days)
    return labelled[labelled["delivery_date"] <= cutoff], labelled[labelled["delivery_date"] > cutoff]


def _logit(probability: np.ndarray) -> np.ndarray:
    """Return the log-odds of probabilities as a one-column matrix."""
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def calibrate(calibrator: LogisticRegression | None, probability: np.ndarray) -> np.ndarray:
    """Map raw model probabilities through the calibrator, if there is one."""
    if calibrator is None:
        return probability
    return calibrator.predict_proba(_logit(probability))[:, 1]


def _fit_threshold(fit_part: pd.DataFrame, check_part: pd.DataFrame, features: list[str], settings: Settings):
    """Fit on one part, then fit the calibrator and pick the flag threshold on the later part."""
    model = _estimator(settings).fit(fit_part[features], fit_part["target"])
    raw = model.predict_proba(check_part[features])[:, 1]
    calibrator = None
    if check_part["target"].nunique() == 2:
        calibrator = LogisticRegression().fit(_logit(raw), check_part["target"])
    threshold = metrics.choose_threshold(
        check_part["target"], calibrate(calibrator, raw), settings.target_precision, settings.threshold
    )
    return model, calibrator, threshold


def train(dataset: pd.DataFrame, features: list[str], settings: Settings) -> tuple[dict, dict]:
    """Pick the threshold on a validation window, score the holdout, then refit on all labels."""
    labelled = dataset[dataset["target"].notna()].reset_index(drop=True)
    if labelled.empty:
        raise ValueError("No labelled rows available for training")

    fit_part, test_part = _split(labelled, settings.test_days)
    report = {"train_rows": int(len(fit_part)), "test_rows": int(len(test_part))}
    threshold, calibrator = settings.threshold, None
    if len(fit_part) and len(test_part):
        early, tune = _split(fit_part, settings.test_days)
        if len(early) and len(tune):
            _, tuned_calibrator, tuned = _fit_threshold(early, tune, features, settings)
            holdout = _estimator(settings).fit(fit_part[features], fit_part["target"])
            probability = calibrate(tuned_calibrator, holdout.predict_proba(test_part[features])[:, 1])
            report["holdout_threshold"] = tuned
            report["holdout"] = metrics.score(test_part["target"], probability, tuned)
        _, calibrator, threshold = _fit_threshold(fit_part, test_part, features, settings)

    estimator = _estimator(settings).fit(labelled[features], labelled["target"])
    trained_at = datetime.now(timezone.utc)
    bundle = {
        "estimator": estimator,
        "calibrator": calibrator,
        "features": list(features),
        "threshold": threshold,
        "model_version": f"hgb-{trained_at:%Y%m%dT%H%M%SZ}",
        "trained_at": trained_at.isoformat(),
    }
    report["threshold"] = threshold
    report["model_version"] = bundle["model_version"]
    return bundle, report


def save(bundle: dict, model_dir: Path) -> Path:
    """Write the model bundle to the model directory."""
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / MODEL_FILE
    joblib.dump(bundle, path)
    return path


def load(model_dir: Path) -> dict:
    """Read the model bundle from the model directory."""
    path = model_dir / MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}; run the train command first")
    return joblib.load(path)


def predict(bundle: dict, dataset: pd.DataFrame) -> np.ndarray:
    """Return calibrated congestion probabilities for each feature row."""
    raw = bundle["estimator"].predict_proba(dataset[bundle["features"]])[:, 1]
    return calibrate(bundle.get("calibrator"), raw)
