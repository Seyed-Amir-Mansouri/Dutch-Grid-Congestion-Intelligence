"""Training, persistence and inference for the congestion risk classifier."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import metrics
from .config import Config
from .features import FEATURE_COLUMNS

MODEL_FILE = "congestion_model.joblib"


def _estimator(config: Config) -> HistGradientBoostingClassifier:
    """Build an untrained classifier from the model settings."""
    settings = config.model
    return HistGradientBoostingClassifier(
        learning_rate=settings.learning_rate,
        max_iter=settings.max_iter,
        max_leaf_nodes=settings.max_leaf_nodes,
        random_state=settings.random_state,
    )


def train(config: Config, dataset: pd.DataFrame) -> tuple[dict, dict]:
    """Fit on history, score the most recent days, then refit on all labels."""
    labelled = dataset[dataset["target"].notna()].reset_index(drop=True)
    if labelled.empty:
        raise ValueError("No labelled rows available for training")

    cutoff = labelled["delivery_date"].max() - pd.Timedelta(days=config.model.test_days)
    fit_part = labelled[labelled["delivery_date"] <= cutoff]
    test_part = labelled[labelled["delivery_date"] > cutoff]

    report = {"train_rows": int(len(fit_part)), "test_rows": int(len(test_part))}
    if len(fit_part) and len(test_part):
        holdout = _estimator(config).fit(fit_part[FEATURE_COLUMNS], fit_part["target"])
        probability = holdout.predict_proba(test_part[FEATURE_COLUMNS])[:, 1]
        report["holdout"] = metrics.score(
            test_part["target"], probability, config.model.threshold
        )

    estimator = _estimator(config).fit(labelled[FEATURE_COLUMNS], labelled["target"])
    trained_at = datetime.now(timezone.utc)
    bundle = {
        "estimator": estimator,
        "features": FEATURE_COLUMNS,
        "areas": list(config.areas),
        "threshold": config.model.threshold,
        "model_version": f"hgb-{trained_at:%Y%m%dT%H%M%SZ}",
        "trained_at": trained_at.isoformat(),
    }
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
    """Return congestion probabilities for each feature row."""
    return bundle["estimator"].predict_proba(dataset[bundle["features"]])[:, 1]
