from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def score(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    """Return Brier, skill against climatology, AUC and flag precision and recall."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    flag = probability >= threshold
    positive_rate = y_true.mean() if len(y_true) else np.nan

    brier = brier_score_loss(y_true, probability) if len(y_true) else np.nan
    climatology = positive_rate * (1 - positive_rate) if len(y_true) else np.nan
    skill = 1 - brier / climatology if climatology and climatology > 0 else np.nan

    has_both = len(np.unique(y_true)) == 2
    auc = roc_auc_score(y_true, probability) if has_both else np.nan

    true_positive = np.sum(flag & (y_true == 1))
    predicted = flag.sum()
    actual = y_true.sum()
    precision = true_positive / predicted if predicted else np.nan
    recall = true_positive / actual if actual else np.nan

    return {
        "n": int(len(y_true)),
        "positive_rate": float(positive_rate),
        "brier": float(brier),
        "skill": float(skill),
        "roc_auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
    }
