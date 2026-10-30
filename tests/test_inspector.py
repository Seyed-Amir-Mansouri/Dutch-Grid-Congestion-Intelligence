import numpy as np
import pandas as pd

from congestion_nl import inspector


def _past(days=600, areas=("A", "B"), seed=0):
    """Walk-forward scores and targets for a few areas, where high scores are usually right."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days)
    rows = [(d, a, p, float(rng.random() < p)) for d in dates for a in areas for p in [rng.random()]]
    return pd.DataFrame(rows, columns=["delivery_date", "area", "peak", "target"])


def _dataset(past):
    """A minimal dataset with the day-level inputs, one quarter per area-day."""
    return past[["delivery_date", "area"]].assign(quarter=0, weekday=past["delivery_date"].dt.weekday, notice_q=0.0)


def test_track_record_uses_only_days_up_to_d_minus_two():
    """A wrong flag on D-1 is not yet in D's record, and is from D+1."""
    past = pd.DataFrame({
        "delivery_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
        "area": "A", "peak": [0.1, 0.1, 0.9, 0.1], "target": [0.0, 0.0, 0.0, 0.0],
    })
    record = inspector.track_record(past, 0.2).set_index("delivery_date")
    assert record.loc["2025-01-04", "wrong_7"] == 0
    longer = pd.concat([past, past.iloc[[-1]].assign(delivery_date=pd.Timestamp("2025-01-05"))])
    assert inspector.track_record(longer, 0.2).set_index("delivery_date").loc["2025-01-05", "wrong_7"] == 1


def test_unlabelled_days_do_not_count_as_wrong():
    """A flag on a day without settled labels is neither correct nor wrong."""
    past = pd.DataFrame({
        "delivery_date": pd.date_range("2025-01-01", periods=4), "area": "A",
        "peak": [0.9, 0.1, 0.1, 0.1], "target": [np.nan, 0.0, 0.0, 0.0],
    })
    record = inspector.track_record(past, 0.2).set_index("delivery_date")
    assert record.loc["2025-01-04", "wrong_7"] == 0 and record.loc["2025-01-04", "seen_7"] == 1


def test_judge_without_inspector_shows_every_flag(config):
    """With no trained inspector every flag is shown and has no confidence."""
    days = pd.DataFrame({"delivery_date": pd.to_datetime(["2025-01-10"] * 2), "area": ["A", "B"],
                         "peak_probability": [0.5, 0.1], "day_flag": [1, 0]})
    judged = inspector.judge(config, {"inspector": None, "day_threshold": 0.2}, pd.DataFrame(), days)
    assert judged["shown"].tolist() == [1, 0]
    assert judged["inspector_confidence"].isna().all()


def test_fit_learns_to_veto_low_scores(config):
    """Trained on past flags, the inspector is more confident in high-scoring flags and never shows an unflagged day."""
    past = _past()
    dataset = _dataset(past)
    veto = inspector.fit(config, past, dataset, 0.2)
    assert veto is not None and veto["trained_flags"] == int((past["peak"] >= 0.2).sum())
    table = inspector.inputs(past, dataset, 0.2)
    rows = pd.DataFrame({"delivery_date": past["delivery_date"].iloc[-2:].tolist(), "area": ["A", "B"], "peak": [0.25, 0.95]})
    rows = rows.merge(table, on=["delivery_date", "area"], how="left")
    low, high = veto["model"].predict_proba(rows[veto["columns"]])[:, 1]
    assert high > low


def test_fit_needs_enough_flags(config):
    """With too few past flags the inspector is not trained."""
    past = _past(days=10, areas=("A",))
    assert inspector.fit(config, past, _dataset(past), 0.2) is None
