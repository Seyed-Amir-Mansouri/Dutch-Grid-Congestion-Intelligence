import numpy as np
import pandas as pd

from congestion_nl import pipeline
from congestion_nl.features import build_dataset
from congestion_nl.rolling import lagged_rolling
from congestion_nl.synthetic import generate


CELL = "53.00N 6.00E"


def _series_frame(values, cell=CELL, hour=0):
    """Build a one-series daily frame for rolling tests."""
    dates = pd.date_range("2025-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"delivery_date": dates, "cell": cell, "hour": hour, "value": values})


def _extra_event(event_id, cell, start, end):
    """Build a single located GOPACS event frame."""
    return pd.DataFrame(
        {
            "event_id": [event_id],
            "cell": [cell],
            "start_time": pd.to_datetime([start]),
            "end_time": pd.to_datetime([end]),
        }
    )


def _build(config, inputs, events, label_end=None):
    """Build the dataset from loaded inputs and the given events."""
    return build_dataset(config, inputs.sdac, inputs.forecasts, events, inputs.cells, label_end)


def test_lagged_mean_excludes_current_day():
    """A lag of one day excludes the current day from the mean."""
    frame = _series_frame([10.0, 20.0, 30.0, 40.0])
    result = lagged_rolling(frame, "value", ["cell", "hour"], window=2, lag=1, how="mean")
    assert np.isnan(result[0])
    assert result[1] == 10.0
    assert result[2] == 15.0
    assert result[3] == 25.0


def test_lag_two_uses_only_days_up_to_t_minus_two():
    """A lag of two days uses values up to t minus two only."""
    frame = _series_frame([1.0, 2.0, 4.0, 8.0])
    result = lagged_rolling(frame, "value", ["cell", "hour"], window=1, lag=2, how="sum")
    assert np.isnan(result[1])
    assert result[2] == 1.0
    assert result[3] == 2.0


def test_event_on_previous_day_does_not_change_features(config):
    """An event on day t minus one does not affect features for day t."""
    generate(config, start="2025-01-01", days=20, seed=3)
    inputs = pipeline.load_inputs(config)
    target_day = pd.Timestamp("2025-01-15")

    base = _build(config, inputs, inputs.events)
    extra = _extra_event("X1", CELL, "2025-01-14 00:00", "2025-01-14 23:00")
    changed = _build(config, inputs, pd.concat([inputs.events, extra]))

    columns = ["hour_congested_hist", "congested_hours_hist", "congested_hours_long"]
    before = base[base["delivery_date"] == target_day].sort_values(["cell", "hour"])
    after = changed[changed["delivery_date"] == target_day].sort_values(["cell", "hour"])
    for column in columns:
        assert np.allclose(before[column].fillna(-1), after[column].fillna(-1))


def test_event_two_days_before_changes_features(config):
    """An event on day t minus two does affect features for day t."""
    generate(config, start="2025-01-01", days=20, seed=3)
    inputs = pipeline.load_inputs(config)
    target_day = pd.Timestamp("2025-01-15")

    base = _build(config, inputs, inputs.events)
    extra = _extra_event("X2", CELL, "2025-01-13 00:00", "2025-01-13 23:00")
    changed = _build(config, inputs, pd.concat([inputs.events, extra]))

    before = base[(base["delivery_date"] == target_day) & (base["cell"] == CELL)]
    after = changed[(changed["delivery_date"] == target_day) & (changed["cell"] == CELL)]
    assert after["congested_hours_hist"].sum() > before["congested_hours_hist"].sum()


def test_target_is_masked_after_label_end(config):
    """Targets are kept up to label_end and empty after it."""
    generate(config, start="2025-01-01", days=10, seed=3)
    inputs = pipeline.load_inputs(config)
    label_end = pd.Timestamp("2025-01-05")
    dataset = _build(config, inputs, inputs.events, label_end)
    assert dataset.loc[dataset["delivery_date"] > label_end, "target"].isna().all()
    assert dataset.loc[dataset["delivery_date"] <= label_end, "target"].notna().all()
