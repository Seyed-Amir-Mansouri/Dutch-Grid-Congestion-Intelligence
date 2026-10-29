import numpy as np
import pandas as pd

from congestion_nl import pipeline
from congestion_nl.features import FEATURE_COLUMNS, build_dataset
from congestion_nl.rolling import lagged_rolling
from congestion_nl.synthetic import generate


AREA = "Area North"
DAY = pd.Timestamp("2025-01-15")


def _series_frame(values, quarter=0):
    """Build a one-series daily frame for rolling tests."""
    dates = pd.date_range("2025-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"delivery_date": dates, "area": AREA, "quarter": quarter, "value": values})


def _extra_event(event_id, start, end, announced, inputs, area=AREA):
    """Build a single located GOPACS event frame for one area."""
    row = inputs.areas.set_index("area").loc[area]
    return pd.DataFrame(
        {
            "event_id": [event_id],
            "area": [area],
            "direction": ["feed-in"],
            "operator": [row["operator"]],
            "cell": [row["cell"]],
            "mw": [2.0],
            "start_time": pd.to_datetime([start]),
            "end_time": pd.to_datetime([end]),
            "announced_at": pd.to_datetime([announced]),
        }
    )


def _day(frame, day=DAY, area=AREA):
    """Return one area's rows for one delivery day, ordered by quarter-hour."""
    return frame[(frame["delivery_date"] == day) & (frame["area"] == area)].sort_values("quarter")


def _build(config, inputs, events, label_end=None):
    """Build the dataset from loaded inputs and the given events."""
    return build_dataset(config, inputs.sdac, inputs.forecasts, inputs.weather, events, inputs.areas, label_end)


def _same_features(before, after):
    """Return True when two frames carry identical feature values."""
    return all(
        np.allclose(before[c].fillna(-1).to_numpy(float), after[c].fillna(-1).to_numpy(float)) for c in FEATURE_COLUMNS
    )


def _inputs(config):
    """Generate a short synthetic history and load it."""
    generate(config, start="2025-01-01", days=20, seed=3)
    return pipeline.load_inputs(config)


def test_lagged_mean_excludes_current_day():
    """A lag of one day excludes the current day from the mean."""
    frame = _series_frame([10.0, 20.0, 30.0, 40.0])
    result = lagged_rolling(frame, "value", ["area", "quarter"], window=2, lag=1, how="mean")
    assert np.isnan(result[0])
    assert result[1] == 10.0
    assert result[2] == 15.0
    assert result[3] == 25.0


def test_lag_two_uses_only_days_up_to_t_minus_two():
    """A lag of two days uses values up to t minus two only."""
    frame = _series_frame([1.0, 2.0, 4.0, 8.0])
    result = lagged_rolling(frame, "value", ["area", "quarter"], window=1, lag=2, how="sum")
    assert np.isnan(result[1])
    assert result[2] == 1.0
    assert result[3] == 2.0


def test_request_for_the_delivery_day_is_never_a_feature(config):
    """A request for day t, even one announced days before, leaves day t's features unchanged."""
    inputs = _inputs(config)
    base = _build(config, inputs, inputs.events)
    extra = _extra_event("X1", "2025-01-15 10:00", "2025-01-15 14:00", "2025-01-12 09:00", inputs)
    changed = _build(config, inputs, pd.concat([inputs.events, extra]))
    assert _same_features(_day(base), _day(changed))


def test_yesterday_request_announced_after_issue_is_not_a_feature(config):
    """A request on day t minus one announced after the issue time leaves day t's features unchanged."""
    inputs = _inputs(config)
    base = _build(config, inputs, inputs.events)
    extra = _extra_event("X2", "2025-01-14 20:00", "2025-01-14 21:00", "2025-01-14 15:00", inputs)
    changed = _build(config, inputs, pd.concat([inputs.events, extra]))
    assert _same_features(_day(base), _day(changed))


def test_yesterday_request_counts_once_announced(config):
    """A request on day t minus one announced before the issue time feeds day t's history."""
    inputs = _inputs(config)
    events = inputs.events[inputs.events["area"] != AREA]
    older = _extra_event("X3", "2025-01-10 10:00", "2025-01-10 10:15", "2025-01-09 09:00", inputs)
    extra = _extra_event("X4", "2025-01-14 10:00", "2025-01-14 13:00", "2025-01-14 08:00", inputs)
    after = _day(_build(config, inputs, pd.concat([events, older, extra])))
    assert (after["area_yesterday"] == 12).all()
    assert (after["run_days"] == 1).all()
    assert (after["days_since_request"] == 1).all()
    assert (after["area_age_days"] == 5).all()
    assert (after["mw_week"] == 4.0).all()
    assert (after.loc[after["quarter"] == 40, "area_slot_week"] == 2).all()
    assert (after.loc[after["quarter"].between(41, 51), "area_slot_week"] == 1).all()
    assert (after.loc[after["quarter"] == 52, "area_slot_week"] == 0).all()


def test_area_without_recent_requests_is_not_forecast(config):
    """An area drops out once its last known request is older than the active window."""
    inputs = _inputs(config)
    events = inputs.events[inputs.events["area"] != AREA]
    extra = _extra_event("X5", "2025-01-02 10:00", "2025-01-02 11:00", "2025-01-01 09:00", inputs)
    short = config.features.__class__(**{**config.features.__dict__, "active_days": 5})
    frame = _build(config.__class__(**{**config.__dict__, "features": short}), inputs, pd.concat([events, extra]))
    days = frame.loc[frame["area"] == AREA, "delivery_date"].unique()
    assert pd.Timestamp("2025-01-07") in days
    assert pd.Timestamp("2025-01-08") not in days
    assert pd.Timestamp("2025-01-02") not in days


def test_rows_carry_their_cell_weather(config):
    """Each row gets the weather of its area's cell for that quarter-hour."""
    inputs = _inputs(config)
    frame = _day(_build(config, inputs, inputs.events))
    cell = inputs.areas.set_index("area").loc[AREA, "cell"]
    noon = inputs.weather[(inputs.weather["cell"] == cell) & (inputs.weather["time"] == DAY + pd.Timedelta(hours=12))]
    assert (frame.loc[frame["quarter"].between(48, 51), "radiation_wm2"] == noon["radiation_wm2"].iloc[0]).all()
    assert frame["rad_day_change"].notna().all()


def test_target_is_masked_after_label_end(config):
    """Targets are kept up to label_end and empty after it."""
    inputs = _inputs(config)
    label_end = pd.Timestamp("2025-01-05")
    dataset = _build(config, inputs, inputs.events, label_end)
    assert dataset.loc[dataset["delivery_date"] > label_end, "target"].isna().all()
    assert dataset.loc[dataset["delivery_date"] <= label_end, "target"].notna().all()


def _notice(postcode, posted, start="2025-01-15 10:00", end="2025-01-15 12:00"):
    """Build one transport-restriction notice for a postcode."""
    return pd.DataFrame({
        "notice_id": ["n1"], "operator": ["Enexis"], "postcodes": [postcode], "direction": ["Opwaarts"],
        "start_time": [pd.Timestamp(start)], "end_time": [pd.Timestamp(end)], "announced_at": [pd.Timestamp(posted)],
    })


def test_notice_known_by_issue_time_marks_its_quarters(config):
    """A notice posted before 14:00 on D-1 for a postcode the area named earlier sets the notice inputs."""
    inputs = _inputs(config)
    events = inputs.events.copy()
    events = events.assign(postcodes=np.where(events["area"] == AREA, "1234AB", None))
    frame = _day(build_dataset(config, inputs.sdac, inputs.forecasts, inputs.weather, events, inputs.areas, None,
                               _notice("1234AB", "2025-01-14 08:00")))
    assert (frame["notice_day"] == 1).all() and (frame["notice_up"] == 1).all()
    assert frame.loc[frame["notice_q"] == 1, "quarter"].tolist() == list(range(40, 48))


def test_notice_posted_after_issue_time_is_not_a_feature(config):
    """A notice posted after 14:00 on D-1 is not used for that delivery day."""
    inputs = _inputs(config)
    events = inputs.events.copy()
    events = events.assign(postcodes=np.where(events["area"] == AREA, "1234AB", None))
    frame = _day(build_dataset(config, inputs.sdac, inputs.forecasts, inputs.weather, events, inputs.areas, None,
                               _notice("1234AB", "2025-01-14 15:00")))
    assert (frame[["notice_day", "notice_q"]] == 0).all().all()
