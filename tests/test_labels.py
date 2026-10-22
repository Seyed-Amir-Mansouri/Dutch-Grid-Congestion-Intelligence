import pandas as pd

from congestion_nl import schemas
from congestion_nl.labels import expand_events, label_grid


def _events(rows):
    """Build an event frame with parsed timestamps."""
    return pd.DataFrame(rows, columns=["event_id", "area", "start_time", "end_time"]).assign(
        start_time=lambda f: pd.to_datetime(f["start_time"]),
        end_time=lambda f: pd.to_datetime(f["end_time"]),
    )


def test_expand_covers_partial_quarters():
    """Partial start and end quarter-hours are both labelled."""
    events = _events([("E1", "C1", "2025-03-01 10:20", "2025-03-01 10:50")])
    quarters = expand_events(events)
    assert sorted(quarters["quarter"].tolist()) == [41, 42, 43]
    assert set(quarters["congested"]) == {1}


def test_end_exactly_on_quarter_excludes_next_quarter():
    """An event ending on a quarter boundary does not label that quarter."""
    events = _events([("E1", "C2", "2025-03-01 08:00", "2025-03-01 08:30")])
    assert expand_events(events)["quarter"].tolist() == [32, 33]


def test_overlapping_events_do_not_duplicate_rows():
    """Overlapping events yield one row per congested quarter-hour."""
    events = _events(
        [
            ("E1", "C1", "2025-03-01 10:00", "2025-03-01 10:30"),
            ("E2", "C1", "2025-03-01 10:15", "2025-03-01 10:45"),
        ]
    )
    quarters = expand_events(events)
    assert len(quarters) == len(quarters.drop_duplicates(subset=schemas.KEY))
    assert sorted(quarters["quarter"].tolist()) == [40, 41, 42]


def test_label_grid_marks_unlabelled_keys_zero():
    """Keys without an event get a zero label."""
    events = _events([("E1", "C1", "2025-03-01 10:00", "2025-03-01 10:15")])
    keys = pd.DataFrame(
        {
            "delivery_date": pd.to_datetime(["2025-03-01", "2025-03-01"]),
            "quarter": [40, 41],
            "area": ["C1", "C1"],
        }
    )
    labelled = label_grid(keys, events)
    assert labelled["congested"].tolist() == [1, 0]
