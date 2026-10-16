import pandas as pd

from congestion_nl import schemas
from congestion_nl.labels import expand_events, label_grid


def _events(rows):
    """Build an event frame with parsed timestamps."""
    return pd.DataFrame(rows, columns=["event_id", "area", "start_time", "end_time"]).assign(
        start_time=lambda f: pd.to_datetime(f["start_time"]),
        end_time=lambda f: pd.to_datetime(f["end_time"]),
    )


def test_expand_covers_partial_hours():
    """Partial start and end hours are both labelled."""
    events = _events([("E1", "NL1", "2025-03-01 10:30", "2025-03-01 12:00")])
    hours = expand_events(events)
    assert sorted(hours["hour"].tolist()) == [10, 11]
    assert set(hours["congested"]) == {1}


def test_end_exactly_on_hour_excludes_next_hour():
    """An event ending on the hour does not label that hour."""
    events = _events([("E1", "NL2", "2025-03-01 08:00", "2025-03-01 09:00")])
    assert expand_events(events)["hour"].tolist() == [8]


def test_overlapping_events_do_not_duplicate_rows():
    """Overlapping events yield one row per congested hour."""
    events = _events(
        [
            ("E1", "NL1", "2025-03-01 10:00", "2025-03-01 12:00"),
            ("E2", "NL1", "2025-03-01 11:00", "2025-03-01 13:00"),
        ]
    )
    hours = expand_events(events)
    assert len(hours) == len(hours.drop_duplicates(subset=schemas.KEY))
    assert sorted(hours["hour"].tolist()) == [10, 11, 12]


def test_label_grid_marks_unlabelled_keys_zero():
    """Keys without an event get a zero label."""
    events = _events([("E1", "NL1", "2025-03-01 10:00", "2025-03-01 11:00")])
    keys = pd.DataFrame(
        {
            "delivery_date": pd.to_datetime(["2025-03-01", "2025-03-01"]),
            "hour": [10, 11],
            "area": ["NL1", "NL1"],
        }
    )
    labelled = label_grid(keys, events)
    assert labelled["congested"].tolist() == [1, 0]
