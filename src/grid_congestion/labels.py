"""Turn GOPACS event intervals into hourly congestion labels."""

from __future__ import annotations

import pandas as pd

from . import schemas


def expand_events(events: pd.DataFrame) -> pd.DataFrame:
    """Return one row per congested clock hour for each event."""
    rows = []
    for event in events.itertuples(index=False):
        first = event.start_time.floor("h")
        last = (event.end_time - pd.Timedelta(seconds=1)).floor("h")
        for stamp in pd.date_range(first, last, freq="h"):
            rows.append((stamp.normalize(), stamp.hour, event.area))
    frame = pd.DataFrame(rows, columns=schemas.KEY)
    frame["congested"] = 1
    return frame.drop_duplicates(subset=schemas.KEY).reset_index(drop=True)


def label_grid(keys: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach a 0/1 congestion flag to each key row using expanded events."""
    hourly = expand_events(events)
    labelled = keys[schemas.KEY].merge(hourly, on=schemas.KEY, how="left")
    labelled["congested"] = labelled["congested"].fillna(0).astype(int)
    return labelled
