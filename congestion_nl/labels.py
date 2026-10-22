from __future__ import annotations

import numpy as np
import pandas as pd

from . import schemas


def expand_quarters(events: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    """Return one row per event, quarter-hour and area, carrying the listed event columns."""
    if events.empty:
        return pd.DataFrame(columns=schemas.KEY + keep)
    first = events["start_time"].dt.floor("15min")
    last = (events["end_time"] - pd.Timedelta(seconds=1)).dt.floor("15min")
    quarters = ((last - first) / pd.Timedelta(minutes=15)).astype(int) + 1
    stamps = np.repeat(first.to_numpy(), quarters) + pd.to_timedelta(
        np.concatenate([np.arange(n) for n in quarters]) * 15, unit="min"
    )
    stamps = pd.DatetimeIndex(stamps)
    return pd.DataFrame(
        {
            "delivery_date": stamps.normalize(),
            "quarter": stamps.hour * 4 + stamps.minute // 15,
            "area": np.repeat(events["area"].to_numpy(), quarters),
            **{column: np.repeat(events[column].to_numpy(), quarters) for column in keep},
        }
    )


def expand_events(events: pd.DataFrame) -> pd.DataFrame:
    """Return one row per congested quarter-hour and area for each event."""
    frame = expand_quarters(events, [])
    frame["congested"] = 1
    return frame.drop_duplicates(subset=schemas.KEY).reset_index(drop=True)


def label_grid(keys: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach a 0/1 congestion flag to each key row using expanded events."""
    quarters = expand_events(events)
    labelled = keys[schemas.KEY].merge(quarters, on=schemas.KEY, how="left")
    labelled["congested"] = labelled["congested"].fillna(0).astype(int)
    return labelled
