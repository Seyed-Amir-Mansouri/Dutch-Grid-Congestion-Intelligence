from __future__ import annotations

import numpy as np
import pandas as pd

from . import schemas


def expand_events(events: pd.DataFrame) -> pd.DataFrame:
    """Return one row per congested clock hour and cell for each event."""
    if events.empty:
        return pd.DataFrame(columns=schemas.KEY + ["congested"])
    first = events["start_time"].dt.floor("h")
    last = (events["end_time"] - pd.Timedelta(seconds=1)).dt.floor("h")
    hours = ((last - first) / pd.Timedelta(hours=1)).astype(int) + 1
    stamps = np.repeat(first.to_numpy(), hours) + pd.to_timedelta(
        np.concatenate([np.arange(n) for n in hours]), unit="h"
    )
    stamps = pd.DatetimeIndex(stamps)
    frame = pd.DataFrame(
        {
            "delivery_date": stamps.normalize(),
            "hour": stamps.hour,
            "cell": np.repeat(events["cell"].to_numpy(), hours),
        }
    )
    frame["congested"] = 1
    return frame.drop_duplicates(subset=schemas.KEY).reset_index(drop=True)


def label_grid(keys: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach a 0/1 congestion flag to each key row using expanded events."""
    hourly = expand_events(events)
    labelled = keys[schemas.KEY].merge(hourly, on=schemas.KEY, how="left")
    labelled["congested"] = labelled["congested"].fillna(0).astype(int)
    return labelled
