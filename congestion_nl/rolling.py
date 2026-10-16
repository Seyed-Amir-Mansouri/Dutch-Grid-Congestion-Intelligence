from __future__ import annotations

import numpy as np
import pandas as pd


def lagged_rolling(
    frame: pd.DataFrame,
    value: str,
    by: list[str],
    window: int,
    lag: int,
    how: str,
) -> np.ndarray:
    """Aggregate a value over trailing days per group, shifted by a lag in days."""
    result = np.full(len(frame), np.nan)
    positions = frame.reset_index(drop=True)
    for _, group in positions.groupby(by, sort=False):
        series = group.set_index("delivery_date")[value].sort_index()
        calendar = pd.date_range(series.index.min(), series.index.max(), freq="D")
        rolled = getattr(series.reindex(calendar).rolling(window, min_periods=1), how)()
        rolled = rolled.shift(lag)
        result[group.index.to_numpy()] = rolled.reindex(group["delivery_date"]).to_numpy()
    return result
