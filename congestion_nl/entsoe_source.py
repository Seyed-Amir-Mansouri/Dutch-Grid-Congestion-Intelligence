from __future__ import annotations

import time

import pandas as pd

from . import schemas

AREA = "NL"
PIECE_MONTHS = 3
PAUSE_SECONDS = 1


def quarter_values(series: pd.Series, column: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Put a tz-aware series on local quarter-hours, repeating an hourly value over its four quarters."""
    series = series[(series.index >= start) & (series.index < end)]
    grid = pd.date_range(start, end, freq="15min", inclusive="left")
    quarters = series.resample("15min").mean().reindex(grid).ffill(limit=3)
    frame = quarters.rename(column).rename_axis("timestamp").reset_index()
    frame["delivery_date"] = frame["timestamp"].dt.normalize().dt.tz_localize(None)
    frame["quarter"] = frame["timestamp"].dt.hour * 4 + frame["timestamp"].dt.minute // 15
    # the repeated hour on the autumn clock change is averaged into one set of quarters
    return frame.groupby(schemas.MARKET_KEY, as_index=False)[column].mean()


def _query(method, start: pd.Timestamp, end: pd.Timestamp, **kwargs):
    """Run an ENTSO-E query in three-month pieces, well inside the one-year limit and the server's time limit."""
    parts = []
    piece_start = start
    while piece_start < end:
        if parts:
            time.sleep(PAUSE_SECONDS)
        piece_end = min(piece_start + pd.DateOffset(months=PIECE_MONTHS), end)
        parts.append(method(AREA, start=piece_start, end=piece_end, **kwargs))
        piece_start = piece_end
    joined = pd.concat(parts)
    return joined[~joined.index.duplicated(keep="last")]


def build_client(api_key: str):
    """Construct an ENTSO-E pandas client from an API key."""
    from entsoe import EntsoePandasClient

    return EntsoePandasClient(api_key=api_key)


def fetch_sdac(client, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch day-ahead prices and net position for NL per quarter-hour."""
    price = _query(client.query_day_ahead_prices, start, end)
    position = _query(client.query_net_position, start, end, dayahead=True)

    price_frame = quarter_values(price, "price_eur_mwh", start, end)
    position_frame = quarter_values(position, "net_position_mw", start, end)
    merged = price_frame.merge(position_frame, on=schemas.MARKET_KEY, how="outer")
    return merged.sort_values(schemas.MARKET_KEY).reset_index(drop=True)


def fetch_forecasts(client, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch load, wind and solar forecasts for NL per quarter-hour."""
    load = _query(client.query_load_forecast, start, end)["Forecasted Load"]
    renewables = _query(client.query_wind_and_solar_forecast, start, end)
    wind = renewables.get("Wind Onshore", 0) + renewables.get("Wind Offshore", 0)
    solar = renewables.get("Solar", 0)

    load_frame = quarter_values(load, "load_forecast_mw", start, end)
    wind_frame = quarter_values(wind, "wind_forecast_mw", start, end)
    solar_frame = quarter_values(solar, "solar_forecast_mw", start, end)

    merged = load_frame.merge(wind_frame, on=schemas.MARKET_KEY, how="outer")
    merged = merged.merge(solar_frame, on=schemas.MARKET_KEY, how="outer")
    return merged.sort_values(schemas.MARKET_KEY).reset_index(drop=True)
