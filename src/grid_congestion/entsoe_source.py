"""Fetch SDAC prices, net position and system forecasts from ENTSO-E for NL."""

from __future__ import annotations

import pandas as pd

from . import schemas

AREA = "NL"


def build_client(api_key: str):
    """Construct an ENTSO-E pandas client from an API key."""
    from entsoe import EntsoePandasClient

    return EntsoePandasClient(api_key=api_key)


def _hourly_frame(series: pd.Series, column: str, end: pd.Timestamp) -> pd.DataFrame:
    """Resample a tz-aware series to hourly means, excluding the end boundary."""
    series = series[series.index < end]
    hourly = series.resample("1h").mean()
    frame = hourly.rename(column).reset_index()
    frame = frame.rename(columns={frame.columns[0]: "timestamp"})
    frame["delivery_date"] = frame["timestamp"].dt.normalize().dt.tz_localize(None)
    frame["hour"] = frame["timestamp"].dt.hour
    return frame[schemas.MARKET_KEY + [column]]


def fetch_sdac(client, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch day-ahead prices and net position for NL and merge them hourly."""
    price = client.query_day_ahead_prices(AREA, start=start, end=end)
    position = client.query_net_position(AREA, start=start, end=end, dayahead=True)

    price_frame = _hourly_frame(price, "price_eur_mwh", end)
    position_frame = _hourly_frame(position, "net_position_mw", end)
    merged = price_frame.merge(position_frame, on=schemas.MARKET_KEY, how="outer")
    return merged.sort_values(schemas.MARKET_KEY).reset_index(drop=True)


def fetch_forecasts(client, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch load, wind and solar forecasts for NL and merge them hourly."""
    load = client.query_load_forecast(AREA, start=start, end=end)["Forecasted Load"]
    renewables = client.query_wind_and_solar_forecast(AREA, start=start, end=end)
    wind = renewables.get("Wind Onshore", 0) + renewables.get("Wind Offshore", 0)
    solar = renewables.get("Solar", 0)

    load_frame = _hourly_frame(load, "load_forecast_mw", end)
    wind_frame = _hourly_frame(wind, "wind_forecast_mw", end)
    solar_frame = _hourly_frame(solar, "solar_forecast_mw", end)

    merged = load_frame.merge(wind_frame, on=schemas.MARKET_KEY, how="outer")
    merged = merged.merge(solar_frame, on=schemas.MARKET_KEY, how="outer")
    return merged.sort_values(schemas.MARKET_KEY).reset_index(drop=True)
