from __future__ import annotations

import pandas as pd

from . import schemas

AREA = "NL"


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


def build_client(api_key: str):
    """Construct an ENTSO-E pandas client from an API key."""
    from entsoe import EntsoePandasClient

    return EntsoePandasClient(api_key=api_key)


def fetch_sdac(client, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch day-ahead prices and net position for NL per quarter-hour."""
    price = client.query_day_ahead_prices(AREA, start=start, end=end)
    position = client.query_net_position(AREA, start=start, end=end, dayahead=True)

    price_frame = quarter_values(price, "price_eur_mwh", start, end)
    position_frame = quarter_values(position, "net_position_mw", start, end)
    merged = price_frame.merge(position_frame, on=schemas.MARKET_KEY, how="outer")
    return merged.sort_values(schemas.MARKET_KEY).reset_index(drop=True)


def fetch_forecasts(client, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch load, wind and solar forecasts for NL per quarter-hour."""
    load = client.query_load_forecast(AREA, start=start, end=end)["Forecasted Load"]
    renewables = client.query_wind_and_solar_forecast(AREA, start=start, end=end)
    wind = renewables.get("Wind Onshore", 0) + renewables.get("Wind Offshore", 0)
    solar = renewables.get("Solar", 0)

    load_frame = quarter_values(load, "load_forecast_mw", start, end)
    wind_frame = quarter_values(wind, "wind_forecast_mw", start, end)
    solar_frame = quarter_values(solar, "solar_forecast_mw", start, end)

    merged = load_frame.merge(wind_frame, on=schemas.MARKET_KEY, how="outer")
    merged = merged.merge(solar_frame, on=schemas.MARKET_KEY, how="outer")
    return merged.sort_values(schemas.MARKET_KEY).reset_index(drop=True)
