MARKET_KEY = ["delivery_date", "quarter"]

KEY = ["delivery_date", "quarter", "area"]

AREA_KEY = ["delivery_date", "area"]

SDAC_COLUMNS = MARKET_KEY + ["price_eur_mwh", "net_position_mw"]

FORECAST_COLUMNS = MARKET_KEY + ["load_forecast_mw", "wind_forecast_mw", "solar_forecast_mw"]

GOPACS_COLUMNS = ["event_id", "area", "direction", "operator", "postcodes", "mw", "start_time", "end_time", "announced_at"]

WEATHER_COLUMNS = ["cell", "time", "temperature_c", "radiation_wm2", "cloud_cover_pct", "wind_100m_kmh", "wind_10m_kmh"]

LOCATION_COLUMNS =["area", "lat", "lon", "cell", "source"]

EXCLUSION_COLUMNS = ["operator", "announced_from", "announced_to", "reason"]

PREDICTION_COLUMNS = KEY + ["issued_at", "model_version", "risk_probability"]

AREA_DAY_COLUMNS = [
    "delivery_date", "area", "issued_at", "model_version",
    "peak_probability", "expected_quarters", "window_start", "window_end", "day_flag",
]
