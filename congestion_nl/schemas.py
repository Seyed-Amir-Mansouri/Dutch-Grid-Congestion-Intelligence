MARKET_KEY = ["delivery_date", "hour"]

KEY = ["delivery_date", "hour", "cell"]

SDAC_COLUMNS = MARKET_KEY + ["price_eur_mwh", "net_position_mw"]

FORECAST_COLUMNS = MARKET_KEY + ["load_forecast_mw", "wind_forecast_mw", "solar_forecast_mw"]

GOPACS_COLUMNS = ["event_id", "area", "direction", "operator", "postcodes", "mw", "start_time", "end_time"]

LOCATION_COLUMNS = ["area", "lat", "lon", "cell", "source"]

PREDICTION_COLUMNS = KEY + ["issued_at", "model_version", "risk_probability", "risk_flag"]
