"""Required columns for each input file."""

KEY = ["delivery_date", "hour", "area"]

SDAC_COLUMNS = KEY + ["price_eur_mwh", "net_position_mw"]

FORECAST_COLUMNS = KEY + ["load_forecast_mw", "wind_forecast_mw", "solar_forecast_mw"]

OUTAGE_COLUMNS = KEY + ["outage_mw"]

GOPACS_COLUMNS = ["event_id", "area", "start_time", "end_time"]

PREDICTION_COLUMNS = KEY + ["issued_at", "model_version", "risk_probability", "risk_flag"]
