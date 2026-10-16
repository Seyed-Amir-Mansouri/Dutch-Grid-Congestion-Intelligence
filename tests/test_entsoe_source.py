import pandas as pd

from congestion_nl import entsoe_source, schemas


class StubClient:

    def __init__(self, price, position, load, renewables):
        self.price = price
        self.position = position
        self.load = load
        self.renewables = renewables

    def query_day_ahead_prices(self, area, start, end):
        """Return the stub price series."""
        return self.price

    def query_net_position(self, area, start, end, dayahead=True):
        """Return the stub net position series."""
        return self.position

    def query_load_forecast(self, area, start, end):
        """Return the stub load forecast frame."""
        return self.load

    def query_wind_and_solar_forecast(self, area, start, end):
        """Return the stub wind and solar forecast frame."""
        return self.renewables


def _quarter_hour_index(start, periods):
    """Build a tz-aware quarter-hourly index in Europe/Amsterdam."""
    return pd.date_range(start, periods=periods, freq="15min", tz="Europe/Amsterdam")


def _hourly_index(start, periods):
    """Build a tz-aware hourly index in Europe/Amsterdam."""
    return pd.date_range(start, periods=periods, freq="1h", tz="Europe/Amsterdam")


def _local(text):
    """Parse a timestamp as tz-aware Europe/Amsterdam time."""
    return pd.Timestamp(text).tz_localize("Europe/Amsterdam")


def test_fetch_sdac_resamples_quarter_hour_prices_to_hourly():
    price_index = _quarter_hour_index("2025-06-01 00:00", 8)
    price = pd.Series([10, 20, 30, 40, 50, 60, 70, 80], index=price_index)
    position_index = _hourly_index("2025-06-01 00:00", 2)
    position = pd.Series([100, 200], index=position_index)
    client = StubClient(price, position, None, None)

    result = entsoe_source.fetch_sdac(client, _local("2025-06-01"), _local("2025-06-02"))

    assert list(result.columns) == schemas.SDAC_COLUMNS
    assert result.loc[0, "hour"] == 0
    assert result.loc[0, "price_eur_mwh"] == 25.0
    assert result.loc[1, "hour"] == 1
    assert result.loc[1, "price_eur_mwh"] == 65.0
    assert result["net_position_mw"].tolist() == [100.0, 200.0]


def test_fetch_sdac_excludes_a_row_exactly_at_the_end_boundary():
    index = _hourly_index("2025-06-01 00:00", 3)
    price = pd.Series([10, 20, 30], index=index)
    position = pd.Series([100, 200], index=index[:2])
    client = StubClient(price, position, None, None)

    result = entsoe_source.fetch_sdac(client, _local("2025-06-01"), index[2])

    assert len(result) == 2
    assert result["net_position_mw"].isna().sum() == 0


def test_fetch_forecasts_sums_onshore_and_offshore_wind():
    index = _hourly_index("2025-06-01 00:00", 2)
    load = pd.DataFrame({"Forecasted Load": [1000.0, 1100.0]}, index=index)
    renewables = pd.DataFrame(
        {"Wind Onshore": [100.0, 110.0], "Wind Offshore": [200.0, 210.0], "Solar": [0.0, 50.0]},
        index=index,
    )
    client = StubClient(None, None, load, renewables)

    result = entsoe_source.fetch_forecasts(client, _local("2025-06-01"), _local("2025-06-02"))

    assert list(result.columns) == schemas.FORECAST_COLUMNS
    assert result["wind_forecast_mw"].tolist() == [300.0, 320.0]
    assert result["solar_forecast_mw"].tolist() == [0.0, 50.0]
    assert result["load_forecast_mw"].tolist() == [1000.0, 1100.0]
