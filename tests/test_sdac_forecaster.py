import numpy as np
import pandas as pd

from congestion_nl import entsoe_source, sdac_forecaster


def _market(days: int) -> pd.DataFrame:
    """Quarter-hours of consecutive days with a net position that equals the day number."""
    dates = pd.date_range("2026-01-01", periods=days)
    frame = pd.DataFrame({"delivery_date": np.repeat(dates, 96), "quarter": np.tile(range(96), days)})
    frame["net_position_mw"] = np.repeat(np.arange(days, dtype=float), 96)
    frame["price_eur_mwh"] = frame["net_position_mw"] * 2
    return frame


def test_history_looks_only_at_earlier_days():
    t = _market(40)
    last = t["delivery_date"] == t["delivery_date"].max()
    t.loc[last, ["net_position_mw", "price_eur_mwh"]] = np.nan  # the day being forecast is not known yet
    row = sdac_forecaster.net_position_history(t).query("delivery_date == @t.delivery_date.max() and quarter == 10").iloc[0]
    assert row["np_d2"] == 37 and row["np_d3"] == 36
    assert row["np_wd4"] == np.mean([25, 18, 11])
    assert row["np_mean3"] == np.mean([37, 38, 36]) and row["np_mean7"] == np.mean(range(32, 39))
    assert row["price_d1"] == 76 and row["price_d1_mean"] == 76


def test_neighbour_weather_averages_the_points_of_each_country():
    hour = pd.Timestamp("2026-03-01 10:00")
    rows = [("DE", "54.00N 9.00E", 10.0), ("DE", "50.00N 9.00E", 20.0), ("BE", "51.00N 4.00E", 5.0)]
    weather = pd.DataFrame({"country": [r[0] for r in rows], "cell": [r[1] for r in rows], "time": hour,
                            "temperature_c": [r[2] for r in rows], "radiation_wm2": 100.0, "cloud_cover_pct": 50.0,
                            "wind_100m_kmh": 36.0, "wind_10m_kmh": 20.0})
    out = sdac_forecaster.neighbour_weather(weather).iloc[0]
    assert out["nb_DE_temperature_c"] == 15.0 and out["nb_BE_temperature_c"] == 5.0
    # northern Germany is the point at 54N only: (36 / 3.6) ** 3
    assert out["nb_DE_north_wind3"] == 1000.0


def test_only_the_net_position_gets_the_neighbour_inputs():
    plain = sdac_forecaster.inputs_for("net_position_mw")
    more = sdac_forecaster.inputs_for("net_position_mw", neighbours=True)
    assert len(more) - len(plain) == len(sdac_forecaster.HISTORY) + len(sdac_forecaster.neighbour_columns()) + 3 + len(sdac_forecaster.DERIVED)
    assert sdac_forecaster.inputs_for("price_eur_mwh", neighbours=True) == sdac_forecaster.inputs_for("price_eur_mwh")


def test_recency_halves_every_half_life():
    train = pd.DataFrame({"delivery_date": pd.to_datetime(["2026-06-01", "2026-06-01", "2026-12-01"])})
    day = pd.Timestamp("2026-12-01")
    weight = sdac_forecaster.recency(train, day)
    assert weight[2] == 1.0 and weight[0] == 0.5 ** (183 / sdac_forecaster.HALF_LIFE_DAYS)


def test_fetch_neighbour_load_gives_one_column_per_country():
    index = pd.date_range("2025-06-01", periods=4, freq="h", tz="Europe/Amsterdam")

    class Client:
        def query_load_forecast(self, area, start, end):
            return pd.DataFrame({"Forecasted Load": [float(len(area))] * 4}, index=index)

    start = index[0]
    out = entsoe_source.fetch_neighbour_load(Client(), start, start + pd.Timedelta(hours=4))
    assert list(out.columns) == ["delivery_date", "quarter", "DE_LU_load", "BE_load", "FR_load"]
    assert len(out) == 16 and out["BE_load"].iloc[0] == 2.0


def test_derived_inputs_total_the_neighbours():
    days = pd.date_range("2026-01-01", periods=400)
    t = pd.DataFrame({"delivery_date": days, "load_forecast_mw": 10.0, "wind3": 100.0, "rad": 50.0,
                      "DE_LU_load": 60.0, "BE_load": 20.0, "FR_load": 20.0})
    for c in ("DE", "BE", "FR"):
        t[f"nb_{c}_wind3"] = {"DE": 300.0, "BE": 100.0, "FR": 200.0}[c]
        t[f"nb_{c}_radiation_wm2"] = {"DE": 30.0, "BE": 60.0, "FR": 90.0}[c]
    out = sdac_forecaster.derived_inputs(t).iloc[0]
    assert out["x_wind3_sum"] == 600 and out["x_load_sum"] == 100 and out["x_nl_share"] == 0.1
    assert out["x_wind3_gap"] == 100 - 200 and out["x_rad_gap"] == 50 - 60
