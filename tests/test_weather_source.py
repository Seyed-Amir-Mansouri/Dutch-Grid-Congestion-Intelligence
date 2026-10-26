import pandas as pd

from congestion_nl import schemas, weather_source
from congestion_nl.features import WEATHER_FEATURES, weather_features


def _response(times, radiation):
    """Build an Open-Meteo previous-runs response for the given hours."""
    n = len(times)
    hourly = {"time": times}
    for name in weather_source.VARIABLES:
        hourly[f"{name}_{weather_source.LEAD}"] = [1.0] * n
    hourly[f"shortwave_radiation_{weather_source.LEAD}"] = radiation
    return {"hourly": hourly}


def test_cell_centre_is_half_a_cell_from_the_corner():
    """A cell named by its south-west corner is fetched at its centre."""
    assert weather_source.cell_centre("52.00N 5.25E", 0.25) == (52.125, 5.375)


def test_fetch_asks_for_the_run_two_days_before_and_keeps_local_hours():
    """Each request names the previous_day2 variables in local time and returns the weather columns."""
    seen = []

    def fake(params):
        seen.append(params)
        return _response(["2026-01-01T00:00", "2026-01-01T01:00"], [0.0, 5.0])

    pauses = []
    frame = weather_source.fetch(["52.00N 5.00E", "52.25N 5.00E"], pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01"),
                                 0.25, get_json=fake, sleep=pauses.append)
    assert list(frame.columns) == schemas.WEATHER_COLUMNS
    assert len(frame) == 4
    assert all(v.endswith("_previous_day2") for v in seen[0]["hourly"].split(","))
    assert seen[0]["timezone"] == "Europe/Amsterdam"
    assert frame.loc[frame["cell"] == "52.25N 5.00E", "radiation_wm2"].tolist() == [0.0, 5.0]
    assert len(pauses) == 1


def _hourly(cell, start, radiation):
    """Build hourly weather rows for one cell with the given radiation per hour."""
    times = pd.date_range(start, periods=len(radiation), freq="h")
    frame = pd.DataFrame({"cell": cell, "time": times, "radiation_wm2": radiation})
    for column in ("temperature_c", "cloud_cover_pct", "wind_100m_kmh", "wind_10m_kmh"):
        frame[column] = 10.0
    return frame[schemas.WEATHER_COLUMNS]


def test_weather_features_repeat_hours_and_compare_with_the_day_before():
    """An hour fills its four quarters, and changes are taken against the same quarter the day before."""
    first = [0.0] * 12 + [400.0] + [0.0] * 11
    second = [0.0] * 12 + [100.0] + [0.0] * 11
    frame = weather_features(_hourly("52.00N 5.00E", "2026-06-01", first + second))
    day2 = frame[frame["delivery_date"] == pd.Timestamp("2026-06-02")].set_index("quarter")

    assert list(frame.columns) == ["delivery_date", "quarter", "cell"] + WEATHER_FEATURES
    assert (day2.loc[48:51, "radiation_wm2"] == 100.0).all()
    assert (day2.loc[48:51, "radiation_change"] == -300.0).all()
    assert day2.loc[0, "rad_day"] == 100.0 / 24
    assert day2.loc[0, "rad_day_change"] == (100.0 - 400.0) / 24
    assert frame.loc[frame["delivery_date"] == pd.Timestamp("2026-06-01"), "radiation_change"].isna().all()


def test_weather_features_average_the_repeated_autumn_hour():
    """The hour that occurs twice on the autumn clock change becomes one set of quarters."""
    radiation = [0.0] * 25
    radiation[3] = 10.0
    weather = _hourly("52.00N 5.00E", "2026-10-25", radiation)
    hours = [0, 1, 2, 2] + list(range(3, 24))
    weather["time"] = [pd.Timestamp("2026-10-25") + pd.Timedelta(hours=h) for h in hours]
    frame = weather_features(weather).set_index("quarter")
    assert len(frame) == 96
    assert (frame.loc[8:11, "radiation_wm2"] == 5.0).all()
