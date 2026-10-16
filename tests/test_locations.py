import pandas as pd
import pytest

from congestion_nl import locations

PLACES = {"waskemeer": (53.05, 6.28), "eindhoven": (51.44, 5.47), "veldhoven": (51.42, 5.40), "limmel": (50.87, 5.71)}
POSTCODES = {"6211": (50.85, 5.69), "6212": (50.84, 5.70)}
CODES = {"LIMM": "Limmel"}


def _events(rows):
    """Build an event frame with area and postcode columns."""
    return pd.DataFrame(rows, columns=["area", "postcodes"])


def test_place_name_strips_station_prefix_and_voltage():
    assert locations.place_name("RS WASKEMEER 10-2i") == "waskemeer"
    assert locations.place_name("Houten Zuid Invoeding") == "houten zuid"


def test_cell_of_names_the_south_west_corner():
    assert locations.cell_of(52.10, 5.12, 0.25) == "52.00N 5.00E"
    assert locations.cell_of(51.99, 4.99, 0.25) == "51.75N 4.75E"


def test_postcodes_take_priority_over_the_name():
    point, source = locations.locate_area("Waskemeer", ["6211AA", "6212BB"], POSTCODES, PLACES, CODES)
    assert source == "postcode"
    assert point == pytest.approx((50.845, 5.695))


def test_town_list_uses_the_median_of_known_towns():
    point, source = locations.locate_area("EINDHOVEN, VELDHOVEN, NOWHERE", [], POSTCODES, PLACES, CODES)
    assert source == "towns"
    assert point == pytest.approx((51.43, 5.435))


def test_station_code_resolves_through_the_csc_report():
    point, source = locations.locate_area("LIMM Congestie11_P", [], {}, PLACES, CODES)
    assert source == "station code"
    assert point == PLACES["limmel"]


def test_manual_rows_override_and_blank_rows_exclude():
    events = _events([("RS WASKEMEER 10-2i", None), ("Friesland", None), ("Unknown place", None)])
    manual = pd.DataFrame(
        {"area": ["RS WASKEMEER 10-2i", "Friesland"], "lat": [53.0, None], "lon": [6.3, None], "evidence": ["x", "y"]}
    )
    located = locations.locate_areas(events, POSTCODES, PLACES, CODES, manual, 0.25)
    assert located["area"].tolist() == ["RS WASKEMEER 10-2i"]
    assert located["source"].tolist() == ["manual"]
    assert located["cell"].tolist() == ["53.00N 6.25E"]
