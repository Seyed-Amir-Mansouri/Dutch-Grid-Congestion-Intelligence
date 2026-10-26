from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request

import pandas as pd

from . import schemas

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
# values from model runs 48-71 hours before the hour they forecast, so all were out before 14:00 on the day before
LEAD = "previous_day2"
VARIABLES = {
    "temperature_2m": "temperature_c",
    "shortwave_radiation": "radiation_wm2",
    "cloud_cover": "cloud_cover_pct",
    "wind_speed_100m": "wind_100m_kmh",
    "wind_speed_10m": "wind_10m_kmh",
}
# Open-Meteo counts one call per location and 14 days; this keeps a full download under 5000 calls an hour
SECONDS_PER_CALL = 0.8


def cell_centre(cell: str, size: float) -> tuple[float, float]:
    """Return the centre of a cell named by its south-west corner, such as '52.00N 5.00E'."""
    lat, lon = cell.split()
    return float(lat[:-1]) + size / 2, float(lon[:-1]) + size / 2


def _get_json(params: dict) -> dict:
    """Fetch one Open-Meteo response."""
    with urllib.request.urlopen(f"{URL}?{urllib.parse.urlencode(params)}", timeout=120) as response:
        return json.load(response)


def fetch_cell(cell: str, start: pd.Timestamp, end: pd.Timestamp, size: float, get_json=_get_json) -> pd.DataFrame:
    """Fetch hourly local-time forecasts for one cell from start to end, both dates included."""
    lat, lon = cell_centre(cell, size)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{start:%Y-%m-%d}",
        "end_date": f"{end:%Y-%m-%d}",
        "timezone": "Europe/Amsterdam",
        "hourly": ",".join(f"{name}_{LEAD}" for name in VARIABLES),
    }
    hourly = get_json(params)["hourly"]
    frame = pd.DataFrame({column: hourly[f"{name}_{LEAD}"] for name, column in VARIABLES.items()})
    frame.insert(0, "time", pd.to_datetime(hourly["time"]))
    frame.insert(0, "cell", cell)
    return frame[schemas.WEATHER_COLUMNS]


def fetch(cells: list[str], start: pd.Timestamp, end: pd.Timestamp, size: float, get_json=_get_json, sleep=time.sleep) -> pd.DataFrame:
    """Fetch every cell in turn, pausing between cells to stay inside the free usage limits."""
    pause = SECONDS_PER_CALL * math.ceil(((end - start).days + 1) / 14)
    parts = []
    for i, cell in enumerate(cells):
        if i:
            sleep(pause)
        parts.append(fetch_cell(cell, start, end, size, get_json))
    return pd.concat(parts, ignore_index=True)
