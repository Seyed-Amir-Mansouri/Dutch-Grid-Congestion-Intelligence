from __future__ import annotations

import csv
import io
import math
import re
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd

from . import schemas

STATION_PREFIX = re.compile(r"^(rs|os|ss|tennet|powersystemresource)\s+")
NAME_NOISE = re.compile(
    r"\b(invoeding|afname|congestie\w*|clc with request|\d+ ?kv|\d+ \d+i?|\d+i|odn|ldn|wls|\d+)\b"
)
CODE_NAME = re.compile(r"^([A-Z]{2,5}) Congestie")


def normalize(text: str) -> str:
    """Lowercase, strip accents and keep only letters, digits and single spaces."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _read_geonames(path: Path) -> pd.DataFrame:
    """Read the NL.txt table from a GeoNames zip download."""
    with zipfile.ZipFile(path) as archive:
        data = archive.read("NL.txt")
    return pd.read_csv(io.BytesIO(data), sep="\t", header=None, dtype=str, keep_default_na=False, quoting=3)


def load_postcodes(path: Path) -> dict[str, tuple[float, float]]:
    """Return the centroid of every four-digit postcode."""
    table = _read_geonames(path)
    return {row[1]: (float(row[9]), float(row[10])) for row in table.itertuples(index=False)}


def load_places(path: Path) -> dict[str, tuple[float, float]]:
    """Return coordinates per normalized place name, preferring the most populous match."""
    table = _read_geonames(path)
    table = table[table[6].isin(["P", "S"])]
    best: dict[str, tuple[float, float, tuple]] = {}
    for row in table.itertuples(index=False):
        rank = (row[6] == "P", int(row[14] or 0))
        for name in {row[1], row[2], *row[3].split(",")}:
            key = normalize(name)
            if key and (key not in best or rank > best[key][2]):
                best[key] = (float(row[4]), float(row[5]), rank)
    return {key: value[:2] for key, value in best.items()}


def load_station_codes(path: Path) -> dict[str, str]:
    """Map Enexis area codes to station names using the CSC call-off report."""
    raw = pd.read_excel(path, header=None, skiprows=1)[0].astype(str)
    codes = {}
    for row in csv.reader(raw):
        if len(row) < 3:
            continue
        match = re.match(r"^([A-Z]{2,5}) (.+)$", row[2])
        if match and match.group(1) not in ("OS", "RS", "SS"):
            codes[match.group(1)] = match.group(2)
    codes.setdefault("OS", "Oss")
    return codes


def load_manual(path: Path) -> pd.DataFrame:
    """Read hand-checked area locations; a blank latitude marks an excluded area."""
    return pd.read_csv(path, dtype={"area": str})


def place_name(area: str) -> str:
    """Strip station prefixes, voltage codes and direction words from an area name."""
    text = normalize(re.sub(r"\(.*?\)", "", area))
    text = STATION_PREFIX.sub("", text)
    return re.sub(r"\s+", " ", NAME_NOISE.sub(" ", text)).strip()


def _lookup(name: str, places: dict) -> tuple[float, float] | None:
    """Find a place by its full cleaned name, without spaces, or by its first words."""
    words = name.split()
    for candidate in (name, name.replace(" ", ""), " ".join(words[:2]), words[0] if words else ""):
        if candidate in places:
            return places[candidate]
    return None


def _median(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Return the coordinate-wise median of a list of points."""
    frame = pd.DataFrame(points, columns=["lat", "lon"])
    return float(frame["lat"].median()), float(frame["lon"].median())


def cell_of(lat: float, lon: float, size: float) -> str:
    """Return the id of the grid cell holding a point, named by its south-west corner."""
    south = math.floor(lat / size) * size
    west = math.floor(lon / size) * size
    return f"{south:.2f}N {west:.2f}E"


def locate_area(
    area: str,
    postcodes: list[str],
    postcode_points: dict,
    places: dict,
    codes: dict,
) -> tuple[tuple[float, float] | None, str | None]:
    """Locate one area from its postcodes, town list, station code or name."""
    points = [postcode_points[code[:4]] for code in postcodes if code[:4] in postcode_points]
    if points:
        return _median(points), "postcode"
    if "," in area:
        towns = [places[normalize(t)] for t in area.split(",") if normalize(t) in places]
        if towns:
            return _median(towns), "towns"
    match = CODE_NAME.match(area)
    if match and match.group(1) in codes:
        point = _lookup(normalize(codes[match.group(1)]), places)
        if point:
            return point, "station code"
    point = _lookup(place_name(area), places)
    if point:
        return point, "name"
    if " - " in area:
        point = _lookup(place_name(area.split(" - ")[0]), places)
        if point:
            return point, "name"
    return None, None


def locate_areas(
    events: pd.DataFrame,
    postcode_points: dict,
    places: dict,
    codes: dict,
    manual: pd.DataFrame,
    size: float,
) -> pd.DataFrame:
    """Locate every area in the events and assign it a grid cell."""
    zips = events.groupby("area")["postcodes"].apply(
        lambda values: [code for value in values.dropna() for code in str(value).split(";") if code]
    )
    checked = manual.set_index("area")
    rows = []
    for area, postcodes in zips.items():
        if area in checked.index:
            entry = checked.loc[area]
            if pd.isna(entry["lat"]):
                continue
            point, source = (float(entry["lat"]), float(entry["lon"])), "manual"
        else:
            point, source = locate_area(area, postcodes, postcode_points, places, codes)
            if point is None:
                continue
        rows.append((area, point[0], point[1], cell_of(point[0], point[1], size), source))
    return pd.DataFrame(rows, columns=schemas.LOCATION_COLUMNS)
