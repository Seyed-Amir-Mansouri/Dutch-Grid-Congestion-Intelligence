from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd

from . import schemas

BASE_URL = "https://idcons.gopacs-services.eu/public/announcements"
PAGE_SIZE = 5000
PLACEHOLDERS = ("all", "all area", "all other areas", "on all voltage levels")
# "Buiten" and "Alle gebieden buiten" mean "outside" and "all areas outside"
NEGATIONS = ("buiten ", "outside ", "alle gebieden buiten")
SIDES = (("requestAreaDescriptionBuyOrders", "feed-in"), ("requestAreaDescriptionSellOrders", "offtake"))


def _fetch_page(page: int, size: int) -> dict:
    """Fetch one page of the public announcements endpoint."""
    url = f"{BASE_URL}?page={page}&size={size}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def fetch_announcements(page_size: int = PAGE_SIZE, fetch_page=_fetch_page) -> list[dict]:
    """Page through the public announcements endpoint and return raw records."""
    records: list[dict] = []
    page = 0
    while True:
        body = fetch_page(page, page_size)
        records.extend(body["content"])
        if not body["content"] or len(records) >= body["totalElements"]:
            break
        page += 1
    return records


def _to_local(epoch_ms: int) -> pd.Timestamp:
    """Convert an epoch millisecond timestamp to naive Europe/Amsterdam time."""
    utc = pd.Timestamp(epoch_ms, unit="ms", tz="UTC")
    return utc.tz_convert("Europe/Amsterdam").tz_localize(None)


def _is_named_area(segment: str) -> bool:
    """Return False for placeholders such as "all other areas" or "Buiten <area>"."""
    text = segment.strip().lower()
    if not text or text in PLACEHOLDERS:
        return False
    return not text.startswith(NEGATIONS)


def _named_areas(value: str | None) -> list[str]:
    """Return the real area names listed on one side of an announcement."""
    return [s.strip() for s in (value or "").split(";") if _is_named_area(s)]


def to_events(records: list[dict]) -> pd.DataFrame:
    """Keep congestion-management announcements as one row per named area."""
    rows = []
    for record in records:
        period = record.get("problemPeriod")
        if record.get("type") != "CONGESTIONMANAGEMENT" or not period:
            continue
        start = _to_local(period["startTime"])
        end = _to_local(period["endTime"])
        profile = record.get("requiredProblemProfileInMW") or []
        mw = max(profile) if profile else np.nan
        postcodes = ";".join(record.get("congestedZipCodes") or [])
        for field, direction in SIDES:
            for area in _named_areas(record.get(field)):
                rows.append((record["id"], area, direction, record.get("organisationName"), postcodes, mw, start, end))
    frame = pd.DataFrame(rows, columns=schemas.GOPACS_COLUMNS)
    return frame.drop_duplicates(subset=["event_id", "area"]).reset_index(drop=True)
