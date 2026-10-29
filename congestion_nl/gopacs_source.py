from __future__ import annotations

import json
import re
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
# Enexis notices: "Periode: <start> - <end> ... Postcode: <list> Richting: Opwaarts|Neerwaarts"
NOTICE_TIME = r"(\d\d-\d\d-\d{4} \d\d:\d\d|\d{4}-\d\d-\d\d \d\d:\d\d)"
NOTICE_PERIOD = re.compile(rf"Periode: {NOTICE_TIME} - {NOTICE_TIME}")
NOTICE_POSTCODES = re.compile(r"Postcode: (.*?) Richting")
NOTICE_DIRECTION = re.compile(r"Richting: (\w+)")


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
        created = record.get("createdTimestamp")
        announced = _to_local(created) if created else pd.NaT
        for field, direction in SIDES:
            for area in _named_areas(record.get(field)):
                rows.append((record["id"], area, direction, record.get("organisationName"), postcodes, mw, start, end, announced))
    frame = pd.DataFrame(rows, columns=schemas.GOPACS_COLUMNS)
    return frame.drop_duplicates(subset=["event_id", "area"]).reset_index(drop=True)


def _notice_time(text: str) -> pd.Timestamp:
    """Read a notice time written as 16-10-2026 08:00 or, before 2025, as 2024-09-01 15:00."""
    return pd.to_datetime(text, format="%Y-%m-%d %H:%M" if text[4] == "-" else "%d-%m-%Y %H:%M")


def to_notices(records: list[dict]) -> pd.DataFrame:
    """Keep the free-text transport-restriction notices as one row each, with period, postcodes and direction."""
    rows = []
    for record in records:
        text = record.get("message") or ""
        if record.get("type") != "FREE_TEXT" or "transportrestrictie" not in text.lower():
            continue
        period, codes, direction = NOTICE_PERIOD.search(text), NOTICE_POSTCODES.search(text), NOTICE_DIRECTION.search(text)
        created = record.get("createdTimestamp")
        if not (period and codes and created):
            continue
        postcodes = sorted({p.strip().replace(" ", "") for p in codes.group(1).split(",") if p.strip()})
        rows.append((
            record["id"], record.get("organisationName"), ";".join(postcodes),
            direction.group(1) if direction else None,
            _notice_time(period.group(1)), _notice_time(period.group(2)), _to_local(created),
        ))
    return pd.DataFrame(rows, columns=schemas.NOTICE_COLUMNS)
