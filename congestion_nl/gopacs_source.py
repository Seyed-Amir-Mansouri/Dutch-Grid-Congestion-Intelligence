from __future__ import annotations

import json
import urllib.request

import pandas as pd

from . import schemas

BASE_URL = "https://idcons.gopacs-services.eu/public/announcements"
PAGE_SIZE = 5000


def _fetch_page(page: int, size: int) -> dict:
    """Fetch one page of the public announcements endpoint."""
    url = f"{BASE_URL}?page={page}&size={size}"
    with urllib.request.urlopen(url, timeout=30) as response:
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
    if not text or text in ("all", "all other areas"):
        return False
    # "Buiten" is Dutch for "outside"
    return not text.startswith("buiten ")


def _named_areas(record: dict) -> list[str]:
    """Return the unique real area names from the buy and sell sides."""
    segments = []
    for field in ("requestAreaDescriptionSellOrders", "requestAreaDescriptionBuyOrders"):
        segments.extend((record.get(field) or "").split(";"))
    named = [s.strip() for s in segments if _is_named_area(s)]
    seen: dict[str, None] = {}
    for area in named:
        seen.setdefault(area, None)
    return list(seen)


def to_events(records: list[dict]) -> pd.DataFrame:
    """Keep congestion-management announcements and expand multi-area rows."""
    rows = []
    for record in records:
        period = record.get("problemPeriod")
        if record.get("type") != "CONGESTIONMANAGEMENT" or not period:
            continue
        start = _to_local(period["startTime"])
        end = _to_local(period["endTime"])
        for area in _named_areas(record):
            rows.append((record["id"], area, start, end))
    return pd.DataFrame(rows, columns=schemas.GOPACS_COLUMNS)
