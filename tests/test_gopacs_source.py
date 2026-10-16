import pandas as pd

from congestion_nl import gopacs_source, schemas


def _announcement(id_, type_, sell=None, buy=None, start_ms=None, end_ms=None, zips=None):
    """Build one raw announcement record as the API would return it."""
    period = None
    if start_ms is not None:
        period = {"startTime": start_ms, "endTime": end_ms}
    return {
        "id": id_,
        "type": type_,
        "organisationName": "Enexis",
        "requestAreaDescriptionSellOrders": sell,
        "requestAreaDescriptionBuyOrders": buy,
        "requiredProblemProfileInMW": [0.5, 1.5],
        "congestedZipCodes": zips or [],
        "problemPeriod": period,
    }


def _window():
    """Return a 15-minute (start_ms, end_ms) pair for 2026-01-01 12:00 UTC."""
    start_ms = int(pd.Timestamp("2026-01-01 12:00", tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp("2026-01-01 12:15", tz="UTC").timestamp() * 1000)
    return start_ms, end_ms


def test_fetch_announcements_pages_until_total_reached():
    pages = {
        0: {"content": [{"id": "a"}, {"id": "b"}], "totalElements": 3},
        1: {"content": [{"id": "c"}], "totalElements": 3},
    }

    def fake_fetch(page, size):
        return pages[page]

    records = gopacs_source.fetch_announcements(page_size=2, fetch_page=fake_fetch)
    assert [r["id"] for r in records] == ["a", "b", "c"]


def test_fetch_announcements_stops_on_empty_page():
    def fake_fetch(page, size):
        return {"content": [], "totalElements": 5}

    assert gopacs_source.fetch_announcements(page_size=2, fetch_page=fake_fetch) == []


def test_to_events_skips_free_text_and_missing_period():
    start_ms, end_ms = _window()
    records = [
        _announcement("a", "FREE_TEXT", sell="Area1", start_ms=start_ms, end_ms=end_ms),
        _announcement("b", "CONGESTIONMANAGEMENT", sell="Area1"),
    ]
    result = gopacs_source.to_events(records)
    assert result.empty
    assert list(result.columns) == schemas.GOPACS_COLUMNS


def test_to_events_takes_the_named_side_when_sell_is_specific():
    start_ms, end_ms = _window()
    records = [
        _announcement(
            "a", "CONGESTIONMANAGEMENT", sell="LIMM Congestie11_P",
            buy="Buiten LIMM Congestie11_P", start_ms=start_ms, end_ms=end_ms,
        ),
    ]
    result = gopacs_source.to_events(records)
    assert result["area"].tolist() == ["LIMM Congestie11_P"]
    assert result["direction"].tolist() == ["offtake"]


def test_to_events_takes_the_named_side_when_buy_is_specific():
    start_ms, end_ms = _window()
    records = [
        _announcement(
            "a", "CONGESTIONMANAGEMENT", sell="all other areas",
            buy="GLTK Congestie3_P", start_ms=start_ms, end_ms=end_ms,
        ),
    ]
    result = gopacs_source.to_events(records)
    assert result["area"].tolist() == ["GLTK Congestie3_P"]
    assert result["direction"].tolist() == ["feed-in"]


def test_to_events_drops_records_with_no_named_area():
    start_ms, end_ms = _window()
    records = [
        _announcement(
            "a", "CONGESTIONMANAGEMENT", sell="all", buy="Outside of Helden, Panningen",
            start_ms=start_ms, end_ms=end_ms,
        ),
    ]
    result = gopacs_source.to_events(records)
    assert result.empty


def test_to_events_splits_multiple_named_areas():
    start_ms, end_ms = _window()
    records = [
        _announcement(
            "a", "CONGESTIONMANAGEMENT", sell="Area1; Area2 ",
            start_ms=start_ms, end_ms=end_ms, zips=["6211AA", "6212BB"],
        ),
    ]
    result = gopacs_source.to_events(records)
    assert sorted(result["area"]) == ["Area1", "Area2"]
    assert (result["event_id"] == "a").all()
    assert result["start_time"].iloc[0] == pd.Timestamp("2026-01-01 13:00")
    assert result["end_time"].iloc[0] == pd.Timestamp("2026-01-01 13:15")
    assert result["postcodes"].iloc[0] == "6211AA;6212BB"
    assert result["mw"].iloc[0] == 1.5


def test_to_events_deduplicates_areas_named_on_both_sides():
    start_ms, end_ms = _window()
    records = [
        _announcement(
            "a", "CONGESTIONMANAGEMENT", sell="Area1", buy="Area1",
            start_ms=start_ms, end_ms=end_ms,
        ),
    ]
    result = gopacs_source.to_events(records)
    assert result["area"].tolist() == ["Area1"]
