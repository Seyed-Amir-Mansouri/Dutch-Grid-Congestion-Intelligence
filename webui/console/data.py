"""Read the tool's files for the web pages, cached until a file changes."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from congestion_nl import estimator, features, ingest, inspector, ledger, pipeline, schemas
from congestion_nl.config import load_config
from congestion_nl.labels import expand_quarters

CONFIG = load_config()
_cache: dict = {}


def path(name: str) -> Path:
    """Return the absolute path of a configured file."""
    return CONFIG.resolve(getattr(CONFIG.paths, name))


def model_dir(hour: int | None = None) -> Path:
    """Return the folder of the saved model for an issue hour, the configured one by default."""
    config = CONFIG.at_hour(hour)
    return config.resolve(config.paths.model_dir)


def hour_label(hour: int) -> str:
    """Whether a forecast at this hour is made after or before the day-ahead market results."""
    return "After Day-Ahead" if hour >= CONFIG.features.market_hour else "Before Day-Ahead"


def hours() -> list[dict]:
    """The issue hours with their labels, the configured one first."""
    return [{"value": h, "text": f"{h:02d}:00", "note": hour_label(h), "early": h < CONFIG.features.market_hour}
            for h in CONFIG.hours]


def _cached(name: str, files: list[Path], build):
    """Return build(), recomputed only when one of the files changes."""
    stamp = tuple(f.stat().st_mtime if f.exists() else None for f in files)
    hit = _cache.get(name)
    if hit is None or hit[0] != stamp:
        hit = (stamp, build())
        _cache[name] = hit
    return hit[1]


def events() -> pd.DataFrame:
    """GOPACS requests in the configured directions, without those announced in error."""
    def build():
        if not path("gopacs").exists():
            return pd.DataFrame(columns=schemas.GOPACS_COLUMNS)
        found = ingest.load_gopacs(path("gopacs"))
        found = pipeline.drop_excluded(found, ingest.load_exclusions(path("excluded_requests")))
        return found[found["direction"].isin(CONFIG.grid.directions)]
    return _cached("events", [path("gopacs"), path("excluded_requests")], build)


def locations() -> pd.DataFrame:
    """Every located area with its coordinates and grid cell."""
    def build():
        if not path("locations").exists():
            return pd.DataFrame(columns=schemas.LOCATION_COLUMNS)
        return ingest.load_locations(path("locations"))
    return _cached("locations", [path("locations")], build)


def request_quarters() -> pd.DataFrame:
    """One row per requested quarter-hour and area, with direction, operator and MW."""
    def build():
        return expand_quarters(events(), ["direction", "operator", "mw"])
    return _cached("quarters", [path("gopacs"), path("excluded_requests")], build)


def request_days() -> pd.DataFrame:
    """Area-days that had at least one request."""
    return request_quarters()[schemas.AREA_KEY].drop_duplicates().assign(had_request=1)


def label_end() -> pd.Timestamp | None:
    """The last delivery day whose GOPACS record is complete."""
    found = events()
    return pipeline.label_cutoff(CONFIG, found) if len(found) else None


def bundle(hour: int | None = None) -> dict | None:
    """The saved model of an issue hour, or None if it has not been trained."""
    file = model_dir(hour) / estimator.MODEL_FILE
    return _cached(f"bundle{hour}", [file], lambda: estimator.load(model_dir(hour)) if file.exists() else None)


def area_ledger() -> pd.DataFrame | None:
    """Every issued area-day forecast, or None before the first forecast."""
    file = path("area_predictions")
    return _cached("ledger", [file], lambda: ledger.load(file) if file.exists() else None)


def forecast_days() -> list[pd.Timestamp]:
    """Delivery days with a forecast, newest first."""
    found = area_ledger()
    if found is None or found.empty:
        return []
    return sorted(found["delivery_date"].unique(), reverse=True)


def status_of(row) -> str:
    """'alert' for a flag kept by the inspector, 'vetoed' for a flag it removed, else 'clear'."""
    if row["day_flag"] == 1 and row["shown"] == 1:
        return "alert"
    return "vetoed" if row["day_flag"] == 1 else "clear"


def forecast(day: pd.Timestamp) -> pd.DataFrame:
    """The latest forecast of each area for a day, with location, status and outcome when known."""
    found = area_ledger()
    rows = found[found["delivery_date"] == day].sort_values("issued_at")
    rows = rows.drop_duplicates(schemas.AREA_KEY, keep="last")
    rows = rows.merge(locations()[["area", "lat", "lon", "cell"]], on="area", how="left")
    rows["status"] = rows.apply(status_of, axis=1) if len(rows) else []
    end = label_end()
    if end is not None and day <= end:
        rows = rows.merge(request_days(), on=schemas.AREA_KEY, how="left")
        rows["had_request"] = rows["had_request"].fillna(0).astype(int)
    else:
        rows["had_request"] = pd.NA
    return rows.sort_values("peak_probability", ascending=False).reset_index(drop=True)


def rolling_test(hour: int | None = None, forecaster: bool = False) -> dict | None:
    """The test of an issue hour per 14-day window: area-days and requests, actual against the kept flags."""
    file = model_dir(hour) / (inspector.FORECASTER_TEST if forecaster else inspector.TEST_FILE)
    if not file.exists():
        return None
    scores = pd.read_csv(file, parse_dates=["delivery_date"])
    if not {"shown", "had_request"} <= set(scores.columns):
        # an older file in the same place, such as the monthly scores before they were renamed
        return None
    start, end = scores["delivery_date"].min(), scores["delivery_date"].max()
    asked, found = request_days(), events()

    def row(first: pd.Timestamp, last: pd.Timestamp) -> dict:
        part = scores[scores["delivery_date"].between(first, last)]
        shown, hit = part["shown"] == 1, part["had_request"] == 1
        flagged = set(zip(part.loc[shown, "delivery_date"], part.loc[shown, "area"]))
        days = asked[asked["delivery_date"].between(first, last)]
        # a request belongs to the area-day it starts on
        starts = found["start_time"].dt.normalize()
        requests = found[starts.between(first, last)]
        caught = sum(key in flagged for key in zip(requests["start_time"].dt.normalize(), requests["area"]))
        right = int((shown & hit).sum())
        out = {"start": first, "end": last, "asked_days": len(days), "requests": len(requests), "flagged": int(shown.sum()),
               "right": right, "wrong": int((shown & ~hit).sum()), "missed_days": len(days) - right, "caught": caught,
               "missed_requests": len(requests) - caught}
        for key, whole in (("right", "flagged"), ("wrong", "flagged"), ("caught", "requests"), ("missed_requests", "requests")):
            out[f"{key}_pct"] = f"{100 * out[key] / out[whole]:.0f}%" if out[whole] else ""
        return out

    # consecutive 14-day windows from the first test day, then the whole test when there is more than one
    window = pd.Timedelta(days=CONFIG.validation.window_days)
    rows = [row(first, min(first + window - pd.Timedelta(days=1), end)) for first in pd.date_range(start, end, freq=window)]
    total = row(start, end) if len(rows) > 1 else None
    return {"rows": rows, "total": total, "start": start, "end": end, "count": scores["delivery_date"].nunique(),
            "made": pd.Timestamp(file.stat().st_mtime, unit="s", tz="UTC").tz_convert("Europe/Amsterdam")}


def notices(limit: int = 12) -> pd.DataFrame:
    """The most recent transport-restriction notices."""
    file = path("notices")
    if not file.exists():
        return pd.DataFrame()
    found = _cached("notices", [file], lambda: ingest.load_notices(file))
    return found.sort_values("announced_at", ascending=False).head(limit)


def activity(days: int = 30) -> dict:
    """GOPACS requests over the last days of the record: daily counts, busiest areas and operators."""
    quarters = request_quarters()
    if quarters.empty:
        return {}
    end = quarters["delivery_date"].max()
    start = end - pd.Timedelta(days=days - 1)
    window = quarters[(quarters["delivery_date"] >= start) & (quarters["delivery_date"] <= end)]
    area_days = window.drop_duplicates(schemas.AREA_KEY + ["direction"])
    daily = (area_days.pivot_table(index="delivery_date", columns="direction", values="area", aggfunc="count")
             .reindex(pd.date_range(start, end)).fillna(0).astype(int))
    busiest = (window.groupby("area").agg(days=("delivery_date", "nunique"), hours=("quarter", "size"),
                                         operator=("operator", "first"), direction=("direction", lambda s: s.mode().iat[0]))
               .assign(hours=lambda t: t["hours"] / 4).sort_values(["days", "hours"], ascending=False).head(12)
               .reset_index())
    operators = (window.drop_duplicates(schemas.AREA_KEY + ["operator"]).groupby("operator")
                 .agg(area_days=("area", "size"), areas=("area", "nunique"))
                 .sort_values("area_days", ascending=False).reset_index())
    found = events()
    in_window = found[(found["start_time"] >= start) & (found["start_time"] < end + pd.Timedelta(days=1))]
    return {
        "start": start, "end": end, "daily": daily, "busiest": busiest, "operators": operators,
        "requests": len(in_window), "areas": window["area"].nunique(),
        "mwh": float((window["mw"].fillna(0) / 4).sum()), "area_days": len(window.drop_duplicates(schemas.AREA_KEY)),
        "downloaded": found["announced_at"].max(),
    }


def feature_groups(columns: list[str]) -> list[tuple[str, list[str]]]:
    """The model's inputs grouped by where they come from."""
    suffixes = tuple(f"_{s}" for s in features.DIRECTIONS.values())
    history = ("yesterday", "week", "month", "days", "active", "hist")

    def group(column: str) -> str:
        if column in ("weekday", "month", "is_weekend", "price_hist_mean"):
            return "Market, calendar and location"
        if column in features.WEATHER_FEATURES:
            return "Weather per grid cell"
        if column in features.NOTICE_FEATURES:
            return "Transport-restriction notices"
        if column in features.DAYTYPE_FEATURES:
            return "Requests on the same day type"
        if column in features.CLOCK_FEATURES + features.SCHOOL_FEATURES:
            return "Clock change and school holidays"
        if column.endswith(suffixes):
            return "Request history by direction"
        if column in features.DIRECTION_HISTORY or any(word in column for word in history):
            return "Area, cell and operator history"
        return "Market, calendar and location"

    out: dict[str, list[str]] = {}
    for column in columns:
        out.setdefault(group(column), []).append(column)
    return list(out.items())


FILE_GROUPS = [
    ("Data", [("gopacs", "GOPACS requests", "Update GOPACS"), ("notices", "Restriction notices", "Update GOPACS"),
              ("weather", "Weather forecasts", "Update weather"), ("sdac", "Day-ahead market", "Update market data"),
              ("forecasts", "System forecasts", "Update market data"), ("locations", "Area locations", "Locate areas")]),
    ("Model", [(f"{name}{suffix}", f"{label}{when}", maker) for suffix, when in
               [("", "")] + [(f"_{h}", f", {h:02d}:00") for h in CONFIG.hours[1:]]
               for name, label, maker in (("model", "Trained model", "Retrain model"), ("walk_forward", "Test (14-day windows)", "Download from Drive"))]
              + [(f"walk_forward_forecaster_{h}", f"Test (14-day windows), {h:02d}:00 with the forecaster", "Download from Drive")
                 for h in CONFIG.hours if h < CONFIG.features.market_hour]),
    ("Outputs", [("predictions", "Quarter-hour forecasts", "Issue forecast"),
                 ("area_predictions", "Area forecasts ledger", "Issue forecast"), ("validation", "Validation history", "Validate"),
                 ("market_forecasts", "Forecaster's day-ahead values", "Issue forecast with the forecaster")]),
]


def file_path(key: str) -> Path:
    """The path of a listed file, including the model files that are not in the config, such as model_11."""
    from congestion_nl import drive_source

    return drive_source.target(CONFIG, key)


def file_status() -> list[dict]:
    """Each data, model and output file by group, with whether it exists, its last change and the step that makes it."""
    groups = []
    for group, names in FILE_GROUPS:
        files = []
        for key, label, maker in names:
            file = file_path(key)
            exists = file.exists()
            files.append({
                "key": key, "label": label, "maker": maker, "exists": exists, "size": file.stat().st_size if exists else 0,
                "path": file.relative_to(CONFIG.root).as_posix() if file.is_relative_to(CONFIG.root) else str(file),
                "changed": pd.Timestamp(file.stat().st_mtime, unit="s", tz="UTC").tz_convert("Europe/Amsterdam") if exists else None,
            })
        groups.append({"name": group, "files": files})
    return groups


def has_history(name: str, column: str, back_days: int) -> bool:
    """Whether a saved file reaches back to the start of the history, so a short fetch is enough."""
    file = path(name)
    if not file.exists():
        return False
    first = _cached(f"first_{name}", [file], lambda: pd.to_datetime(pd.read_csv(file, usecols=[column])[column]).min())
    return pd.notna(first) and first <= CONFIG.grid.start_date - pd.Timedelta(days=back_days)


def last_day(name: str, column: str) -> pd.Timestamp:
    """The last day a saved file has data for."""
    file = path(name)
    return _cached(f"last_{name}", [file], lambda: pd.to_datetime(pd.read_csv(file, usecols=[column])[column]).max().normalize())


def weather_start(day, days_back: int) -> str:
    """Where a weather fetch starts: some days before the day, but no later than the last saved day."""
    # a week before the history, for the inputs that compare a day with the three before it
    if has_history("weather", "time", 7):
        start = min(pd.Timestamp(day) - pd.Timedelta(days=days_back), last_day("weather", "time") - pd.Timedelta(days=1))
        return start.strftime("%Y-%m-%d")
    return (CONFIG.grid.start_date - pd.Timedelta(days=7)).strftime("%Y-%m-%d")


def market_start(day) -> str:
    """Where a market fetch starts: a week before the day, but no later than the last saved day."""
    # a month before the history, for the price history input
    if has_history("sdac", "delivery_date", 31):
        start = min(pd.Timestamp(day) - pd.Timedelta(days=7), last_day("sdac", "delivery_date") - pd.Timedelta(days=1))
        return start.strftime("%Y-%m-%d")
    return (CONFIG.grid.start_date - pd.Timedelta(days=31)).strftime("%Y-%m-%d")
