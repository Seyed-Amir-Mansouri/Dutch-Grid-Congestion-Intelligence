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


def model_dir() -> Path:
    """Return the folder of the saved model."""
    return path("model_dir")


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


def bundle() -> dict | None:
    """The saved model, or None if it has not been trained."""
    file = model_dir() / estimator.MODEL_FILE
    return _cached("bundle", [file], lambda: estimator.load(model_dir()) if file.exists() else None)


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


def track_record() -> pd.DataFrame:
    """Per settled forecast day: flags shown by the inspector, how many were right, and requests missed."""
    found = area_ledger()
    end = label_end()
    if found is None or end is None:
        return pd.DataFrame()
    latest = found[found["delivery_date"] <= end].sort_values("issued_at").drop_duplicates(schemas.AREA_KEY, keep="last")
    if latest.empty:
        return pd.DataFrame()
    latest = latest.merge(request_days(), on=schemas.AREA_KEY, how="left").fillna({"had_request": 0})
    hit = latest["had_request"] == 1
    shown, flagged = latest["shown"] == 1, latest["day_flag"] == 1
    out = latest.assign(
        shown_correct=shown & hit, shown_wrong=shown & ~hit, vetoed_wrong=flagged & ~shown & ~hit,
        vetoed_correct=flagged & ~shown & hit, missed=~shown & hit, requests=hit,
    )
    columns = ["shown_correct", "shown_wrong", "vetoed_wrong", "vetoed_correct", "missed", "requests"]
    return out.groupby("delivery_date")[columns].sum().astype(int).sort_index(ascending=False).reset_index()


def validation_history() -> pd.DataFrame:
    """Rows written by the validate command, newest first."""
    file = path("validation")
    if not file.exists():
        return pd.DataFrame()
    return pd.read_csv(file).iloc[::-1].reset_index(drop=True)


def rolling_test() -> dict | None:
    """The 14-day test: per day and in total, the predictor's flags and those the inspector kept, against requests."""
    file = model_dir() / inspector.TEST_FILE
    if not file.exists():
        return None
    scores = pd.read_csv(file, parse_dates=["delivery_date"])
    if not {"day_flag", "shown", "had_request"} <= set(scores.columns):
        # an older file in the same place, such as the monthly scores before they were renamed
        return None
    flag, shown, hit = scores["day_flag"] == 1, scores["shown"] == 1, scores["had_request"] == 1
    table = scores.assign(requests=hit, flag_correct=flag & hit, flag_wrong=flag & ~hit, shown_correct=shown & hit,
                          shown_wrong=shown & ~hit, saved=flag & ~shown & ~hit, lost=flag & ~shown & hit,
                          missed=~shown & hit)
    columns = ["requests", "flag_correct", "flag_wrong", "shown_correct", "shown_wrong", "saved", "lost", "missed"]
    days = table.groupby("delivery_date")[columns].sum().astype(int).reset_index()
    total = days[columns].sum().to_dict()
    for kind in ("flag", "shown"):
        flags = total[f"{kind}_flags"] = total[f"{kind}_correct"] + total[f"{kind}_wrong"]
        total[f"{kind}_missed"] = total["requests"] - total[f"{kind}_correct"]
        total[f"{kind}_precision"] = total[f"{kind}_correct"] / flags if flags else None
        total[f"{kind}_recall"] = total[f"{kind}_correct"] / total["requests"] if total["requests"] else None
    timing = []
    if "inside_quarters" in scores:
        # how well the risky window matched the requested hours, on the flags that were right about the day
        for label, kept in (("Predictor alone", flag & hit), ("With the inspector", shown & hit)):
            right = scores[kept]
            asked = int(right["request_quarters"].sum())
            timing.append((label, {"area_days": len(right), "overlap": int((right["inside_quarters"] > 0).sum()),
                                   "asked_hours": asked / 4, "inside_hours": right["inside_quarters"].sum() / 4,
                                   "share": right["inside_quarters"].sum() / asked if asked else None,
                                   "window_hours": right["window_quarters"].mean() / 4 if len(right) else None}))
    return {"days": days.to_dict("records"), "total": total, "timing": timing, "start": days["delivery_date"].min(),
            "end": days["delivery_date"].max(), "count": len(days),
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
        if column in features.WEATHER_FEATURES:
            return "Weather per grid cell"
        if column in features.NOTICE_FEATURES:
            return "Transport-restriction notices"
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
    ("Model", [("model", "Trained model", "Retrain model"), ("walk_forward", "14-day test", "Run 14-day test")]),
    ("Outputs", [("predictions", "Quarter-hour forecasts", "Issue forecast"),
                 ("area_predictions", "Area forecasts ledger", "Issue forecast"), ("validation", "Validation history", "Validate")]),
]


def file_path(key: str) -> Path:
    """The path of a listed file, including the two model files that are not in the config."""
    special = {"model": model_dir() / estimator.MODEL_FILE, "walk_forward": model_dir() / inspector.TEST_FILE,
               "inspector_history": model_dir() / inspector.HISTORY_FILE}
    return special.get(key) or path(key)


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


def weather_start(day, days_back: int) -> str:
    """Where a weather fetch starts: some days before the day, or the start of the history when the file does not reach it."""
    # a week before the history, for the inputs that compare a day with the three before it
    if has_history("weather", "time", 7):
        return (pd.Timestamp(day) - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
    return (CONFIG.grid.start_date - pd.Timedelta(days=7)).strftime("%Y-%m-%d")


def market_start(day) -> str:
    """Where a market fetch starts: a week before the day, or the start of the history when the file does not reach it."""
    # a month before the history, for the price history input
    if has_history("sdac", "delivery_date", 31):
        return (pd.Timestamp(day) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    return (CONFIG.grid.start_date - pd.Timedelta(days=31)).strftime("%Y-%m-%d")
