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


def walk_forward() -> pd.DataFrame:
    """Out-of-sample predictor scores per settled area-day (monthly refits), with whether a request came."""
    file = model_dir() / inspector.WALK_FORWARD_FILE

    def build():
        if not file.exists():
            return pd.DataFrame()
        scores = pd.read_csv(file, usecols=["delivery_date", "area", "peak"], parse_dates=["delivery_date"])
        end = label_end()
        if end is not None:
            scores = scores[scores["delivery_date"] <= end]
        scores = scores.merge(request_days(), on=schemas.AREA_KEY, how="left")
        scores["had_request"] = scores["had_request"].fillna(0).astype(int)
        return scores
    return _cached("walk_forward", [file, path("gopacs")], build)


def monthly_skill(threshold: float) -> pd.DataFrame:
    """Per month: active area-days, area-days with requests, flags, correct and wrong flags, and misses."""
    scores = walk_forward()
    if scores.empty:
        return scores
    flag, hit = scores["peak"] >= threshold, scores["had_request"] == 1
    table = scores.assign(month=scores["delivery_date"].dt.to_period("M").dt.to_timestamp(), area_days=1,
                          requests=hit, flagged=flag, correct=flag & hit, wrong=flag & ~hit, missed=~flag & hit)
    columns = ["area_days", "requests", "flagged", "correct", "wrong", "missed"]
    table = table.groupby("month")[columns].sum().astype(int).reset_index()
    table["precision"] = table["correct"] / table["flagged"].where(table["flagged"] > 0)
    table["recall"] = table["correct"] / table["requests"].where(table["requests"] > 0)
    return table


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


def file_status() -> list[dict]:
    """Each input and output file with its size and last change."""
    names = [("gopacs", "GOPACS requests"), ("notices", "Restriction notices"), ("weather", "Weather forecasts"),
             ("sdac", "Day-ahead market"), ("forecasts", "System forecasts"), ("locations", "Area locations"),
             ("area_predictions", "Area forecasts ledger"), ("validation", "Validation history")]
    out = []
    for key, label in names:
        file = path(key)
        exists = file.exists()
        out.append({
            "label": label, "path": str(file.relative_to(CONFIG.root)) if file.is_relative_to(CONFIG.root) else str(file),
            "exists": exists, "size": file.stat().st_size if exists else 0,
            "changed": pd.Timestamp(file.stat().st_mtime, unit="s", tz="UTC").tz_convert("Europe/Amsterdam") if exists else None,
        })
    return out


def market_start(day) -> str:
    """Where a market fetch starts: a week before the day, or the start of the history when there is no file yet."""
    if path("sdac").exists():
        return (pd.Timestamp(day) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    return (CONFIG.grid.start_date - pd.Timedelta(days=31)).strftime("%Y-%m-%d")
