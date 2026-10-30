from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import estimator, schemas
from .config import Config
from .features import NOTICE_FEATURES, WEATHER_DAY

WALK_FORWARD_FILE = "walk_forward.csv"
WINDOWS = (7, 30, 90)
MIN_FLAGS = 50
# area-day inputs from the predictor's dataset: values that are the same for every quarter of the day
DAY_VALUES = (
    ["weekday", "is_weekend", "feed_in_area", "days_since_request", "run_days", "area_yesterday", "area_week", "area_month"]
    + [c for c in NOTICE_FEATURES if c != "notice_q"]
    + list(WEATHER_DAY)
    + [f"{c}_change" for c in WEATHER_DAY]
    + ["rad_day_vs3", "wind_day_vs3"]
)
DAY_SUMMARIES = {
    "notice_q": "sum",
    "price_eur_mwh": "max",
    "residual_load_mw": "max",
    "solar_forecast_mw": "max",
    "wind_forecast_mw": "max",
}


def _features_key(features: list[str]) -> str:
    """Return a short hash of the predictor's input list, so cached scores are dropped when it changes."""
    return hashlib.sha1(",".join(features).encode()).hexdigest()[:12]


def walk_forward(config: Config, dataset: pd.DataFrame, features: list[str], cache: Path) -> pd.DataFrame:
    """Score every area-day from the inspector start with a predictor trained up to two days before its month."""
    key = _features_key(features)
    cached = pd.DataFrame()
    if cache.exists():
        cached = pd.read_csv(cache, parse_dates=["month", "delivery_date"])
        cached = cached[cached["features_key"] == key]
    labelled = dataset[dataset["target"].notna()]
    start = max(config.inspector.start_date, dataset["delivery_date"].min()).to_period("M").to_timestamp()
    parts = []
    for month in pd.date_range(start, dataset["delivery_date"].max(), freq="MS"):
        rows = dataset[(dataset["delivery_date"] >= month) & (dataset["delivery_date"] <= month + pd.offsets.MonthEnd(0))]
        old = cached[cached["month"] == month] if len(cached) else cached
        if len(old) and len(old) == len(rows[schemas.AREA_KEY].drop_duplicates()):
            parts.append(old)
            continue
        known = labelled[labelled["delivery_date"] <= month - pd.Timedelta(days=estimator.LABEL_GAP_DAYS)]
        if known.empty or known["target"].nunique() < 2:
            continue
        bundle = estimator.fit_bundle(known, features, config.model)
        peak = rows.assign(peak=estimator.predict(bundle, rows)).groupby(schemas.AREA_KEY, as_index=False)["peak"].max()
        parts.append(peak.assign(month=month, features_key=key))
    if not parts:
        return pd.DataFrame(columns=schemas.AREA_KEY + ["peak", "target"])
    scores = pd.concat(parts, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(cache, index=False)
    target = dataset.groupby(schemas.AREA_KEY, as_index=False)["target"].max()
    return scores[schemas.AREA_KEY + ["peak"]].merge(target, on=schemas.AREA_KEY, how="left")


def track_record(past: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per area and day, the predictor's correct flags, wrong flags, misses and bias over recent windows up to D-2."""
    days = pd.date_range(past["delivery_date"].min(), past["delivery_date"].max())
    known = past[past["target"].notna()]
    flag, hit = known["peak"] >= threshold, known["target"] == 1
    known = known.assign(correct=(flag & hit).astype(float), wrong=(flag & ~hit).astype(float),
                         missed=(~flag & hit).astype(float), err=known["peak"] - known["target"], seen=1.0)
    out = []
    for column in ("correct", "wrong", "missed", "err", "seen"):
        wide = known.pivot_table(index="delivery_date", columns="area", values=column).reindex(days).fillna(0)
        for n in WINDOWS:
            # shift(2): the requests of D-1 are not complete at 14:00 on D-1
            out.append(wide.rolling(n, min_periods=1).sum().shift(2).stack().rename(f"{column}_{n}"))
    table = pd.concat(out, axis=1)
    table.index.names = schemas.AREA_KEY
    table = table.reset_index()
    for n in WINDOWS:
        table[f"bias_{n}"] = table[f"err_{n}"] / table[f"seen_{n}"].where(table[f"seen_{n}"] > 0)
    return table.drop(columns=[f"err_{n}" for n in WINDOWS])


def recent_scores(past: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per area and day, the predictor's scores for D-1 and D-2 and how many days in a row before D it flagged."""
    days = pd.date_range(past["delivery_date"].min(), past["delivery_date"].max())
    wide = past.pivot_table(index="delivery_date", columns="area", values="peak").reindex(days)
    flagged = (wide >= threshold).astype(float)
    streak = flagged.copy()
    for i in range(1, len(streak)):
        streak.iloc[i] = (streak.iloc[i - 1] + 1) * flagged.iloc[i]
    table = pd.concat([wide.shift(1).stack().rename("peak_d1"), wide.shift(2).stack().rename("peak_d2"),
                       streak.shift(1).stack().rename("flag_streak")], axis=1)
    table.index.names = schemas.AREA_KEY
    return table.reset_index()


def day_inputs(dataset: pd.DataFrame) -> pd.DataFrame:
    """One row per area-day with the day-level inputs and daily market and notice summaries."""
    agg = {c: "first" for c in DAY_VALUES if c in dataset.columns}
    agg.update({c: how for c, how in DAY_SUMMARIES.items() if c in dataset.columns})
    return dataset.groupby(schemas.AREA_KEY, as_index=False).agg(agg)


def inputs(past: pd.DataFrame, dataset: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Every area-day of the dataset with the inspector's inputs, apart from the predictor's own score."""
    table = day_inputs(dataset)
    if past.empty:
        return table
    table = table.merge(track_record(past, threshold), on=schemas.AREA_KEY, how="left")
    return table.merge(recent_scores(past, threshold), on=schemas.AREA_KEY, how="left")


def _model() -> HistGradientBoostingClassifier:
    """A small boosted-tree model: few rows per day, so shallow trees and large leaves."""
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_depth=3, min_samples_leaf=100,
                                          l2_regularization=1.0, random_state=42)


def fit(config: Config, past: pd.DataFrame, dataset: pd.DataFrame, threshold: float) -> dict | None:
    """Train the veto on past flags scored out of sample, target whether the flag was correct; None if too few."""
    if past.empty:
        return None
    table = inputs(past, dataset, threshold)
    columns = ["peak"] + [c for c in table.columns if c not in schemas.AREA_KEY]
    rows = past[(past["peak"] >= threshold) & past["target"].notna()].merge(table, on=schemas.AREA_KEY, how="left")
    if len(rows) < MIN_FLAGS or rows["target"].nunique() < 2:
        return None
    return {
        "model": _model().fit(rows[columns], rows["target"]),
        "columns": columns,
        "cut": config.inspector.veto_cut,
        "trained_flags": int(len(rows)),
        "trained_correct": int(rows["target"].sum()),
    }


def judge(config: Config, bundle: dict, dataset: pd.DataFrame, days: pd.DataFrame) -> pd.DataFrame:
    """Add the inspector's chance that each flag is correct, and whether the flag is shown after the veto."""
    days = days.assign(inspector_confidence=np.nan, shown=days["day_flag"])
    veto = bundle.get("inspector")
    flagged = days["day_flag"] == 1
    if veto is None or not flagged.any():
        return days
    cache = config.resolve(config.paths.model_dir) / WALK_FORWARD_FILE
    past = walk_forward(config, dataset, bundle["features"], cache)
    table = inputs(past, dataset, bundle["day_threshold"])
    rows = days.loc[flagged, schemas.AREA_KEY + ["peak_probability"]].rename(columns={"peak_probability": "peak"})
    rows = rows.merge(table, on=schemas.AREA_KEY, how="left").reindex(columns=schemas.AREA_KEY + veto["columns"])
    confidence = veto["model"].predict_proba(rows[veto["columns"]])[:, 1]
    days.loc[flagged, "inspector_confidence"] = confidence
    days.loc[flagged, "shown"] = (confidence >= veto["cut"]).astype(int)
    return days
