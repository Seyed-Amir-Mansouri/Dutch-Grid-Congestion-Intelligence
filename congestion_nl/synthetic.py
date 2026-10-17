from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import schemas
from .config import Config
from .locations import cell_of

AREAS = {
    "Area North": (53.20, 6.10),
    "Area East": (52.30, 6.70),
    "Area West": (52.05, 4.60),
    "Area South": (51.45, 5.50),
    "Area Coast": (51.55, 3.70),
}


def _national_grid(start: pd.Timestamp, days: int) -> pd.DataFrame:
    """Return every delivery date and hour combination."""
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.MultiIndex.from_product([dates, range(24)], names=schemas.MARKET_KEY).to_frame(index=False)


def _runs_to_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive congested hours per area into event intervals."""
    congested = frame[frame["congested"] == 1].copy()
    congested["stamp"] = congested["delivery_date"] + pd.to_timedelta(congested["hour"], unit="h")
    events = []
    for area, group in congested.sort_values(["area", "stamp"]).groupby("area"):
        stamps = group["stamp"].tolist()
        run_start = previous = stamps[0]
        for stamp in stamps[1:]:
            if stamp - previous != pd.Timedelta(hours=1):
                events.append((area, run_start, previous + pd.Timedelta(hours=1)))
                run_start = stamp
            previous = stamp
        events.append((area, run_start, previous + pd.Timedelta(hours=1)))
    out = pd.DataFrame(events, columns=["area", "start_time", "end_time"])
    out.insert(0, "event_id", [f"GOPACS-{i:05d}" for i in range(len(out))])
    out["direction"] = "feed-in"
    out["operator"] = "Synthetic"
    out["postcodes"] = ""
    out["mw"] = 1.0
    out["announced_at"] = out["start_time"] - pd.Timedelta(hours=24)
    return out[schemas.GOPACS_COLUMNS]


def generate(config: Config, start: str, days: int, seed: int = 7) -> dict[str, Path]:
    """Write synthetic SDAC, forecast, GOPACS and location files to the configured paths."""
    rng = np.random.default_rng(seed)
    national = _national_grid(pd.Timestamp(start), days)
    hour = national["hour"].to_numpy()
    weekday = national["delivery_date"].dt.weekday.to_numpy()
    n = len(national)

    daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    load = 11000 + 2500 * np.sin((hour - 8) / 24 * 2 * np.pi) - 900 * (weekday >= 5)
    load = load + rng.normal(0, 300, n)
    wind = rng.gamma(2.0, 1500.0, n)
    solar = 6000 * daylight * rng.uniform(0.4, 1.0, n)
    residual = load - wind - solar

    price = 40 + 0.012 * residual + rng.normal(0, 6, n)
    net_position = rng.normal(0, 800, n) - 0.05 * residual

    sdac = national.assign(price_eur_mwh=price.round(2), net_position_mw=net_position.round(1))
    forecasts = national.assign(
        load_forecast_mw=load.round(1),
        wind_forecast_mw=wind.round(1),
        solar_forecast_mw=solar.round(1),
    )

    names = list(AREAS)
    area_grid = national.assign(solar=solar).merge(pd.DataFrame({"area": names}), how="cross")
    m = len(area_grid)
    area_idx = area_grid["area"].map({a: i for i, a in enumerate(names)}).to_numpy()
    logit = (area_grid["solar"].to_numpy() - 3500) / 700 + rng.normal(0, 0.7, m) + area_idx * 0.2 - 1.0
    congested = (rng.random(m) < 1 / (1 + np.exp(-logit))).astype(int)
    gopacs = _runs_to_events(area_grid.assign(congested=congested))

    size = config.grid.cell_degrees
    locations = pd.DataFrame(
        [(a, lat, lon, cell_of(lat, lon, size), "synthetic") for a, (lat, lon) in AREAS.items()],
        columns=schemas.LOCATION_COLUMNS,
    )

    targets = {
        "sdac": (config.resolve(config.paths.sdac), sdac),
        "forecasts": (config.resolve(config.paths.forecasts), forecasts),
        "gopacs": (config.resolve(config.paths.gopacs), gopacs),
        "locations": (config.resolve(config.paths.locations), locations),
    }
    written = {}
    for name, (path, frame) in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        written[name] = path
    return written
