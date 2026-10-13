"""Generate synthetic input files in the expected schema for demos and tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import schemas
from .config import Config


def _national_grid(start: pd.Timestamp, days: int) -> pd.DataFrame:
    """Return every delivery date and hour combination."""
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.MultiIndex.from_product([dates, range(24)], names=schemas.MARKET_KEY).to_frame(
        index=False
    )


def _runs_to_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive congested hours per area into event intervals."""
    congested = frame[frame["congested"] == 1].copy()
    congested["stamp"] = congested["delivery_date"] + pd.to_timedelta(congested["hour"], unit="h")
    events = []
    for area, group in congested.sort_values(["area", "stamp"]).groupby("area"):
        stamps = group["stamp"].tolist()
        run_start = stamps[0] if stamps else None
        previous = run_start
        for stamp in stamps[1:]:
            if stamp - previous != pd.Timedelta(hours=1):
                events.append((area, run_start, previous + pd.Timedelta(hours=1)))
                run_start = stamp
            previous = stamp
        if run_start is not None:
            events.append((area, run_start, previous + pd.Timedelta(hours=1)))
    frame_out = pd.DataFrame(events, columns=["area", "start_time", "end_time"])
    frame_out.insert(0, "event_id", [f"GOPACS-{i:05d}" for i in range(len(frame_out))])
    return frame_out


def generate(config: Config, start: str, days: int, seed: int = 7) -> dict[str, Path]:
    """Write synthetic SDAC, forecast, outage and GOPACS files to the configured paths."""
    rng = np.random.default_rng(seed)
    areas = config.areas
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
    net_position = rng.normal(0, 800, n) + 0.05 * residual

    sdac = national.assign(price_eur_mwh=price.round(2), net_position_mw=net_position.round(1))
    forecasts = national.assign(
        load_forecast_mw=load.round(1),
        wind_forecast_mw=wind.round(1),
        solar_forecast_mw=solar.round(1),
    )

    area_grid = national.merge(pd.DataFrame({"area": list(areas)}), how="cross")
    m = len(area_grid)
    area_idx = area_grid["area"].map({a: i for i, a in enumerate(areas)}).to_numpy()
    residual_by_key = national.assign(residual=residual)
    area_grid = area_grid.merge(residual_by_key, on=schemas.MARKET_KEY, how="left")

    local_pressure = rng.normal(0, 700, m) + area_idx * 150
    outage = rng.gamma(1.5, 200.0, m) * (rng.random(m) < 0.2)
    logit = (
        (area_grid["residual"].to_numpy() - 9000) / 900
        + local_pressure / 1000
        + outage / 500
        - 1.5
    )
    congested = (rng.random(m) < 1 / (1 + np.exp(-logit))).astype(int)

    outages = area_grid.assign(outage_mw=outage.round(1))[schemas.KEY + ["outage_mw"]]
    gopacs = _runs_to_events(area_grid.assign(congested=congested)[schemas.KEY + ["congested"]])

    targets = {
        "sdac": (config.resolve(config.paths.sdac), sdac),
        "forecasts": (config.resolve(config.paths.forecasts), forecasts),
        "gopacs": (config.resolve(config.paths.gopacs), gopacs),
    }
    if config.paths.outages is not None:
        targets["outages"] = (config.resolve(config.paths.outages), outages)

    written = {}
    for name, (path, frame) in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        written[name] = path
    return written
