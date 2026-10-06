"""Generate synthetic input files in the expected schema for demos and tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config


def _hourly_grid(start: pd.Timestamp, days: int, areas: tuple[str, ...]) -> pd.DataFrame:
    """Return every delivery date, hour and area combination."""
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.MultiIndex.from_product(
        [dates, range(24), areas], names=["delivery_date", "hour", "area"]
    ).to_frame(index=False)


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
    grid = _hourly_grid(pd.Timestamp(start), days, areas)
    hour = grid["hour"].to_numpy()
    weekday = grid["delivery_date"].dt.weekday.to_numpy()
    area_shift = grid["area"].map({area: i * 150 for i, area in enumerate(areas)}).to_numpy()

    daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    load = 11000 + 2500 * np.sin((hour - 8) / 24 * 2 * np.pi) - 900 * (weekday >= 5)
    load = load + area_shift + rng.normal(0, 300, len(grid))
    wind = rng.gamma(2.0, 1500.0, len(grid))
    solar = 6000 * daylight * rng.uniform(0.4, 1.0, len(grid))
    residual = load - wind - solar

    price = 40 + 0.012 * residual + rng.normal(0, 6, len(grid))
    net_position = rng.normal(0, 800, len(grid)) + 0.05 * residual
    outage = rng.gamma(1.5, 200.0, len(grid)) * (rng.random(len(grid)) < 0.2)

    logit = (residual - 9000) / 900 + (price - 90) / 12 - 1.5
    congested = (rng.random(len(grid)) < 1 / (1 + np.exp(-logit))).astype(int)

    sdac = grid.assign(price_eur_mwh=price.round(2), net_position_mw=net_position.round(1))
    forecasts = grid.assign(
        load_forecast_mw=load.round(1),
        wind_forecast_mw=wind.round(1),
        solar_forecast_mw=solar.round(1),
    )
    outages = grid.assign(outage_mw=outage.round(1))
    gopacs = _runs_to_events(grid.assign(congested=congested))

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
