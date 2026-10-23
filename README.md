# Dutch Grid Congestion Risk

This estimator predicts, for each delivery quarter-hour and each GOPACS congestion area in the Netherlands, the probability that the grid operator publishes a congestion-management request for that area. It also sums these up per area and day: how likely the area is to get a request at all, and in which hours. It is trained and checked against the requests that GOPACS publishes, and cross-checked against the CSC call-off report of the grid operators.

## What the code does

1. **Fetch.** It downloads the NL day-ahead price, net position and load, wind and solar forecasts from ENTSO-E, and every public GOPACS announcement.
2. **Direction.** Each congestion-management request names its area on one side only. An area on the buy side is **feed-in** congestion (parties inside should produce less), and an area on the sell side is **offtake** congestion (parties inside should use less).
3. **Locate.** Each GOPACS area is placed at a point and given a 0.25° grid cell (about 28 × 17 km), using in order:
   - a hand-checked row in `reference/manual_locations.csv`
   - the postcodes that GOPACS lists for the area (median of the postcode centroids)
   - the towns in a town-list area name ("EINDHOVEN, VELDHOVEN")
   - an Enexis station code ("LIMM Congestie11_P" is Limmel), read from the CSC report
   - the substation or town in the name ("RS WASKEMEER 10-2i" is Waskemeer)

   TenneT's province-wide requests have no local area and are left out. The cell is used only to group nearby areas.
4. **Build.** It builds one row per delivery date, quarter-hour and active area. An area is active while it has had a request in the last `active_days` days, so new areas join after their first request and dormant ones drop out. Each row uses only information known at the issue time, 14:00 on the day before delivery (`issue_hour`):
   - the area's position, operator and main direction
   - the national day-ahead price, net position, forecasts and residual load (load minus wind and solar), and the price history for the same quarter-hour
   - the area's own history: request quarter-hours yesterday and over the last 7 and 28 days, how often this quarter-hour had a request in the last 7 days, the run of days with requests, days since the last request, days since the first request, and the requested MW over the last 7 days
   - the neighbourhood: request quarter-hours yesterday and over the last 7 days in the area's cell, and how many areas of the cell and of the operator had requests yesterday

   A past day counts only the requests that had been announced by the issue time of the forecast for the following day. Requests for the delivery day itself are never used, even when they were announced before the issue time. Because the model learns from features shared by all areas, an area with few requests of its own still gets a forecast. GOPACS gives each announcement a creation time, which is what makes the issue-time cut possible. The period of an announcement can still be changed after it is created, and only the final period is published, so these features can be slightly optimistic.
5. **Predict.** A gradient-boosted classifier gives a probability per row. In backtests its probabilities are close to calibrated as they are (a forecast of 30% comes true about 30% of the time), so by default they are used directly with fixed thresholds:
   - a quarter-hour is flagged when its probability reaches `threshold`
   - an area is flagged for the day when its highest quarter-hour probability reaches `day_threshold`. The day forecast also gives the expected number of requested quarter-hours (the sum of the probabilities) and the risky window, from the first to the last quarter-hour at or above `day_threshold`.

   The quarter-hours go to `outputs/predictions.csv` and the area-days to `outputs/area_predictions.csv`, each with its issue time and model version. Setting `threshold_days` above 0 instead calibrates the probabilities with a logistic fit on the log-odds (Platt scaling) over that many recent days, and picks the quarter-hour threshold as the lowest one whose precision there reaches `target_precision`. Short windows hold few requests and made the threshold jump from day to day, so this is off by default.
6. **Validate.** Once a day has settled, each logged prediction is joined to the requests that came. The validation step computes Brier score, skill against the base rate, ROC AUC, precision and recall, over the full history and over a recent window. If recent skill falls below the configured minimum, the command exits with status 2.

## Layout

```
README.md
pyproject.toml
config.toml               paths, grid, feature windows, model and validation settings
reference/
    manual_locations.csv  hand-checked area locations, each with its evidence
congestion_nl/
    config.py             loads config.toml into typed objects
    schemas.py            required columns for every input and output file
    entsoe_source.py      fetches NL day-ahead prices and forecasts from ENTSO-E
    gopacs_source.py      fetches GOPACS announcements and turns them into events with a direction
    locations.py          places GOPACS areas and assigns grid cells
    ingest.py             reads and checks the CSV inputs
    labels.py             expands GOPACS intervals into quarter-hourly 0/1 labels per area
    features.py           builds features and the target, with lags to prevent look-ahead
    rolling.py            trailing-window aggregates shifted by a reporting lag
    estimator.py          gradient-boosted classifier: rolling backtest, thresholds, area-day summary, save, load, predict
    pipeline.py           loads inputs, builds datasets, trains and issues daily predictions
    validate.py           scores the latest logged predictions against settled GOPACS requests
    ledger.py             reads and appends the prediction and validation CSVs
    metrics.py            Brier score, skill, ROC AUC, precision and recall
    synthetic.py          generates demo inputs in the expected format
    cli.py                command line entry point
tests/                    unit and end-to-end tests
data/ models/ outputs/    local files, not tracked
```

Relative paths in `config.toml` are resolved from this folder.

## Setup

Requires Python 3.11 or newer. From this folder:

```
python -m pip install -e ".[test,entsoe]"
python -m pytest
```

The ENTSO-E fetch needs a free Web API token from `transparency.entsoe.eu`, set as the environment variable `ENTSOE_API_KEY`. It is never written to config or data files.

The location step needs two GeoNames files (CC BY 4.0) in `data/raw/geonames/`:

- `places.zip` from `https://download.geonames.org/export/dump/NL.zip`
- `postcodes.zip` from `https://download.geonames.org/export/zip/NL.zip`

It also reads the CSC call-off report (`CSC_public_report_*.xlsx`) to resolve Enexis station codes.

## Inputs

| File | Columns | Source |
| --- | --- | --- |
| SDAC results | `delivery_date, quarter, price_eur_mwh, net_position_mw` | `congestion-nl fetch-entsoe` |
| System forecasts | `delivery_date, quarter, load_forecast_mw, wind_forecast_mw, solar_forecast_mw` | `congestion-nl fetch-entsoe` |
| GOPACS events | `event_id, area, direction, operator, postcodes, mw, start_time, end_time, announced_at` | `congestion-nl fetch-gopacs` |
| Area locations | `area, lat, lon, cell, source` | `congestion-nl locate` |

- **Time and quarter-hours.** All times are Dutch local clock time. `quarter` runs from 0 (00:00–00:15) to 95 (23:45–24:00). The repeated hour on the autumn clock change is averaged into one set of quarters. A request is labelled on every quarter-hour it touches.
- **Resolution.** The day-ahead market moved to 15-minute prices on 1 October 2025, so `grid.start_date` is set to that day and every input is native 15-minute data. Market data before that date is ignored, including for the price history. If an input ever arrives hourly, each hourly value is repeated over its four quarters.
- **Market data.** The Netherlands is a single bidding zone and only day-ahead results are used, so the market inputs are national and the same for every area.
- **Areas and cells.** Only located areas with at least one request are modelled. A cell id names its south-west corner, for example `52.00N 5.00E`.
- **Missing quarter-hours.** A quarter-hour without a request is treated as not congested.

## Commands

```
congestion-nl fetch-entsoe --start 2023-07-01 --end 2026-10-18
congestion-nl fetch-gopacs
congestion-nl locate
congestion-nl train
congestion-nl predict --date 2026-10-17 --issued-at 2026-10-16T11:00:00+00:00
congestion-nl validate --as-of 2026-10-19
congestion-nl demo --start 2025-01-01 --days 400
```

- **`locate`** writes `data/processed/area_locations.csv` and prints how many requests could be placed.
- **`train`** saves the model to `models/`.
- **`predict`** writes quarter-hours to `outputs/predictions.csv` and area-days to `outputs/area_predictions.csv`, and prints the flagged areas with their window. `--issued-at` records when the forecast was made.
- **`validate`** scores settled days. Days within `labels.settle_days` of `--as-of` are not scored yet.
- **`demo`** writes synthetic inputs, including area locations, in the expected format.

## Configuration

- `[grid]` sets the cell size in degrees, the first delivery date used, and which request directions count as congestion.
- `[features]` sets the short and long history windows in days, the lag on price history, the hour on the day before delivery at which forecasts are issued, and how many quiet days an area stays active. Requests for the day before delivery are used only if they were announced before the issue time, and requests for the delivery day are never used. Set `issue_hour` to the hour you actually run `predict`, because the features assume everything announced before it is known.
- `[labels]` sets how many days after a delivery date the GOPACS record is treated as complete. Days after the last GOPACS announcement in the downloaded file are never labelled, so a stale download does not turn into quiet days.
- `[model]` sets the number of rolling test days, the quarter-hour and area-day thresholds, the optional threshold window and its target precision, and the boosting parameters.
- `[validation]` sets the recent window length and the minimum acceptable skill.

## Notes on the design

- Training is scored with a rolling daily backtest over the last `test_days` labelled days. Each test day gets its own model, trained on every label up to two days before it, because at 14:00 on the day before delivery that is the last complete day of requests. If `threshold_days` is set, its threshold and calibration are picked on the `threshold_days` before that cut-off, by a model trained on the data before them. `train` reports the quarter-hour score per day and in total, and the area-day score, then fits the final model the same way on all labelled data.
- Targets after `label_end` are masked, so a feature table built for a prediction day never carries a label.
- The demo data follows a simple rule, so its scores are much higher than real performance will be.
- About 45% of requested quarter-hours repeat a quarter that was already requested the day before. These are forecast well. The rest are new windows, which come up at a rate of 0.1–3%, so they can rarely be flagged without many false alarms. Local weather forecasts, longer slot histories, other model types (random forest, extra trees, logistic regression) and a separate area-day model were tested in a rolling backtest and did not improve on this model.
