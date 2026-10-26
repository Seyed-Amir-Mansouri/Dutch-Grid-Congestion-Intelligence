# Dutch Grid Congestion Risk

This estimator predicts which GOPACS congestion areas in the Netherlands will get a congestion-management request on the next day. The area is the congested area named in the request, where the grid operator needs flexibility, not the balancing area on the other side of the order. A forecast counts as correct when a flagged area gets any request that day, whatever the hour.

Inside, the model gives a probability per quarter-hour. Per area and day it reports the highest of these, which ranked area-days better in backtests than a model trained on area-days directly. It is trained and checked against the requests that GOPACS publishes, and cross-checked against the CSC call-off report of the grid operators.

## What the code does

1. **Fetch.** It downloads the NL day-ahead price, net position and load, wind and solar forecasts from ENTSO-E, an hourly weather forecast for every grid cell with an area from Open-Meteo, and every public GOPACS announcement.
2. **Clean and direct.** Requests listed in `reference/excluded_requests.csv` are dropped. Each row names an operator and an announcement time window, with the reason, for example the erroneous Liander requests of 15 October 2026 that GOPACS reported in a free-text notice. Each congestion-management request names its area on one side only. An area on the buy side is **feed-in** congestion (parties inside should produce less), and an area on the sell side is **offtake** congestion (parties inside should use less).
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
   - the area's own history: request quarter-hours yesterday and over the last 7 and 28 days, how often this quarter-hour had a request in the last 7 days, the run of days with requests, days since the last request, days since the first request, and the requested MW over the last 7 days. These are counted once for all requests and again separately for feed-in and offtake requests
   - the neighbourhood: request quarter-hours yesterday and over the last 7 days in the area's cell, and how many areas of the cell and of the operator had requests yesterday
   - the weather forecast for the area's cell: temperature, solar radiation, cloud cover and wind speed at 100 m and 10 m for the quarter-hour, their daily means, and how radiation, wind, temperature and cloud change against the day before and, for radiation and wind, against the mean of the three days before

   A past day counts only the requests that had been announced by the issue time of the forecast for the following day. Requests for the delivery day itself are never used, even when they were announced before the issue time. Because the model learns from features shared by all areas, an area with few requests of its own still gets a forecast. GOPACS gives each announcement a creation time, which is what makes the issue-time cut possible. The period of an announcement can still be changed after it is created, and only the final period is published, so these features can be slightly optimistic.
5. **Predict.** A gradient-boosted classifier gives a probability per row. An area is flagged for the day when its highest quarter-hour probability reaches `day_threshold`. The day forecast also gives the expected number of requested quarter-hours (the sum of the probabilities) and the risky window, from the first to the last quarter-hour at or above `day_threshold`. The area-days go to `outputs/area_predictions.csv` and the quarter-hour probabilities to `outputs/predictions.csv`, each with its issue time and model version.
6. **Validate.** Once a day has settled, each logged area-day is marked by whether the area got any request that day. The validation step computes Brier score, skill against the base rate, ROC AUC, the counts of flagged, correct, wrong and missed area-days, precision and recall, over the full history and over a recent window. It also counts area-days with a request in an area that had no forecast (`untracked`). If recent skill falls below the configured minimum, the command exits with status 2.

## Layout

```
README.md
pyproject.toml
config.toml               paths, grid, feature windows, model and validation settings
reference/
    manual_locations.csv  hand-checked area locations, each with its evidence
    excluded_requests.csv erroneous requests to drop, by operator and announcement window
congestion_nl/
    config.py             loads config.toml into typed objects
    schemas.py            required columns for every input and output file
    entsoe_source.py      fetches NL day-ahead prices and forecasts from ENTSO-E
    weather_source.py     fetches Open-Meteo forecasts per grid cell from the model run two days before
    gopacs_source.py      fetches GOPACS announcements and turns them into events with a direction
    locations.py          places GOPACS areas and assigns grid cells
    ingest.py             reads and checks the CSV inputs
    labels.py             expands GOPACS intervals into 0/1 labels per area and quarter-hour, and per area and day
    features.py           builds features and the target, with lags to prevent look-ahead
    rolling.py            trailing-window aggregates shifted by a reporting lag
    estimator.py          gradient-boosted classifier: rolling backtest, area-day summary and score, save, load, predict
    pipeline.py           loads inputs, builds datasets, trains and issues daily predictions
    validate.py           scores the latest logged area-day predictions against settled GOPACS requests
    ledger.py             reads and appends the prediction and validation CSVs
    metrics.py            Brier score, skill, ROC AUC, flag counts, precision and recall
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
| Weather | `cell, time, temperature_c, radiation_wm2, cloud_cover_pct, wind_100m_kmh, wind_10m_kmh` | `congestion-nl fetch-weather` |
| GOPACS events | `event_id, area, direction, operator, postcodes, mw, start_time, end_time, announced_at` | `congestion-nl fetch-gopacs` |
| Area locations | `area, lat, lon, cell, source` | `congestion-nl locate` |

- **Time and quarter-hours.** All times are Dutch local clock time. `quarter` runs from 0 (00:00–00:15) to 95 (23:45–24:00). The repeated hour on the autumn clock change is averaged into one set of quarters. A request is labelled on every quarter-hour it touches.
- **Resolution and history.** `grid.start_date` is 1 April 2024, when GOPACS requests became frequent. The day-ahead market moved to 15-minute prices on 1 October 2025; earlier hourly values are repeated over their four quarters. Market data before `grid.start_date` is ignored. In the 13-week rolling test, starting in April 2024 instead of October 2025 raised area-day recall at 70% precision from 0.62 to 0.67, mostly through better autumn forecasts.
- **Market data.** The Netherlands is a single bidding zone and only day-ahead results are used, so the market inputs are national and the same for every area.
- **Weather.** Open-Meteo's previous-runs API keeps past forecasts. The fetch asks for the `previous_day2` values, which come from model runs 48 to 71 hours before the hour they forecast, so every value was published before 14:00 on the day before delivery. Each cell is fetched at its centre with Open-Meteo's default model, in local time; hourly values are repeated over their four quarters. The free service needs no key and limits use per hour, so the fetch pauses between cells and a full history takes about an hour.
- **Areas and cells.** Only located areas with at least one request are modelled. A cell id names its south-west corner, for example `52.00N 5.00E`.
- **Missing quarter-hours.** A quarter-hour without a request is treated as not congested.

## Commands

```
congestion-nl fetch-entsoe --start 2023-07-01 --end 2026-10-18
congestion-nl fetch-weather --start 2024-03-25 --end 2026-10-18
congestion-nl fetch-gopacs
congestion-nl locate
congestion-nl train
congestion-nl predict --date 2026-10-17 --issued-at 2026-10-16T11:00:00+00:00
congestion-nl validate --as-of 2026-10-19
congestion-nl demo --start 2025-01-01 --days 400
```

- **`fetch-weather`** fetches every cell in `data/processed/area_locations.csv`, so run it after `locate`. It merges the new hours into `data/raw/weather.csv`, replacing hours it already had. Before `predict`, fetch up to the delivery day.
- **`locate`** writes `data/processed/area_locations.csv` and prints how many requests could be placed.
- **`train`** saves the model to `models/`.
- **`predict`** writes area-days to `outputs/area_predictions.csv` and quarter-hours to `outputs/predictions.csv`, and prints the flagged areas with their window. `--issued-at` records when the forecast was made.
- **`validate`** scores the settled area-days in `outputs/area_predictions.csv`. Days within `labels.settle_days` of `--as-of` are not scored yet.
- **`demo`** writes synthetic inputs, including area locations, in the expected format.

## Configuration

- `[grid]` sets the cell size in degrees, the first delivery date used, and which request directions count as congestion.
- `[features]` sets the short and long history windows in days, the lag on price history, the hour on the day before delivery at which forecasts are issued, and how many quiet days an area stays active. Requests for the day before delivery are used only if they were announced before the issue time, and requests for the delivery day are never used. Set `issue_hour` to the hour you actually run `predict`, because the features assume everything announced before it is known.
- `[labels]` sets how many days after a delivery date the GOPACS record is treated as complete. Days after the last GOPACS announcement in the downloaded file are never labelled, so a stale download does not turn into quiet days.
- `[model]` sets the number of rolling test days, the area-day threshold and the boosting parameters. The threshold is 0.2, chosen to keep missed area-days low. Over 16 July to 15 October 2026, with the weather inputs, a threshold of 0.2 gave precision 0.63 and recall 0.77, 0.3 gave 0.71 and 0.65, and 0.4 gave 0.74 and 0.55.
- `[validation]` sets the recent window length and the minimum acceptable skill.

## Notes on the design

- Training is scored with a rolling daily backtest over the last `test_days` labelled days. Each test day gets its own model, trained on every label up to two days before it, because at 14:00 on the day before delivery that is the last complete day of requests. `train` reports the area-day score per day and in total, then fits the final model the same way on all labelled data. Its score covers only areas that were active, so `validate` and its `untracked` count show requests in areas that had no forecast.
- Targets after `label_end` are masked, so a feature table built for a prediction day never carries a label.
- The demo data follows a simple rule, so its scores are much higher than real performance will be.
- An area that had requests yesterday is much more likely to get one tomorrow, and this is forecast well. Areas that start again after quiet days are the main misses. Longer slot histories, other model types (random forest, extra trees, logistic regression) and a separate area-day model were tested in a rolling backtest and did not improve on this model.
- The local weather forecast was added after a 13-week rolling test (16 July to 15 October 2026, retrained weekly). At the 0.2 threshold it found the same 158 of 204 area-days with requests as the model without it, with 94 wrong flags instead of 119, and caught 350 of 422 requests instead of 344; AP rose from 0.740 to 0.751. At 0.3 it found fewer area-days (132 against 144). Calendar inputs (holidays, school holidays), details of the latest request, sun position and extra ENTSO-E inputs were tested the same way and left out because they lowered AP.
