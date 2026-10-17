# Dutch Grid Congestion Risk

This estimator predicts, for each delivery hour and 0.25° grid cell of the Netherlands (about 28 × 17 km), the probability that a Dutch grid operator publishes a GOPACS congestion-management request covering that cell. It is trained and checked against the requests that GOPACS publishes, and cross-checked against the CSC call-off report of the grid operators.

## What the code does

1. **Fetch.** It downloads the NL day-ahead price, net position and load, wind and solar forecasts from ENTSO-E, and every public GOPACS announcement.
2. **Direction.** Each congestion-management request names its area on one side only. An area on the buy side is **feed-in** congestion (parties inside should produce less), and an area on the sell side is **offtake** congestion (parties inside should use less).
3. **Locate.** Each GOPACS area is placed at a point and given a grid cell, using in order:
   - a hand-checked row in `reference/manual_locations.csv`
   - the postcodes that GOPACS lists for the area (median of the postcode centroids)
   - the towns in a town-list area name ("EINDHOVEN, VELDHOVEN")
   - an Enexis station code ("LIMM Congestie11_P" is Limmel), read from the CSC report
   - the substation or town in the name ("RS WASKEMEER 10-2i" is Waskemeer)

   TenneT's province-wide requests are too broad for a cell and are left out.
4. **Build.** It builds one row per delivery date, hour and cell, using only information known before the delivery day:
   - the national day-ahead price, net position, forecasts and residual load (load minus wind and solar)
   - the price history for the same hour
   - how often that cell, and that hour in that cell, had a request over the last 7 and 28 days, lagged by two days
5. **Predict.** A gradient-boosted classifier gives a probability per row. A row is flagged when its probability reaches the model's threshold. Training picks that threshold as the lowest one whose precision on the most recent window reaches `target_precision`, so the flag trades recall for fewer false alarms. The predictions go to a ledger with their issue time, model version and flag.
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
    labels.py             expands GOPACS intervals into hourly 0/1 labels per cell
    features.py           builds features and the target, with lags to prevent look-ahead
    rolling.py            trailing-window aggregates shifted by a reporting lag
    estimator.py          gradient-boosted classifier: holdout scoring, refit, save, load, predict
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
| SDAC results | `delivery_date, hour, price_eur_mwh, net_position_mw` | `congestion-nl fetch-entsoe` |
| System forecasts | `delivery_date, hour, load_forecast_mw, wind_forecast_mw, solar_forecast_mw` | `congestion-nl fetch-entsoe` |
| GOPACS events | `event_id, area, direction, operator, postcodes, mw, start_time, end_time` | `congestion-nl fetch-gopacs` |
| Area locations | `area, lat, lon, cell, source` | `congestion-nl locate` |

- **Time and hours.** All times are Dutch local clock time, and hours run from 0 to 23. The repeated hour on the autumn clock change is averaged into one row.
- **Market data.** The Netherlands is a single bidding zone and only day-ahead results are used, so the market inputs are national and the same for every cell.
- **Cells.** A cell id names its south-west corner, for example `52.00N 5.00E`. Only cells holding at least one located area are modelled.
- **Missing hours.** An hour without a request is treated as not congested.

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
- **`predict`** writes to `outputs/predictions.csv`. `--issued-at` records when the forecast was made.
- **`validate`** scores settled days. Days within `labels.settle_days` of `--as-of` are not scored yet.
- **`demo`** writes synthetic inputs, including area locations, in the expected format.

## Configuration

- `[grid]` sets the cell size in degrees, the first delivery date used, and which request directions count as congestion.
- `[features]` sets the short and long history windows in days, the lag on price history and the lag on request history. The request lag is two days, so requests for the day before delivery are never used.
- `[labels]` sets how many days after a delivery date the GOPACS record is treated as complete.
- `[model]` sets the holdout length, the target precision for the flag, the fallback threshold used when that target cannot be reached, and the boosting parameters. The holdout score uses a threshold picked on the window before the holdout, so the holdout never informs its own threshold.
- `[validation]` sets the recent window length and the minimum acceptable skill.

## Notes on the design

- Training uses a time-based split. The last `test_days` of labelled data are held out for scoring, then the final model is refit on all labelled data.
- Targets after `label_end` are masked, so a feature table built for a prediction day never carries a label.
- GOPACS was little used before 2024, so training starts on 2024-01-01.
- The set of cells comes from all located areas, so a cell that only appears later is still present, with no history, in earlier days.
- The demo data follows a simple rule, so its scores are much higher than real performance will be.
