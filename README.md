# Dutch Grid Congestion Risk

This estimator predicts, for each delivery date, hour and GOPACS congestion area, the probability that a Dutch grid operator will publish a congestion-management request. It is checked against the congestion-management events that GOPACS actually publishes.

## What the code does

Each day, after the Single Day-Ahead Coupling (SDAC) results for the next day are known, the estimator:

1. Reads the national day-ahead prices and net position, the load, wind and solar forecasts, optional planned outages per area, and the GOPACS event history.
2. Builds one row per delivery date, hour and area, using only information available before the delivery day: national residual load (load minus wind and solar), the price history for the same hour, and how often that hour and area were congested in the recent past.
3. Scores each row with a gradient-boosted classifier that outputs a congestion probability.
4. Writes the predictions to a ledger with the time they were issued and the model version.

Once GOPACS events for those days are settled, the validation step joins each logged prediction to what actually happened. It computes Brier score, skill against a climatology baseline, ROC AUC, precision and recall, both over the full history and over a recent window. It then appends the result to a history file. If recent skill falls below the configured minimum, the command exits with status 2 so a scheduler can raise an alert.

## Layout

```
README.md
pyproject.toml
config.toml             paths, areas, feature windows, model and validation settings
congestion_nl/
    config.py           loads config.toml into typed objects
    schemas.py          required columns for every input and output file
    ingest.py           reads and checks the raw CSV inputs
    labels.py           expands GOPACS intervals into hourly 0/1 congestion labels
    features.py         builds features and the target, with lags to prevent look-ahead
    pipeline.py         loads inputs, builds datasets, trains and issues daily predictions
    validate.py         scores the latest logged predictions against settled GOPACS events
    entsoe_source.py    fetches NL SDAC prices and forecasts from ENTSO-E
    gopacs_source.py    fetches the public GOPACS announcements and turns them into events
    synthetic.py        generates demo input files in the expected format
    estimator.py        gradient-boosted classifier: holdout scoring, refit, save, load, predict
    metrics.py          Brier score, skill, ROC AUC, precision and recall
    rolling.py          trailing-window aggregates shifted by a reporting lag
    ledger.py           reads and appends the prediction and validation CSVs
    cli.py              command line entry point
tests/                  unit and end-to-end tests on synthetic data
data/ models/ outputs/  local files, not tracked
```

Relative paths in `config.toml` are resolved from this folder.

## Setup

Requires Python 3.11 or newer. From this folder:

```
python -m pip install -e ".[test,entsoe]"
python -m pytest
```

The ENTSO-E fetch needs a free Web API token from `transparency.entsoe.eu` (Account Settings → Web API Access), set as the environment variable `ENTSOE_API_KEY`. It is never written to config or data files.

## Inputs

| File | Columns | Source |
| --- | --- | --- |
| SDAC results (national) | `delivery_date, hour, price_eur_mwh, net_position_mw` | `congestion-nl fetch-entsoe` |
| System forecasts (national) | `delivery_date, hour, load_forecast_mw, wind_forecast_mw, solar_forecast_mw` | `congestion-nl fetch-entsoe` |
| Planned outages (optional, per area) | `delivery_date, hour, area, outage_mw` | own source |
| GOPACS events (per area) | `event_id, area, start_time, end_time` | `congestion-nl fetch-gopacs` |

- **Time and hours.** All times are Dutch local clock time, and hours run from 0 to 23.
- **Geography.** The Netherlands is a single SDAC bidding zone, so the market and forecast files are national. Outages and GOPACS events carry a congestion `area`.
- **Area codes.** `area` values must match `[areas] codes` in `config.toml`. The codes there are placeholders until they are replaced with real GOPACS area names.
- **Missing hours.** An hour without an event row is treated as not congested, so gaps in the feed show up as false negatives.

`fetch-entsoe` resamples everything to hourly means, because some prices are published at 15-minute resolution. `fetch-gopacs` pages through the public GOPACS announcements endpoint and keeps the congestion-management announcements. Each announcement names the congestion area on either its buy side or its sell side. The other side is a generic placeholder ("all other areas", or "Buiten <area>"), which is dropped.

## Commands

```
congestion-nl fetch-entsoe --start 2024-01-01 --end 2026-01-01
congestion-nl fetch-gopacs
congestion-nl demo --start 2025-01-01 --days 400
congestion-nl train
congestion-nl predict --date 2026-02-01 --issued-at 2026-01-31T11:00:00+00:00
congestion-nl validate --as-of 2026-02-03
```

- **`demo`** writes synthetic inputs in the expected format.
- **`train`** saves the model to `models/`.
- **`predict`** writes to `outputs/predictions.csv`. `--issued-at` records when the forecast was made.
- **`validate`** scores settled days. Days within `labels.settle_days` of `--as-of` are not scored yet.

## Configuration

- `[features]` sets the trailing window in days (`history_days`), the lag on price history (`price_lag_days`) and the lag on congestion history (`event_lag_days`). The event lag is two days, so events from the day before delivery are never used.
- `[labels]` sets how many days after a delivery date the GOPACS record is treated as complete.
- `[model]` sets the holdout length, the decision threshold and the boosting parameters.
- `[validation]` sets the recent window length and the minimum acceptable skill.

## Notes on the design

- Training uses a time-based split. The last `test_days` of labelled data are held out for scoring, then the final model is refit on all labelled data.
- Targets after `label_end` are masked, so a feature table built for a prediction day never carries a label.
- The demo data follows a simple rule, so its scores are much higher than real performance will be.
