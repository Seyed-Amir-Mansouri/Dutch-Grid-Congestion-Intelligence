# Dutch Grid Congestion Risk

This project predicts whether the Dutch transmission grid will need congestion management on the next delivery day, and checks those predictions against the congestion-management events that GOPACS actually publishes.

## What the code does

Each day, after the Single Day-Ahead Coupling (SDAC) results for the next day are known, the framework:

1. Reads the day-ahead clearing prices and net positions, the system load, wind and solar forecasts, and any planned grid outages.
2. Builds one row per delivery date, hour and congestion area with features that only use information available before the delivery day. This includes the national residual load (load minus wind and solar), the historical price for the same hour, and how often that hour and area were congested in the recent past.
3. Scores each row with a gradient-boosted classifier that outputs a congestion probability.
4. Writes the predictions to a ledger together with the time they were issued and the model version.

Later, once GOPACS events for those days are settled, the validation step joins each logged prediction to the events that actually happened. It computes Brier score, skill against a climatology baseline, ROC AUC, precision and recall over the full history and a recent window, and appends the result to a history file. If the recent skill drops below a configured minimum, the command exits with status 2 so a scheduler can raise an alert.

## Layout

```
config.toml                 paths, areas, feature windows, model and validation settings
src/grid_congestion/
    config.py               loads config.toml into typed objects
    schemas.py              required columns for every input and output file
    ingest.py               reads and checks the raw CSV inputs
    labels.py               expands GOPACS intervals into hourly 0/1 congestion labels
    features.py             builds features and the target, with lags to prevent look-ahead
    metrics.py              Brier, skill, AUC, precision and recall
    model.py                training, holdout scoring, saving and loading the model
    pipeline.py             loads inputs, builds datasets and issues daily predictions
    validate.py             scores the prediction ledger against settled GOPACS events
    synthetic.py            generates demo input files in the expected format
    entsoe_source.py        fetches NL SDAC prices and forecasts from ENTSO-E
    cli.py                  command line entry point
tests/                      unit and end-to-end tests on synthetic data
```

## Input data

Place these files at the paths set in `config.toml`. All times are Dutch local clock time. Hours are integers from 0 to 23.

The Netherlands is a single SDAC bidding zone, so market data has no area breakdown. GOPACS congestion happens at specific grid locations, which is a separate and finer geography. The SDAC and forecast files are therefore national (one row per date and hour); the outage and GOPACS files carry a congestion `area`.

| File | Columns |
| --- | --- |
| SDAC results (national) | `delivery_date, hour, price_eur_mwh, net_position_mw` |
| System forecasts (national) | `delivery_date, hour, load_forecast_mw, wind_forecast_mw, solar_forecast_mw` |
| Planned outages (optional, per area) | `delivery_date, hour, area, outage_mw` |
| GOPACS events (per area) | `event_id, area, start_time, end_time` |

`area` values must match the codes in `[areas] codes`. Those codes are placeholders; replace them with the real GOPACS congestion-location codes once you have access to that feed. The GOPACS file must cover the full history you train on. An hour with no event row is treated as not congested, so gaps in the feed will show up as false negatives.

The prediction ledger and validation history are written by the framework to `outputs/`.

## Fetching NL market data from ENTSO-E

The SDAC price, net position and forecast files can be pulled directly from the ENTSO-E Transparency Platform.

1. Register a free account at `transparency.entsoe.int` and generate a Web API token from Account Settings → Web API Access.
2. Set it as an environment variable named `ENTSOE_API_KEY`. Do not commit it or put it in `config.toml`.
3. Install the optional dependency: `python -m pip install -e ".[entsoe]"`
4. Run:

```
grid-congestion fetch-entsoe --start 2024-01-01 --end 2026-01-01
```

This writes the `sdac` and `forecasts` files at the paths set in `config.toml`. Prices on some zones are published at 15-minute resolution; the fetch step resamples everything to hourly means to match this project's schema. Planned outages and GOPACS events are not available from ENTSO-E and need their own source.

## Setup

Requires Python 3.11 or newer.

```
python -m pip install -e ".[test]"
```

## Usage

Generate demo inputs in the expected format:

```
grid-congestion demo --start 2025-01-01 --days 400
```

Train on the labelled history and save the model to `models/`:

```
grid-congestion train
```

Issue predictions for a delivery day. The default is tomorrow. The `--issued-at` option records the time the forecast was made:

```
grid-congestion predict --date 2026-02-01 --issued-at 2026-01-31T11:00:00+00:00
```

Score settled predictions against GOPACS events. `--as-of` sets the evaluation date. Days within `labels.settle_days` of it are not yet scored:

```
grid-congestion validate --as-of 2026-02-03
```

Run the tests:

```
python -m pytest
```

## Configuration

`config.toml` controls everything tunable:

- `[features]` sets the trailing window in days (`history_days`), the lag that applies to price history (`price_lag_days`), and the lag that applies to congestion history (`event_lag_days`). The event lag is set to two days so that events from the day before the delivery day are never used.
- `[labels]` sets how many days after a delivery date the GOPACS record is treated as complete.
- `[model]` sets the holdout length, the decision threshold and the boosting parameters.
- `[validation]` sets the recent window length and the minimum acceptable skill.

## Notes on the design

- Training uses a time-based split. The last `test_days` of labelled data are held out for scoring, and the final model is refit on all labelled data.
- Targets after `label_end` are masked, so a feature table built for a prediction day never carries a label.
- The demo data is generated from a simple rule, so its scores are much higher than real performance will be. Use real SDAC, forecast and GOPACS files before drawing any conclusions about model quality.
