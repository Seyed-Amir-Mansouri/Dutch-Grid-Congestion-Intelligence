# Dutch Grid Congestion Intelligence

This tool guesses which areas of the Dutch electricity grid will get a congestion request tomorrow.

When a grid operator expects a local grid to be overloaded, it asks the market for help through GOPACS: some parties are asked to produce less (feed-in congestion) or to use less (offtake congestion). Each request names the congested area, the place where the operator needs the help. The tool predicts, for every active area and every quarter-hour of the next day, how likely such a request is. For each area it then keeps the highest chance of the day. If that reaches 0.2, the area is flagged. A flag counts as right when the area gets any request that day, whatever the hour.

It comes with a small web app for running the steps and reading the results, and a command line for the same steps.

## How a forecast is made

There are two steps, and a third that helps with early forecasts.

1. **The predictor.** A gradient-boosted tree model (scikit-learn's `HistGradientBoostingClassifier`) gives a chance for each area and quarter-hour. It was trained on history from 1 April 2024. It uses 78 facts: the day-ahead market, the weather around the area, the area's own request history, notices from the grid operator, and the calendar.
2. **The inspector.** A second, smaller model looks at every flag the predictor raises and removes the ones that look wrong. It learns from past flags that were scored by a model that had not seen those days. It removes a flag when it is less than 50% sure the flag is right, unless the predictor's own score is already 0.45 or higher.
3. **The market estimator.** See the next section.

The tool only uses what was known when the forecast is made. Training stops two days before the day being forecast, because that is the last day whose requests are complete. A past day only counts requests that were announced before the issue time.

### Before and after the day-ahead results

The day-ahead price, net position and wind and solar forecasts come out around 13:00 on the day before. So there are two ways to issue a forecast, and each has its own model:

| Option | Issue time | What it uses |
| --- | --- | --- |
| After Day-Ahead | 14:00 | Everything, including the day-ahead results. 78 facts. |
| Before Day-Ahead | 11:00 | The same, without those five market facts (price, net position, wind, solar and residual load). 73 facts. |
| Before Day-Ahead with the market estimator | 11:00 | The 14:00 model, with the five market facts replaced by estimates. |

The market estimator is a set of small models, one each for price, net position, wind and solar. It works from the weather forecast, the load forecast, the calendar and the values of the day before and a week before. It is retrained on all earlier days each time it runs, so it has no model file. Residual load is then the load forecast minus the estimated wind and solar.

## How well it works

The numbers below come from a test on 42 days (20 September to 31 October 2026). For every day the model was retrained on the data that was available two days before, then forecast the day as if it was live. In that period 68 area-days had a request (150 requests in total), and 6 of those area-days were in areas the tool does not forecast.

| Option | Flagged | Right | Wrong | Missed | Requests in flagged areas |
| --- | --- | --- | --- | --- | --- |
| After Day-Ahead (14:00) | 41 | 37 | 4 | 31 | 109 |
| Before Day-Ahead (11:00) | 35 | 30 | 5 | 38 | 96 |
| Before Day-Ahead with the market estimator | 36 | 32 | 4 | 36 | 101 |

The market estimator helps overall but is weaker in the first two weeks of the test. How well it guesses the market values (correlation with the real values over the 42 days, where 1 is a perfect match): price 0.86, net position 0.39, wind 0.97, solar 0.98, residual load 0.98. Net position is the weak one.

The results are also on the Performance page of the app, split into 14-day windows. Most of the misses are areas that start up again after a few quiet days, and Enexis notices are the inspector's most useful input.

## The web app

Run `webui\app.bat` on Windows. The first time it sets up a virtual environment, installs what the app needs, downloads the shared data files from Google Drive and opens http://localhost:9110/ in the browser.

The pages:

- **Forecast.** The forecast for one day: a map, the flagged areas, and which flags the inspector kept or removed. Once the day is complete it also shows which flags were right.
- **Grid activity.** What GOPACS has been requesting lately, by day, area and grid operator.
- **Performance.** The test results described above, one section per option.
- **Model.** The trained model, the facts it uses and the settings.
- **Operations.** The steps you can run, a live log, and the list of data files.

The steps on the Operations page:

- **Daily forecast run.** Updates GOPACS, weather and market data, then makes the forecast for the day you pick. This is the one to use every day. Choose Before or After Day-Ahead. With Before you can tick "Use the market estimator".
- **Update GOPACS, Update weather, Update market data.** Download new data, starting from the last day you already have. Market data needs an ENTSO-E key (see Setup).
- **Locate areas.** Finds each GOPACS area on the map and gives it a weather grid cell. Needed when new areas appear.
- **Retrain model.** Trains the predictor and the inspector again on all complete days. It takes about half an hour.
- **Issue forecast.** Makes the forecast again for a day that already has one, without downloading anything. It is off until a first forecast exists.
- **Validate.** Compares past forecasts with the requests that really came and saves the result. It is also off until there is a forecast.

## Setup

You need Python 3.11 or newer. For the command line, from this folder:

```
python -m pip install -e ".[test,entsoe]"
python -m pytest
```

The market data comes from ENTSO-E. You need a free Web API token from transparency.entsoe.eu, set as the environment variable `ENTSOE_API_KEY`. The tool never writes it to a file.

To find where each GOPACS area is, the locate step uses three files that are in the repository: the GeoNames places and postcodes lists for the Netherlands (CC BY 4.0, geonames.org) and the CSC call-off report, which is used to read Enexis station codes. Hand-checked locations are in `reference/manual_locations.csv`.

### Sharing data between computers

The data, the trained models and the test results are not in git. They are kept in one zip on Google Drive. The `[drive]` section of `config.toml` holds its file ID, and `python -m congestion_nl fetch-drive` (or the Download from Drive button) unpacks it into place. If you upload a new zip, either upload it as a new version of the same Drive file or put the new ID in `config.toml`.

## Command line

Everything the app does can be run from the project folder with `python -m congestion_nl <command>`. Dates are written YYYY-MM-DD.

```
fetch-gopacs                          new requests and notices from GOPACS
fetch-weather --start D1 --end D2     weather per grid cell, after locate
fetch-entsoe --start D1 --end D2      prices, net position and forecasts
fetch-drive [--missing]               the shared zip from Drive
locate                                find the areas on the map
train [--issue-hour 11]               train the predictor and the inspector
estimate-market --start D1 [--end D2] estimate the day-ahead values for those days
predict --date D [--issue-hour 11 [--market-estimator]]
rolling-test --start D1 --end D2 [--issue-hour 11 [--market-estimator]]
validate [--as-of D]
demo --start D --days N               made-up input files for trying the tool out
```

`predict` writes one row per area to `outputs/area_predictions.csv` and one row per quarter-hour to `outputs/predictions.csv`, and prints the flagged areas with their risk window. The risk window runs from the first to the last quarter-hour with a chance of 0.1 or more. `rolling-test` is the test described above. It takes about an hour for 42 days and uses several GB of memory, so run one at a time.

`validate` marks each past forecast as right or wrong once the day is complete. It reports the counts of flagged, right, wrong and missed area-days, precision and recall, and how much better the model is than guessing the usual rate. If recent skill falls below the limit in `config.toml`, the command exits with status 2.

## The data

| File | What it holds | Comes from |
| --- | --- | --- |
| `data/raw/gopacs_events.csv` | every GOPACS request: area, direction, operator, MW, time span, when it was announced | fetch-gopacs |
| `data/raw/gopacs_notices.csv` | Enexis transport restriction notices | fetch-gopacs |
| `data/raw/sdac_results.csv` | day-ahead price and net position per quarter-hour | fetch-entsoe |
| `data/raw/system_forecasts.csv` | load, wind and solar forecasts | fetch-entsoe |
| `data/raw/weather.csv` | temperature, sun, cloud and wind per grid cell and hour | fetch-weather |
| `data/processed/area_locations.csv` | each area's point and grid cell | locate |

A few things worth knowing about it:

- All times are Dutch local time. A day has 96 quarter-hours, numbered 0 to 95. The extra hour when the clocks go back is averaged into one.
- The day-ahead market moved to 15-minute prices on 1 October 2025. Earlier hourly values are repeated over their four quarter-hours.
- The weather comes from Open-Meteo. The tool asks for the forecast that was made two days ahead, so every value was already published before the forecast is issued. Each grid cell is 0.25 degrees (about 28 by 17 km) and is only used to group nearby areas.
- GOPACS requests that were announced by mistake are listed in `reference/excluded_requests.csv` and dropped.
- A quarter-hour without a request counts as not congested. Days after the last downloaded announcement are never treated as quiet, so an old download does not turn into empty days.
- An area is active while it had a request in the last 90 days. New areas join after their first request.
- GOPACS can still change the period of a request after it is announced, and only the final period is published. The history facts can therefore look slightly better than they would have been live.

## Settings

Everything is in `config.toml`. The ones you are most likely to change:

- `[features]` has the issue hours (`issue_hours = [14, 11]`), the hour the day-ahead results are assumed to come out (`market_hour = 13`) and how many quiet days an area stays active.
- `[model]` has the flag threshold (`day_threshold = 0.2`) and the start of the risk window (`window_threshold = 0.1`). A lower threshold finds more requests but flags more areas that get none.
- `[inspector]` has the confidence needed to keep a flag (`veto_cut = 0.5`) and the predictor score above which a flag is never removed (`keep_above = 0.45`).
- `[validation]` has the length of the recent window in days and the lowest acceptable skill.
- `[drive]` has the Google Drive link of the shared zip.

## Folders

```
config.toml          settings and file locations
congestion_nl/       the forecasting code and the command line
webui/               the web app (Django), started with app.bat
reference/           hand-made lists: area locations, excluded requests, school holidays
tests/               unit and end-to-end tests
data/ models/ outputs/ upload/   local files, not in git
```

## Things we tried that did not help

Over the months many ideas were tested on the same kind of rolling test and left out because they did not improve the results: facts about neighbouring areas and operators, nationwide request counts, ENTSO-E outages, JAO shadow prices, sun position, onshore and offshore wind split, price differences with Germany and Belgium, other model types (random forest, extra trees, logistic regression), a separate area-day model, and weighting recent days more. Working day against weekend and public holiday was the last change that did help, and it is part of the 78 facts.
