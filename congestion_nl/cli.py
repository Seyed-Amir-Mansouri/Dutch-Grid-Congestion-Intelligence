from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

from . import estimator, ledger, pipeline, synthetic, validate
from .config import DEFAULT_CONFIG, load_config


def _timestamp(text: str) -> pd.Timestamp:
    """Parse a YYYY-MM-DD argument into a normalized timestamp."""
    return pd.Timestamp(text).normalize()


def _today() -> pd.Timestamp:
    """Return today's date in UTC as a normalized timestamp."""
    return pd.Timestamp(datetime.now(timezone.utc).date())


def cmd_demo(args: argparse.Namespace) -> None:
    """Write synthetic input files to the configured paths."""
    config = load_config(args.config)
    written = synthetic.generate(config, start=args.start, days=args.days, seed=args.seed)
    for name, path in written.items():
        print(f"{name}: {path}")


def cmd_fetch_entsoe(args: argparse.Namespace) -> None:
    """Fetch SDAC prices, net position and forecasts from ENTSO-E and write CSVs."""
    from . import entsoe_source

    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        print("ENTSOE_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)
    client = entsoe_source.build_client(api_key)
    start = _timestamp(args.start).tz_localize("Europe/Amsterdam")
    end = _timestamp(args.end).tz_localize("Europe/Amsterdam")

    sdac = entsoe_source.fetch_sdac(client, start, end)
    forecasts = entsoe_source.fetch_forecasts(client, start, end)

    sdac_path = config.resolve(config.paths.sdac)
    forecasts_path = config.resolve(config.paths.forecasts)
    sdac_path.parent.mkdir(parents=True, exist_ok=True)
    forecasts_path.parent.mkdir(parents=True, exist_ok=True)
    sdac.to_csv(sdac_path, index=False)
    forecasts.to_csv(forecasts_path, index=False)
    print(f"sdac: {sdac_path} ({len(sdac)} rows)")
    print(f"forecasts: {forecasts_path} ({len(forecasts)} rows)")


def cmd_fetch_gopacs(args: argparse.Namespace) -> None:
    """Fetch historical congestion-management announcements from GOPACS."""
    from . import gopacs_source

    config = load_config(args.config)
    records = gopacs_source.fetch_announcements()
    events = gopacs_source.to_events(records)

    path = config.resolve(config.paths.gopacs)
    path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(path, index=False)
    print(f"gopacs: {path} ({len(events)} rows from {len(records)} announcements)")


def cmd_train(args: argparse.Namespace) -> None:
    """Build the dataset from history and save a trained model."""
    config = load_config(args.config)
    inputs = pipeline.load_inputs(config)
    dataset = pipeline.build(config, inputs, pipeline.label_cutoff(config))
    bundle, report = pipeline.train(config, dataset)
    path = estimator.save(bundle, config.resolve(config.paths.model_dir))
    print(json.dumps(report, indent=2, default=str))
    print(f"saved: {path}")


def cmd_predict(args: argparse.Namespace) -> None:
    """Issue risk predictions for one delivery day and log them."""
    config = load_config(args.config)
    bundle = estimator.load(config.resolve(config.paths.model_dir))
    inputs = pipeline.load_inputs(config)
    dataset = pipeline.build(config, inputs, None)

    if args.date:
        day = _timestamp(args.date)
    else:
        day = _today() + pd.Timedelta(days=1)
    issued_at = datetime.fromisoformat(args.issued_at) if args.issued_at else None
    predictions = pipeline.predict_day(config, bundle, dataset, day, issued_at)

    ledger_path = config.resolve(config.paths.predictions)
    ledger.append(predictions, ledger_path)
    flagged = int(predictions["risk_flag"].sum())
    threshold = bundle["threshold"]
    print(f"{len(predictions)} rows for {day:%Y-%m-%d}, {flagged} flagged at threshold {threshold}")
    print(f"logged: {ledger_path}")


def cmd_validate(args: argparse.Namespace) -> None:
    """Score logged predictions against settled GOPACS events and append the result."""
    config = load_config(args.config)
    as_of = _timestamp(args.as_of) if args.as_of else _today()
    label_end = as_of - pd.Timedelta(days=config.labels.settle_days)

    predictions = ledger.load(config.resolve(config.paths.predictions))
    inputs = pipeline.load_inputs(config)
    result = validate.evaluate(config, predictions, inputs.events, label_end, as_of)
    history = ledger.append_history(result, config.resolve(config.paths.validation))

    print(json.dumps(result, indent=2, default=str))
    print(f"history rows: {len(history)}")
    if result["alert"]:
        print("ALERT: window skill is below the configured minimum", file=sys.stderr)
        sys.exit(2)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser with one subcommand per workflow step."""
    parser = argparse.ArgumentParser(prog="congestion-nl")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to config.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="write synthetic input files")
    demo.add_argument("--start", default="2025-01-01")
    demo.add_argument("--days", type=int, default=365)
    demo.add_argument("--seed", type=int, default=7)
    demo.set_defaults(func=cmd_demo)

    fetch = sub.add_parser("fetch-entsoe", help="fetch NL SDAC and forecast data from ENTSO-E")
    fetch.add_argument("--start", required=True, help="start date YYYY-MM-DD")
    fetch.add_argument("--end", required=True, help="end date YYYY-MM-DD")
    fetch.set_defaults(func=cmd_fetch_entsoe)

    fetch_gopacs = sub.add_parser("fetch-gopacs", help="fetch congestion events from GOPACS")
    fetch_gopacs.set_defaults(func=cmd_fetch_gopacs)

    train = sub.add_parser("train", help="train and save the risk model")
    train.set_defaults(func=cmd_train)

    predict = sub.add_parser("predict", help="issue predictions for a delivery day")
    predict.add_argument("--date", help="delivery date YYYY-MM-DD, default tomorrow")
    predict.add_argument("--issued-at", help="ISO timestamp recorded as issue time")
    predict.set_defaults(func=cmd_predict)

    check = sub.add_parser("validate", help="score predictions against GOPACS events")
    check.add_argument("--as-of", help="evaluation date YYYY-MM-DD, default today")
    check.set_defaults(func=cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the selected command."""
    args = build_parser().parse_args(argv)
    args.func(args)
