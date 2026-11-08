"""Run the tool's command-line steps in the background, one job at a time, with a log per job."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta

from django.conf import settings

PYTHON = os.environ.get("PIPELINE_PYTHON", sys.executable)
STAGES = {
    "fetch-gopacs": ("GOPACS", "Requests and notices"),
    "fetch-weather": ("Weather", "Open-Meteo per grid cell"),
    "fetch-entsoe": ("Market", "ENTSO-E prices and forecasts"),
    "locate": ("Locate", "Areas to grid cells"),
    "train": ("Train", "Predictor and inspector"),
    "predict": ("Forecast", "Score areas, apply the veto"),
    "validate": ("Validate", "Score against GOPACS"),
    "forecast-market": ("Forecaster", "Day-ahead price, wind and solar"),
}
# short stage names for the files a Drive download replaces
DRIVE_STAGES = {"gopacs": "Requests", "notices": "Notices", "sdac": "Market", "forecasts": "Forecasts",
                "weather": "Weather", "locations": "Locations", "model": "Model", "inspector_history": "Inspector",
                "walk_forward": "Test", "walk_forward_forecaster": "Test with forecaster", "bundle": "All files"}
PROGRESS = re.compile(r"progress: (\d+)%")
# files the steps read: a label and the step that makes the file, or None for reference files
FILES = {
    "gopacs": ("GOPACS requests", "Update GOPACS"),
    "weather": ("Weather forecasts", "Update weather"),
    "sdac": ("Day-ahead market", "Update market data"),
    "forecasts": ("System forecasts", "Update market data"),
    "locations": ("Area locations", "Locate areas"),
    "model": ("Trained model", "Retrain model"),
    "area_predictions": ("Area forecasts ledger", "Daily forecast run or Issue forecast"),
    "postcodes": ("GeoNames postcodes", None),
    "places": ("GeoNames places", None),
    "csc_report": ("Enexis CSC report", None),
    "manual_locations": ("Manual locations", None),
}
INPUTS = ["gopacs", "weather", "sdac", "forecasts", "locations"]
NEEDS = {
    "gopacs": [],
    "weather": ["locations"],
    "entsoe": [],
    "locate": ["gopacs", "postcodes", "places", "csc_report", "manual_locations"],
    "train": INPUTS,
    "predict": ["model"] + INPUTS,
    "validate": ["area_predictions"] + INPUTS,
}
SECRET = re.compile(r"(securityToken=)[^&\s\"']+", re.IGNORECASE)
_lock = threading.Lock()
_running: dict = {}
# the command process of each running job, and the jobs asked to stop
_process: dict = {}
_stopping: set = set()
# the last saved state of each running job, so the page never reads a file that is being rewritten
_saved: dict = {}


def _today():
    """Today's date in Amsterdam."""
    return datetime.now().date()


def _now() -> str:
    """The local time as an ISO string, to the second."""
    return datetime.now().isoformat(timespec="seconds")


def steps() -> list[dict]:
    """The steps an operator can start, each with its form fields and default values."""
    from . import data

    today = _today()
    tomorrow = today + timedelta(days=1)
    forecast_days = [d.strftime("%Y-%m-%d") for d in data.forecast_days()]
    options = [{**o, "text": o["note"], "note": f"Issued at {o['text']}"} for o in data.hours()]
    hour = {"name": "hour", "label": "Issued at", "type": "select", "options": options}
    # offered only for an issue hour before the day-ahead results
    use = {"name": "forecaster", "label": "Use the forecaster", "type": "check", "note": ""}
    found = [
        {"key": "daily", "title": "Daily forecast run", "kind": "chain", "lead": True,
         "text": "Refresh GOPACS, weather and market data, then issue tomorrow's forecast with the inspector's verdict.",
         "fields": [{"name": "date", "label": "Delivery day", "type": "date", "value": tomorrow}, hour, use]},
        {"key": "gopacs", "title": "Update GOPACS", "kind": "data",
         "text": "Download congestion requests and transport-restriction notices from GOPACS.", "fields": []},
        {"key": "weather", "title": "Update weather", "kind": "data",
         "text": "Fetch Open-Meteo forecasts for every grid cell and merge them into the weather file.",
         "fields": []},
        {"key": "entsoe", "title": "Update market data", "kind": "data", "needs_key": True,
         "text": "Fetch day-ahead prices, net position and system forecasts from ENTSO-E and merge them into the "
                 "market files.",
         "fields": []},
        {"key": "locate", "title": "Locate areas", "kind": "model",
         "text": "Place every GOPACS area on the map and assign its grid cell.", "fields": []},
        {"key": "train", "title": "Retrain model", "kind": "model",
         "text": "Train the predictor and the inspector on all settled history. Takes about half an hour.", "fields": [hour]},
        {"key": "predict", "title": "Issue forecast", "kind": "model",
         "text": "Score every active area for one delivery day and log it in the forecast ledger. Unlike the Daily forecast run, it downloads nothing, so it is fast, works without GOPACS or ENTSO-E, and can redo an older forecast day.",
         "fields": [{"name": "date", "label": "Delivery day", "type": "days",
                     "options": forecast_days}, hour, use]},
        {"key": "validate", "title": "Validate", "kind": "model",
         "text": "Score settled forecasts against GOPACS requests and append to the validation history.",
         "fields": [{"name": "as_of", "label": "As of", "type": "days", "options": forecast_days}]},
    ]
    for step in found:
        step["missing"] = missing(step["key"])
        if step["key"] == "predict" and not data.forecast_days():
            step["missing"].append("No forecast yet: run Daily forecast run first")
    return found


def entsoe_key_set() -> bool:
    """Whether the ENTSO-E key is in the environment; its value is never read here."""
    return bool(os.environ.get("ENTSOE_API_KEY"))


def needs(key: str) -> list[str]:
    """The files a step reads before it can run."""
    if key == "daily":
        # the chain writes the GOPACS and weather files itself, and the market files when the key is set
        return ["locations", "model"] + ([] if entsoe_key_set() else ["sdac", "forecasts"])
    return NEEDS[key]


def missing(key: str) -> list[str]:
    """A warning for each file a step needs that is not there, naming the step that makes it."""
    from congestion_nl import estimator

    from . import data

    out = []
    for name in needs(key):
        file = data.model_dir() / estimator.MODEL_FILE if name == "model" else data.path(name)
        if file.exists():
            continue
        label, maker = FILES[name]
        if maker:
            out.append(f"{label} missing: run {maker} first")
        else:
            where = file.relative_to(data.CONFIG.root) if file.is_relative_to(data.CONFIG.root) else file
            out.append(f"{label} missing: add {where.as_posix()}")
    return out


def drive_ready() -> bool:
    """Whether config.toml has Drive links to download from."""
    from . import data

    return bool(data.CONFIG.drive)


def commands(key: str, form: dict) -> list[list[str]]:
    """The command lines for a step, from the submitted form."""
    day = lambda name: datetime.strptime(form[name], "%Y-%m-%d").strftime("%Y-%m-%d")
    # the issue hour as a checked number, for the steps that ask for it
    at = lambda: ["--issue-hour", str(int(form["hour"]))] if form.get("hour") else []
    # the forecaster only applies before the day-ahead results
    early = lambda: form.get("forecaster") and form.get("hour") and int(form["hour"]) < _market_hour()
    use = lambda: [*at(), "--forecaster"] if early() else at()
    # with the forecaster, its own stage first
    market = lambda first, last: [["forecast-market", "--start", first, "--end", last]] if early() else []
    single = {
        "gopacs": lambda: [["fetch-gopacs"]],
        "weather": lambda: [["fetch-weather", "--start", _data().weather_start(_today(), 2),
                             "--end", str(_today() + timedelta(days=1))]],
        "entsoe": lambda: [["fetch-entsoe", "--start", _data().market_start(_today()),
                            "--end", str(_today() + timedelta(days=2))]],
        "locate": lambda: [["locate"]],
        "train": lambda: [["train", *at()]],
        "predict": lambda: [*market(day("date"), day("date")), ["predict", "--date", day("date"), *use()]],
        "validate": lambda: [["validate", "--as-of", day("as_of")]],
    }
    if key in single:
        return single[key]()
    if key == "drive":
        from . import data

        from congestion_nl import drive_source

        # with a bundle link, the single files are not downloaded as well
        names = [drive_source.BUNDLE] if drive_source.BUNDLE in data.CONFIG.drive else list(data.CONFIG.drive)
        return [["fetch-drive", name] for name in names]
    if key == "daily":
        from . import data

        target = datetime.strptime(form["date"], "%Y-%m-%d").date()
        chain = [["fetch-gopacs"],
                 ["fetch-weather", "--start", data.weather_start(target, 3), "--end", str(target)]]
        if entsoe_key_set():
            chain.append(["fetch-entsoe", "--start", data.market_start(target), "--end", str(target + timedelta(days=1))])
        return chain + market(str(target), str(target)) + [["predict", "--date", str(target), *use()]]
    raise KeyError(key)


def _data():
    from . import data

    return data


def _market_hour() -> int:
    from . import data

    return data.CONFIG.features.market_hour


def _meta_path(job_id: str):
    return settings.RUNS_DIR / f"{job_id}.json"


def _write(meta: dict) -> None:
    """Save a job's state next to its log, swapping in a whole new copy so no reader sees half a file."""
    text = json.dumps(meta, indent=1)
    with _lock:
        _saved[meta["id"]] = text
    path = _meta_path(meta["id"])
    part = path.with_suffix(".tmp")
    part.write_text(text, encoding="utf-8")
    for _ in range(20):
        try:
            os.replace(part, path)
            return
        except PermissionError:
            # Windows refuses while another process, such as Drive sync, has the file open
            time.sleep(0.1)
    os.replace(part, path)


def _load(file) -> dict | None:
    """A job's state: from memory while it runs, otherwise from its file; None for a file cut off mid-write.

    Jobs cut off by a restart count as stopped.
    """
    with _lock:
        text = _saved.get(file.stem) if file.stem in _running else None
    try:
        meta = json.loads(text or file.read_text(encoding="utf-8"))
    except ValueError:
        return None
    if meta["status"] == "running" and meta["id"] not in _running:
        meta["status"] = "stopped"
    return meta


def busy() -> dict | None:
    """The job that is running now, if any."""
    with _lock:
        return next(iter(_running.values()), None)


def start(key: str, title: str, form: dict) -> dict:
    """Start a step in a background thread; refuses while another job runs."""
    lines = commands(key, form)
    with _lock:
        if _running:
            raise RuntimeError("Another job is still running")
        settings.RUNS_DIR.mkdir(exist_ok=True)
        now = datetime.now()
        meta = {"id": now.strftime("%Y%m%d-%H%M%S") + f"-{key}", "key": key, "title": title,
                "commands": [" ".join(["congestion-nl"] + line) for line in lines],
                "status": "running", "started": now.isoformat(timespec="seconds"), "ended": None, "step": 0,
                "timing": []}
        _running[meta["id"]] = meta
    _write(meta)
    threading.Thread(target=_run, args=(meta, lines), daemon=True).start()
    return meta


def _run(meta: dict, lines: list[list[str]]) -> None:
    """Run the command lines in order and stop at the first failure."""
    log = settings.RUNS_DIR / f"{meta['id']}.log"
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    status = "done"
    try:
        with log.open("w", encoding="utf-8") as out:
            meta["timing"] = []
            for i, line in enumerate(lines):
                meta["step"] = i + 1
                meta["timing"].append({"started": _now(), "ended": None, "code": None})
                _write(meta)
                out.write(f"$ congestion-nl {' '.join(line)}\n")
                out.flush()
                with _lock:
                    # asked to stop between two commands: do not start the next one
                    if meta["id"] in _stopping:
                        meta["timing"][-1].update(ended=_now(), stopped=True)
                        status = "stopped"
                        break
                    process = subprocess.Popen([PYTHON, "-m", "congestion_nl", *line], cwd=settings.PROJECT_ROOT,
                                               stdout=out, stderr=subprocess.STDOUT, env=env,
                                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    _process[meta["id"]] = process
                code = process.wait()
                stopped = meta["id"] in _stopping
                meta["timing"][-1].update(ended=_now(), code=code, stopped=stopped)
                _write(meta)
                out.write("[stopped by the user]\n\n" if stopped else f"[exit {code}]\n\n")
                out.flush()
                if stopped:
                    status = "stopped"
                    break
                if code != 0:
                    status = "failed"
                    break
    except Exception as error:  # noqa: BLE001 - the page must show why a job died
        with log.open("a", encoding="utf-8") as out:
            out.write(f"\n{type(error).__name__}: {error}\n")
        status = "failed"
    meta.update(status=status, ended=_now())
    _write(meta)
    with _lock:
        _running.pop(meta["id"], None)
        _process.pop(meta["id"], None)
        _stopping.discard(meta["id"])
        _saved.pop(meta["id"], None)


def stop(job_id: str) -> None:
    """Stop a running job: end the command that is running now and skip the rest."""
    with _lock:
        if job_id not in _running:
            raise RuntimeError("This job is not running")
        _stopping.add(job_id)
        process = _process.get(job_id)
        if process and process.poll() is None:
            process.terminate()


def clear(with_model: bool) -> int:
    """Delete the downloaded data, the outputs and the job history, and the model too if asked; returns the files deleted.

    Reference files (GeoNames, the CSC report, reference/) and the config are never touched.
    """
    from congestion_nl import inspector

    from . import data

    groups = ["Data", "Outputs"] + (["Model"] if with_model else [])
    files = [data.file_path(key) for group, names in data.FILE_GROUPS if group in groups for key, _, _ in names]
    if with_model:
        # not listed in the Files panel, but made by training like the model
        files += [data.file_path(f"inspector_history{'' if i == 0 else f'_{h}'}") for i, h in enumerate(data.CONFIG.hours)]
        files.append(data.model_dir() / inspector.FORECASTER_HISTORY)
    if settings.RUNS_DIR.exists():
        files += [file for kind in ("*.json", "*.tmp", "*.log") for file in settings.RUNS_DIR.glob(kind)]
    with _lock:
        if _running:
            raise RuntimeError("A job is running; wait for it to finish first")
        found = [file for file in files if file.exists()]
        for file in found:
            file.unlink()
    return len(found)


def history(limit: int = 15) -> list[dict]:
    """Recent jobs, newest first; jobs cut off by a restart are shown as stopped."""
    if not settings.RUNS_DIR.exists():
        return []
    out = []
    for file in sorted(settings.RUNS_DIR.glob("*.json"), reverse=True)[:limit]:
        meta = _load(file)
        if meta:
            out.append(meta)
    return out


def log_tail(job_id: str, lines: int | None = 400) -> str:
    """The end of a job's log, or all of it when lines is None, with any API token hidden."""
    if not re.fullmatch(r"[\w-]+", job_id):
        return ""
    file = settings.RUNS_DIR / f"{job_id}.log"
    if not file.exists():
        return ""
    text = file.read_text(encoding="utf-8", errors="replace").splitlines()
    text = text[-lines:] if lines else text
    return SECRET.sub(r"\1***", "\n".join(text))


def seconds(start: str | None, end: str | None) -> float | None:
    """Seconds between two ISO times; up to now when the end is missing."""
    if not start:
        return None
    finish = datetime.fromisoformat(end) if end else datetime.now()
    return max((finish - datetime.fromisoformat(start)).total_seconds(), 0)


def typical_seconds() -> dict[str, float]:
    """The median time each command took in past successful jobs."""
    found: dict[str, list[float]] = {}
    if settings.RUNS_DIR.exists():
        for file in sorted(settings.RUNS_DIR.glob("*.json"), reverse=True)[:60]:
            meta = _load(file) or {"commands": []}
            for command, timing in zip(meta["commands"], meta.get("timing", [])):
                if timing.get("code") == 0:
                    found.setdefault(_kind(command), []).append(seconds(timing["started"], timing["ended"]))
    return {name: sorted(values)[len(values) // 2] for name, values in found.items()}


def _kind(command: str) -> str:
    """The command name that past times are kept by; Drive downloads by file, since their sizes differ a lot."""
    parts = command.split()
    return " ".join(parts[1:3]) if parts[1] == "fetch-drive" else parts[1]


def _progress(job_id: str) -> int | None:
    """The last download percentage the running command printed."""
    log = settings.RUNS_DIR / f"{job_id}.log"
    if not log.exists():
        return None
    text = log.read_text(encoding="utf-8", errors="replace").rsplit("$ congestion-nl", 1)[-1]
    found = PROGRESS.findall(text)
    return int(found[-1]) if found else None


def stages(meta: dict) -> list[dict]:
    """Each command of a job with its label, state, time taken and the usual time for it."""
    typical = typical_seconds()
    timing = meta.get("timing", [])
    out = []
    for i, command in enumerate(meta["commands"]):
        name = _kind(command)
        label, detail = STAGES.get(name, (name, ""))
        icon = label
        if name.startswith("fetch-drive "):
            from . import data

            file = name.split()[1]
            # model files carry their issue hour, such as model_11
            base, _, hour = file.rpartition("_")
            label = f"{DRIVE_STAGES[base]} {hour}:00" if hour.isdigit() and base in DRIVE_STAGES else DRIVE_STAGES.get(file, file)
            detail, icon = data.file_path(file).name, "Download"
        record = timing[i] if i < len(timing) else None
        ended_early = meta["status"] in ("failed", "stopped")
        if record and record.get("stopped"):
            state = "stopped"
        elif record and record.get("code") is not None:
            state = "done" if record["code"] == 0 else "failed"
        elif record:
            state = meta["status"]
        elif not timing and meta["status"] == "done":
            # jobs saved before step timing was recorded
            state = "done"
        elif not timing and i + 1 <= meta["step"]:
            state = "done" if i + 1 < meta["step"] else meta["status"]
        else:
            state = "skipped" if ended_early else "pending"
        started = record["started"] if record else None
        out.append({"label": label, "detail": detail, "icon": icon, "command": command, "state": state,
                    "seconds": seconds(started, record.get("ended") if record else None) if started else None,
                    "typical": typical.get(name),
                    "progress": _progress(meta["id"]) if state == "running" and icon == "Download" else None})
    return out


def status(job_id: str) -> dict | None:
    """A job's saved state."""
    file = _meta_path(job_id) if re.fullmatch(r"[\w-]+", job_id) else None
    if file is None or not file.exists():
        return None
    return _load(file)
