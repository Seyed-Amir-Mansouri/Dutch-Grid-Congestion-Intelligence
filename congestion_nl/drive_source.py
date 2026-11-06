"""Download the shared data and model files from Google Drive and swap them in only when complete."""
from __future__ import annotations

import os
import re
import urllib.request
import zipfile
from pathlib import Path

URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
CHUNK = 1 << 20
# one zip holding every shared file; the [drive] key "bundle" links to it
BUNDLE = "bundle"
DATA_FILES = ["gopacs", "notices", "sdac", "forecasts", "weather", "locations"]
MODEL_FILES = ["model", "walk_forward", "inspector_history"]


def file_id(link: str) -> str:
    """The Drive file ID from a share link, or the text itself when it is already an ID."""
    found = re.search(r"/d/([\w-]{20,})|[?&]id=([\w-]{20,})", link)
    return next(g for g in found.groups() if g) if found else link.strip()


def target(config, name: str) -> Path:
    """Where a shared file goes: its configured path, or the model folder for the model files.

    Model files of another issue hour carry it as a suffix, such as model_11.
    """
    from . import estimator, inspector

    if name == BUNDLE:
        return config.resolve(Path("data") / "drive_bundle.zip")
    hour = re.fullmatch(r"(model|walk_forward|inspector_history)_(\d+)", name)
    if hour:
        name, config = hour.group(1), config.at_hour(int(hour.group(2)))
    model_dir = config.resolve(config.paths.model_dir)
    special = {"model": model_dir / estimator.MODEL_FILE, "walk_forward": model_dir / inspector.TEST_FILE,
               "inspector_history": model_dir / inspector.HISTORY_FILE}
    return special.get(name) or config.resolve(getattr(config.paths, name))


def download(link: str, target: Path, report=print) -> int:
    """Download one file next to its target and replace the target only once the whole file arrived; returns bytes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    try:
        with urllib.request.urlopen(URL.format(file_id(link)), timeout=60) as response, part.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            if "text/html" in response.headers.get("Content-Type", ""):
                raise ValueError("Drive sent a web page, not the file: check that it is shared with anyone with the link")
            size, shown = 0, -1
            while chunk := response.read(CHUNK):
                out.write(chunk)
                size += len(chunk)
                percent = size * 100 // total if total else None
                # the web UI reads these lines for the progress bar
                if percent is not None and percent // 5 != shown // 5:
                    shown = percent
                    report(f"progress: {percent}% of {total / CHUNK:.1f} MB", flush=True)
        if size == 0 or (total and size != total):
            raise ValueError(f"download incomplete: {size} of {total} bytes")
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)
    return size


def bundle_names(config) -> dict[str, Path]:
    """The file names a bundle may hold and where each goes: data files by name, model files with their hour."""
    names = {target(config, key).name: target(config, key) for key in DATA_FILES}
    for hour in config.hours:
        for key in MODEL_FILES:
            path = target(config, f"{key}_{hour}")
            names[f"{path.stem}_{hour}{path.suffix}"] = path
    return names


def unpack(config, bundle: Path, report=print) -> int:
    """Put each known file of a downloaded bundle in its place, each swapped in whole; unknown names are skipped."""
    names = bundle_names(config)
    placed = 0
    with zipfile.ZipFile(bundle) as archive:
        for member in archive.infolist():
            # only the bare file name counts, so no member can point outside its folder
            name = Path(member.filename).name
            if member.is_dir() or name not in names:
                if not member.is_dir():
                    report(f"skipped: {member.filename}", flush=True)
                continue
            path = names[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_name(path.name + ".part")
            try:
                with archive.open(member) as source, part.open("wb") as out:
                    while chunk := source.read(CHUNK):
                        out.write(chunk)
                os.replace(part, path)
            finally:
                part.unlink(missing_ok=True)
            report(f"{name}: {path} ({member.file_size / 2**20:.1f} MB)", flush=True)
            placed += 1
    return placed
