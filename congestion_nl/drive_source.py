"""Download the shared data and model files from Google Drive and swap them in only when complete."""
from __future__ import annotations

import os
import re
import urllib.request
from pathlib import Path

URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
CHUNK = 1 << 20


def file_id(link: str) -> str:
    """The Drive file ID from a share link, or the text itself when it is already an ID."""
    found = re.search(r"/d/([\w-]{20,})|[?&]id=([\w-]{20,})", link)
    return next(g for g in found.groups() if g) if found else link.strip()


def target(config, name: str) -> Path:
    """Where a shared file goes: its configured path, or the model folder for the two model files."""
    from . import estimator, inspector

    model_dir = config.resolve(config.paths.model_dir)
    special = {"model": model_dir / estimator.MODEL_FILE, "walk_forward": model_dir / inspector.WALK_FORWARD_FILE}
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
