"""Shared fixtures for the test suite."""

import dataclasses
from pathlib import Path

import pytest

from grid_congestion.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    """Load a copy of the config in a temp directory with fewer boosting rounds."""
    text = (REPO_ROOT / "config.toml").read_text(encoding="utf-8")
    (tmp_path / "config.toml").write_text(text, encoding="utf-8")
    loaded = load_config(tmp_path / "config.toml")
    fast = dataclasses.replace(loaded.model, max_iter=60)
    return dataclasses.replace(loaded, model=fast)
