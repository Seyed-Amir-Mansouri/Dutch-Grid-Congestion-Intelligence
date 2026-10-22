import dataclasses

import pandas as pd
import pytest

from congestion_nl.config import DEFAULT_CONFIG, load_config


@pytest.fixture
def config(tmp_path):
    """Load a copy of the config in a temp directory with fewer boosting rounds."""
    text = DEFAULT_CONFIG.read_text(encoding="utf-8")
    (tmp_path / "config.toml").write_text(text, encoding="utf-8")
    loaded = load_config(tmp_path / "config.toml")
    fast = dataclasses.replace(loaded.model, max_iter=60, min_samples_leaf=20)
    grid = dataclasses.replace(loaded.grid, start_date=pd.Timestamp("2025-01-01"))
    return dataclasses.replace(loaded, model=fast, grid=grid)
