"""
Global fixtures usable across the entire test‑suite.

Keeps external libs quiet, injects a temporary DATA_ROOT so your
code never touches the real filesystem, and provides a fake
Transfermarkt‑API via requests‑mock.

Author : Peter Morris
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest

import logging

# ---------------------------------------------------------------------------#
#  Directories
# ---------------------------------------------------------------------------#
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "tests" / "_sample_data"


# ---------------------------------------------------------------------------#
#  Silence noisy third‑party loggers
# ---------------------------------------------------------------------------#
@pytest.fixture(scope="session", autouse=True)
def _quiet_external_loggers() -> None:
    for name in ("urllib3", "selenium", "requests"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


# ---------------------------------------------------------------------------#
#  Sample blobs
# ---------------------------------------------------------------------------#
@pytest.fixture(scope="session")
def sample_raw_csv() -> Path:
    """A minimal-but-valid raw FBref export (2 rows)."""
    return DATA_DIR / "raw_premier_league_2023.csv"


@pytest.fixture(scope="session")
def sample_cleaned_parquet() -> Path:
    """A tiny cleaned parquet used for model-utils unit tests."""
    return DATA_DIR / "cleaned_premier_league_2023.parquet"


# ---------------------------------------------------------------------------#
#  Monkey‑patch env so code writes into tmpdir
# ---------------------------------------------------------------------------#
@pytest.fixture(autouse=True)
def _patch_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Generator[None, None, None]:
    """
    Force DATA_ROOT and API url to a safe temp‑dir/fake‑host for every test.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("API_BASE_URL", "http://localhost:9999")
    yield


# ---------------------------------------------------------------------------#
#  Fake transfermarkt‑api
# ---------------------------------------------------------------------------#
@pytest.fixture
def fake_api(requests_mock):
    """
    A simple stub of GET /lookup?name=… returning a single JSON obj.
    """
    payload = {"player": "Erling Haaland", "market_value": 180_000_000}
    requests_mock.get("http://localhost:9999/lookup", json=payload)
    return requests_mock
