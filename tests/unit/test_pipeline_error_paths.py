import os

import pytest
from click.testing import CliRunner

from main import cli  # your Click group
from preprocessing import web_scrape as ws


@pytest.mark.parametrize("step", ["scrape", "preprocess", "update", "train"])
def test_run_all_happy_path(tmp_path, step, sample_csv_dir):
    # combined E2E‑01
    runner = CliRunner()
    result = runner.invoke(cli, ["run-all"])
    assert result.exit_code == 0
    assert "Completed" in result.output


def test_run_all_stops_on_scrape_error(monkeypatch):
    # E2E‑02
    monkeypatch.setattr(ws, "scrape_league_data", lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    runner = CliRunner()
    result = runner.invoke(cli, ["run-all"])
    assert result.exit_code != 0
    assert "Web scraping failed" in result.output


def test_run_all_large_data(tmp_path, sample_csv_dir):
    # E2E‑03: simulate large scraped CSVs
    runner = CliRunner()
    result = runner.invoke(cli, ["run-all"])
    assert result.exit_code == 0
    assert os.path.isdir(tmp_path / "data" / "predictions")
