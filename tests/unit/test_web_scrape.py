import os
import stat
import time

import pandas as pd
import pytest

import preprocessing.web_scrape as ws

GOOD_HTML = """
<table id="stats_standard">
  <thead><tr><th>Player</th><th>Goals</th></tr></thead>
  <tbody><tr><td>Test Player</td><td>5</td></tr></tbody>
</table>
"""
COMMENT_WRAPPED = f"<!--\n{GOOD_HTML}\n-->"


def test_build_league_season_url():
    url = ws.build_league_season_url(
        "Premier-League",
        "https://fbref.com/en/comps/9/",
        "2023-2024",
    )
    expected = "https://fbref.com/en/comps/9/2023-2024/stats/2023_2024-Premier_League-Stats"
    assert url == expected


@pytest.mark.parametrize("html", [GOOD_HTML, COMMENT_WRAPPED])
def test_get_player_data_selenium_parses_table(monkeypatch, html):
    # Monkey‑patch WebDriverWait and sleep
    monkeypatch.setattr(ws, "WebDriverWait", lambda d, t: type("W", (), {"until": lambda s, f: None})())
    monkeypatch.setattr(time, "sleep", lambda s: None)

    class DummyDriver:
        def __init__(self, html): self.page_source = html

        def get(self, url): pass

    df = ws.get_player_data_selenium("any", DummyDriver(html))
    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0]["Player"] == "Test Player"
    assert df.iloc[0]["Goals"] == 5


def test_scrape_league_data_single_league(tmp_path, monkeypatch):
    # WS‑04: override LEAGUES and SEASONS to one minimal case
    monkeypatch.setattr(ws, "LEAGUES", {"X": "base/"})
    monkeypatch.setattr(ws, "SEASONS", ["2020-2021"])

    # patch driver and get_player_data_selenium
    class DummyDriver:
        def quit(self): pass

    monkeypatch.setattr(ws, "configure_driver", lambda: DummyDriver())
    monkeypatch.setattr(ws, "get_player_data_selenium", lambda url, d: pd.DataFrame([{"A": 1}]))
    out = tmp_path / "data" / "scraped";
    monkeypatch.chdir(tmp_path);
    os.makedirs(out, exist_ok=True)
    ws.OUTPUT_DIR = str(out)
    ws.scrape_league_data()
    # one file created
    files = os.listdir(out)
    assert len(files) == 1 and files[0].endswith("_player_data.csv")


def test_scrape_no_write_permission(tmp_path, monkeypatch):
    # WS‑03: OUTPUT_DIR unwritable
    monkeypatch.setattr(ws, "OUTPUT_DIR", str(tmp_path / "protected"))
    os.makedirs(ws.OUTPUT_DIR, exist_ok=True)
    # remove write perms
    os.chmod(ws.OUTPUT_DIR, stat.S_IREAD)
    # stub driver and scraper
    monkeypatch.setattr(ws, "configure_driver", lambda: type("D", (), {"quit": lambda s: None})())
    monkeypatch.setattr(ws, "get_player_data_selenium", lambda url, d: pd.DataFrame([{"A": 1}]))
    # should not crash
    ws.scrape_league_data()
