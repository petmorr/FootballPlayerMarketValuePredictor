import subprocess

import requests

from main import check_api_running, start_local_api


class DummyResponse:
    def __init__(self, status_code):
        self.status_code = status_code


def test_check_api_running_success(monkeypatch):
    # API‑01 healthy API returns HTTP 200
    monkeypatch.setattr(requests, "get", lambda *a, **k: DummyResponse(200))
    assert check_api_running() is True


def test_check_api_running_failure(monkeypatch):
    # API‑01+API‑04: connection error => API not running
    def bad_get(*args, **kwargs):
        raise requests.RequestException("fail")

    monkeypatch.setattr(requests, "get", bad_get)
    assert check_api_running() is False


def test_start_local_api_valid(monkeypatch, tmp_path):
    # API‑01: repository present and starts successfully
    repo = tmp_path / "transfermarkt-api"
    repo.mkdir()
    monkeypatch.setenv("TRANSFERMARKT_API_PATH", str(repo))
    # simulate first check=False then check=True
    calls = {"count": 0}

    def fake_check():
        calls["count"] += 1
        return calls["count"] > 1

    monkeypatch.setattr("main.check_api_running", fake_check)
    # stub out poetry install and subprocess calls
    monkeypatch.setattr(subprocess, "check_call", lambda *a, **k: 0)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: None)
    # ensure cwd is tmp_path
    monkeypatch.chdir(tmp_path)
    assert start_local_api() is True


def test_start_local_api_missing_repo(monkeypatch, tmp_path):
    # API‑02: missing repo => should return False, no crash
    missing = tmp_path / "transfermarkt-api"
    monkeypatch.setenv("TRANSFERMARKT_API_PATH", str(missing))

    # force a clone failure
    def fail_clone(*a, **k):
        raise subprocess.CalledProcessError(1, a[0])

    monkeypatch.setattr(subprocess, "check_call", fail_clone)
    monkeypatch.setattr("main.check_api_running", lambda: False)
    monkeypatch.chdir(tmp_path)
    assert start_local_api() is False


def test_start_local_api_poetry_corrupt(monkeypatch, tmp_path):
    # API‑03: poetry install error => False
    repo = tmp_path / "transfermarkt-api"
    repo.mkdir()
    monkeypatch.setenv("TRANSFERMARKT_API_PATH", str(repo))

    # fail on any "poetry" cmd, succeed on others
    def fake_call(cmd, **kwargs):
        if "poetry" in cmd[0]:
            raise subprocess.CalledProcessError(1, cmd)
        return 0

    monkeypatch.setattr(subprocess, "check_call", fake_call)
    monkeypatch.setattr("main.check_api_running", lambda: False)
    monkeypatch.chdir(tmp_path)
    assert start_local_api() is False


def test_minimal_no_csv_at_startup(tmp_path, monkeypatch):
    # API‑04: even with empty scraped/ csv folder, check_api_running is safe
    data_dir = tmp_path / "data" / "scraped"
    data_dir.mkdir(parents=True)
    # Should not raise, returns either True/False
    monkeypatch.chdir(tmp_path)
    res = check_api_running()
    assert isinstance(res, bool)
