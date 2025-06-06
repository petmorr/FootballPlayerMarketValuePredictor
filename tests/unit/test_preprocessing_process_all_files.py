import pandas as pd
from pathlib import Path


def test_process_all_files_absolute_paths(tmp_path, monkeypatch):
    """process_all_files should send absolute paths for each CSV."""
    f1 = tmp_path / "one.csv"
    f1.write_text("player\nA")
    f2 = tmp_path / "two.csv"
    f2.write_text("player\nB")

    seen = []

    def fake_process_single_file(path):
        seen.append(Path(path))
        return pd.DataFrame()

    class DummyPool:
        def __init__(self, *_, **__):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
        def map(self, func, iterable):
            for item in iterable:
                func(item)

    monkeypatch.setattr("preprocessing.preprocessing.process_single_file", fake_process_single_file)
    monkeypatch.setattr("preprocessing.preprocessing.Pool", DummyPool)

    from preprocessing.preprocessing import process_all_files

    process_all_files(tmp_path)

    assert set(seen) == {f1, f2}
    for p in seen:
        assert p.is_absolute()
        assert str(p).startswith(str(tmp_path))
