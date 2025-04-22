import pandas as pd
import pytest

from preprocessing import preprocessing as pp


def test_flatten_columns_creates_single_level():
    arrays = [["a", "a"], ["x", "y"]]
    df = pd.DataFrame([[1, 2]], columns=pd.MultiIndex.from_arrays(arrays))
    df_flat = pp.flatten_columns(df)
    assert all(not isinstance(col, tuple) for col in df_flat.columns)

def test_data_integrity_checks_raises_on_missing_essential():
    df = pd.DataFrame({"goals": [1, 2]})
    with pytest.raises(ValueError, match="Essential columns missing"):
        pp.data_integrity_checks(df, "any.csv")


def _make_minimal_df():
    # One row, essential columns + a dummy feature
    data = {c: [1] for c in pp.ESSENTIAL_COLUMNS}
    data["dummy_feat"] = [42]
    return pd.DataFrame(data)


def test_process_single_file_minimal_row(tmp_path):
    df = _make_minimal_df()
    csv = tmp_path / "X_2020-2021_player_data.csv"
    df.to_csv(csv, index=False)

    out_df = pp.process_single_file(str(csv))
    assert isinstance(out_df, pd.DataFrame)
    assert len(out_df) == 1
    assert "dummy_feat" in out_df.columns


def test_process_single_file_merges_duplicates(tmp_path):
    # Duplicate the one-row minimal df
    base = _make_minimal_df()
    dup = pd.concat([base, base], ignore_index=True)
    # both rows have same player by default
    csv = tmp_path / "X_2020-2021_player_data.csv"
    dup.to_csv(csv, index=False)

    out_df = pp.process_single_file(str(csv))
    assert out_df["player"].nunique() == 1


def test_process_all_files_iterates_correctly(monkeypatch, tmp_path):
    # Stub out the Pool so we don't actually fork
    class DummyPool:
        def __init__(self, *args, **kwargs): pass

        def map(self, fn, iterable):
            for item in iterable:
                fn(item)

        def __enter__(self): return self

        def __exit__(self, *args): pass

    monkeypatch.setattr(pp, "Pool", DummyPool)

    # Create two empty CSV stubs
    (tmp_path / "A_2020-2021_player_data.csv").write_text("")
    (tmp_path / "B_2020-2021_player_data.csv").write_text("")

    called = []

    def fake_single(fname):
        called.append(fname)

    # Monkey‑patch process_single_file to our fake
    monkeypatch.setattr(pp, "process_single_file", fake_single)

    # Run
    pp.process_all_files(tmp_path)

    # Pool.map passes only the filename (f.name), so we expect just those two names
    assert sorted(called) == sorted([
        "A_2020-2021_player_data.csv",
        "B_2020-2021_player_data.csv",
    ])
