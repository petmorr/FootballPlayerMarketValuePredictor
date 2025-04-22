import pandas as pd
import pyarrow.parquet as pq
import pytest

from preprocessing import preprocessing as pp


# Helper to set up test environment and patch module constants
def _setup_env(tmp_path):
    inp = tmp_path / "data" / "scraped"
    out = tmp_path / "data" / "cleaned"
    inp.mkdir(parents=True, exist_ok=True)
    for variant in ["enhanced_feature_engineering", "feature_engineering", "no_feature_engineering"]:
        (out / variant).mkdir(parents=True, exist_ok=True)
    # Patch module-level folder constants
    pp.RAW_DATA_FOLDER = inp
    pp.CLEANED_DATA_FOLDER = out
    pp.ENHANCED_FE_FOLDER = out / "enhanced_feature_engineering"
    pp.BASIC_FE_FOLDER = out / "feature_engineering"
    pp.NO_FE_FOLDER = out / "no_feature_engineering"
    return inp, out


# Builds a DataFrame with all essential columns populated
def _make_essential_df(rows):
    df_rows = []
    for idx, row in enumerate(rows):
        new_row = {}
        for col in pp.ESSENTIAL_COLUMNS:
            if col in row:
                new_row[col] = row[col]
            else:
                # Provide default dummy values
                if col == "player":
                    new_row[col] = f"Player{idx}"
                elif col in {"country_code", "position", "squad", "born"}:
                    new_row[col] = row.get(col, "NA")
                else:
                    new_row[col] = row.get(col, 1)
        df_rows.append(new_row)
    return pd.DataFrame(df_rows)


def test_flatten_columns_creates_single_level():
    arrays = [["a", "a"], ["x", "y"]]
    df = pd.DataFrame([[1, 2]], columns=pd.MultiIndex.from_arrays(arrays))
    df_flat = pp.flatten_columns(df)
    assert all(not isinstance(col, tuple) for col in df_flat.columns)


def test_data_integrity_checks_raises_on_missing_essential():
    df = pd.DataFrame({"goals": [1, 2]})
    with pytest.raises(ValueError, match="Essential columns missing"):
        pp.data_integrity_checks(df, "any.csv")


def test_preprocess_standard_csv(tmp_path, monkeypatch):
    # PP-01: simulate a scraped CSV and verify all variants output
    inp, out = _setup_env(tmp_path)
    rows = [
        {"matches_played": 10, "minutes_played": 900, "goals": 1, "rank": 1, "age": 25, "player": "A",
         "country_code": "ENG", "position": "Mid", "squad": "Team", "born": "1990-01-01"},
        {"matches_played": 20, "minutes_played": 1800, "goals": 2, "rank": 2, "age": 26, "player": "B",
         "country_code": "ENG", "position": "Mid", "squad": "Team", "born": "1990-01-01"}
    ]
    df = _make_essential_df(rows)
    fname = "Premier-League_2023-2024_player_data.csv"
    df.to_csv(inp / fname, index=False)

    monkeypatch.chdir(tmp_path)
    pp.process_all_files(inp)

    # Check each variant for non-empty Parquet files
    for variant in ["enhanced_feature_engineering", "feature_engineering", "no_feature_engineering"]:
        folder = out / variant
        files = list(folder.glob("*.parquet"))
        assert files, f"No parquet files found in {folder}"
        for f in files:
            tbl = pq.read_table(str(f)).to_pandas()
            assert not tbl.empty


def test_minimal_csv_one_row(tmp_path, monkeypatch):
    # PP-03: one-row CSV yields a single record in enhanced output
    inp, out = _setup_env(tmp_path)
    rows = [{"matches_played": 1, "minutes_played": 90, "goals": 0, "rank": 1, "age": 30, "player": "Solo",
             "country_code": "ENG", "position": "Mid", "squad": "Team", "born": "1990-01-01"}]
    df = _make_essential_df(rows)
    df.to_csv(inp / "X_2020-2021_player_data.csv", index=False)

    monkeypatch.chdir(tmp_path)
    pp.process_all_files(inp)

    enhanced_folder = out / "enhanced_feature_engineering"
    files = list(enhanced_folder.glob("*.parquet"))
    assert files, "No enhanced parquet file generated"
    tbl = pq.read_table(str(files[0])).to_pandas()
    assert len(tbl) == 1


def test_duplicate_players_merged(tmp_path, monkeypatch):
    # PP-04: duplicate player rows merge into one
    inp, out = _setup_env(tmp_path)
    rows = [
        {"matches_played": 2, "minutes_played": 180, "goals": 0, "rank": 1, "age": 30, "player": "Dup",
         "country_code": "ENG", "position": "Mid", "squad": "Team", "born": "1990-01-01"},
        {"matches_played": 4, "minutes_played": 360, "goals": 1, "rank": 1, "age": 30, "player": "Dup",
         "country_code": "ENG", "position": "Mid", "squad": "Team", "born": "1990-01-01"}
    ]
    df = _make_essential_df(rows)
    df.to_csv(inp / "X_2020-2021_player_data.csv", index=False)

    monkeypatch.chdir(tmp_path)
    pp.process_all_files(inp)

    enhanced_folder = out / "enhanced_feature_engineering"
    files = list(enhanced_folder.glob("*.parquet"))
    assert files, "No enhanced parquet file generated for duplicates"
    tbl = pd.read_parquet(str(files[0]))
    assert tbl["player"].tolist().count("Dup") == 1
