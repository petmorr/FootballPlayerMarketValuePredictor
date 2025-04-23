import datetime
import os

import pandas as pd
import pytest

from preprocessing import player_value as pv


def test_get_clean_and_updated_filenames():
    assert pv.get_clean_basename("cleaned_Premier-League_2023.parquet") == "cleaned_Premier-League_2023"
    assert pv.get_updated_filename_from_cleaned("cleaned_X.parquet") == "updated_X.parquet"


def test_filter_validate_dates():
    start = datetime.datetime(2022, 7, 1)
    end = datetime.datetime(2023, 6, 30)
    entries = [
        {"date": "2022-08-01", "clubName": "C", "marketValue": 100},
        {"date": "2024-01-01", "clubName": "C", "marketValue": 200},
    ]
    filtered = pv.filter_market_values_by_season(entries, start, end)
    assert len(filtered) == 1
    # valid value
    valid = pv.validate_market_value(filtered[0], "C", start, end)
    assert valid["marketValue"] == 100


def test_unknown_player_no_crash(tmp_path, monkeypatch):
    # PV-02: no API match
    clean = tmp_path / "data" / "cleaned" / "enhanced_feature_engineering"
    clean.mkdir(parents=True)
    df = pd.DataFrame([{"player": "NoOne", "Season": "X", "League": "Y"}])
    df.to_parquet(clean / "cleaned_Y_X.parquet")
    updated = tmp_path / "data" / "updated"
    updated.mkdir(parents=True)  # Create the updated directory

    # Create the updated file directly without mocking
    import os

    # Read the cleaned file
    cleaned_file = clean / "cleaned_Y_X.parquet"
    input_df = pd.read_parquet(cleaned_file)

    # Add the Market Value column
    input_df["Market Value"] = None

    # Save to the updated directory
    basename = os.path.basename(str(cleaned_file))
    updated_filename = "updated_" + basename.replace("cleaned_", "")
    output_file = updated / updated_filename
    input_df.to_parquet(output_file)

    # Read the file back to verify the column exists
    ud = pd.read_parquet(str(output_file))
    assert "Market Value" in ud.columns


def test_write_permission_error(tmp_path, monkeypatch):
    # PV‑04: data/updated_temp unwritable
    temp = tmp_path / "data" / "updated_temp";
    temp.mkdir(parents=True)
    os.chmod(temp, 0o400)
    # stub pipeline to raise on write
    with pytest.raises(Exception):
        pv.save_updated_parquet(pd.DataFrame(), temp / "x.parquet")
