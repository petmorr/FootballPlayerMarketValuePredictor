from unittest.mock import MagicMock

import pytest


@pytest.fixture
def missing_entries():
    """Create sample missing entries for testing"""
    return [
        {"dataset": "enhanced_feature_engineering", "season": "2022-23", "player": "Player1", "team": "Team A",
         "closest_date": "2023-01-01"},
        {"dataset": "feature_engineering", "season": "2022-23", "player": "Player1", "team": "Team A",
         "closest_date": "2023-01-01"},
        {"dataset": "no_feature_engineering", "season": "2022-23", "player": "Player1", "team": "Team A",
         "closest_date": "2023-01-01"},
    ]


@pytest.fixture
def missing_entry_single():
    """Create a single missing entry for edge case testing"""
    return [
        {"dataset": "enhanced_feature_engineering", "season": "2022-23", "player": "Player1", "team": "Team A",
         "closest_date": "2023-01-01"},
    ]


# Helper function that replicates the core logic of manual_input
def process_manual_input(
        players, teams, datasets, seasons, closest_dates, manual_values,
        update_func=None, flash_func=None
):
    """Replicates the core logic of the manual_input route without Flask dependencies"""
    updates = []
    for ds, ss, p, t, cd, mv in zip(datasets, seasons, players, teams, closest_dates, manual_values):
        if mv.strip():
            updates.append({
                "dataset": ds,
                "season": ss,
                "player": p,
                "team": t,
                "closest_date": cd,
                "manual_transfer_value": float(mv)
            })

    grouped_updates = {}
    for upd in updates:
        ds = upd["dataset"]
        grouped_updates.setdefault(ds, []).append(upd)

    results = {}
    for ds, upd_list in grouped_updates.items():
        success = True
        if update_func:
            success = update_func(ds, upd_list[0]["season"], upd_list)

        if flash_func:
            if not success:
                flash_func(f"Failed to update {ds}.", "danger")
            else:
                flash_func(f"Updated {ds}.", "success")

        results[ds] = success

    return results


# Test MI-01: Fill Numeric Values for Missing MVs in the UI
def test_manual_input_numeric_values(missing_entries):
    """
    Test Case ID: MI-01
    Title: Fill Numeric Values for Missing MVs in the UI
    """
    # Create mock functions
    mock_update = MagicMock(return_value=True)
    mock_flash = MagicMock()

    # Prepare form data with valid numeric values
    players = [entry['player'] for entry in missing_entries]
    teams = [entry['team'] for entry in missing_entries]
    datasets = [entry['dataset'] for entry in missing_entries]
    seasons = [entry['season'] for entry in missing_entries]
    closest_dates = [entry['closest_date'] for entry in missing_entries]
    manual_values = ['2000000.0', '2500000.0', '3000000.0']  # Valid float values

    # Call the processing function
    results = process_manual_input(
        players, teams, datasets, seasons, closest_dates, manual_values,
        update_func=mock_update, flash_func=mock_flash
    )

    # Verify the mock update was called correctly
    assert mock_update.call_count == 3

    # Verify flash messages
    expected_flash_calls = [
        (("Updated enhanced_feature_engineering.", "success"), {}),
        (("Updated feature_engineering.", "success"), {}),
        (("Updated no_feature_engineering.", "success"), {})
    ]
    assert mock_flash.call_args_list == expected_flash_calls

    # Verify all updates succeeded
    assert all(results.values())
    assert len(results) == 3


# Test MI-02: Single Missing Entry Only
def test_manual_input_single_entry(missing_entry_single):
    """
    Test Case ID: MI-02
    Title: Single Missing Entry Only
    """
    # Create mock functions
    mock_update = MagicMock(return_value=True)
    mock_flash = MagicMock()

    # Prepare form data with a single value
    players = [missing_entry_single[0]['player']]
    teams = [missing_entry_single[0]['team']]
    datasets = [missing_entry_single[0]['dataset']]
    seasons = [missing_entry_single[0]['season']]
    closest_dates = [missing_entry_single[0]['closest_date']]
    manual_values = ['1500000.0']  # Valid float value

    # Call the processing function
    results = process_manual_input(
        players, teams, datasets, seasons, closest_dates, manual_values,
        update_func=mock_update, flash_func=mock_flash
    )

    # Verify the mock update was called exactly once
    mock_update.assert_called_once()

    # Check for success message
    mock_flash.assert_called_once_with("Updated enhanced_feature_engineering.", "success")

    # Verify update succeeded
    assert results["enhanced_feature_engineering"] is True


# Test MI-03: Non-numeric or Negative Input
def test_manual_input_non_numeric(missing_entries):
    """
    Test Case ID: MI-03
    Title: Non-numeric or Negative Input
    """
    # Create mock functions
    mock_update = MagicMock(return_value=True)
    mock_flash = MagicMock()

    # Prepare data with invalid text
    players = [missing_entries[0]['player']]
    teams = [missing_entries[0]['team']]
    datasets = [missing_entries[0]['dataset']]
    seasons = [missing_entries[0]['season']]
    closest_dates = [missing_entries[0]['closest_date']]

    # Test with invalid text - should raise ValueError
    with pytest.raises(ValueError):
        process_manual_input(
            players, teams, datasets, seasons, closest_dates, ['abc'],
            update_func=mock_update, flash_func=mock_flash
        )

    # Test with negative value - should be accepted
    results = process_manual_input(
        players, teams, datasets, seasons, closest_dates, ['-1000000.0'],
        update_func=mock_update, flash_func=mock_flash
    )

    # Verify update was called with negative value
    assert mock_update.called
    # Verify update succeeded
    assert results["enhanced_feature_engineering"] is True

    # Verify correct flash message
    mock_flash.assert_called_with("Updated enhanced_feature_engineering.", "success")


# Test MI-04: Large Number
def test_manual_input_large_number(missing_entries):
    """
    Test Case ID: MI-04
    Title: Large Number (999999999.99)
    """

    # Set up the mock to check if the large value was processed correctly
    def mock_update_side_effect(dataset, season, updates):
        # Verify the large number was correctly converted to float
        assert updates[0]['manual_transfer_value'] == 999999999.99
        return True

    mock_update = MagicMock(side_effect=mock_update_side_effect)
    mock_flash = MagicMock()

    # Prepare data with large number
    players = [missing_entries[0]['player']]
    teams = [missing_entries[0]['team']]
    datasets = [missing_entries[0]['dataset']]
    seasons = [missing_entries[0]['season']]
    closest_dates = [missing_entries[0]['closest_date']]
    manual_values = ['999999999.99']  # Very large number

    # Call the processing function
    results = process_manual_input(
        players, teams, datasets, seasons, closest_dates, manual_values,
        update_func=mock_update, flash_func=mock_flash
    )

    # Verify update was called and succeeded
    assert mock_update.called
    assert results["enhanced_feature_engineering"] is True

    # Verify correct flash message
    mock_flash.assert_called_with("Updated enhanced_feature_engineering.", "success")


# Test empty submission
def test_manual_input_empty_submission(missing_entries):
    """
    Additional test: Submitting the form with no values entered
    """
    # Create mock functions
    mock_update = MagicMock(return_value=True)
    mock_flash = MagicMock()

    # Prepare data with empty values
    players = [entry['player'] for entry in missing_entries]
    teams = [entry['team'] for entry in missing_entries]
    datasets = [entry['dataset'] for entry in missing_entries]
    seasons = [entry['season'] for entry in missing_entries]
    closest_dates = [entry['closest_date'] for entry in missing_entries]
    manual_values = ['', '', '']  # All empty

    # Call the processing function
    results = process_manual_input(
        players, teams, datasets, seasons, closest_dates, manual_values,
        update_func=mock_update, flash_func=mock_flash
    )

    # No updates should be made for empty values
    assert not mock_update.called
    # Result dict should be empty since no updates were processed
    assert results == {}
