from datetime import datetime
from unittest.mock import patch

import pytest

from preprocessing.player_value import (
    filter_market_values_by_season,
    fetch_player_market_value  # This function was visible in the snippets
)


@pytest.fixture
def sample_market_values():
    """Fixture for sample market values from API"""
    return [
        {"date": "2023-08-01", "value": 5000000},
        {"date": "2023-05-01", "value": 4500000},
        {"date": "2022-12-01", "value": 4000000},
        {"date": "2022-08-01", "value": 3800000}
    ]


def test_filter_market_values_by_season(sample_market_values):
    """
    Test Case ID: PV-01
    Title: Filter Market Values by Season

    Tests normal filtering of market values within a season's date range.
    """
    # Define a season date range
    season_start = datetime(2023, 7, 1)
    season_end = datetime(2024, 6, 30)

    # Get market values for this season
    filtered = filter_market_values_by_season(
        sample_market_values,
        season_start,
        season_end
    )

    # There should be 1 market value in this season range
    assert len(filtered) == 1
    assert filtered[0]["value"] == 5000000
    assert filtered[0]["date"] == "2023-08-01"


def test_filter_market_values_previous_season(sample_market_values):
    """
    Test Case ID: PV-01
    Title: Filter Market Values for Previous Season

    Tests filtering market values for a previous season.
    """
    # Define a previous season
    season_start = datetime(2022, 7, 1)
    season_end = datetime(2023, 6, 30)

    filtered = filter_market_values_by_season(
        sample_market_values,
        season_start,
        season_end
    )

    # There should be 3 market values in this season range
    assert len(filtered) == 3
    assert filtered[0]["value"] == 4500000
    assert filtered[1]["value"] == 4000000
    assert filtered[2]["value"] == 3800000


def test_player_not_found():
    """
    Test Case ID: PV-02
    Title: Unknown Players (API Mismatch)

    Tests handling when a player ID is not found in the API.
    """
    with patch('preprocessing.player_value.make_request_with_retry') as mock_request, \
            patch('preprocessing.player_value.logger') as mock_logger:
        # Simulate an empty response
        mock_request.return_value = {"marketValueHistory": []}

        # Call function with invalid player ID
        result = fetch_player_market_value("0")

        # Verify result is empty list
        assert result == []

        # Check if logger recorded the API call
        mock_logger.info.assert_called_with("Fetching market value for player ID 0")


def test_negative_market_value():
    """
    Test Case ID: PV-03
    Title: Handling Extreme Market Value

    Tests that negative market values are processed correctly.
    """
    with patch('preprocessing.player_value.make_request_with_retry') as mock_request:
        # Return negative market value
        mock_request.return_value = {
            "marketValueHistory": [
                {"date": "2023-08-01", "value": -1000000}
            ]
        }

        result = fetch_player_market_value("123")

        # Verify the negative value is returned
        assert len(result) == 1
        assert result[0]["value"] == -1000000


def test_extremely_large_market_value():
    """
    Test Case ID: PV-03
    Title: Handling Extreme Market Value

    Tests that extremely large market values are processed correctly.
    """
    with patch('preprocessing.player_value.make_request_with_retry') as mock_request:
        # Return a very large market value
        mock_request.return_value = {
            "marketValueHistory": [
                {"date": "2023-08-01", "value": 999999999}
            ]
        }

        result = fetch_player_market_value("123")

        # Verify the large value is returned
        assert len(result) == 1
        assert result[0]["value"] == 999999999


def test_api_connection_error():
    """
    Test Case ID: PV-04
    Title: API Connection Error

    Tests that API connection errors are properly raised and can be caught by caller.
    """
    with patch('preprocessing.player_value.make_request_with_retry',
               side_effect=Exception("Connection error")), \
            patch('preprocessing.player_value.logger') as mock_logger:
        # The function should raise the exception when API call fails
        with pytest.raises(Exception) as excinfo:
            fetch_player_market_value("123")

        # Verify exception message
        assert "Connection error" in str(excinfo.value)

        # Verify the API call was logged
        mock_logger.info.assert_called_with("Fetching market value for player ID 123")
