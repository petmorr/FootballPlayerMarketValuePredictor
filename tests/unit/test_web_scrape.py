import os
import tempfile
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

import preprocessing.web_scrape as ws


@pytest.fixture
def setup_output_dir(tmp_path):
    """Create a temporary output directory for scraped data"""
    output_dir = tmp_path / "data" / "scraped"
    output_dir.mkdir(parents=True, exist_ok=True)
    original_output_dir = ws.OUTPUT_DIR
    ws.OUTPUT_DIR = str(output_dir)
    yield output_dir
    ws.OUTPUT_DIR = original_output_dir


@pytest.fixture
def mock_driver():
    """Create a mock Selenium driver"""
    driver = MagicMock()
    driver.quit = MagicMock()
    return driver


def test_valid_league_seasons(setup_output_dir, mock_driver):
    """
    Test Case ID: WS-01
    Title: Scrape Valid Leagues/Seasons

    Ensures that web_scrape.py successfully fetches CSVs for known leagues/seasons.
    """
    output_dir = setup_output_dir
    # Create a sample DataFrame that would be returned by the scraper
    sample_df = pd.DataFrame({
        'player': ['Player1', 'Player2', 'Player3'],
        'team': ['Team A', 'Team B', 'Team C'],
        'goals': [10, 5, 1],
        'assists': [5, 8, 3]
    })

    # Patch the methods to simulate a successful scrape
    with patch('preprocessing.web_scrape.configure_driver', return_value=mock_driver), \
            patch('preprocessing.web_scrape.get_player_data_selenium', return_value=sample_df), \
            patch('preprocessing.web_scrape.logger') as mock_logger:

        # Override test leagues and seasons to a smaller subset
        original_leagues = ws.LEAGUES
        original_seasons = ws.SEASONS

        try:
            # Set test data
            ws.LEAGUES = {"Premier-League": "https://fbref.com/en/comps/9/"}
            ws.SEASONS = ["2023-2024"]

            # Run the scraper
            ws.scrape_league_data()

            # Check that the CSV file was created
            expected_csv = output_dir / "Premier-League_2023-2024_player_data.csv"
            assert expected_csv.exists(), f"Expected CSV file was not created: {expected_csv}"

            # Verify the content
            df = pd.read_csv(expected_csv)
            assert len(df) == 3, "CSV should contain 3 rows"
            assert "League" in df.columns, "League column should be added"
            assert "Season" in df.columns, "Season column should be added"

            # Verify logs
            mock_logger.info.assert_any_call("Web scraping completed. WebDriver closed.")
            mock_logger.info.assert_any_call(f"Saved data to {expected_csv}")

            # Verify no error logs
            assert not mock_logger.error.called, "No errors should be logged"

        finally:
            # Restore original settings
            ws.LEAGUES = original_leagues
            ws.SEASONS = original_seasons


def test_malformed_html(setup_output_dir, mock_driver):
    """
    Test Case ID: WS-02
    Title: Partial or Malformed HTML

    Verifies scraping logic if FBref returns incomplete or broken HTML.
    """
    output_dir = setup_output_dir

    # Patch the methods to simulate malformed HTML
    with patch('preprocessing.web_scrape.configure_driver', return_value=mock_driver), \
            patch('preprocessing.web_scrape.get_player_data_selenium', return_value=None), \
            patch('preprocessing.web_scrape.logger') as mock_logger:

        # Override test leagues and seasons
        original_leagues = ws.LEAGUES
        original_seasons = ws.SEASONS

        try:
            # Set test data
            ws.LEAGUES = {"Broken-League": "https://fbref.com/en/broken/"}
            ws.SEASONS = ["2023-2024"]

            # Run the scraper
            ws.scrape_league_data()

            # Verify logs
            mock_logger.warning.assert_called_with("No data for Broken-League - 2023-2024.")
            mock_logger.info.assert_called_with("Web scraping completed. WebDriver closed.")

            # Verify no CSV was created
            files = list(output_dir.glob("*.csv"))
            assert len(files) == 0, "No CSV should be created for malformed HTML"

        finally:
            # Restore original settings
            ws.LEAGUES = original_leagues
            ws.SEASONS = original_seasons


def test_no_write_permission(mock_driver):
    """
    Test Case ID: WS-03
    Title: No Write Permission to data/scraped

    Check if the script fails gracefully when a directory is unwritable.
    """
    # Create a temp directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Store original output dir
        original_output_dir = ws.OUTPUT_DIR

        # Create a sample DataFrame
        sample_df = pd.DataFrame({'player': ['Player1'], 'team': ['Team A']})

        # Instead of trying to change permissions (which doesn't work reliably on Windows),
        # we'll mock DataFrame.to_csv to raise a PermissionError
        def mock_to_csv(*args, **kwargs):
            raise PermissionError("Access is denied")

        # Patch methods for the test
        with patch('preprocessing.web_scrape.configure_driver', return_value=mock_driver), \
                patch('preprocessing.web_scrape.get_player_data_selenium', return_value=sample_df), \
                patch('pandas.DataFrame.to_csv', mock_to_csv), \
                patch('preprocessing.web_scrape.logger') as mock_logger:

            try:
                # Set minimal test data
                original_leagues = ws.LEAGUES
                original_seasons = ws.SEASONS
                ws.LEAGUES = {"Test-League": "https://fbref.com/en/test/"}
                ws.SEASONS = ["2023-2024"]
                ws.OUTPUT_DIR = os.path.join(tmpdir, "data", "scraped")

                # Run the scraper
                ws.scrape_league_data()

                # Verify error was logged
                assert mock_logger.error.called, "Error should be logged when write permission is denied"
                error_msg = mock_logger.error.call_args[0][0]
                assert "Error in Test-League - 2023-2024" in error_msg

            finally:
                # Restore original settings
                ws.LEAGUES = original_leagues
                ws.SEASONS = original_seasons
                ws.OUTPUT_DIR = original_output_dir


def test_scrape_league_data_single_league(setup_output_dir):
    """
    Test Case ID: WS-04
    Title: Single League with Minimal Data

    Tests scraping when only one league or 1 table row is available.
    """
    output_dir = setup_output_dir

    # Create a minimal DataFrame with just one row
    minimal_df = pd.DataFrame([{'player': 'Player1', 'team': 'Team A', 'goals': 1}])

    # Create a dummy driver
    class DummyDriver:
        def quit(self):
            pass

    # Patch the needed functions
    with patch('preprocessing.web_scrape.configure_driver', return_value=DummyDriver()), \
            patch('preprocessing.web_scrape.get_player_data_selenium', return_value=minimal_df), \
            patch('preprocessing.web_scrape.logger') as mock_logger:

        # Override test leagues and seasons
        original_leagues = ws.LEAGUES
        original_seasons = ws.SEASONS

        try:
            # Set minimal test data
            ws.LEAGUES = {"X": "base/"}
            ws.SEASONS = ["2020-2021"]

            # Run the scraper
            ws.scrape_league_data()

            # Check for CSV file
            expected_csv = output_dir / "X_2020-2021_player_data.csv"
            assert expected_csv.exists(), f"Expected CSV file not created: {expected_csv}"

            # Verify content
            df = pd.read_csv(expected_csv)
            assert len(df) == 1, "CSV should contain exactly 1 row"
            assert "League" in df.columns and df["League"].iloc[0] == "X"
            assert "Season" in df.columns and df["Season"].iloc[0] == "2020-2021"

            # Verify logs
            mock_logger.info.assert_any_call(f"Saved data to {expected_csv}")
            mock_logger.info.assert_any_call("Web scraping completed. WebDriver closed.")

        finally:
            # Restore original settings
            ws.LEAGUES = original_leagues
            ws.SEASONS = original_seasons
