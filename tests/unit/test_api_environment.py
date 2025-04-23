import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Make sure we're importing the correct module for testing
try:
    from main import start_local_api, check_api_running
except ImportError:
    # If we're running from a different directory, try to find the module
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from main import start_local_api, check_api_running


@pytest.fixture
def setup_environment():
    """Set up a clean environment for API testing"""
    # Store original paths and settings
    original_dir = os.getcwd()

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Change to temporary directory
        os.chdir(tmpdir)

        # Set up mock folder structure
        os.makedirs(os.path.join(tmpdir, "data", "scraped"), exist_ok=True)

        # Create empty transfermarkt-api folder
        api_dir = os.path.join(tmpdir, "transfermarkt-api")
        os.makedirs(api_dir, exist_ok=True)

        yield tmpdir

        # Restore original directory
        os.chdir(original_dir)


def test_start_local_api_valid_env():
    """
    Test Case ID: API-01
    Title: Start Local API (Valid Env)

    Check that the transfermarkt-api repository is cloned and that the Poetry env is okay.
    """
    # Mock subprocess operations and API checks
    with patch('subprocess.check_call') as mock_check_call, \
            patch('subprocess.Popen') as mock_popen, \
            patch('main.check_api_running', side_effect=[False, True]) as mock_check_api, \
            patch('main.logger') as mock_logger:
        # Setup mock process
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        # Call the function
        result = start_local_api()

        # Verify results
        assert result is True, "API server should start successfully"

        # Check that installation commands were called
        assert mock_check_call.call_count >= 2
        mock_check_call.assert_any_call(["poetry", "install", "--no-root"],
                                        cwd=str(Path("transfermarkt-api")))
        mock_check_call.assert_any_call(["poetry", "check"],
                                        cwd=str(Path("transfermarkt-api")))

        # Verify API started
        assert mock_popen.called
        mock_popen.assert_called_with(
            ["poetry", "run", "python", "app/main.py"],
            cwd=str(Path("transfermarkt-api"))
        )

        # Verify logs
        mock_logger.info.assert_any_call("API server started successfully.")


def test_missing_or_broken_repo(setup_environment):
    """
    Test Case ID: API-02
    Title: Missing or Broken Repo
    
    Test behaviour if the transfermarkt-api folder is missing or corrupted.
    """
    tmpdir = setup_environment

    # Make sure transfermarkt-api folder doesn't exist
    api_dir = os.path.join(tmpdir, "transfermarkt-api")
    if os.path.exists(api_dir):
        shutil.rmtree(api_dir)

    # Mock git clone to fail AND ensure check_api_running returns False
    with patch('subprocess.check_call', side_effect=Exception("Git repo not found")) as mock_check_call, \
            patch('main.check_api_running', return_value=False) as mock_check_api, \
            patch('main.logger') as mock_logger:
        # Call the function
        result = start_local_api()

        # Verify results
        assert result is False, "API server should fail to start"

        # Verify clone attempt
        mock_check_call.assert_called_once_with(
            ["git", "clone", "https://github.com/felipeall/transfermarkt-api.git"]
        )

        # Verify error log
        mock_logger.error.assert_called_once()
        error_msg = mock_logger.error.call_args[0][0]
        assert "Error cloning repository" in error_msg


def test_poetry_environment_corrupted(setup_environment):
    """
    Test Case ID: API-03
    Title: Poetry Environment Corrupted
    
    Checks handling if Poetry dependencies or lock files are missing.
    """
    tmpdir = setup_environment

    # Create empty poetry.lock file to simulate corruption
    api_dir = os.path.join(tmpdir, "transfermarkt-api")
    with open(os.path.join(api_dir, "poetry.lock"), "w") as f:
        f.write("corrupted lock file")

    # Mock subprocess operations to simulate Poetry install failure
    def mock_subprocess_check(*args, **kwargs):
        if "poetry" in args[0] and ("install" in args[0] or "check" in args[0]):
            raise subprocess.CalledProcessError(1, args[0], output="Dependency resolution failed")
        return 0

    with patch('subprocess.check_call', side_effect=mock_subprocess_check) as mock_check_call, \
            patch('main.check_api_running', return_value=False) as mock_check_api, \
            patch('main.logger') as mock_logger:
        # Call the function
        result = start_local_api()

        # Verify results
        assert result is False, "API server should fail to start with corrupted poetry environment"

        # Verify installation attempt was made
        assert mock_check_call.called
        assert any("poetry" in str(call) and "install" in str(call) for call in mock_check_call.call_args_list)

        # Verify error log
        mock_logger.error.assert_called_once()
        error_msg = mock_logger.error.call_args[0][0]
        assert "Error installing or verifying dependencies" in error_msg


def test_minimal_or_no_csv_data(setup_environment):
    """
    Test Case ID: API-04
    Title: Minimal or No CSV Data at Startup

    Verifies system can still start with zero local data
    """
    tmpdir = setup_environment

    # Ensure data/scraped exists but is empty
    scraped_dir = os.path.join(tmpdir, "data", "scraped")
    if os.path.exists(scraped_dir):
        # Remove any files but keep the directory
        for item in os.listdir(scraped_dir):
            item_path = os.path.join(scraped_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)

    # Mock subprocess and API checks
    with patch('subprocess.check_call') as mock_check_call, \
            patch('subprocess.Popen') as mock_popen, \
            patch('main.check_api_running', side_effect=[False, True]) as mock_check_api, \
            patch('main.logger') as mock_logger:

        # Setup mock process
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        # Call the function
        result = start_local_api()

        # Verify API starts even with no data
        assert result is True, "API server should start even with no CSV data"

        # Verify API was started
        assert mock_popen.called

        # Check if the directory is still empty
        assert os.path.exists(scraped_dir), "data/scraped directory should still exist"
        assert len(os.listdir(scraped_dir)) == 0, "data/scraped directory should be empty"

        # Verify logs
        mock_logger.info.assert_any_call("API server started successfully.")
