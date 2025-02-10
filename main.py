import subprocess
from pathlib import Path
from sys import executable

from logging_config import configure_logger

# ------------------------------------------------------------------------------
# Logger Configuration
# ------------------------------------------------------------------------------
logger = configure_logger("main", "main.log")

# ------------------------------------------------------------------------------
# Scripts to Execute (in order)
# ------------------------------------------------------------------------------
SCRIPTS = [
    "web_scrape.py",  # Script to scrape player data.
    "preprocessing.py",  # Script to preprocess scraped data.
    "player_value.py"  # Script to add market values to the preprocessed data.
]

# ------------------------------------------------------------------------------
# Function: execute_script
# ------------------------------------------------------------------------------
def execute_script(script_name: str) -> None:
    """
    Execute a Python script using subprocess.

    This function builds the absolute path for the given script, verifies
    its existence, and then runs it using the current Python interpreter.
    Output and errors are captured and logged.

    Args:
        script_name (str): Name of the Python script to execute.

    Raises:
        FileNotFoundError: If the script is not found in the current directory.
    """
    script_path = Path.cwd() / script_name
    if not script_path.is_file():
        raise FileNotFoundError(
            f"Script '{script_name}' not found in the current directory: {Path.cwd()}"
        )

    logger.info(f"Executing script: {script_name}")
    try:
        # Use sys.executable to ensure the current Python interpreter is used.
        result = subprocess.run(
            [executable, str(script_path)],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode == 0:
            logger.info(f"Script '{script_name}' completed successfully.")
            logger.debug(f"Output:\n{result.stdout}")
        else:
            logger.error(
                f"Script '{script_name}' encountered an error (Return code: {result.returncode})."
            )
            logger.error(f"Error output:\n{result.stderr}")
    except Exception as e:
        logger.exception(f"An error occurred while executing '{script_name}': {e}")

# ------------------------------------------------------------------------------
# Function: main
# ------------------------------------------------------------------------------
def main() -> None:
    """
    Execute all specified scripts sequentially.

    Iterates over the list of scripts and executes each one using the execute_script function.
    Any errors during execution are logged.
    """
    logger.info("Starting the execution of all scripts...")
    for script in SCRIPTS:
        try:
            execute_script(script)
        except FileNotFoundError as fnf_error:
            logger.error(fnf_error)
        except Exception as e:
            logger.exception(f"An unexpected error occurred while processing '{script}': {e}")
    logger.info("All scripts have been executed.")

# ------------------------------------------------------------------------------
# Script Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()