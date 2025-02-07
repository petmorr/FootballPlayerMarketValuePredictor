import os
import subprocess

from logging_config import configure_logger

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
# Configure the logger. You can replace this basic configuration with your own
# (e.g., using a custom configure_logger function if available).
logger = configure_logger("main", "logging/main.log")

# ------------------------------------------------------------------------------
# Scripts to Execute
# ------------------------------------------------------------------------------
# Define the list of Python scripts to be executed sequentially.
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

    This function constructs the full path to the script, checks whether the file exists,
    and then runs the script using the system's Python interpreter. The output and errors
    are captured and logged accordingly.

    Args:
        script_name (str): Name of the Python script to execute.

    Raises:
        FileNotFoundError: If the script is not found in the current directory.
    """
    # Construct the absolute path of the script.
    script_path = os.path.join(os.getcwd(), script_name)

    # Verify that the script exists.
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Script '{script_name}' not found in the current directory: {os.getcwd()}")

    try:
        logger.info(f"Executing script: {script_name}")
        # Run the script using subprocess and capture its output.
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True
        )

        # Check if the script executed successfully.
        if result.returncode == 0:
            logger.info(f"Script '{script_name}' completed successfully.")
            logger.debug(f"Output:\n{result.stdout}")
        else:
            logger.error(f"Script '{script_name}' encountered an error (Return code: {result.returncode}).")
            logger.error(f"Error output:\n{result.stderr}")
    except Exception as e:
        logger.exception(f"An error occurred while executing '{script_name}': {e}")


# ------------------------------------------------------------------------------
# Function: main
# ------------------------------------------------------------------------------
def main() -> None:
    """
    Main function to execute all scripts sequentially.

    Iterates over the list of scripts and executes each one using the execute_script function.
    If a script is not found or an error occurs during its execution, the error is logged,
    and the script continues with the next file.
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