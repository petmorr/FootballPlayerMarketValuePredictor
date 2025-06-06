import os
import sys

import logging

# ------------------------------------------------------------------------------
# Logger Directory Setup
# ------------------------------------------------------------------------------
# Define the directory for log files and create it if it doesn't exist.
LOG_DIR = "./logging"
os.makedirs(LOG_DIR, exist_ok=True)


def configure_logger(name: str, log_file: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Configure and return a logger instance with separate file and console handlers.

    This function sets up a logger with two handlers:
      - A file handler that writes logs to a specified file using UTF-16 encoding.
      - A stream (console) handler that outputs logs to stdout using the system's default encoding (typically UTF-8).

    If the logger already has handlers attached, the function will not add duplicate handlers.

    Args:
        name (str): Name of the logger.
        log_file (str): Log file name (will be placed under the LOG_DIR).
        level (int): Logging level (default: DEBUG).

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Construct the full path for the log file.
    log_file_path = os.path.join(LOG_DIR, log_file)

    # Retrieve (or create) the logger by name.
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if handlers have already been added to avoid duplication.
    if not logger.handlers:
        # ------------------------------------------------------------------------------
        # File Handler Setup
        # ------------------------------------------------------------------------------
        # Create a file handler that writes to the log file with UTF-8 encoding.
        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        # Define a common log message format.
        file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(file_formatter)

        # ------------------------------------------------------------------------------
        # Stream (Console) Handler Setup
        # ------------------------------------------------------------------------------
        # Create a stream handler that outputs to sys.stdout.
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(file_formatter)

        # Add both handlers to the logger.
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
