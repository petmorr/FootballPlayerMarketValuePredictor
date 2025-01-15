import os

import logging

# Constants for log directory and default log file name
LOG_DIR = "./logging"
os.makedirs(LOG_DIR, exist_ok=True)


def configure_logger(name: str, log_file: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Configure and return a logger instance.

    Args:
        name (str): Name of the logger.
        log_file (str): File path for the log file.
        level (int): Logging level (default: DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_file_path = os.path.join(LOG_DIR, log_file)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # File handler
    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.setLevel(level)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)

    # Stream handler (console output)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(file_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
