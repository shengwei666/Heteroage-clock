"""
heteroage_clock.utils.logging

This module defines logging utilities that are used across the pipeline.
It ensures consistent logging, including timestamped log entries and configurable verbosity.
"""

import logging
from datetime import datetime


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Sets up a logger for pipeline processes, providing timestamped and level-controlled logs.

    Args:
        name (str): Name of the logger.
        log_level (str, optional): The log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is 'INFO'.

    Returns:
        logging.Logger: A logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level.upper())

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level.upper())
    ch.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(ch)

    return logger


def log(msg: str, level: str = "INFO") -> None:
    """
    Log a message with a specified level.

    Args:
        msg (str): The log message.
        level (str, optional): The level of logging (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is 'INFO'.
    """
    logger = setup_logger("heteroage_logger", log_level="INFO")
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(msg)
