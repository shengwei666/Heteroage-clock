"""
heteroage_clock.utils.logging

This module defines logging utilities that are used across the pipeline.
It ensures consistent logging with a singleton logger to prevent duplicate outputs.
"""

import logging
import sys
from typing import Optional

# Define a global logger name
LOGGER_NAME = "heteroage_clock"

def setup_logger(name: str = LOGGER_NAME, log_level: str = "INFO") -> logging.Logger:
    """
    Sets up a logger if it hasn't been configured yet.
    
    Args:
        name (str): Name of the logger.
        log_level (str): Log level.
        
    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level.upper())
    
    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        # Create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level.upper())
        ch.setFormatter(formatter)
        
        logger.addHandler(ch)
        
        # Prevent propagation to root logger (optional, keeps output clean)
        logger.propagate = False

    return logger

# Initialize the logger once at module level
_logger = setup_logger()

def log(msg: str, level: str = "INFO") -> None:
    """
    Log a message using the global singleton logger.

    Args:
        msg (str): The log message.
        level (str, optional): The level of logging (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is 'INFO'.
    """
    # Map string level to method
    log_func = getattr(_logger, level.lower(), _logger.info)
    log_func(msg)