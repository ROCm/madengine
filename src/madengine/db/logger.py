"""Module of logging functions.

This module provides the functions to setup the logger for the MAD Engine.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import logging
import os
import sys


# Get the log level, if it is not set, set it to INFO.
if "LOG_LEVEL" not in os.environ:
    LOG_LEVEL = "INFO"
else:
    LOG_LEVEL = os.environ["LOG_LEVEL"]


def setup_logger():
    """Setup the logger for the MAD Engine.

    This function sets up the logger for the MAD Engine.

    Returns:
        logging.Logger: The logger for the MAD Engine.
    """
    logging.basicConfig(level=LOG_LEVEL)
    # Create a logger
    logger = logging.getLogger("madengine")
    # logger.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )

    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.propagate = False
    logger.addHandler(console_handler)

    # Create a file handler
    log_file = os.path.join(os.getcwd(), "madengine.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
