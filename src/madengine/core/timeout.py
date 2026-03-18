#!/usr/bin/env python3
"""Module to define the Timeout class.

This module provides the Timeout class to handle timeouts.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import signal
import typing


class Timeout:
    """Class to handle timeouts.

    Attributes:
        seconds (int): The timeout in seconds.
    """

    def __init__(self, seconds: int = 15) -> None:
        """Constructor of the Timeout class.

        Args:
            seconds (int): The timeout in seconds.
        """
        self.seconds = seconds

    def handle_timeout(self, signum, frame) -> None:
        """Handle timeout.

        Args:
            signum: The signal number.
            frame: The frame.

        Returns:
            None

        Raises:
            TimeoutError: If the program times out.
        """
        raise TimeoutError("Program timed out. Requested timeout=" + str(self.seconds))

    def __enter__(self) -> None:
        """Enter the context manager."""
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback) -> None:
        """Exit the context manager."""
        signal.alarm(0)
