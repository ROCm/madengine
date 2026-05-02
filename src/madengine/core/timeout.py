#!/usr/bin/env python3
"""Module to define the Timeout class.

This module provides the Timeout class to handle timeouts.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import signal
from typing import Optional


class Timeout:
    """Class to handle timeouts.

    Attributes:
        seconds (Optional[int]): The timeout in seconds, or None/0 to disable.
    """

    def __init__(self, seconds: Optional[int] = 15) -> None:
        """Constructor of the Timeout class.

        Args:
            seconds (Optional[int]): The timeout in seconds. None or 0 disables
                the timeout. Negative values are treated as no timeout.
        """
        self.seconds = seconds if seconds and seconds > 0 else None

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
        if not self.seconds:
            return
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback) -> None:
        """Exit the context manager."""
        if not self.seconds:
            return
        signal.alarm(0)
