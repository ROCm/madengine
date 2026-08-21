#!/usr/bin/env python3
"""Module to define the Timeout class and run-timeout resolution.

This module provides the Timeout class to handle timeouts, plus the single
definition of how a run timeout is resolved and how the sentinel maps onto
subprocess semantics.

The sentinel contract, applied at every layer:

    -1   not specified — fall through to the next precedence level
     0   no timeout — run unbounded
    > 0  explicit timeout in seconds

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import signal
import typing
from typing import Optional

# Default run timeout (2 hours). Single source of truth for local and
# distributed execution alike.
DEFAULT_RUN_TIMEOUT = 7200


def resolve_run_timeout(
    model_info: typing.Dict,
    cli_timeout: typing.Optional[int],
    default_timeout: int = DEFAULT_RUN_TIMEOUT,
) -> int:
    """Resolve the effective run timeout.

    Precedence, lowest to highest: default < model card < explicit CLI. A value
    of -1 at either level means "not specified" and falls through; 0 means "no
    timeout" and is a real choice that wins over the levels below it.

    ``None`` is accepted as a synonym for -1 so that manifests written by older
    builds (which store ``null`` for an absent timeout) still load.

    Args:
        model_info: Model info dict; may have a "timeout" key.
        cli_timeout: Timeout from the CLI, using the sentinel contract.
        default_timeout: Value used when neither level specifies one.

    Returns:
        int: Effective timeout in seconds; 0 means no timeout.
    """
    timeout = default_timeout

    model_timeout = model_info.get("timeout", -1)
    if model_timeout is not None and model_timeout >= 0:
        timeout = model_timeout

    if cli_timeout is not None and cli_timeout >= 0:
        timeout = cli_timeout

    return timeout


def subprocess_timeout(timeout: typing.Optional[int]) -> Optional[int]:
    """Map a sentinel timeout onto ``subprocess``/``communicate`` semantics.

    ``subprocess`` treats ``timeout=0`` as "expire immediately", not as "no
    timeout", so the sentinel cannot be passed through directly. Both 0 (no
    timeout) and -1 (unspecified) become ``None``.

    Args:
        timeout: Timeout using the sentinel contract.

    Returns:
        Optional[int]: Seconds to pass to subprocess, or None for no timeout.
    """
    if timeout is None or timeout <= 0:
        return None
    return timeout


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
