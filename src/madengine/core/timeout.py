#!/usr/bin/env python3
"""Module to define the Timeout class and run-timeout resolution.

This module provides the Timeout class to handle timeouts, plus the single
definition of how a run timeout is resolved and how it maps onto subprocess
semantics.

Resolution follows madengine v1: the default is overridden by the model card,
which is overridden by an explicit ``--timeout``. Only the CLI has a sentinel,
``-1``, meaning "not passed". Any non-positive resolved timeout runs unbounded,
which both consumers below already implement.

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

    Precedence, lowest to highest: default < model card < CLI. A model card's
    ``timeout`` is taken as-is, including a non-positive one, which means the
    author asked for no timeout. A CLI timeout of -1 means ``--timeout`` was
    not passed and falls through to the level below.

    ``None`` in the model card is ignored so that manifests written by older
    builds (which store ``null`` for an absent timeout) still load.

    Args:
        model_info: Model info dict; may have a "timeout" key.
        cli_timeout: Timeout from the CLI; -1 if not passed.
        default_timeout: Value used when neither level specifies one.

    Returns:
        int: Effective timeout in seconds; non-positive means no timeout.
    """
    timeout = default_timeout

    model_timeout = model_info.get("timeout")
    if model_timeout is not None:
        timeout = model_timeout

    if cli_timeout is not None and cli_timeout >= 0:
        timeout = cli_timeout

    return timeout


def subprocess_timeout(timeout: typing.Optional[int]) -> Optional[int]:
    """Map a resolved timeout onto ``subprocess``/``communicate`` semantics.

    ``subprocess`` treats ``timeout=0`` as "expire immediately", not as "no
    timeout", so a non-positive timeout cannot be passed through directly and
    becomes ``None`` instead.

    Args:
        timeout: Resolved timeout in seconds; non-positive means no timeout.

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
