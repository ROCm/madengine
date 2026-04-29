#!/usr/bin/env python3
"""Utility functions for madengine

This module contains utility functions for madengine.

functions:
    PythonicTee: Class to both write and display stream, in "live" mode
    file_print: Write and flush file

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import typing
import sys


# Class to both write and display stream, in "live" mode
class PythonicTee(object):
    """Class to both write and display stream, in 'live' mode."""

    def __init__(self, logf: typing.Any, live_output: bool = False) -> None:
        """Initialize the PythonicTee class.

        Args:
            logf: The log file.
            live_output: The live output flag. Default is False.
        """
        self.logf = logf
        self.stdio = None
        # if live_output is True, then set the actual stdout for printing
        if live_output:
            self.stdio = sys.__stdout__  # actual stdout for printing

    def write(self, data: str) -> None:
        """Write the data.

        Args:
            data: The data to write.
        """
        self.logf.write(data)
        # write to stdout
        if self.stdio:
            self.stdio.write(data)

    def flush(self) -> None:
        """Flush the data."""
        self.logf.flush()
        # flush the stdout buffer
        if self.stdio:
            self.stdio.flush()


def file_print(write_str: str, filename: str, mode: str = "a") -> None:
    """Write and flush file.

    Args:
        write_str (str): The string to write.
        filename (str): The name of the file.
        mode (str): The mode of the file. Default is "a".
    """
    with open(filename, mode) as perf_csv:
        print(write_str, file=perf_csv)
        perf_csv.flush()
