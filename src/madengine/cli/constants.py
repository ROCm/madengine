#!/usr/bin/env python3
"""
Constants and configuration for madengine CLI

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

from enum import IntEnum


# Exit codes
class ExitCode(IntEnum):
    """Exit codes for CLI commands."""

    SUCCESS = 0
    FAILURE = 1
    BUILD_FAILURE = 2
    RUN_FAILURE = 3
    INVALID_ARGS = 4
    #: The workload finished but produced no performance metric. Distinct from
    #: RUN_FAILURE so a caller can tell a crashed run from a broken result contract.
    NO_METRIC = 5


# Valid values for validation
VALID_GPU_VENDORS = ["AMD", "NVIDIA"]
VALID_GUEST_OS = ["UBUNTU", "CENTOS"]

# Default file paths and values
DEFAULT_MANIFEST_FILE = "build_manifest.json"
DEFAULT_PERF_OUTPUT = "perf.csv"
DEFAULT_DATA_CONFIG = "data.json"
DEFAULT_TOOLS_CONFIG = "./scripts/common/tools.json"
DEFAULT_TIMEOUT = -1

