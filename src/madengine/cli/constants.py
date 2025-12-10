#!/usr/bin/env python3
"""
Constants and configuration for madengine CLI

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""


# Exit codes
class ExitCode:
    """Exit codes for CLI commands."""
    
    SUCCESS = 0
    FAILURE = 1
    BUILD_FAILURE = 2
    RUN_FAILURE = 3
    INVALID_ARGS = 4


# Valid values for validation
VALID_GPU_VENDORS = ["AMD", "NVIDIA", "INTEL"]
VALID_GUEST_OS = ["UBUNTU", "CENTOS", "ROCKY"]

# Default file paths and values
DEFAULT_MANIFEST_FILE = "build_manifest.json"
DEFAULT_PERF_OUTPUT = "perf.csv"
DEFAULT_DATA_CONFIG = "data.json"
DEFAULT_TOOLS_CONFIG = "./scripts/common/tools.json"
DEFAULT_TIMEOUT = -1

