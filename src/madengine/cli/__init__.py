#!/usr/bin/env python3
"""
CLI Package for madengine

This package contains the modular CLI implementation split from the
monolithic mad_cli.py for better maintainability.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# Import for backward compatibility
from .app import app, cli_main
from .constants import ExitCode, VALID_GPU_VENDORS, VALID_GUEST_OS
from .constants import (
    DEFAULT_MANIFEST_FILE,
    DEFAULT_PERF_OUTPUT,
    DEFAULT_DATA_CONFIG,
    DEFAULT_TOOLS_CONFIG,
    DEFAULT_TIMEOUT,
)
from .utils import (
    setup_logging,
    split_comma_separated_tags,
    create_args_namespace,
    save_summary_with_feedback,
    display_results_table,
    display_performance_table,
)
from .validators import (
    validate_additional_context,
    process_batch_manifest,
    process_batch_manifest_entries,
)

__all__ = [
    "app",
    "cli_main",
    "ExitCode",
    "VALID_GPU_VENDORS",
    "VALID_GUEST_OS",
    "DEFAULT_MANIFEST_FILE",
    "DEFAULT_PERF_OUTPUT",
    "DEFAULT_DATA_CONFIG",
    "DEFAULT_TOOLS_CONFIG",
    "DEFAULT_TIMEOUT",
    "setup_logging",
    "split_comma_separated_tags",
    "create_args_namespace",
    "save_summary_with_feedback",
    "display_results_table",
    "display_performance_table",
    "validate_additional_context",
    "process_batch_manifest",
    "process_batch_manifest_entries",
]

