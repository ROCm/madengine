"""Test the mad_cli module.

This module tests the modern Typer-based command-line interface functionality.

GPU Hardware Support:
- Tests automatically detect if the machine has GPU hardware
- GPU-dependent tests are skipped on CPU-only machines using @requires_gpu decorator
- Tests use auto-generated additional context appropriate for the current machine
- CPU-only machines default to AMD GPU vendor for build compatibility

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import json
import os
import sys
import tempfile
import unittest.mock
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open

# third-party modules
import pytest
import typer
from typer.testing import CliRunner

# project modules
from madengine.cli import (
    app,
    setup_logging,
    create_args_namespace,
    validate_additional_context,
    save_summary_with_feedback,
    display_results_table,
    ExitCode,
    VALID_GPU_VENDORS,
    VALID_GUEST_OS,
    DEFAULT_MANIFEST_FILE,
    DEFAULT_PERF_OUTPUT,
    DEFAULT_DATA_CONFIG,
    DEFAULT_TOOLS_CONFIG,
    DEFAULT_TIMEOUT,
)
from tests.fixtures.utils import (
    BASE_DIR,
    MODEL_DIR,
    has_gpu,
    requires_gpu,
    generate_additional_context_for_machine,
)


class TestConstants:
    """Test module constants."""

    def test_exit_codes(self):
        """Test exit code constants."""
        assert ExitCode.SUCCESS == 0
        assert ExitCode.FAILURE == 1
        assert ExitCode.BUILD_FAILURE == 2
        assert ExitCode.RUN_FAILURE == 3
        assert ExitCode.INVALID_ARGS == 4

    def test_valid_values(self):
        """Test valid value constants."""
        assert "AMD" in VALID_GPU_VENDORS
        assert "NVIDIA" in VALID_GPU_VENDORS
        assert "INTEL" in VALID_GPU_VENDORS

        assert "UBUNTU" in VALID_GUEST_OS
        assert "CENTOS" in VALID_GUEST_OS
        assert "ROCKY" in VALID_GUEST_OS

    def test_default_values(self):
        """Test default value constants."""
        assert DEFAULT_MANIFEST_FILE == "build_manifest.json"
        assert DEFAULT_PERF_OUTPUT == "perf.csv"
        assert DEFAULT_DATA_CONFIG == "data.json"
        assert DEFAULT_TOOLS_CONFIG == "./scripts/common/tools.json"
        assert DEFAULT_TIMEOUT == -1






