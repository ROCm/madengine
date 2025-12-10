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


class TestValidateAdditionalContext:
    """Test the validate_additional_context function."""

    def test_validate_additional_context_valid_string(self):
        """Test validation with valid additional context from string."""
        # Use auto-generated context for current machine
        context = generate_additional_context_for_machine()
        context_json = json.dumps(context)

        with patch("madengine.cli.validators.console") as mock_console:
            result = validate_additional_context(context_json)

            assert result == context
            mock_console.print.assert_called()

    def test_validate_additional_context_valid_file(self):
        """Test validation with valid additional context from file."""
        # Use auto-generated context for current machine
        context = generate_additional_context_for_machine()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(context, f)
            temp_file = f.name

        try:
            with patch("madengine.cli.validators.console") as mock_console:
                result = validate_additional_context("{}", temp_file)

                assert result == context
                mock_console.print.assert_called()
        finally:
            os.unlink(temp_file)

    def test_validate_additional_context_string_overrides_file(self):
        """Test that string context overrides file context."""
        # Use auto-generated context for current machine
        context = generate_additional_context_for_machine()
        context_json = json.dumps(context)

        # Create file with different context
        file_context = {"gpu_vendor": "NVIDIA", "guest_os": "CENTOS"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(file_context, f)
            temp_file = f.name

        try:
            with patch("madengine.cli.validators.console") as mock_console:
                result = validate_additional_context(context_json, temp_file)

                assert result == context
        finally:
            os.unlink(temp_file)

    def test_validate_additional_context_invalid_json(self):
        """Test validation with invalid JSON."""
        with patch("madengine.cli.validators.console") as mock_console:
            with pytest.raises(typer.Exit) as exc_info:
                validate_additional_context("invalid json")

            assert exc_info.value.exit_code == ExitCode.INVALID_ARGS
            mock_console.print.assert_called()

    def test_validate_additional_context_missing_gpu_vendor(self):
        """Test validation with missing gpu_vendor."""
        with patch("madengine.cli.validators.console") as mock_console:
            with pytest.raises(typer.Exit) as exc_info:
                validate_additional_context('{"guest_os": "UBUNTU"}')

            assert exc_info.value.exit_code == ExitCode.INVALID_ARGS
            mock_console.print.assert_called()

    def test_validate_additional_context_missing_guest_os(self):
        """Test validation with missing guest_os."""
        with patch("madengine.cli.validators.console") as mock_console:
            with pytest.raises(typer.Exit) as exc_info:
                validate_additional_context('{"gpu_vendor": "AMD"}')

            assert exc_info.value.exit_code == ExitCode.INVALID_ARGS
            mock_console.print.assert_called()

    def test_validate_additional_context_invalid_gpu_vendor(self):
        """Test validation with invalid gpu_vendor."""
        with patch("madengine.cli.validators.console") as mock_console:
            with pytest.raises(typer.Exit) as exc_info:
                validate_additional_context(
                    '{"gpu_vendor": "INVALID", "guest_os": "UBUNTU"}'
                )

            assert exc_info.value.exit_code == ExitCode.INVALID_ARGS
            mock_console.print.assert_called()

    def test_validate_additional_context_invalid_guest_os(self):
        """Test validation with invalid guest_os."""
        with patch("madengine.cli.validators.console") as mock_console:
            with pytest.raises(typer.Exit) as exc_info:
                validate_additional_context(
                    '{"gpu_vendor": "AMD", "guest_os": "INVALID"}'
                )

            assert exc_info.value.exit_code == ExitCode.INVALID_ARGS
            mock_console.print.assert_called()

    def test_validate_additional_context_case_insensitive(self):
        """Test validation with case insensitive values."""
        with patch("madengine.cli.validators.console") as mock_console:
            result = validate_additional_context(
                '{"gpu_vendor": "amd", "guest_os": "ubuntu"}'
            )

            assert result == {"gpu_vendor": "amd", "guest_os": "ubuntu"}
            mock_console.print.assert_called()

    def test_validate_additional_context_empty_context(self):
        """Test validation with empty context."""
        with patch("madengine.cli.validators.console") as mock_console:
            with pytest.raises(typer.Exit) as exc_info:
                validate_additional_context("{}")

            assert exc_info.value.exit_code == ExitCode.INVALID_ARGS
            mock_console.print.assert_called()

    def test_validate_additional_context_file_not_found(self):
        """Test validation with non-existent file."""
        with patch("madengine.cli.validators.console") as mock_console:
            with pytest.raises(typer.Exit) as exc_info:
                validate_additional_context("{}", "non_existent_file.json")

            assert exc_info.value.exit_code == ExitCode.INVALID_ARGS
            mock_console.print.assert_called()






