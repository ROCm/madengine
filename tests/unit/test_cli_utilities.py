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


class TestSetupLogging:
    """Test the setup_logging function."""

    @patch("madengine.cli.utils.logging.basicConfig")
    def test_setup_logging_verbose(self, mock_basic_config):
        """Test logging setup with verbose mode enabled."""
        setup_logging(verbose=True)

        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args
        assert call_args[1]["level"] == 10  # logging.DEBUG

    @patch("madengine.cli.utils.logging.basicConfig")
    def test_setup_logging_normal(self, mock_basic_config):
        """Test logging setup with normal mode."""
        setup_logging(verbose=False)

        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args
        assert call_args[1]["level"] == 20  # logging.INFO






class TestCreateArgsNamespace:
    """Test the create_args_namespace function."""

    def test_create_args_namespace_basic(self):
        """Test creating args namespace with basic parameters."""
        args = create_args_namespace(
            tags=["dummy"], registry="localhost:5000", verbose=True
        )

        assert args.tags == ["dummy"]
        assert args.registry == "localhost:5000"
        assert args.verbose is True

    def test_create_args_namespace_empty(self):
        """Test creating args namespace with no parameters."""
        args = create_args_namespace()

        # Should create an object with no attributes
        assert not hasattr(args, "tags")

    def test_create_args_namespace_complex(self):
        """Test creating args namespace with complex parameters."""
        args = create_args_namespace(
            tags=["model1", "model2"],
            additional_context='{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}',
            timeout=300,
            keep_alive=True,
            verbose=False,
        )

        assert args.tags == ["model1", "model2"]
        assert args.additional_context == '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
        assert args.timeout == 300
        assert args.keep_alive is True
        assert args.verbose is False






class TestSaveSummaryWithFeedback:
    """Test the save_summary_with_feedback function."""

    def test_save_summary_success(self):
        """Test successful summary saving."""
        summary = {"successful_builds": ["model1", "model2"], "failed_builds": []}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            with patch("madengine.cli.utils.console") as mock_console:
                save_summary_with_feedback(summary, temp_file, "Build")

                # Verify file was written
                with open(temp_file, "r") as f:
                    saved_data = json.load(f)
                assert saved_data == summary

                mock_console.print.assert_called()
        finally:
            os.unlink(temp_file)

    def test_save_summary_no_output_path(self):
        """Test summary saving with no output path."""
        summary = {"successful_builds": ["model1"], "failed_builds": []}

        with patch("madengine.cli.utils.console") as mock_console:
            save_summary_with_feedback(summary, None, "Build")

            # Should not call console.print for saving
            mock_console.print.assert_not_called()

    def test_save_summary_io_error(self):
        """Test summary saving with IO error."""
        summary = {"successful_builds": ["model1"], "failed_builds": []}

        with patch("madengine.cli.utils.console") as mock_console:
            with pytest.raises(typer.Exit) as exc_info:
                save_summary_with_feedback(summary, "/invalid/path/file.json", "Build")

            assert exc_info.value.exit_code == ExitCode.FAILURE
            mock_console.print.assert_called()






class TestDisplayResultsTable:
    """Test the display_results_table function."""

    def test_display_results_table_build_success(self):
        """Test displaying build results table with successes."""
        summary = {"successful_builds": ["model1", "model2"], "failed_builds": []}

        with patch("madengine.cli.utils.console") as mock_console:
            display_results_table(summary, "Build Results")

            mock_console.print.assert_called()

    def test_display_results_table_build_failures(self):
        """Test displaying build results table with failures."""
        summary = {
            "successful_builds": ["model1"],
            "failed_builds": ["model2", "model3"],
        }

        with patch("madengine.cli.utils.console") as mock_console:
            display_results_table(summary, "Build Results")

            mock_console.print.assert_called()

    def test_display_results_table_run_results(self):
        """Test displaying run results table."""
        summary = {
            "successful_runs": [
                {"model": "model1", "status": "success"},
                {"model": "model2", "status": "success"},
            ],
            "failed_runs": [{"model": "model3", "status": "failed"}],
        }

        with patch("madengine.cli.utils.console") as mock_console:
            display_results_table(summary, "Run Results")

            mock_console.print.assert_called()

    def test_display_results_table_empty_results(self):
        """Test displaying empty results table."""
        summary = {"successful_builds": [], "failed_builds": []}

        with patch("madengine.cli.utils.console") as mock_console:
            display_results_table(summary, "Empty Results")

            mock_console.print.assert_called()

    def test_display_results_table_many_items(self):
        """Test displaying results table with many items (truncation)."""
        summary = {
            "successful_builds": [f"model{i}" for i in range(10)],
            "failed_builds": [],
        }

        with patch("madengine.cli.utils.console") as mock_console:
            display_results_table(summary, "Many Results")

            mock_console.print.assert_called()






