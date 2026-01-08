"""Test the CLI module.

This module tests the modern Typer-based command-line interface functionality
including utilities, validation, and argument processing.

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


# ============================================================================
# CLI Utilities Tests
# ============================================================================

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


# ============================================================================
# CLI Validation Tests
# ============================================================================

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

    def test_validate_additional_context_invalid_json(self):
        """Test validation with invalid JSON."""
        with patch("madengine.cli.validators.console") as mock_console:
            with pytest.raises(typer.Exit) as exc_info:
                validate_additional_context("invalid json")

            assert exc_info.value.exit_code == ExitCode.INVALID_ARGS
            mock_console.print.assert_called()

    def test_validate_additional_context_missing_required_fields(self):
        """Test validation with missing required fields."""
        with patch("madengine.cli.validators.console") as mock_console:
            # Missing gpu_vendor
            with pytest.raises(typer.Exit) as exc_info:
                validate_additional_context('{"guest_os": "UBUNTU"}')
            assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

            # Missing guest_os
            with pytest.raises(typer.Exit) as exc_info:
                validate_additional_context('{"gpu_vendor": "AMD"}')
            assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_validate_additional_context_invalid_values(self):
        """Test validation with invalid field values."""
        with patch("madengine.cli.validators.console") as mock_console:
            # Invalid gpu_vendor
            with pytest.raises(typer.Exit) as exc_info:
                validate_additional_context(
                    '{"gpu_vendor": "INVALID", "guest_os": "UBUNTU"}'
                )
            assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

            # Invalid guest_os
            with pytest.raises(typer.Exit) as exc_info:
                validate_additional_context(
                    '{"gpu_vendor": "AMD", "guest_os": "INVALID"}'
                )
            assert exc_info.value.exit_code == ExitCode.INVALID_ARGS


class TestProcessBatchManifest:
    """Test the process_batch_manifest function."""

    def test_process_batch_manifest_valid_mixed_build_new(self):
        """Test processing batch manifest with mixed build_new values - core functionality."""
        from madengine.cli.validators import process_batch_manifest
        
        batch_data = [
            {"model_name": "model1", "build_new": True},
            {"model_name": "model2", "build_new": False},
            {"model_name": "model3", "build_new": True},
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(batch_data, f)
            temp_file = f.name
        
        try:
            result = process_batch_manifest(temp_file)
            
            # Only models with build_new=True should be in build_tags
            assert result["build_tags"] == ["model1", "model3"]
            # All models should be in all_tags
            assert result["all_tags"] == ["model1", "model2", "model3"]
            assert len(result["manifest_data"]) == 3
        finally:
            os.unlink(temp_file)

    def test_process_batch_manifest_default_build_new_false(self):
        """Test that build_new defaults to false when not specified."""
        from madengine.cli.validators import process_batch_manifest
        
        batch_data = [
            {"model_name": "model1"},  # No build_new field
            {"model_name": "model2", "build_new": True},
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(batch_data, f)
            temp_file = f.name
        
        try:
            result = process_batch_manifest(temp_file)
            
            # model1 should not be in build_tags (defaults to false)
            assert result["build_tags"] == ["model2"]
            assert result["all_tags"] == ["model1", "model2"]
        finally:
            os.unlink(temp_file)

    def test_process_batch_manifest_with_registry_fields(self):
        """Test per-model registry override - key feature."""
        from madengine.cli.validators import process_batch_manifest
        
        batch_data = [
            {
                "model_name": "model1",
                "build_new": True,
                "registry": "docker.io/myorg",
                "registry_image": "myorg/model1"
            },
            {
                "model_name": "model2",
                "build_new": True,
                "registry": "gcr.io/myproject"
            },
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(batch_data, f)
            temp_file = f.name
        
        try:
            result = process_batch_manifest(temp_file)
            
            # Verify registry metadata is preserved
            assert result["manifest_data"][0]["registry"] == "docker.io/myorg"
            assert result["manifest_data"][0]["registry_image"] == "myorg/model1"
            assert result["manifest_data"][1]["registry"] == "gcr.io/myproject"
        finally:
            os.unlink(temp_file)

    def test_process_batch_manifest_error_handling(self):
        """Test error handling for various invalid inputs."""
        from madengine.cli.validators import process_batch_manifest
        
        # File not found
        with pytest.raises(FileNotFoundError) as exc_info:
            process_batch_manifest("non_existent_file.json")
        assert "Batch manifest file not found" in str(exc_info.value)
        
        # Invalid JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content{")
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                process_batch_manifest(temp_file)
            assert "Invalid JSON" in str(exc_info.value)
        finally:
            os.unlink(temp_file)

    def test_process_batch_manifest_validation(self):
        """Test validation rules for batch manifest."""
        from madengine.cli.validators import process_batch_manifest
        
        # Not a list
        batch_data = {"model_name": "model1", "build_new": True}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(batch_data, f)
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                process_batch_manifest(temp_file)
            assert "must be a list" in str(exc_info.value)
        finally:
            os.unlink(temp_file)
        
        # Missing model_name
        batch_data = [{"build_new": True}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(batch_data, f)
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                process_batch_manifest(temp_file)
            assert "missing required 'model_name' field" in str(exc_info.value)
        finally:
            os.unlink(temp_file)
