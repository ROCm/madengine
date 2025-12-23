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

    def test_process_batch_manifest_file_not_found(self):
        """Test error handling for non-existent file."""
        from madengine.cli.validators import process_batch_manifest
        
        with pytest.raises(FileNotFoundError) as exc_info:
            process_batch_manifest("non_existent_file.json")
        
        assert "Batch manifest file not found" in str(exc_info.value)

    def test_process_batch_manifest_invalid_json(self):
        """Test error handling for invalid JSON."""
        from madengine.cli.validators import process_batch_manifest
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content{")
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                process_batch_manifest(temp_file)
            
            assert "Invalid JSON" in str(exc_info.value)
        finally:
            os.unlink(temp_file)

    def test_process_batch_manifest_not_a_list(self):
        """Test validation that manifest must be a list."""
        from madengine.cli.validators import process_batch_manifest
        
        batch_data = {"model_name": "model1", "build_new": True}  # Dict instead of list
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(batch_data, f)
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                process_batch_manifest(temp_file)
            
            assert "must be a list" in str(exc_info.value)
        finally:
            os.unlink(temp_file)

    def test_process_batch_manifest_missing_model_name(self):
        """Test validation for required model_name field."""
        from madengine.cli.validators import process_batch_manifest
        
        batch_data = [
            {"build_new": True},  # Missing model_name
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(batch_data, f)
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                process_batch_manifest(temp_file)
            
            assert "missing required 'model_name' field" in str(exc_info.value)
        finally:
            os.unlink(temp_file)




