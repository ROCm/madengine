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
import importlib
import json
import os
import sys
from io import StringIO
import tempfile
import unittest.mock
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open

# third-party modules
import pytest
import typer
from rich.console import Console as RichConsole
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

    def test_display_results_table_build_shows_gpu_arch_from_docker_builder(self):
        """Multi-arch builds record gpu_architecture; table must show it, not N/A."""
        summary = {
            "successful_builds": [
                {"model": "dummy", "docker_image": "ci-dummy_dummy.ubuntu.amd_gfx90a", "gpu_architecture": "gfx90a"},
                {"model": "dummy", "docker_image": "ci-dummy_dummy.ubuntu.amd_gfx942", "gpu_architecture": "gfx942"},
            ],
            "failed_builds": [],
        }

        with patch("madengine.cli.utils.console") as mock_console:
            display_results_table(summary, "Build Results", show_gpu_arch=True)

            mock_console.print.assert_called()
            table_arg = mock_console.print.call_args[0][0]
            buf = StringIO()
            RichConsole(file=buf, width=120, force_terminal=True).print(table_arg)
            rendered = buf.getvalue()
            assert "gfx90a" in rendered
            assert "gfx942" in rendered
            assert "N/A" not in rendered

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

    def test_validate_additional_context_defaults_fill_partial_fields(self):
        """Missing gpu_vendor or guest_os is filled from defaults (no error)."""
        from madengine.core.additional_context_defaults import (
            DEFAULT_GPU_VENDOR,
            DEFAULT_GUEST_OS,
        )

        with patch("madengine.cli.validators.console") as mock_console:
            r1 = validate_additional_context('{"guest_os": "UBUNTU"}')
            assert r1["gpu_vendor"] == DEFAULT_GPU_VENDOR
            assert r1["guest_os"] == "UBUNTU"

            r2 = validate_additional_context('{"gpu_vendor": "AMD"}')
            assert r2["gpu_vendor"] == "AMD"
            assert r2["guest_os"] == DEFAULT_GUEST_OS
            mock_console.print.assert_called()

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


# ============================================================================
# CLI exit code and error handling tests (CI / Jenkins smoke)
# ============================================================================

@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestExitCodeConstants:
    """Exit codes used by Jenkins/scripts for success and failure."""

    def test_build_failure_exit_code(self):
        """BUILD_FAILURE is 2 for build-only failures."""
        assert ExitCode.BUILD_FAILURE == 2

    def test_run_failure_exit_code(self):
        """RUN_FAILURE is 3 when at least one model run fails."""
        assert ExitCode.RUN_FAILURE == 3


class TestCliMainPreservesTyperExit:
    """cli_main must re-raise typer.Exit so process exit code is correct for Jenkins."""

    def test_typer_exit_preserves_exit_code(self):
        """When a command raises typer.Exit(3), cli_main re-raises and runner sees exit_code 3."""
        # Use importlib to get the app module (not the Typer bound as madengine.cli.app)
        app_module = importlib.import_module("madengine.cli.app")
        mock_app = MagicMock(side_effect=typer.Exit(ExitCode.RUN_FAILURE))
        with patch.object(app_module, "app", mock_app):
            with pytest.raises(typer.Exit) as exc_info:
                app_module.cli_main()
            assert exc_info.value.exit_code == ExitCode.RUN_FAILURE


class TestRunCommandExitCodes:
    """Smoke tests and run command exit codes for Jenkins.

    Uses Typer CliRunner (in-process); no Docker/GPU required for timeout/help tests.
    """

    def test_run_invalid_timeout_exits_invalid_args(self, runner: CliRunner) -> None:
        """timeout < -1 is rejected before orchestration (ExitCode.INVALID_ARGS)."""
        result = runner.invoke(
            app,
            ["run", "--timeout", "-2"],
        )
        assert result.exit_code == ExitCode.INVALID_ARGS

    def test_run_help_exits_zero(self, runner: CliRunner) -> None:
        """CLI help is reachable in CI without GPU."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == ExitCode.SUCCESS
        assert "run" in result.stdout.lower() or "model" in result.stdout.lower()
        assert "--skip-model-run" in result.stdout

    def test_run_command_build_error_returns_build_failure_exit_code(
        self, runner: CliRunner
    ) -> None:
        """When BuildError is raised (e.g. all builds failed), run command exits with BUILD_FAILURE."""
        from madengine.core.errors import BuildError

        run_module = importlib.import_module("madengine.cli.commands.run")
        mock_orch = MagicMock()
        mock_orch.execute.side_effect = BuildError("All builds failed")
        with patch.object(run_module, "RunOrchestrator", return_value=mock_orch):
            result = runner.invoke(
                app,
                [
                    "run",
                    "--tags",
                    "some_model",
                    "--additional-context",
                    '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}',
                ],
            )

        assert result.exit_code == ExitCode.BUILD_FAILURE
