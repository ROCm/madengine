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
from madengine import mad_cli
from madengine.mad_cli import (
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
    DEFAULT_ANSIBLE_OUTPUT,
    DEFAULT_TIMEOUT,
)
from .fixtures.utils import (
    BASE_DIR,
    MODEL_DIR,
    has_gpu,
    requires_gpu,
    generate_additional_context_for_machine,
)


class TestSetupLogging:
    """Test the setup_logging function."""

    @patch("madengine.mad_cli.logging.basicConfig")
    def test_setup_logging_verbose(self, mock_basic_config):
        """Test logging setup with verbose mode enabled."""
        setup_logging(verbose=True)

        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args
        assert call_args[1]["level"] == 10  # logging.DEBUG

    @patch("madengine.mad_cli.logging.basicConfig")
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


class TestValidateAdditionalContext:
    """Test the validate_additional_context function."""

    def test_validate_additional_context_valid_string(self):
        """Test validation with valid additional context from string."""
        # Use auto-generated context for current machine
        context = generate_additional_context_for_machine()
        context_json = json.dumps(context)

        with patch("madengine.mad_cli.console") as mock_console:
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
            with patch("madengine.mad_cli.console") as mock_console:
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
            with patch("madengine.mad_cli.console") as mock_console:
                result = validate_additional_context(context_json, temp_file)

                assert result == context
        finally:
            os.unlink(temp_file)

    def test_validate_additional_context_invalid_json(self):
        """Test validation with invalid JSON."""
        with patch("madengine.mad_cli.console") as mock_console:
            with pytest.raises(typer.Exit) as exc_info:
                validate_additional_context("invalid json")

            assert exc_info.value.exit_code == ExitCode.INVALID_ARGS
            mock_console.print.assert_called()

    def test_validate_additional_context_missing_gpu_vendor(self):
        """Test validation with missing gpu_vendor."""
        with patch("madengine.mad_cli.console") as mock_console:
            with pytest.raises(typer.Exit) as exc_info:
                validate_additional_context('{"guest_os": "UBUNTU"}')

            assert exc_info.value.exit_code == ExitCode.INVALID_ARGS
            mock_console.print.assert_called()

    def test_validate_additional_context_missing_guest_os(self):
        """Test validation with missing guest_os."""
        with patch("madengine.mad_cli.console") as mock_console:
            with pytest.raises(typer.Exit) as exc_info:
                validate_additional_context('{"gpu_vendor": "AMD"}')

            assert exc_info.value.exit_code == ExitCode.INVALID_ARGS
            mock_console.print.assert_called()

    def test_validate_additional_context_invalid_gpu_vendor(self):
        """Test validation with invalid gpu_vendor."""
        with patch("madengine.mad_cli.console") as mock_console:
            with pytest.raises(typer.Exit) as exc_info:
                validate_additional_context(
                    '{"gpu_vendor": "INVALID", "guest_os": "UBUNTU"}'
                )

            assert exc_info.value.exit_code == ExitCode.INVALID_ARGS
            mock_console.print.assert_called()

    def test_validate_additional_context_invalid_guest_os(self):
        """Test validation with invalid guest_os."""
        with patch("madengine.mad_cli.console") as mock_console:
            with pytest.raises(typer.Exit) as exc_info:
                validate_additional_context(
                    '{"gpu_vendor": "AMD", "guest_os": "INVALID"}'
                )

            assert exc_info.value.exit_code == ExitCode.INVALID_ARGS
            mock_console.print.assert_called()

    def test_validate_additional_context_case_insensitive(self):
        """Test validation with case insensitive values."""
        with patch("madengine.mad_cli.console") as mock_console:
            result = validate_additional_context(
                '{"gpu_vendor": "amd", "guest_os": "ubuntu"}'
            )

            assert result == {"gpu_vendor": "amd", "guest_os": "ubuntu"}
            mock_console.print.assert_called()

    def test_validate_additional_context_empty_context(self):
        """Test validation with empty context."""
        with patch("madengine.mad_cli.console") as mock_console:
            with pytest.raises(typer.Exit) as exc_info:
                validate_additional_context("{}")

            assert exc_info.value.exit_code == ExitCode.INVALID_ARGS
            mock_console.print.assert_called()

    def test_validate_additional_context_file_not_found(self):
        """Test validation with non-existent file."""
        with patch("madengine.mad_cli.console") as mock_console:
            with pytest.raises(typer.Exit) as exc_info:
                validate_additional_context("{}", "non_existent_file.json")

            assert exc_info.value.exit_code == ExitCode.INVALID_ARGS
            mock_console.print.assert_called()


class TestSaveSummaryWithFeedback:
    """Test the save_summary_with_feedback function."""

    def test_save_summary_success(self):
        """Test successful summary saving."""
        summary = {"successful_builds": ["model1", "model2"], "failed_builds": []}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            with patch("madengine.mad_cli.console") as mock_console:
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

        with patch("madengine.mad_cli.console") as mock_console:
            save_summary_with_feedback(summary, None, "Build")

            # Should not call console.print for saving
            mock_console.print.assert_not_called()

    def test_save_summary_io_error(self):
        """Test summary saving with IO error."""
        summary = {"successful_builds": ["model1"], "failed_builds": []}

        with patch("madengine.mad_cli.console") as mock_console:
            with pytest.raises(typer.Exit) as exc_info:
                save_summary_with_feedback(summary, "/invalid/path/file.json", "Build")

            assert exc_info.value.exit_code == ExitCode.FAILURE
            mock_console.print.assert_called()


class TestDisplayResultsTable:
    """Test the display_results_table function."""

    def test_display_results_table_build_success(self):
        """Test displaying build results table with successes."""
        summary = {"successful_builds": ["model1", "model2"], "failed_builds": []}

        with patch("madengine.mad_cli.console") as mock_console:
            display_results_table(summary, "Build Results")

            mock_console.print.assert_called()

    def test_display_results_table_build_failures(self):
        """Test displaying build results table with failures."""
        summary = {
            "successful_builds": ["model1"],
            "failed_builds": ["model2", "model3"],
        }

        with patch("madengine.mad_cli.console") as mock_console:
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

        with patch("madengine.mad_cli.console") as mock_console:
            display_results_table(summary, "Run Results")

            mock_console.print.assert_called()

    def test_display_results_table_empty_results(self):
        """Test displaying empty results table."""
        summary = {"successful_builds": [], "failed_builds": []}

        with patch("madengine.mad_cli.console") as mock_console:
            display_results_table(summary, "Empty Results")

            mock_console.print.assert_called()

    def test_display_results_table_many_items(self):
        """Test displaying results table with many items (truncation)."""
        summary = {
            "successful_builds": [f"model{i}" for i in range(10)],
            "failed_builds": [],
        }

        with patch("madengine.mad_cli.console") as mock_console:
            display_results_table(summary, "Many Results")

            mock_console.print.assert_called()


class TestBuildCommand:
    """Test the build command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("madengine.mad_cli.DistributedOrchestrator")
    @patch("madengine.mad_cli.validate_additional_context")
    def test_build_command_success(self, mock_validate, mock_orchestrator_class):
        """Test successful build command."""
        # Use auto-generated context for current machine
        context = generate_additional_context_for_machine()
        context_json = json.dumps(context)

        # Mock validation
        mock_validate.return_value = context

        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.build_phase.return_value = {
            "successful_builds": ["model1"],
            "failed_builds": [],
        }
        mock_orchestrator_class.return_value = mock_orchestrator

        result = self.runner.invoke(
            app, ["build", "--tags", "dummy", "--additional-context", context_json]
        )

        assert result.exit_code == ExitCode.SUCCESS
        mock_validate.assert_called_once()
        mock_orchestrator.build_phase.assert_called_once()

    @patch("madengine.mad_cli.DistributedOrchestrator")
    @patch("madengine.mad_cli.validate_additional_context")
    def test_build_command_failure(self, mock_validate, mock_orchestrator_class):
        """Test build command with failures."""
        # Use auto-generated context for current machine
        context = generate_additional_context_for_machine()
        context_json = json.dumps(context)

        # Mock validation
        mock_validate.return_value = context

        # Mock orchestrator with failures
        mock_orchestrator = MagicMock()
        mock_orchestrator.build_phase.return_value = {
            "successful_builds": [],
            "failed_builds": ["model1", "model2"],
        }
        mock_orchestrator_class.return_value = mock_orchestrator

        result = self.runner.invoke(
            app, ["build", "--tags", "dummy", "--additional-context", context_json]
        )

        assert result.exit_code == ExitCode.BUILD_FAILURE

    def test_build_command_invalid_context(self):
        """Test build command with invalid context."""
        result = self.runner.invoke(
            app, ["build", "--tags", "dummy", "--additional-context", "invalid json"]
        )

        assert result.exit_code == ExitCode.INVALID_ARGS

    def test_build_command_missing_context(self):
        """Test build command with missing context."""
        result = self.runner.invoke(app, ["build", "--tags", "dummy"])

        assert result.exit_code == ExitCode.INVALID_ARGS

    @patch("madengine.mad_cli.DistributedOrchestrator")
    @patch("madengine.mad_cli.validate_additional_context")
    def test_build_command_with_registry(self, mock_validate, mock_orchestrator_class):
        """Test build command with registry option."""
        # Use auto-generated context for current machine
        context = generate_additional_context_for_machine()
        context_json = json.dumps(context)

        # Mock validation
        mock_validate.return_value = context

        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.build_phase.return_value = {
            "successful_builds": ["model1"],
            "failed_builds": [],
        }
        mock_orchestrator_class.return_value = mock_orchestrator

        result = self.runner.invoke(
            app,
            [
                "build",
                "--tags",
                "dummy",
                "--registry",
                "localhost:5000",
                "--additional-context",
                context_json,
            ],
        )

        assert result.exit_code == ExitCode.SUCCESS
        # Verify registry was passed to build_phase
        mock_orchestrator.build_phase.assert_called_once()
        call_args = mock_orchestrator.build_phase.call_args
        assert call_args[1]["registry"] == "localhost:5000"

    @patch("madengine.mad_cli.DistributedOrchestrator")
    @patch("madengine.mad_cli.validate_additional_context")
    def test_build_command_exception_handling(
        self, mock_validate, mock_orchestrator_class
    ):
        """Test build command exception handling."""
        # Use auto-generated context for current machine
        context = generate_additional_context_for_machine()
        context_json = json.dumps(context)

        # Mock validation
        mock_validate.return_value = context

        # Mock orchestrator to raise exception
        mock_orchestrator_class.side_effect = Exception("Test error")

        result = self.runner.invoke(
            app, ["build", "--tags", "dummy", "--additional-context", context_json]
        )

        assert result.exit_code == ExitCode.FAILURE


class TestRunCommand:
    """Test the run command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("madengine.mad_cli.os.path.exists")
    @patch("madengine.mad_cli.DistributedOrchestrator")
    def test_run_command_execution_only(self, mock_orchestrator_class, mock_exists):
        """Test run command in execution-only mode (manifest exists)."""
        # Mock manifest file exists
        mock_exists.return_value = True

        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_phase.return_value = {
            "successful_runs": [{"model": "model1"}],
            "failed_runs": [],
        }
        mock_orchestrator_class.return_value = mock_orchestrator

        result = self.runner.invoke(
            app, ["run", "--manifest-file", "test_manifest.json"]
        )

        assert result.exit_code == ExitCode.SUCCESS
        mock_orchestrator.run_phase.assert_called_once()

    @patch("madengine.mad_cli.os.path.exists")
    @patch("madengine.mad_cli.DistributedOrchestrator")
    @patch("madengine.mad_cli.validate_additional_context")
    def test_run_command_full_workflow(
        self, mock_validate, mock_orchestrator_class, mock_exists
    ):
        """Test run command in full workflow mode (no manifest)."""
        # Mock manifest file doesn't exist
        mock_exists.return_value = False

        # Use auto-generated context for current machine
        context = generate_additional_context_for_machine()
        context_json = json.dumps(context)

        # Mock validation
        mock_validate.return_value = context

        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.build_phase.return_value = {
            "successful_builds": ["model1"],
            "failed_builds": [],
        }
        mock_orchestrator.run_phase.return_value = {
            "successful_runs": [{"model": "model1"}],
            "failed_runs": [],
        }
        mock_orchestrator_class.return_value = mock_orchestrator

        result = self.runner.invoke(
            app, ["run", "--tags", "dummy", "--additional-context", context_json]
        )

        assert result.exit_code == ExitCode.SUCCESS
        mock_orchestrator.build_phase.assert_called_once()
        mock_orchestrator.run_phase.assert_called_once()

    @patch("madengine.mad_cli.os.path.exists")
    @patch("madengine.mad_cli.DistributedOrchestrator")
    @patch("madengine.mad_cli.validate_additional_context")
    def test_run_command_build_failure(
        self, mock_validate, mock_orchestrator_class, mock_exists
    ):
        """Test run command with build failure in full workflow."""
        # Mock manifest file doesn't exist
        mock_exists.return_value = False

        # Use auto-generated context for current machine
        context = generate_additional_context_for_machine()
        context_json = json.dumps(context)

        # Mock validation
        mock_validate.return_value = context

        # Mock orchestrator with build failure
        mock_orchestrator = MagicMock()
        mock_orchestrator.build_phase.return_value = {
            "successful_builds": [],
            "failed_builds": ["model1"],
        }
        mock_orchestrator_class.return_value = mock_orchestrator

        result = self.runner.invoke(
            app, ["run", "--tags", "dummy", "--additional-context", context_json]
        )

        assert result.exit_code == ExitCode.BUILD_FAILURE
        mock_orchestrator.build_phase.assert_called_once()
        # run_phase should not be called if build fails
        mock_orchestrator.run_phase.assert_not_called()

    @requires_gpu("GPU execution tests require GPU hardware")
    @patch("madengine.mad_cli.os.path.exists")
    @patch("madengine.mad_cli.DistributedOrchestrator")
    def test_run_command_execution_failure(self, mock_orchestrator_class, mock_exists):
        """Test run command with execution failure."""
        # Mock manifest file exists
        mock_exists.return_value = True

        # Mock orchestrator with execution failure
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_phase.return_value = {
            "successful_runs": [],
            "failed_runs": [{"model": "model1"}],
        }
        mock_orchestrator_class.return_value = mock_orchestrator

        result = self.runner.invoke(
            app, ["run", "--manifest-file", "test_manifest.json"]
        )

        assert result.exit_code == ExitCode.RUN_FAILURE

    def test_run_command_invalid_timeout(self):
        """Test run command with invalid timeout."""
        result = self.runner.invoke(app, ["run", "--timeout", "-5"])

        assert result.exit_code == ExitCode.INVALID_ARGS

    @requires_gpu("GPU execution tests require GPU hardware")
    @patch("madengine.mad_cli.os.path.exists")
    @patch("madengine.mad_cli.DistributedOrchestrator")
    def test_run_command_with_options(self, mock_orchestrator_class, mock_exists):
        """Test run command with various options."""
        # Mock manifest file exists
        mock_exists.return_value = True

        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_phase.return_value = {
            "successful_runs": [{"model": "model1"}],
            "failed_runs": [],
        }
        mock_orchestrator_class.return_value = mock_orchestrator

        result = self.runner.invoke(
            app,
            [
                "run",
                "--manifest-file",
                "test_manifest.json",
                "--timeout",
                "300",
                "--keep-alive",
                "--keep-model-dir",
                "--verbose",
            ],
        )

        assert result.exit_code == ExitCode.SUCCESS
        # Verify options were passed
        call_args = mock_orchestrator.run_phase.call_args
        assert call_args[1]["timeout"] == 300
        assert call_args[1]["keep_alive"] is True


class TestGenerateAnsibleCommand:
    """Test the generate ansible command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("madengine.mad_cli.generate_ansible_setup")
    @patch("madengine.mad_cli.os.path.exists")
    def test_generate_ansible_success(self, mock_exists, mock_generate_ansible):
        """Test successful ansible generation."""
        # Mock manifest file exists
        mock_exists.return_value = True

        # Mock the return value of generate_ansible_setup
        mock_generate_ansible.return_value = {
            "playbook": "ansible-setup/madengine_playbook.yml"
        }

        result = self.runner.invoke(
            app,
            [
                "generate",
                "ansible",
                "--manifest-file",
                "test_manifest.json",
                "--output",
                "test_playbook.yml",
            ],
        )

        assert result.exit_code == ExitCode.SUCCESS
        mock_generate_ansible.assert_called_once_with(
            manifest_file="test_manifest.json", environment="default", output_dir="."
        )

    @patch("madengine.mad_cli.os.path.exists")
    def test_generate_ansible_manifest_not_found(self, mock_exists):
        """Test ansible generation with missing manifest."""
        # Mock manifest file doesn't exist
        mock_exists.return_value = False

        result = self.runner.invoke(
            app, ["generate", "ansible", "--manifest-file", "missing_manifest.json"]
        )

        assert result.exit_code == ExitCode.FAILURE

    @patch("madengine.mad_cli.generate_ansible_setup")
    @patch("madengine.mad_cli.os.path.exists")
    def test_generate_ansible_exception(self, mock_exists, mock_generate_ansible):
        """Test ansible generation with exception."""
        # Mock manifest file exists
        mock_exists.return_value = True

        # Mock exception in ansible generation
        mock_generate_ansible.side_effect = Exception("Test error")

        result = self.runner.invoke(
            app, ["generate", "ansible", "--manifest-file", "test_manifest.json"]
        )

        assert result.exit_code == ExitCode.FAILURE

    @patch("madengine.mad_cli.generate_ansible_setup")
    @patch("madengine.mad_cli.os.path.exists")
    def test_generate_ansible_default_values(self, mock_exists, mock_generate_ansible):
        """Test ansible generation with default values."""
        # Mock manifest file exists
        mock_exists.return_value = True

        # Mock the return value of generate_ansible_setup
        mock_generate_ansible.return_value = {
            "playbook": "ansible-setup/madengine_playbook.yml"
        }

        result = self.runner.invoke(app, ["generate", "ansible"])

        assert result.exit_code == ExitCode.SUCCESS
        mock_generate_ansible.assert_called_once_with(
            manifest_file=DEFAULT_MANIFEST_FILE, environment="default", output_dir="."
        )


class TestGenerateK8sCommand:
    """Test the generate k8s command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("madengine.mad_cli.generate_k8s_setup")
    @patch("madengine.mad_cli.os.path.exists")
    def test_generate_k8s_success(self, mock_exists, mock_generate_k8s):
        """Test successful k8s generation."""
        # Mock manifest file exists
        mock_exists.return_value = True

        # Mock the return value of generate_k8s_setup
        mock_generate_k8s.return_value = {
            "deployment": ["k8s-setup/deployment.yml"],
            "service": ["k8s-setup/service.yml"],
        }

        result = self.runner.invoke(
            app,
            [
                "generate",
                "k8s",
                "--manifest-file",
                "test_manifest.json",
                "--output-dir",
                "test-k8s",
            ],
        )

        assert result.exit_code == ExitCode.SUCCESS
        mock_generate_k8s.assert_called_once_with(
            manifest_file="test_manifest.json",
            environment="default",
            output_dir="test-k8s",
        )

    @patch("madengine.mad_cli.os.path.exists")
    def test_generate_k8s_manifest_not_found(self, mock_exists):
        """Test k8s generation with missing manifest."""
        # Mock manifest file doesn't exist
        mock_exists.return_value = False

        result = self.runner.invoke(
            app, ["generate", "k8s", "--manifest-file", "missing_manifest.json"]
        )

        assert result.exit_code == ExitCode.FAILURE

    @patch("madengine.mad_cli.generate_k8s_setup")
    @patch("madengine.mad_cli.os.path.exists")
    def test_generate_k8s_exception(self, mock_exists, mock_generate_k8s):
        """Test k8s generation with exception."""
        # Mock manifest file exists
        mock_exists.return_value = True

        # Mock exception in k8s generation
        mock_generate_k8s.side_effect = Exception("Test error")

        result = self.runner.invoke(
            app, ["generate", "k8s", "--manifest-file", "test_manifest.json"]
        )

        assert result.exit_code == ExitCode.FAILURE

    @patch("madengine.mad_cli.generate_k8s_setup")
    @patch("madengine.mad_cli.os.path.exists")
    def test_generate_k8s_default_values(self, mock_exists, mock_generate_k8s):
        """Test k8s generation with default values."""
        # Mock manifest file exists
        mock_exists.return_value = True

        # Mock the return value of generate_k8s_setup
        mock_generate_k8s.return_value = {
            "deployment": ["k8s-setup/deployment.yml"],
            "service": ["k8s-setup/service.yml"],
        }

        result = self.runner.invoke(app, ["generate", "k8s"])

        assert result.exit_code == ExitCode.SUCCESS
        mock_generate_k8s.assert_called_once_with(
            manifest_file=DEFAULT_MANIFEST_FILE,
            environment="default",
            output_dir="k8s-setup",
        )


class TestMainCallback:
    """Test the main callback function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_main_version_flag(self):
        """Test main callback with version flag."""
        result = self.runner.invoke(app, ["--version"])

        assert result.exit_code == ExitCode.SUCCESS
        assert "madengine-cli" in result.stdout
        assert "version" in result.stdout

    def test_main_help(self):
        """Test main callback shows help when no command."""
        result = self.runner.invoke(app, [])

        # Should show help and exit
        assert "madengine Distributed Orchestrator" in result.stdout


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
        assert DEFAULT_ANSIBLE_OUTPUT == "madengine_distributed.yml"
        assert DEFAULT_TIMEOUT == -1


class TestCliMain:
    """Test the cli_main function."""

    @patch("madengine.mad_cli.app")
    def test_cli_main_success(self, mock_app):
        """Test successful cli_main execution."""
        mock_app.return_value = None

        # Should not raise any exception
        mad_cli.cli_main()

        mock_app.assert_called_once()

    @patch("madengine.mad_cli.app")
    @patch("madengine.mad_cli.sys.exit")
    def test_cli_main_keyboard_interrupt(self, mock_exit, mock_app):
        """Test cli_main with keyboard interrupt."""
        mock_app.side_effect = KeyboardInterrupt()

        mad_cli.cli_main()

        mock_exit.assert_called_once_with(ExitCode.FAILURE)

    @patch("madengine.mad_cli.app")
    @patch("madengine.mad_cli.sys.exit")
    @patch("madengine.mad_cli.console")
    def test_cli_main_unexpected_exception(self, mock_console, mock_exit, mock_app):
        """Test cli_main with unexpected exception."""
        mock_app.side_effect = Exception("Test error")

        mad_cli.cli_main()

        mock_exit.assert_called_once_with(ExitCode.FAILURE)
        mock_console.print.assert_called()
        mock_console.print_exception.assert_called_once()


class TestIntegration:
    """Integration tests for the CLI."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_help_command(self):
        """Test help command works."""
        result = self.runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "madengine Distributed Orchestrator" in result.stdout

    def test_build_help(self):
        """Test build command help."""
        result = self.runner.invoke(app, ["build", "--help"])

        assert result.exit_code == 0
        assert "Build Docker images" in result.stdout

    def test_run_help(self):
        """Test run command help."""
        result = self.runner.invoke(app, ["run", "--help"])

        assert result.exit_code == 0
        assert "Run model containers" in result.stdout

    def test_generate_help(self):
        """Test generate command help."""
        result = self.runner.invoke(app, ["generate", "--help"])

        assert result.exit_code == 0
        assert "Generate orchestration files" in result.stdout

    def test_generate_ansible_help(self):
        """Test generate ansible command help."""
        result = self.runner.invoke(app, ["generate", "ansible", "--help"])

        assert result.exit_code == 0
        assert "Generate Ansible playbook" in result.stdout

    def test_generate_k8s_help(self):
        """Test generate k8s command help."""
        result = self.runner.invoke(app, ["generate", "k8s", "--help"])

        assert result.exit_code == 0
        assert "Generate Kubernetes manifests" in result.stdout


class TestCpuOnlyMachine:
    """Tests specifically for CPU-only machines."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_cpu_only_machine_detection(self):
        """Test that GPU detection works."""
        # This test should always pass, regardless of hardware
        has_gpu_available = has_gpu()
        assert isinstance(has_gpu_available, bool)

    def test_auto_context_generation_cpu_only(self):
        """Test that auto-generated context is appropriate for CPU-only machines."""
        context = generate_additional_context_for_machine()

        # Should always have required fields
        assert "gpu_vendor" in context
        assert "guest_os" in context

        # On CPU-only machines, should use default AMD for build compatibility
        if not has_gpu():
            assert context["gpu_vendor"] == "AMD"
            assert context["guest_os"] == "UBUNTU"

    @patch("madengine.mad_cli.DistributedOrchestrator")
    @patch("madengine.mad_cli.validate_additional_context")
    def test_build_on_cpu_only_machine(self, mock_validate, mock_orchestrator_class):
        """Test build command works on CPU-only machines."""
        # Use auto-generated context for current machine
        context = generate_additional_context_for_machine()
        context_json = json.dumps(context)

        # Mock validation
        mock_validate.return_value = context

        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.build_phase.return_value = {
            "successful_builds": ["model1"],
            "failed_builds": [],
        }
        mock_orchestrator_class.return_value = mock_orchestrator

        result = self.runner.invoke(
            app, ["build", "--tags", "dummy", "--additional-context", context_json]
        )

        # Should work on CPU-only machines for build phase
        assert result.exit_code == ExitCode.SUCCESS
        mock_validate.assert_called_once()
        mock_orchestrator.build_phase.assert_called_once()


class TestGpuRequiredTests:
    """Tests that require GPU hardware."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @requires_gpu("Test requires GPU hardware")
    @patch("madengine.mad_cli.os.path.exists")
    @patch("madengine.mad_cli.DistributedOrchestrator")
    def test_run_with_gpu_required(self, mock_orchestrator_class, mock_exists):
        """Test run command that requires GPU hardware."""
        # Mock manifest file exists
        mock_exists.return_value = True

        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_phase.return_value = {
            "successful_runs": [{"model": "model1"}],
            "failed_runs": [],
        }
        mock_orchestrator_class.return_value = mock_orchestrator

        result = self.runner.invoke(
            app, ["run", "--manifest-file", "test_manifest.json"]
        )

        assert result.exit_code == ExitCode.SUCCESS
        mock_orchestrator.run_phase.assert_called_once()

    @requires_gpu("Test requires AMD GPU hardware")
    @patch("madengine.mad_cli.os.path.exists")
    @patch("madengine.mad_cli.DistributedOrchestrator")
    def test_run_with_amd_gpu_required(self, mock_orchestrator_class, mock_exists):
        """Test run command that requires AMD GPU hardware."""
        # Mock manifest file exists
        mock_exists.return_value = True

        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_phase.return_value = {
            "successful_runs": [{"model": "model1"}],
            "failed_runs": [],
        }
        mock_orchestrator_class.return_value = mock_orchestrator

        result = self.runner.invoke(
            app, ["run", "--manifest-file", "test_manifest.json"]
        )

        assert result.exit_code == ExitCode.SUCCESS
        mock_orchestrator.run_phase.assert_called_once()

    @requires_gpu("Test requires NVIDIA GPU hardware")
    @patch("madengine.mad_cli.os.path.exists")
    @patch("madengine.mad_cli.DistributedOrchestrator")
    def test_run_with_nvidia_gpu_required(self, mock_orchestrator_class, mock_exists):
        """Test run command that requires NVIDIA GPU hardware."""
        # Mock manifest file exists
        mock_exists.return_value = True

        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_phase.return_value = {
            "successful_runs": [{"model": "model1"}],
            "failed_runs": [],
        }
        mock_orchestrator_class.return_value = mock_orchestrator

        result = self.runner.invoke(
            app, ["run", "--manifest-file", "test_manifest.json"]
        )

        assert result.exit_code == ExitCode.SUCCESS
        mock_orchestrator.run_phase.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_build_empty_tags(self):
        """Test build command with empty tags list."""
        # Use auto-generated context for current machine
        context = generate_additional_context_for_machine()
        context_json = json.dumps(context)

        result = self.runner.invoke(
            app, ["build", "--additional-context", context_json]
        )

        # Should handle empty tags gracefully
        assert result.exit_code in [
            ExitCode.SUCCESS,
            ExitCode.BUILD_FAILURE,
            ExitCode.INVALID_ARGS,
        ]

    def test_run_zero_timeout(self):
        """Test run command with zero timeout."""
        result = self.runner.invoke(app, ["run", "--timeout", "0"])

        # Zero timeout should be valid (no timeout)
        # Exit code depends on other factors but shouldn't be INVALID_ARGS for timeout
        assert (
            result.exit_code != ExitCode.INVALID_ARGS or "Timeout" not in result.stdout
        )

    @patch("madengine.mad_cli.validate_additional_context")
    def test_context_file_and_string_both_provided(self, mock_validate):
        """Test providing both context file and string."""
        # Use auto-generated context for current machine
        context = generate_additional_context_for_machine()
        context_json = json.dumps(context)

        mock_validate.return_value = context

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"gpu_vendor": "NVIDIA", "guest_os": "CENTOS"}, f)
            temp_file = f.name

        try:
            result = self.runner.invoke(
                app,
                [
                    "build",
                    "--additional-context",
                    context_json,
                    "--additional-context-file",
                    temp_file,
                ],
            )

            # Should call validate with both parameters
            mock_validate.assert_called_once()
        finally:
            os.unlink(temp_file)
