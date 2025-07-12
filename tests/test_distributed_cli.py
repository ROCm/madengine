"""Test the distributed CLI module.

This module tests the distributed command-line interface functionality.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import os
import sys
import json
import logging
import tempfile
import subprocess
import unittest.mock
from unittest.mock import patch, MagicMock
# third-party modules
import pytest
# project modules
from madengine import distributed_cli
from madengine.tools.distributed_orchestrator import DistributedOrchestrator
from .fixtures.utils import (
    BASE_DIR, MODEL_DIR, has_gpu,
    requires_gpu, generate_additional_context_for_machine, create_mock_args_with_auto_context
)


class TestValidateAdditionalContext:
    """Test the validate_additional_context function."""

    def test_validate_additional_context_valid_string(self):
        """Test validation with valid additional context from string."""
        mock_args = MagicMock()
        mock_args.additional_context = '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
        mock_args.additional_context_file = None
        
        result = distributed_cli.validate_additional_context(mock_args)
        assert result is True

    def test_validate_additional_context_valid_case_insensitive(self):
        """Test validation with valid additional context (case insensitive)."""
        mock_args = MagicMock()
        mock_args.additional_context = '{"gpu_vendor": "amd", "guest_os": "ubuntu"}'
        mock_args.additional_context_file = None
        
        result = distributed_cli.validate_additional_context(mock_args)
        assert result is True

    def test_validate_additional_context_valid_all_vendors(self):
        """Test validation with all valid GPU vendors."""
        vendors = ["AMD", "NVIDIA", "INTEL"]
        for vendor in vendors:
            mock_args = MagicMock()
            mock_args.additional_context = f'{{"gpu_vendor": "{vendor}", "guest_os": "UBUNTU"}}'
            mock_args.additional_context_file = None
            
            result = distributed_cli.validate_additional_context(mock_args)
            assert result is True

    def test_validate_additional_context_valid_all_os(self):
        """Test validation with all valid operating systems."""
        operating_systems = ["UBUNTU", "CENTOS", "ROCKY"]
        for os_name in operating_systems:
            mock_args = MagicMock()
            mock_args.additional_context = f'{{"gpu_vendor": "AMD", "guest_os": "{os_name}"}}'
            mock_args.additional_context_file = None
            
            result = distributed_cli.validate_additional_context(mock_args)
            assert result is True

    def test_validate_additional_context_valid_from_file(self):
        """Test validation with valid additional context from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump({"gpu_vendor": "NVIDIA", "guest_os": "CENTOS"}, tmp_file)
            tmp_file_path = tmp_file.name

        try:
            mock_args = MagicMock()
            mock_args.additional_context = '{}'
            mock_args.additional_context_file = tmp_file_path
            
            result = distributed_cli.validate_additional_context(mock_args)
            assert result is True
        finally:
            os.unlink(tmp_file_path)

    def test_validate_additional_context_string_overrides_file(self):
        """Test that string parameter overrides file parameter."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump({"gpu_vendor": "NVIDIA", "guest_os": "CENTOS"}, tmp_file)
            tmp_file_path = tmp_file.name

        try:
            mock_args = MagicMock()
            mock_args.additional_context = '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
            mock_args.additional_context_file = tmp_file_path
            
            result = distributed_cli.validate_additional_context(mock_args)
            assert result is True
        finally:
            os.unlink(tmp_file_path)

    def test_validate_additional_context_missing_context(self):
        """Test validation with no additional context provided."""
        mock_args = MagicMock()
        mock_args.additional_context = '{}'
        mock_args.additional_context_file = None
        
        result = distributed_cli.validate_additional_context(mock_args)
        assert result is False

    def test_validate_additional_context_missing_gpu_vendor(self):
        """Test validation with missing gpu_vendor field."""
        mock_args = MagicMock()
        mock_args.additional_context = '{"guest_os": "UBUNTU"}'
        mock_args.additional_context_file = None
        
        result = distributed_cli.validate_additional_context(mock_args)
        assert result is False

    def test_validate_additional_context_missing_guest_os(self):
        """Test validation with missing guest_os field."""
        mock_args = MagicMock()
        mock_args.additional_context = '{"gpu_vendor": "AMD"}'
        mock_args.additional_context_file = None
        
        result = distributed_cli.validate_additional_context(mock_args)
        assert result is False

    def test_validate_additional_context_invalid_gpu_vendor(self):
        """Test validation with invalid gpu_vendor value."""
        mock_args = MagicMock()
        mock_args.additional_context = '{"gpu_vendor": "INVALID", "guest_os": "UBUNTU"}'
        mock_args.additional_context_file = None
        
        result = distributed_cli.validate_additional_context(mock_args)
        assert result is False

    def test_validate_additional_context_invalid_guest_os(self):
        """Test validation with invalid guest_os value."""
        mock_args = MagicMock()
        mock_args.additional_context = '{"gpu_vendor": "AMD", "guest_os": "INVALID"}'
        mock_args.additional_context_file = None
        
        result = distributed_cli.validate_additional_context(mock_args)
        assert result is False

    def test_validate_additional_context_invalid_json_string(self):
        """Test validation with invalid JSON in string parameter."""
        mock_args = MagicMock()
        mock_args.additional_context = '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"'  # Missing closing brace
        mock_args.additional_context_file = None
        
        result = distributed_cli.validate_additional_context(mock_args)
        assert result is False

    def test_validate_additional_context_file_not_found(self):
        """Test validation with non-existent context file."""
        mock_args = MagicMock()
        mock_args.additional_context = '{}'
        mock_args.additional_context_file = '/nonexistent/file.json'
        
        result = distributed_cli.validate_additional_context(mock_args)
        assert result is False

    def test_validate_additional_context_invalid_json_file(self):
        """Test validation with invalid JSON in file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_file.write('{"gpu_vendor": "AMD", "guest_os": "UBUNTU"')  # Invalid JSON
            tmp_file_path = tmp_file.name

        try:
            mock_args = MagicMock()
            mock_args.additional_context = '{}'
            mock_args.additional_context_file = tmp_file_path
            
            result = distributed_cli.validate_additional_context(mock_args)
            assert result is False
        finally:
            os.unlink(tmp_file_path)

    def test_validate_additional_context_exception_handling(self):
        """Test that exceptions are properly handled."""
        mock_args = MagicMock()
        # Remove the attributes to cause an AttributeError
        del mock_args.additional_context
        del mock_args.additional_context_file
        
        result = distributed_cli.validate_additional_context(mock_args)
        assert result is False


class TestValidateCommonArgs:
    """Test the validate_common_args function."""

    def test_validate_common_args_valid_timeout(self):
        """Test validation with valid timeout values."""
        mock_args = MagicMock()
        mock_args.timeout = 3600
        mock_args.output = "test_output.json"
        
        # Mock the output directory exists
        with patch('os.path.exists', return_value=True), patch('os.path.dirname', return_value='/tmp'):
            result = distributed_cli.validate_common_args(mock_args)
            assert result is True

    def test_validate_common_args_valid_default_timeout(self):
        """Test validation with default timeout (-1)."""
        mock_args = MagicMock()
        mock_args.timeout = -1
        mock_args.output = None
        
        result = distributed_cli.validate_common_args(mock_args)
        assert result is True

    def test_validate_common_args_invalid_timeout(self):
        """Test validation with invalid timeout."""
        mock_args = MagicMock()
        mock_args.timeout = -5  # Invalid timeout
        mock_args.output = None
        
        result = distributed_cli.validate_common_args(mock_args)
        assert result is False

    def test_validate_common_args_missing_timeout_attribute(self):
        """Test validation when timeout attribute is missing."""
        mock_args = MagicMock()
        del mock_args.timeout  # Remove timeout attribute
        mock_args.output = None
        
        result = distributed_cli.validate_common_args(mock_args)
        assert result is True  # Should pass when timeout is not present

    @patch('os.path.exists')
    @patch('os.path.dirname')
    def test_validate_common_args_output_directory_missing(self, mock_dirname, mock_exists):
        """Test that validation fails when output directory doesn't exist."""
        mock_args = MagicMock()
        mock_args.timeout = 1800
        mock_args.output = "/tmp/new_dir/output.json"
        
        mock_dirname.return_value = "/tmp/new_dir"
        mock_exists.return_value = False
        
        result = distributed_cli.validate_common_args(mock_args)
        
        assert result is False

    @patch('os.path.exists')
    @patch('os.path.dirname')
    def test_validate_common_args_output_directory_exists(self, mock_dirname, mock_exists):
        """Test that validation passes when output directory exists."""
        mock_args = MagicMock()
        mock_args.timeout = 1800
        mock_args.output = "/tmp/existing_dir/output.json"
        
        mock_dirname.return_value = "/tmp/existing_dir"
        mock_exists.return_value = True
        
        result = distributed_cli.validate_common_args(mock_args)
        
        assert result is True

    def test_validate_common_args_no_output_file(self):
        """Test validation when no output file is specified."""
        mock_args = MagicMock()
        mock_args.timeout = 600
        mock_args.output = None
        
        result = distributed_cli.validate_common_args(mock_args)
        assert result is True

    def test_validate_common_args_empty_output_file(self):
        """Test validation when output file is empty string."""
        mock_args = MagicMock()
        mock_args.timeout = 600
        mock_args.output = ""
        
        result = distributed_cli.validate_common_args(mock_args)
        assert result is True


class TestSetupLogging:
    """Test the setup_logging function."""
    
    @patch('logging.basicConfig')
    def test_setup_logging_default(self, mock_basic_config):
        """Test setup_logging with default verbosity."""
        distributed_cli.setup_logging()
        
        mock_basic_config.assert_called_once_with(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    @patch('logging.basicConfig')
    def test_setup_logging_verbose(self, mock_basic_config):
        """Test setup_logging with verbose enabled."""
        distributed_cli.setup_logging(verbose=True)
        
        mock_basic_config.assert_called_once_with(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    @patch('logging.basicConfig')
    def test_setup_logging_not_verbose(self, mock_basic_config):
        """Test setup_logging with verbose explicitly disabled."""
        distributed_cli.setup_logging(verbose=False)
        
        mock_basic_config.assert_called_once_with(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


class TestExitCodes:
    """Test that the correct exit codes are defined."""
    
    def test_exit_codes_defined(self):
        """Test that all required exit codes are defined."""
        assert distributed_cli.EXIT_SUCCESS == 0
        assert distributed_cli.EXIT_FAILURE == 1
        assert distributed_cli.EXIT_BUILD_FAILURE == 2
        assert distributed_cli.EXIT_RUN_FAILURE == 3
        assert distributed_cli.EXIT_INVALID_ARGS == 4

    def test_exit_codes_unique(self):
        """Test that all exit codes are unique."""
        exit_codes = [
            distributed_cli.EXIT_SUCCESS,
            distributed_cli.EXIT_FAILURE,
            distributed_cli.EXIT_BUILD_FAILURE,
            distributed_cli.EXIT_RUN_FAILURE,
            distributed_cli.EXIT_INVALID_ARGS
        ]
        assert len(set(exit_codes)) == len(exit_codes)


class TestDefaultConstants:
    """Test that default constants are properly defined."""
    
    def test_default_constants_defined(self):
        """Test that all default constants are defined."""
        assert distributed_cli.DEFAULT_MANIFEST_FILE == 'build_manifest.json'
        assert distributed_cli.DEFAULT_PERF_OUTPUT == 'perf.csv'
        assert distributed_cli.DEFAULT_DATA_CONFIG == 'data.json'
        assert distributed_cli.DEFAULT_TOOLS_CONFIG == './scripts/common/tools.json'
        assert distributed_cli.DEFAULT_ANSIBLE_OUTPUT == 'madengine_distributed.yml'
        assert distributed_cli.DEFAULT_K8S_NAMESPACE == 'madengine'
        assert distributed_cli.DEFAULT_TIMEOUT == -1


class TestDistributedCLI:
    """Test the distributed CLI module."""

    def test_distributed_cli_help(self):
        """Test the distributed CLI --help command."""
        script_path = os.path.join(BASE_DIR, "src/madengine", "distributed_cli.py")
        result = subprocess.run([sys.executable, script_path, "--help"], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert result.returncode == 0
        assert b"madengine Distributed Orchestrator" in result.stdout

    def test_build_command_help(self):
        """Test the build command --help."""
        script_path = os.path.join(BASE_DIR, "src/madengine", "distributed_cli.py")
        result = subprocess.run([sys.executable, script_path, "build", "--help"], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert result.returncode == 0
        assert b"build" in result.stdout

    def test_run_command_help(self):
        """Test the run command --help."""
        script_path = os.path.join(BASE_DIR, "src/madengine", "distributed_cli.py")
        result = subprocess.run([sys.executable, script_path, "run", "--help"], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert result.returncode == 0
        assert b"run" in result.stdout

    def test_generate_command_help(self):
        """Test the generate command --help."""
        script_path = os.path.join(BASE_DIR, "src/madengine", "distributed_cli.py")
        result = subprocess.run([sys.executable, script_path, "generate", "--help"], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert result.returncode == 0
        assert b"generate" in result.stdout

    @patch('madengine.distributed_cli.DistributedOrchestrator')
    def test_build_models_function(self, mock_orchestrator):
        """Test the build_models function."""
        # Mock args with valid additional context
        mock_args = MagicMock()
        mock_args.registry = "localhost:5000"
        mock_args.clean_docker_cache = True
        mock_args.manifest_output = "test_manifest.json"
        mock_args.summary_output = "test_summary.json"
        mock_args.additional_context = '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
        mock_args.additional_context_file = None

        # Mock orchestrator instance and build phase
        mock_instance = MagicMock()
        mock_orchestrator.return_value = mock_instance
        mock_instance.build_phase.return_value = {
            "successful_builds": ["model1", "model2"],
            "failed_builds": []
        }

        # Test build command
        result = distributed_cli.build_models(mock_args)
        
        # Verify orchestrator was called correctly with build_only_mode=True
        mock_orchestrator.assert_called_once_with(mock_args, build_only_mode=True)
        mock_instance.build_phase.assert_called_once_with(
            registry="localhost:5000",
            clean_cache=True,
            manifest_output="test_manifest.json"
        )
        
        # Should return EXIT_SUCCESS for successful builds
        assert result == distributed_cli.EXIT_SUCCESS

    @patch('madengine.distributed_cli.DistributedOrchestrator')
    def test_build_models_with_failures(self, mock_orchestrator):
        """Test the build_models function with build failures."""
        mock_args = MagicMock()
        mock_args.registry = None
        mock_args.clean_docker_cache = False
        mock_args.manifest_output = "manifest.json"
        mock_args.summary_output = None
        mock_args.additional_context = '{"gpu_vendor": "NVIDIA", "guest_os": "CENTOS"}'
        mock_args.additional_context_file = None

        mock_instance = MagicMock()
        mock_orchestrator.return_value = mock_instance
        mock_instance.build_phase.return_value = {
            "successful_builds": ["model1"],
            "failed_builds": ["model2"]
        }

        result = distributed_cli.build_models(mock_args)
        
        # Should return EXIT_BUILD_FAILURE due to failures
        assert result == distributed_cli.EXIT_BUILD_FAILURE

    def test_build_models_invalid_additional_context(self):
        """Test the build_models function with invalid additional context."""
        mock_args = MagicMock()
        mock_args.registry = "localhost:5000"
        mock_args.clean_docker_cache = True
        mock_args.manifest_output = "test_manifest.json"
        mock_args.summary_output = None
        mock_args.additional_context = '{"gpu_vendor": "INVALID"}'  # Missing guest_os and invalid vendor
        mock_args.additional_context_file = None

        result = distributed_cli.build_models(mock_args)
        
        # Should return EXIT_INVALID_ARGS due to invalid context
        assert result == distributed_cli.EXIT_INVALID_ARGS

    def test_build_models_function_auto_context(self):
        """Test the build_models function with automatically detected context."""
        # Use utility function to create mock args with auto-generated context
        mock_args = create_mock_args_with_auto_context(
            registry="localhost:5000",
            clean_docker_cache=True,
            manifest_output="test_manifest.json",
            summary_output="test_summary.json"
        )

        # Mock orchestrator instance and build phase
        mock_instance = MagicMock()
        with patch('madengine.distributed_cli.DistributedOrchestrator', return_value=mock_instance):
            mock_instance.build_phase.return_value = {
                "successful_builds": ["model1", "model2"],
                "failed_builds": []
            }

            # Test build command
            result = distributed_cli.build_models(mock_args)
            
            # Should return EXIT_SUCCESS for successful builds
            assert result == distributed_cli.EXIT_SUCCESS

    @patch('madengine.distributed_cli.DistributedOrchestrator')
    @patch('os.path.exists')
    def test_run_models_execution_only(self, mock_exists, mock_orchestrator):
        """Test the run_models function in execution-only mode."""
        mock_args = MagicMock()
        mock_args.manifest_file = "manifest.json"
        mock_args.registry = "localhost:5000"
        mock_args.timeout = 3600
        mock_args.keep_alive = False
        mock_args.summary_output = None

        # Mock that manifest file exists (execution-only mode)
        mock_exists.return_value = True

        mock_instance = MagicMock()
        mock_orchestrator.return_value = mock_instance
        mock_instance.run_phase.return_value = {
            "successful_runs": ["model1", "model2"],
            "failed_runs": []
        }

        result = distributed_cli.run_models(mock_args)
        
        mock_orchestrator.assert_called_once_with(mock_args)
        mock_instance.run_phase.assert_called_once_with(
            manifest_file="manifest.json",
            registry="localhost:5000",
            timeout=3600,
            keep_alive=False
        )
        
        assert result == distributed_cli.EXIT_SUCCESS

    @patch('madengine.distributed_cli.DistributedOrchestrator')
    @patch('os.path.exists')
    def test_run_models_complete_workflow(self, mock_exists, mock_orchestrator):
        """Test the run_models function in complete workflow mode (build + run)."""
        mock_args = MagicMock()
        mock_args.manifest_file = None
        mock_args.registry = "localhost:5000"
        mock_args.timeout = 1800
        mock_args.keep_alive = True
        mock_args.summary_output = None
        mock_args.manifest_output = "build_manifest.json"
        mock_args.clean_docker_cache = False

        # Mock that manifest file doesn't exist (complete workflow mode)
        mock_exists.return_value = False

        mock_instance = MagicMock()
        mock_orchestrator.return_value = mock_instance
        
        # Mock successful build phase
        mock_instance.build_phase.return_value = {
            "successful_builds": ["model1"], 
            "failed_builds": []
        }
        
        # Mock successful run phase
        mock_instance.run_phase.return_value = {
            "successful_runs": ["model1"], 
            "failed_runs": []
        }

        result = distributed_cli.run_models(mock_args)
        
        mock_orchestrator.assert_called_once_with(mock_args)
        
        # Verify build phase was called
        mock_instance.build_phase.assert_called_once_with(
            registry="localhost:5000",
            clean_cache=False,
            manifest_output="build_manifest.json"
        )
        
        # Verify run phase was called
        mock_instance.run_phase.assert_called_once_with(
            manifest_file="build_manifest.json",
            registry="localhost:5000",
            timeout=1800,
            keep_alive=True
        )
        
        assert result == distributed_cli.EXIT_SUCCESS

    @requires_gpu("Test run models that requires GPU")
    def test_run_models_with_gpu_requirement(self):
        """Test run models that requires GPU (should be skipped on CPU-only)."""
        mock_args = MagicMock()
        mock_args.manifest_file = "manifest.json"
        mock_args.registry = "localhost:5000"
        mock_args.timeout = 3600
        mock_args.keep_alive = False
        mock_args.summary_output = None

        # Mock that manifest file exists (execution-only mode)
        mock_instance = MagicMock()
        with patch('madengine.distributed_cli.DistributedOrchestrator', return_value=mock_instance), \
             patch('os.path.exists', return_value=True):
            
            mock_instance.run_phase.return_value = {
                "successful_runs": ["model1", "model2"],
                "failed_runs": []
            }

            result = distributed_cli.run_models(mock_args)
            assert result == distributed_cli.EXIT_SUCCESS

    @patch('madengine.distributed_cli.create_ansible_playbook')
    @patch('os.path.exists')
    def test_generate_ansible_function(self, mock_exists, mock_create_ansible):
        """Test the generate_ansible function."""
        mock_args = MagicMock()
        mock_args.manifest_file = "manifest.json"
        mock_args.output = "playbook.yml"

        # Mock that the manifest file exists
        mock_exists.return_value = True

        result = distributed_cli.generate_ansible(mock_args)
        
        mock_exists.assert_called_once_with("manifest.json")
        mock_create_ansible.assert_called_once_with(
            manifest_file="manifest.json",
            playbook_file="playbook.yml"
        )
        
        assert result == distributed_cli.EXIT_SUCCESS

    @patch('madengine.distributed_cli.create_ansible_playbook')
    @patch('os.path.exists')
    def test_generate_ansible_function_missing_manifest(self, mock_exists, mock_create_ansible):
        """Test the generate_ansible function when manifest file doesn't exist."""
        mock_args = MagicMock()
        mock_args.manifest_file = "nonexistent.json"
        mock_args.output = "playbook.yml"

        # Mock that the manifest file doesn't exist
        mock_exists.return_value = False

        result = distributed_cli.generate_ansible(mock_args)
        
        mock_exists.assert_called_once_with("nonexistent.json")
        mock_create_ansible.assert_not_called()
        
        assert result == distributed_cli.EXIT_FAILURE

    @patch('madengine.distributed_cli.create_kubernetes_manifests')
    @patch('os.path.exists')
    def test_generate_k8s_function(self, mock_exists, mock_create_k8s):
        """Test the generate_k8s function."""
        mock_args = MagicMock()
        mock_args.manifest_file = "manifest.json"
        mock_args.namespace = "madengine-test"

        # Mock that the manifest file exists
        mock_exists.return_value = True

        result = distributed_cli.generate_k8s(mock_args)
        
        mock_exists.assert_called_once_with("manifest.json")
        mock_create_k8s.assert_called_once_with(
            manifest_file="manifest.json",
            namespace="madengine-test"
        )
        
        assert result == distributed_cli.EXIT_SUCCESS

    @patch('madengine.distributed_cli.create_kubernetes_manifests')
    @patch('os.path.exists')
    def test_generate_k8s_function_missing_manifest(self, mock_exists, mock_create_k8s):
        """Test the generate_k8s function when manifest file doesn't exist."""
        mock_args = MagicMock()
        mock_args.manifest_file = "nonexistent.json"
        mock_args.namespace = "madengine-test"

        # Mock that the manifest file doesn't exist
        mock_exists.return_value = False

        result = distributed_cli.generate_k8s(mock_args)
        
        mock_exists.assert_called_once_with("nonexistent.json")
        mock_create_k8s.assert_not_called()
        
        assert result == distributed_cli.EXIT_FAILURE

    @patch('madengine.distributed_cli.DistributedOrchestrator')
    @patch('os.path.exists')
    def test_run_models_with_build_failure(self, mock_exists, mock_orchestrator):
        """Test the run_models function when build phase fails in complete workflow."""
        mock_args = MagicMock()
        mock_args.manifest_file = None
        mock_args.registry = "localhost:5000"
        mock_args.timeout = 1800
        mock_args.keep_alive = False
        mock_args.summary_output = None
        mock_args.manifest_output = "build_manifest.json"
        mock_args.clean_docker_cache = False

        # Mock that manifest file doesn't exist (complete workflow mode)
        mock_exists.return_value = False

        mock_instance = MagicMock()
        mock_orchestrator.return_value = mock_instance
        
        # Mock failed build phase
        mock_instance.build_phase.return_value = {
            "successful_builds": [], 
            "failed_builds": ["model1"]
        }

        result = distributed_cli.run_models(mock_args)
        
        # Should return EXIT_BUILD_FAILURE and not call run phase
        assert result == distributed_cli.EXIT_BUILD_FAILURE
        mock_instance.build_phase.assert_called_once()
        mock_instance.run_phase.assert_not_called()

    @patch('madengine.distributed_cli.DistributedOrchestrator')
    @patch('os.path.exists')
    def test_run_models_with_run_failure(self, mock_exists, mock_orchestrator):
        """Test the run_models function when run phase fails in execution-only mode."""
        mock_args = MagicMock()
        mock_args.manifest_file = "manifest.json"
        mock_args.registry = "localhost:5000"
        mock_args.timeout = 3600
        mock_args.keep_alive = False
        mock_args.summary_output = None

        # Mock that manifest file exists (execution-only mode)
        mock_exists.return_value = True

        mock_instance = MagicMock()
        mock_orchestrator.return_value = mock_instance
        mock_instance.run_phase.return_value = {
            "successful_runs": [],
            "failed_runs": ["model1"]
        }

        result = distributed_cli.run_models(mock_args)
        
        # Should return EXIT_RUN_FAILURE
        assert result == distributed_cli.EXIT_RUN_FAILURE

    @patch('madengine.distributed_cli.DistributedOrchestrator')
    def test_run_models_invalid_timeout(self, mock_orchestrator):
        """Test the run_models function with invalid timeout."""
        mock_args = MagicMock()
        mock_args.timeout = -5  # Invalid timeout
        mock_args.manifest_file = None

        result = distributed_cli.run_models(mock_args)
        
        # Should return EXIT_INVALID_ARGS without calling orchestrator
        assert result == distributed_cli.EXIT_INVALID_ARGS
        mock_orchestrator.assert_not_called()

    def test_automatic_context_generation(self):
        """Test automatic generation of additional context for build-only operations."""
        # Test that validation works with mock context for any machine
        mock_context = {
            "gpu_vendor": "AMD",  # Default for build-only
            "guest_os": "UBUNTU"  # Default OS
        }
        
        # Test that validation works with mock context
        mock_args = MagicMock()
        mock_args.additional_context = json.dumps(mock_context)
        mock_args.additional_context_file = None
        
        result = distributed_cli.validate_additional_context(mock_args)
        assert result is True
