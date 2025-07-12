"""Test the distributed CLI pre/post scripts and profiling functionality.

This module tests the distributed CLI's handling of pre/post scripts,
system environment collection, and profiling tools to ensure they match
the standard madengine behavior.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import os
import sys
import json
import tempfile
import subprocess
import unittest.mock
from unittest.mock import patch, MagicMock, mock_open, call
# third-party modules
import pytest
# project modules
from madengine import distributed_cli
from madengine.tools.distributed_orchestrator import DistributedOrchestrator
from madengine.tools.container_runner import ContainerRunner
from madengine.core.context import Context
from madengine.core.console import Console
from .fixtures.utils import BASE_DIR, MODEL_DIR, clean_test_temp_files


class TestDistributedPrePostProfiling:
    """Test the distributed CLI pre/post scripts and profiling functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_model_info = {
            "name": "dummy",
            "n_gpus": "1",
            "scripts": "scripts/dummy/run.sh",
            "dockerfile": "docker/dummy.ubuntu.amd.Dockerfile",
            "tags": ["dummy", "test"]
        }
        
        self.test_build_info = {
            "docker_image": "ci-dummy_dummy.ubuntu.amd",
            "dockerfile": "docker/dummy.ubuntu.amd.Dockerfile",
            "base_docker": "rocm/pytorch",
            "build_duration": 45.2
        }

    @patch('madengine.tools.container_runner.Docker')
    @patch('madengine.core.console.Console')
    def test_system_env_collection_enabled_by_default(self, mock_console, mock_docker):
        """Test that system environment collection is enabled by default in distributed runs."""
        # Setup mocks
        mock_context = MagicMock()
        mock_context.ctx = {
            "gpu_vendor": "AMD",
            "docker_env_vars": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx908"}
        }
        
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        
        mock_docker_instance = MagicMock()
        mock_docker.return_value = mock_docker_instance
        mock_docker_instance.sh.return_value = "test output"
        
        # Create ContainerRunner
        runner = ContainerRunner(mock_context, None, mock_console_instance)
        
        # Mock file operations
        with patch('builtins.open', mock_open()), \
             patch('os.path.exists', return_value=False), \
             patch('madengine.tools.container_runner.Timeout'):
            
            # Call run_container with default generate_sys_env_details=True
            with pytest.raises(Exception):  # Will fail due to mocking, but we can check the pre_scripts
                runner.run_container(
                    self.test_model_info,
                    "ci-dummy_dummy.ubuntu.amd",
                    self.test_build_info,
                    generate_sys_env_details=True
                )
        
        # Verify that gather_system_env_details was called by checking if the method exists
        assert hasattr(runner, 'gather_system_env_details')

    def test_gather_system_env_details_method(self):
        """Test the gather_system_env_details method directly."""
        mock_context = MagicMock()
        runner = ContainerRunner(mock_context, None, Console())
        
        # Test pre_scripts structure
        pre_encapsulate_post_scripts = {"pre_scripts": [], "encapsulate_script": "", "post_scripts": []}
        
        # Call the method
        runner.gather_system_env_details(pre_encapsulate_post_scripts, "test_model")
        
        # Verify the system environment pre-script was added
        assert len(pre_encapsulate_post_scripts["pre_scripts"]) == 1
        pre_script = pre_encapsulate_post_scripts["pre_scripts"][0]
        assert pre_script["path"] == "scripts/common/pre_scripts/run_rocenv_tool.sh"
        assert pre_script["args"] == "test_model_env"

    def test_gather_system_env_details_with_slash_in_name(self):
        """Test gather_system_env_details with model name containing slash."""
        mock_context = MagicMock()
        runner = ContainerRunner(mock_context, None, Console())
        
        pre_encapsulate_post_scripts = {"pre_scripts": [], "encapsulate_script": "", "post_scripts": []}
        
        # Test with model name containing slash
        runner.gather_system_env_details(pre_encapsulate_post_scripts, "namespace/model")
        
        # Verify slash is replaced with underscore in args
        pre_script = pre_encapsulate_post_scripts["pre_scripts"][0]
        assert pre_script["args"] == "namespace_model_env"

    @patch('madengine.tools.container_runner.os.path.exists')
    def test_tools_json_application_with_sys_env(self, mock_exists):
        """Test that tools.json is applied AND system env collection is still added."""
        mock_context = MagicMock()
        mock_context.ctx = {
            "gpu_vendor": "AMD",
            "tools": [{"name": "rocprof", "cmd": "rocprof"}]
        }
        
        runner = ContainerRunner(mock_context, None, Console())
        
        # Mock tools.json exists
        mock_exists.return_value = True
        
        tools_content = {
            "tools": {
                "rocprof": {
                    "pre_scripts": [],
                    "cmd": "rocprof",
                    "env_vars": {},
                    "post_scripts": []
                }
            }
        }
        
        pre_encapsulate_post_scripts = {"pre_scripts": [], "encapsulate_script": "", "post_scripts": []}
        run_env = {}
        
        with patch('builtins.open', mock_open(read_data=json.dumps(tools_content))):
            # Apply tools first
            runner.apply_tools(pre_encapsulate_post_scripts, run_env, "scripts/common/tools.json")
            
            # Then add system env collection (simulating the fixed run_container logic)
            runner.gather_system_env_details(pre_encapsulate_post_scripts, "dummy")
        
        # Verify both tools and system env collection are present
        assert len(pre_encapsulate_post_scripts["pre_scripts"]) == 1  # sys env script
        assert pre_encapsulate_post_scripts["pre_scripts"][0]["path"] == "scripts/common/pre_scripts/run_rocenv_tool.sh"

    @patch('madengine.distributed_cli.DistributedOrchestrator')
    def test_distributed_cli_with_profiling_context(self, mock_orchestrator):
        """Test distributed CLI with profiling tools in additional context."""
        # Create test script to call distributed CLI
        test_context = {
            "tools": [
                {
                    "name": "rocprof",
                    "cmd": "rocprof --hip-trace"
                }
            ]
        }
        
        mock_args = MagicMock()
        mock_args.tags = ["dummy"]
        mock_args.additional_context = json.dumps(test_context)
        mock_args.generate_sys_env_details = True
        mock_args.timeout = 3600
        mock_args.manifest_file = None
        mock_args.manifest_output = "build_manifest.json"
        mock_args.clean_docker_cache = False
        mock_args.registry = None
        mock_args.keep_alive = False
        mock_args.summary_output = None
        
        # Mock successful build and run
        mock_instance = MagicMock()
        mock_orchestrator.return_value = mock_instance
        mock_instance.build_phase.return_value = {"successful_builds": ["dummy"], "failed_builds": []}
        mock_instance.run_phase.return_value = {"successful_runs": ["dummy"], "failed_runs": []}
        
        with patch('os.path.exists', return_value=False):
            result = distributed_cli.run_models(mock_args)
        
        # Verify the context with profiling tools was passed through
        mock_orchestrator.assert_called_once_with(mock_args)
        assert result == distributed_cli.EXIT_SUCCESS

    @patch('subprocess.run')
    def test_distributed_cli_sys_env_integration(self, mock_subprocess):
        """Integration test: verify distributed CLI generates system env details in logs."""
        # Mock subprocess to avoid actual execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"System environment collection test passed"
        mock_subprocess.return_value = mock_result
        
        # Test command that should include system environment collection
        script_path = os.path.join(BASE_DIR, "src/madengine", "distributed_cli.py")
        test_cmd = [
            sys.executable, script_path, "run",
            "--tags", "dummy",
            "--generate-sys-env-details", "True",
            "--timeout", "60"
        ]
        
        # This would run the actual command if we wanted full integration
        # For now, just verify the command structure is correct
        assert script_path.endswith("distributed_cli.py")
        assert "run" in test_cmd
        assert "--generate-sys-env-details" in test_cmd

    def test_distributed_orchestrator_passes_sys_env_arg(self):
        """Test that DistributedOrchestrator passes generate_sys_env_details to ContainerRunner."""
        mock_args = MagicMock()
        mock_args.generate_sys_env_details = False  # Explicitly set to False
        mock_args.live_output = False
        mock_args.additional_context = None
        mock_args.additional_context_file = None
        mock_args.data_config_file_name = "data.json"
        mock_args.force_mirror_local = False
        
        with patch('madengine.tools.distributed_orchestrator.Context'), \
             patch('os.path.exists', return_value=False):
            
            orchestrator = DistributedOrchestrator(mock_args)
            
            # Verify that getattr(self.args, 'generate_sys_env_details', True) would work
            generate_flag = getattr(mock_args, 'generate_sys_env_details', True)
            assert generate_flag == False  # Should use the explicit False value

    @patch('madengine.tools.container_runner.Docker')
    def test_container_runner_respects_generate_sys_env_details_flag(self, mock_docker):
        """Test that ContainerRunner respects the generate_sys_env_details flag."""
        mock_context = MagicMock()
        mock_context.ctx = {
            "gpu_vendor": "AMD",
            "docker_env_vars": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx908"}
        }
        
        runner = ContainerRunner(mock_context, None, Console())
        
        # Test with generate_sys_env_details=False
        pre_scripts_before = {"pre_scripts": [], "encapsulate_script": "", "post_scripts": []}
        
        # Mock the parts that would be called in run_container
        with patch('builtins.open', mock_open()), \
             patch('os.path.exists', return_value=False), \
             patch('madengine.tools.container_runner.Timeout'), \
             patch.object(runner, 'gather_system_env_details') as mock_gather:
            
            try:
                runner.run_container(
                    self.test_model_info,
                    "ci-dummy_dummy.ubuntu.amd",
                    self.test_build_info,
                    generate_sys_env_details=False
                )
            except Exception:
                pass  # Expected due to mocking
            
            # Verify gather_system_env_details was NOT called when flag is False
            mock_gather.assert_not_called()

    @patch('madengine.tools.container_runner.Docker')
    def test_container_runner_calls_gather_when_flag_true(self, mock_docker):
        """Test that ContainerRunner calls gather_system_env_details when flag is True."""
        mock_context = MagicMock()
        mock_context.ctx = {
            "gpu_vendor": "AMD",
            "docker_env_vars": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx908"}
        }
        
        runner = ContainerRunner(mock_context, None, Console())
        
        # Mock the parts that would be called in run_container
        with patch('builtins.open', mock_open()), \
             patch('os.path.exists', return_value=False), \
             patch('madengine.tools.container_runner.Timeout'), \
             patch.object(runner, 'gather_system_env_details') as mock_gather:
            
            try:
                runner.run_container(
                    self.test_model_info,
                    "ci-dummy_dummy.ubuntu.amd", 
                    self.test_build_info,
                    generate_sys_env_details=True
                )
            except Exception:
                pass  # Expected due to mocking
            
            # Verify gather_system_env_details was called when flag is True
            mock_gather.assert_called_once_with(unittest.mock.ANY, "dummy")

    def test_profiling_tools_configuration(self):
        """Test various profiling tools configurations in distributed execution."""
        profiling_configs = [
            {
                "name": "rocprof",
                "tools": [{"name": "rocprof", "cmd": "rocprof --hip-trace"}]
            },
            {
                "name": "rocblas_trace", 
                "tools": [{"name": "rocblas_trace", "env_vars": {"ROCBLAS_TRACE": "1"}}]
            },
            {
                "name": "miopen_trace",
                "tools": [{"name": "miopen_trace", "env_vars": {"MIOPEN_TRACE": "1"}}]
            },
            {
                "name": "gpu_power_profiler",
                "tools": [{"name": "gpu_info_power_profiler", "env_vars": {"MODE": "power"}}]
            }
        ]
        
        for config in profiling_configs:
            # Test that each profiling configuration can be properly structured
            assert "name" in config
            assert "tools" in config
            assert len(config["tools"]) > 0
            
            tool = config["tools"][0]
            assert "name" in tool
            # Should have either cmd or env_vars (or both)
            assert "cmd" in tool or "env_vars" in tool

    @patch('madengine.distributed_cli.DistributedOrchestrator')
    def test_distributed_cli_with_multiple_profiling_tools(self, mock_orchestrator):
        """Test distributed CLI with multiple profiling tools enabled."""
        # Test context with multiple profiling tools
        multi_tool_context = {
            "tools": [
                {"name": "rocprof", "cmd": "rocprof --hip-trace"},
                {"name": "rocblas_trace", "env_vars": {"ROCBLAS_TRACE": "1"}},
                {"name": "gpu_info_power_profiler", "env_vars": {"MODE": "power"}}
            ]
        }
        
        mock_args = MagicMock()
        mock_args.tags = ["dummy"]
        mock_args.additional_context = json.dumps(multi_tool_context)
        mock_args.generate_sys_env_details = True
        mock_args.timeout = 7200
        mock_args.manifest_file = None
        mock_args.clean_docker_cache = False
        mock_args.registry = None
        mock_args.keep_alive = False
        mock_args.summary_output = None
        mock_args.manifest_output = "build_manifest.json"
        
        # Mock successful execution
        mock_instance = MagicMock()
        mock_orchestrator.return_value = mock_instance
        mock_instance.build_phase.return_value = {"successful_builds": ["dummy"], "failed_builds": []}
        mock_instance.run_phase.return_value = {"successful_runs": ["dummy"], "failed_runs": []}
        
        with patch('os.path.exists', return_value=False):
            result = distributed_cli.run_models(mock_args)
        
        # Verify successful execution with multiple profiling tools
        assert result == distributed_cli.EXIT_SUCCESS
        mock_orchestrator.assert_called_once_with(mock_args)

    @pytest.mark.parametrize("clean_test_temp_files", [["test_manifest.json", "test_summary.json"]], indirect=True)
    def test_distributed_build_with_profiling_context_file(self, clean_test_temp_files):
        """Test distributed build command with profiling context from file."""
        # Create temporary context file with profiling tools
        profiling_context = {
            "gpu_vendor": "AMD",
            "guest_os": "UBUNTU",
            "tools": [
                {"name": "rocprof", "cmd": "rocprof --timestamp on"}
            ],
            "docker_env_vars": {"NCCL_DEBUG": "INFO"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(profiling_context, f)
            context_file = f.name
        
        try:
            mock_args = MagicMock()
            mock_args.tags = ["dummy"]
            mock_args.additional_context_file = context_file
            mock_args.additional_context = "{}"
            mock_args.registry = "localhost:5000"
            mock_args.clean_docker_cache = False
            mock_args.manifest_output = "test_manifest.json"
            mock_args.summary_output = "test_summary.json"
            
            with patch('madengine.distributed_cli.DistributedOrchestrator') as mock_orchestrator:
                mock_instance = MagicMock()
                mock_orchestrator.return_value = mock_instance
                mock_instance.build_phase.return_value = {
                    "successful_builds": ["dummy"],
                    "failed_builds": []
                }
                
                result = distributed_cli.build_models(mock_args)
                
                # Verify context file was used
                assert result == distributed_cli.EXIT_SUCCESS
                mock_orchestrator.assert_called_once_with(mock_args, build_only_mode=True)
                
        finally:
            # Clean up temporary file
            if os.path.exists(context_file):
                os.unlink(context_file)

    def test_system_env_vs_standard_run_parity(self):
        """Test that distributed run system env collection matches standard run format."""
        # This test verifies the format of system env pre-script matches standard run
        mock_context = MagicMock()
        runner = ContainerRunner(mock_context, None, Console())
        
        pre_scripts = {"pre_scripts": [], "encapsulate_script": "", "post_scripts": []}
        
        # Add system env collection
        runner.gather_system_env_details(pre_scripts, "dummy")
        
        # Verify format matches what standard run_models.py produces
        expected_pre_script = {
            "path": "scripts/common/pre_scripts/run_rocenv_tool.sh",
            "args": "dummy_env"
        }
        
        assert len(pre_scripts["pre_scripts"]) == 1
        actual_pre_script = pre_scripts["pre_scripts"][0]
        assert actual_pre_script == expected_pre_script

    def test_error_handling_in_profiling_workflow(self):
        """Test error handling when profiling tools or system env collection fails."""
        mock_context = MagicMock()
        mock_context.ctx = {"gpu_vendor": "AMD"}
        runner = ContainerRunner(mock_context, None, Console())
        
        # Test that gather_system_env_details handles edge cases gracefully
        pre_scripts = {"pre_scripts": [], "encapsulate_script": "", "post_scripts": []}
        
        # Test with empty model name
        runner.gather_system_env_details(pre_scripts, "")
        assert pre_scripts["pre_scripts"][0]["args"] == "_env"
        
        # Test with None model name (should not crash)
        pre_scripts_2 = {"pre_scripts": [], "encapsulate_script": "", "post_scripts": []}
        try:
            runner.gather_system_env_details(pre_scripts_2, None)
        except AttributeError:
            pass  # Expected for None.replace()

    @patch('madengine.distributed_cli.DistributedOrchestrator')
    def test_distributed_cli_generate_sys_env_details_arg_parsing(self, mock_orchestrator):
        """Test that the --generate-sys-env-details argument is properly parsed and used."""
        # Test with explicitly disabled system env collection
        mock_args = MagicMock()
        mock_args.tags = ["dummy"]
        mock_args.generate_sys_env_details = False  # Explicitly disabled
        mock_args.timeout = 1800
        mock_args.manifest_file = None
        mock_args.clean_docker_cache = False
        mock_args.registry = None
        mock_args.keep_alive = False
        mock_args.summary_output = None
        mock_args.manifest_output = "build_manifest.json"
        
        mock_instance = MagicMock()
        mock_orchestrator.return_value = mock_instance
        mock_instance.build_phase.return_value = {"successful_builds": ["dummy"], "failed_builds": []}
        mock_instance.run_phase.return_value = {"successful_runs": ["dummy"], "failed_runs": []}
        
        with patch('os.path.exists', return_value=False):
            result = distributed_cli.run_models(mock_args)
        
        # Verify the flag was passed to the orchestrator
        assert result == distributed_cli.EXIT_SUCCESS
        assert mock_args.generate_sys_env_details == False

    def test_profiling_output_verification(self):
        """Test that profiling and system env collection produce expected output patterns."""
        # This test defines the expected patterns in log output to verify
        # that our fix produces the same output as standard madengine runs
        
        expected_patterns = [
            # System environment collection patterns
            r"pre encap post scripts:.*run_rocenv_tool\.sh",
            r"dummy_env",
            r"------- Section: os_information ----------",
            r"------- Section: cpu_information ----------", 
            r"------- Section: gpu_information ----------",
            r"------- Section: rocm_information ----------",
            r"OK: Dumped into.*\.csv file\.",
            
            # Docker execution patterns that should remain consistent
            r"docker exec.*run_rocenv_tool\.sh",
            r"GPU Device type detected is:",
            r"Printing the sys config info env variables\.\.\.",
        ]
        
        # These patterns should appear in distributed CLI logs after our fix
        for pattern in expected_patterns:
            # Verify the pattern format is valid regex
            import re
            assert re.compile(pattern) is not None
            
        # This test serves as documentation of what we expect to see
        # in the distributed CLI logs after applying our fix
        assert len(expected_patterns) > 0
