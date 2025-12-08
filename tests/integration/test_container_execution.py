"""Test the container runner module.

This module tests the Docker container execution functionality for distributed execution.

UPDATED: Now uses execution/container_runner.py (madengine-cli architecture).
Previous: Used deprecated tools/container_runner.py (removed).

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import os
import json
import tempfile
import unittest.mock
from unittest.mock import patch, MagicMock, mock_open

# third-party modules
import pytest

# project modules
from madengine.execution.container_runner import ContainerRunner
from madengine.core.context import Context
from madengine.core.console import Console
from madengine.core.dataprovider import Data
from tests.fixtures.utils import BASE_DIR, MODEL_DIR


class TestContainerRunner:
    """Test the container runner module."""

    @patch("madengine.core.context.Context")
    def test_container_runner_initialization(self, mock_context_class):
        """Test ContainerRunner initialization."""
        mock_context = MagicMock()
        mock_context_class.return_value = mock_context
        context = mock_context_class()
        console = Console()
        data = MagicMock()

        runner = ContainerRunner(context, data, console)

        assert runner.context == context
        assert runner.data == data
        assert runner.console == console
        assert runner.credentials is None

    def test_container_runner_initialization_minimal(self):
        """Test ContainerRunner initialization with minimal parameters."""
        runner = ContainerRunner()

        assert runner.context is None
        assert runner.data is None
        assert isinstance(runner.console, Console)
        assert runner.credentials is None

    def test_load_build_manifest(self):
        """Test loading build manifest from file."""
        runner = ContainerRunner()

        manifest_data = {
            "images": {
                "model1": "localhost:5000/ci-model1:latest",
                "model2": "localhost:5000/ci-model2:latest",
            },
            "metadata": {
                "build_time": "2023-01-01T12:00:00Z",
                "registry": "localhost:5000",
            },
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(manifest_data))):
            result = runner.load_build_manifest("test_manifest.json")

        assert result == manifest_data
        assert "images" in result
        assert "model1" in result["images"]

    @patch.object(Console, "sh")
    def test_pull_image(self, mock_sh):
        """Test pulling image from registry."""
        runner = ContainerRunner()

        mock_sh.return_value = "Pull successful"

        result = runner.pull_image("localhost:5000/test:latest")

        assert result == "localhost:5000/test:latest"
        mock_sh.assert_called_with("docker pull localhost:5000/test:latest")

    @patch.object(Console, "sh")
    def test_pull_image_with_local_name(self, mock_sh):
        """Test pulling image with local name tagging."""
        runner = ContainerRunner()

        mock_sh.return_value = "Success"

        result = runner.pull_image("localhost:5000/test:latest", "local-test")

        assert result == "local-test"
        # Should have called pull and tag
        expected_calls = [
            unittest.mock.call("docker pull localhost:5000/test:latest"),
            unittest.mock.call("docker tag localhost:5000/test:latest local-test"),
        ]
        mock_sh.assert_has_calls(expected_calls)

    @patch("madengine.core.context.Context")
    def test_get_gpu_arg_all_gpus(self, mock_context_class):
        """Test get_gpu_arg with all GPUs requested."""
        mock_context = MagicMock()
        mock_context.ctx = {
            "docker_env_vars": {"MAD_GPU_VENDOR": "AMD", "MAD_SYSTEM_NGPUS": "4"},
            "docker_gpus": "0,1,2,3",
            "gpu_renderDs": [128, 129, 130, 131],  # Mock render device IDs for AMD GPUs
        }
        mock_context_class.return_value = mock_context
        runner = ContainerRunner(mock_context)

        result = runner.get_gpu_arg("-1")

        # Should return GPU args for all available GPUs
        assert "--device=/dev/kfd" in result and "renderD" in result

    @patch("madengine.core.context.Context")
    def test_get_gpu_arg_specific_gpus(self, mock_context_class):
        """Test get_gpu_arg with specific GPUs requested."""
        mock_context = MagicMock()
        mock_context.ctx = {
            "docker_env_vars": {"MAD_GPU_VENDOR": "NVIDIA", "MAD_SYSTEM_NGPUS": "4"},
            "docker_gpus": "0,1,2,3",
        }
        mock_context_class.return_value = mock_context
        runner = ContainerRunner(mock_context)

        result = runner.get_gpu_arg("2")

        # Should return GPU args for 2 GPUs
        assert "gpu" in result.lower()

    @patch("madengine.core.context.Context")
    def test_get_gpu_arg_range_format(self, mock_context_class):
        """Test get_gpu_arg with range format."""
        mock_context = MagicMock()
        mock_context.ctx = {
            "docker_env_vars": {"MAD_GPU_VENDOR": "NVIDIA", "MAD_SYSTEM_NGPUS": "4"},
            "docker_gpus": "0-3",
        }
        mock_context_class.return_value = mock_context
        runner = ContainerRunner(mock_context)

        result = runner.get_gpu_arg("2")

        # Should handle range format correctly
        assert isinstance(result, str)

    @patch("madengine.core.context.Context")
    @patch.object(Console, "sh")
    @patch("madengine.execution.container_runner.Docker")
    def test_run_container_success(
        self, mock_docker_class, mock_sh, mock_context_class
    ):
        """Test successful container run."""
        # Mock context to avoid GPU detection
        mock_context = MagicMock()
        mock_context.ctx = {
            "docker_env_vars": {"MAD_GPU_VENDOR": "NVIDIA", "MAD_SYSTEM_NGPUS": "2"},
            "docker_gpus": "0,1",
            "gpu_vendor": "NVIDIA",
        }
        mock_context_class.return_value = mock_context
        runner = ContainerRunner(mock_context)

        # Mock Docker instance
        mock_docker = MagicMock()
        mock_docker.sh.return_value = "Command output"
        mock_docker_class.return_value = mock_docker

        mock_sh.return_value = "hostname"

        model_info = {
            "name": "test_model",
            "n_gpus": "1",
            "scripts": "test_script.sh",
            "args": "",
        }

        with patch.object(runner, "get_gpu_arg", return_value="--gpus device=0"):
            with patch.object(runner, "get_cpu_arg", return_value=""):
                with patch.object(runner, "get_env_arg", return_value=""):
                    with patch.object(runner, "get_mount_arg", return_value=""):
                        result = runner.run_container(
                            model_info, "test-image", timeout=300
                        )

        assert result["status"] == "SUCCESS"
        assert "test_duration" in result
        assert mock_docker_class.called

    @patch("madengine.core.context.Context")
    @patch.object(Console, "sh")
    @patch("madengine.execution.container_runner.Docker")
    def test_run_container_timeout(
        self, mock_docker_class, mock_sh, mock_context_class
    ):
        """Test container run with timeout."""
        # Mock context to avoid GPU detection
        mock_context = MagicMock()
        mock_context.ctx = {
            "docker_env_vars": {"MAD_GPU_VENDOR": "NVIDIA", "MAD_SYSTEM_NGPUS": "2"},
            "docker_gpus": "0,1",
            "gpu_vendor": "NVIDIA",
        }
        mock_context_class.return_value = mock_context
        runner = ContainerRunner(mock_context)

        # Mock Docker instance that raises TimeoutError
        mock_docker = MagicMock()
        mock_docker.sh.side_effect = TimeoutError("Timeout occurred")
        mock_docker_class.return_value = mock_docker

        mock_sh.return_value = "hostname"

        model_info = {
            "name": "test_model",
            "n_gpus": "1",
            "scripts": "test_script.sh",
            "args": "",
        }

        with patch.object(runner, "get_gpu_arg", return_value="--gpus device=0"):
            with patch.object(runner, "get_cpu_arg", return_value=""):
                with patch.object(runner, "get_env_arg", return_value=""):
                    with patch.object(runner, "get_mount_arg", return_value=""):
                        # run_container catches exceptions and returns results with status
                        result = runner.run_container(
                            model_info, "test-image", timeout=10
                        )
                        assert result["status"] == "FAILURE"

    @patch("madengine.core.context.Context")
    @patch.object(Console, "sh")
    @patch("madengine.execution.container_runner.Docker")
    def test_run_container_failure(
        self, mock_docker_class, mock_sh, mock_context_class
    ):
        """Test container run failure."""
        # Mock context to avoid GPU detection
        mock_context = MagicMock()
        mock_context.ctx = {
            "docker_env_vars": {"MAD_GPU_VENDOR": "NVIDIA", "MAD_SYSTEM_NGPUS": "2"},
            "docker_gpus": "0,1",
            "gpu_vendor": "NVIDIA",
        }
        mock_context_class.return_value = mock_context
        runner = ContainerRunner(mock_context)

        # Mock Docker instance that raises RuntimeError
        mock_docker = MagicMock()
        mock_docker.sh.side_effect = RuntimeError("Container failed to start")
        mock_docker_class.return_value = mock_docker

        mock_sh.return_value = "hostname"

        model_info = {
            "name": "test_model",
            "n_gpus": "1",
            "scripts": "test_script.sh",
            "args": "",
        }

        with patch.object(runner, "get_gpu_arg", return_value="--gpus device=0"):
            with patch.object(runner, "get_cpu_arg", return_value=""):
                with patch.object(runner, "get_env_arg", return_value=""):
                    with patch.object(runner, "get_mount_arg", return_value=""):
                        # run_container catches exceptions and returns results with status
                        result = runner.run_container(
                            model_info, "test-image", timeout=300
                        )
                        assert result["status"] == "FAILURE"

    @patch("madengine.core.context.Context")
    def test_load_credentials(self, mock_context_class):
        """Test setting credentials for container runner."""
        # Mock context to avoid GPU detection
        mock_context = MagicMock()
        mock_context_class.return_value = mock_context
        runner = ContainerRunner(mock_context)

        credentials = {"github": {"username": "testuser", "password": "testpass"}}

        runner.set_credentials(credentials)

        assert runner.credentials == credentials

    @patch("madengine.core.context.Context")
    def test_login_to_registry(self, mock_context_class):
        """Test login to Docker registry."""
        # Mock context to avoid GPU detection
        mock_context = MagicMock()
        mock_context_class.return_value = mock_context
        runner = ContainerRunner(mock_context)

        credentials = {
            "localhost:5000": {"username": "testuser", "password": "testpass"}
        }

        with patch.object(runner.console, "sh") as mock_sh:
            mock_sh.return_value = "Login Succeeded"
            runner.login_to_registry("localhost:5000", credentials)

            # Verify login command was called
            assert mock_sh.called

    @patch("madengine.core.context.Context")
    def test_get_gpu_arg_specific_gpu(self, mock_context_class):
        """Test getting GPU arguments for specific GPU count."""
        # Mock context to avoid GPU detection
        mock_context = MagicMock()
        mock_context.ctx = {
            "docker_env_vars": {"MAD_GPU_VENDOR": "NVIDIA", "MAD_SYSTEM_NGPUS": "4"},
            "docker_gpus": "0,1,2,3",
        }
        mock_context_class.return_value = mock_context
        runner = ContainerRunner(mock_context)

        result = runner.get_gpu_arg("2")

        # Should return GPU args for 2 GPUs
        assert "gpu" in result.lower() or "device" in result.lower()

    @patch("madengine.core.context.Context")
    def test_get_cpu_arg(self, mock_context_class):
        """Test getting CPU arguments for docker run."""
        # Mock context to avoid GPU detection
        mock_context = MagicMock()
        mock_context.ctx = {"docker_cpus": "0,1,2,3"}
        mock_context_class.return_value = mock_context
        runner = ContainerRunner(mock_context)

        result = runner.get_cpu_arg()

        assert "--cpuset-cpus" in result
        assert "0,1,2,3" in result

    @patch("madengine.core.context.Context")
    def test_get_env_arg(self, mock_context_class):
        """Test getting environment variables for container."""
        # Mock context to avoid GPU detection
        mock_context = MagicMock()
        mock_context.ctx = {
            "docker_env_vars": {
                "MAD_GPU_VENDOR": "NVIDIA",
                "MAD_MODEL_NAME": "test_model",
                "CUSTOM_VAR": "custom_value",
            }
        }
        mock_context_class.return_value = mock_context
        runner = ContainerRunner(mock_context)

        custom_env = {"EXTRA_VAR": "extra_value"}
        result = runner.get_env_arg(custom_env)

        assert "--env MAD_GPU_VENDOR=" in result
        assert "--env EXTRA_VAR=" in result

    @patch("madengine.core.context.Context")
    def test_get_mount_arg(self, mock_context_class):
        """Test getting mount arguments for container."""
        # Mock context to avoid GPU detection
        mock_context = MagicMock()
        mock_context.ctx = {
            "docker_mounts": {
                "/container/data": "/host/data",
                "/container/output": "/host/output",
            }
        }
        mock_context_class.return_value = mock_context
        runner = ContainerRunner(mock_context)

        mount_datapaths = [
            {"path": "/host/input", "home": "/container/input", "readwrite": "false"}
        ]

        result = runner.get_mount_arg(mount_datapaths)

        assert "-v /host/input:/container/input:ro" in result
        assert "-v /host/data:/container/data" in result

    def test_apply_tools_without_tools_config(self):
        """Test applying tools when no tools configuration exists."""
        runner = ContainerRunner()

        # Mock context without tools
        runner.context = MagicMock()
        runner.context.ctx = {}

        pre_encapsulate_post_scripts = {
            "pre_scripts": [],
            "encapsulate_script": "",
            "post_scripts": [],
        }
        run_env = {}

        # Should not raise any exception
        runner.apply_tools(pre_encapsulate_post_scripts, run_env, "nonexistent.json")

        # Scripts should remain unchanged
        assert pre_encapsulate_post_scripts["pre_scripts"] == []
        assert pre_encapsulate_post_scripts["encapsulate_script"] == ""
        assert run_env == {}

    def test_run_pre_post_script(self):
        """Test running pre/post scripts."""
        runner = ContainerRunner()

        # Mock Docker instance
        mock_docker = MagicMock()
        mock_docker.sh = MagicMock()

        scripts = [
            {"path": "/path/to/script1.sh", "args": "arg1 arg2"},
            {"path": "/path/to/script2.sh"},
        ]

        runner.run_pre_post_script(mock_docker, "model_dir", scripts)

        # Verify scripts were copied and executed
        assert mock_docker.sh.call_count == 4  # 2 copies + 2 executions

        # Check if copy commands were called
        copy_calls = [
            call for call in mock_docker.sh.call_args_list if "cp -vLR" in str(call)
        ]
        assert len(copy_calls) == 2

    def test_initialization_with_all_parameters(self):
        """Test ContainerRunner initialization with all parameters."""
        context = MagicMock()
        console = Console()
        data = MagicMock()

        runner = ContainerRunner(context, data, console)

        assert runner.context == context
        assert runner.data == data
        assert runner.console == console
        assert runner.credentials is None
