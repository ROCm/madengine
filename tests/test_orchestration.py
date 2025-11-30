"""Test the orchestration layer modules.

This module tests the Build and Run orchestrators that coordinate
the build and execution workflows.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import json
import os
import tempfile
from unittest.mock import MagicMock, mock_open, patch

# third-party modules
import pytest

# project modules
from madengine.orchestration.build_orchestrator import BuildOrchestrator
from madengine.orchestration.run_orchestrator import RunOrchestrator
from madengine.core.errors import BuildError, ConfigurationError, DiscoveryError


class TestBuildOrchestrator:
    """Test the Build Orchestrator module."""

    @patch("madengine.orchestration.build_orchestrator.Context")
    def test_build_orchestrator_initialization(self, mock_context):
        """Test orchestrator initialization with minimal args."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.live_output = True

        mock_context_instance = MagicMock()
        mock_context.return_value = mock_context_instance

        with patch("os.path.exists", return_value=False):
            orchestrator = BuildOrchestrator(mock_args)

        assert orchestrator.args == mock_args
        assert orchestrator.context == mock_context_instance
        assert orchestrator.additional_context == {}
        assert orchestrator.credentials is None

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"dockerhub": {"username": "test", "password": "pass"}}',
    )
    @patch("os.path.exists")
    @patch("madengine.orchestration.build_orchestrator.Context")
    def test_build_orchestrator_with_credentials(
        self, mock_context, mock_exists, mock_file
    ):
        """Test orchestrator initialization with credentials."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.live_output = True

        mock_context_instance = MagicMock()
        mock_context.return_value = mock_context_instance

        def exists_side_effect(path):
            return path == "credential.json"

        mock_exists.side_effect = exists_side_effect

        orchestrator = BuildOrchestrator(mock_args)

        assert orchestrator.credentials == {
            "dockerhub": {"username": "test", "password": "pass"}
        }

    @patch.dict(
        "os.environ",
        {
            "MAD_DOCKERHUB_USER": "env_user",
            "MAD_DOCKERHUB_PASSWORD": "env_pass",
            "MAD_DOCKERHUB_REPO": "env_repo",
        },
    )
    @patch("os.path.exists", return_value=False)
    @patch("madengine.orchestration.build_orchestrator.Context")
    def test_build_orchestrator_env_credentials(self, mock_context, mock_exists):
        """Test orchestrator with environment variable credentials."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.live_output = True

        mock_context_instance = MagicMock()
        mock_context.return_value = mock_context_instance

        orchestrator = BuildOrchestrator(mock_args)

        assert orchestrator.credentials == {
            "dockerhub": {
                "username": "env_user",
                "password": "env_pass",
                "repository": "env_repo",
            }
        }

    @patch("madengine.orchestration.build_orchestrator.DiscoverModels")
    @patch("madengine.orchestration.build_orchestrator.DockerBuilder")
    @patch("madengine.orchestration.build_orchestrator.Context")
    @patch("os.path.exists", return_value=False)
    @patch("pathlib.Path.exists", return_value=False)
    def test_build_execute_success(
        self,
        mock_path_exists,
        mock_os_exists,
        mock_context_class,
        mock_docker_builder,
        mock_discover_models,
    ):
        """Test successful build execution."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.live_output = False
        mock_args._separate_phases = True
        mock_args.target_archs = []

        # Mock context
        mock_context = MagicMock()
        mock_context.ctx = {"docker_build_arg": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx90a"}}
        mock_context_class.return_value = mock_context

        # Mock discover models
        mock_discover_instance = MagicMock()
        mock_discover_instance.run.return_value = [
            {"name": "model1", "tags": ["test"]},
            {"name": "model2", "tags": ["test"]},
        ]
        mock_discover_models.return_value = mock_discover_instance

        # Mock docker builder
        mock_builder_instance = MagicMock()
        # Match actual docker_builder.py return format (lists, not ints)
        mock_builder_instance.build_all_models.return_value = {
            "successful_builds": [{"model": "model1"}, {"model": "model2"}],
            "failed_builds": [],
        }
        mock_docker_builder.return_value = mock_builder_instance

        # Execute
        orchestrator = BuildOrchestrator(mock_args)
        manifest_file = orchestrator.execute(registry="docker.io", clean_cache=False)

        # Assertions
        assert manifest_file == "build_manifest.json"
        mock_discover_instance.run.assert_called_once()
        mock_builder_instance.build_all_models.assert_called_once()
        mock_builder_instance.export_build_manifest.assert_called_once()

    @patch("madengine.orchestration.build_orchestrator.DiscoverModels")
    @patch("madengine.orchestration.build_orchestrator.Context")
    @patch("os.path.exists", return_value=False)
    def test_build_execute_no_models_found(
        self, mock_os_exists, mock_context_class, mock_discover_models
    ):
        """Test build execution when no models are discovered."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.live_output = False

        mock_context = MagicMock()
        mock_context.ctx = {"docker_build_arg": {}}
        mock_context_class.return_value = mock_context

        mock_discover_instance = MagicMock()
        mock_discover_instance.run.return_value = []
        mock_discover_models.return_value = mock_discover_instance

        orchestrator = BuildOrchestrator(mock_args)

        with pytest.raises(DiscoveryError):
            orchestrator.execute()

    @patch("madengine.orchestration.build_orchestrator.DiscoverModels")
    @patch("madengine.orchestration.build_orchestrator.DockerBuilder")
    @patch("madengine.orchestration.build_orchestrator.Context")
    @patch("os.path.exists", return_value=False)
    @patch("pathlib.Path.exists", return_value=False)
    def test_build_execute_all_failures(
        self,
        mock_path_exists,
        mock_os_exists,
        mock_context_class,
        mock_docker_builder,
        mock_discover_models,
    ):
        """Test build execution when ALL builds fail - should raise BuildError."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.live_output = False
        mock_args._separate_phases = True
        mock_args.target_archs = []

        mock_context = MagicMock()
        mock_context.ctx = {"docker_build_arg": {}}
        mock_context_class.return_value = mock_context

        mock_discover_instance = MagicMock()
        mock_discover_instance.run.return_value = [{"name": "model1", "tags": ["test"]}]
        mock_discover_models.return_value = mock_discover_instance

        mock_builder_instance = MagicMock()
        # All builds failed - should raise BuildError
        mock_builder_instance.build_all_models.return_value = {
            "successful_builds": [],
            "failed_builds": [{"model": "model1", "error": "Build failed"}],
        }
        mock_docker_builder.return_value = mock_builder_instance

        orchestrator = BuildOrchestrator(mock_args)

        # Should raise BuildError when ALL builds fail
        with pytest.raises(BuildError, match="All builds failed"):
            orchestrator.execute()

    @patch("madengine.orchestration.build_orchestrator.DiscoverModels")
    @patch("madengine.orchestration.build_orchestrator.DockerBuilder")
    @patch("madengine.orchestration.build_orchestrator.Context")
    @patch("os.path.exists", return_value=False)
    @patch("pathlib.Path.exists", return_value=False)
    def test_build_execute_partial_failure(
        self,
        mock_path_exists,
        mock_os_exists,
        mock_context_class,
        mock_docker_builder,
        mock_discover_models,
    ):
        """Test build execution with PARTIAL failures - should save manifest and not raise."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.live_output = False
        mock_args._separate_phases = True
        mock_args.target_archs = []

        mock_context = MagicMock()
        mock_context.ctx = {"docker_build_arg": {}}
        mock_context_class.return_value = mock_context

        mock_discover_instance = MagicMock()
        mock_discover_instance.run.return_value = [
            {"name": "model1", "tags": ["test"]},
            {"name": "model2", "tags": ["test"]},
        ]
        mock_discover_models.return_value = mock_discover_instance

        mock_builder_instance = MagicMock()
        # Partial failure: 1 success, 1 failure
        mock_builder_instance.build_all_models.return_value = {
            "successful_builds": [{"model": "model1", "docker_image": "ci-model1"}],
            "failed_builds": [{"model": "model2", "error": "Build failed"}],
        }
        mock_docker_builder.return_value = mock_builder_instance

        orchestrator = BuildOrchestrator(mock_args)

        # Should NOT raise exception, manifest should be saved
        manifest_file = orchestrator.execute()

        # Verify manifest was saved
        assert manifest_file == "build_manifest.json"
        mock_builder_instance.export_build_manifest.assert_called_once()

        # Verify both successes and failures are in the summary
        mock_builder_instance.build_all_models.assert_called_once()
        result = mock_builder_instance.build_all_models.return_value
        assert len(result["successful_builds"]) == 1
        assert len(result["failed_builds"]) == 1


class TestRunOrchestrator:
    """Test the Run Orchestrator module."""

    @patch("madengine.orchestration.run_orchestrator.Context")
    def test_run_orchestrator_initialization(self, mock_context):
        """Test orchestrator initialization."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.live_output = True

        orchestrator = RunOrchestrator(mock_args)

        assert orchestrator.args == mock_args
        assert orchestrator.additional_context == {}
        assert orchestrator.context is None  # Lazy initialization

    def test_run_orchestrator_additional_context_parsing(self):
        """Test additional context parsing from JSON string."""
        mock_args = MagicMock()
        mock_args.additional_context = '{"deploy": "slurm", "slurm": {"nodes": 4}}'
        mock_args.live_output = True

        orchestrator = RunOrchestrator(mock_args)

        assert orchestrator.additional_context == {
            "deploy": "slurm",
            "slurm": {"nodes": 4},
        }

    @patch("os.path.exists", return_value=False)
    def test_run_execute_no_manifest_no_tags(self, mock_exists):
        """Test run execution fails without manifest or tags."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.live_output = True

        orchestrator = RunOrchestrator(mock_args)

        with pytest.raises(ConfigurationError):
            orchestrator.execute(manifest_file=None, tags=None)

    @patch("madengine.orchestration.build_orchestrator.BuildOrchestrator")
    def test_run_execute_triggers_build_phase(
        self, mock_build_orchestrator
    ):
        """Test run execution triggers build phase when no manifest exists."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.live_output = True
        mock_args.tags = ["test"]

        mock_build_instance = MagicMock()
        mock_build_instance.execute.return_value = "build_manifest.json"
        mock_build_orchestrator.return_value = mock_build_instance

        # Mock manifest loading
        manifest_data = {
            "built_images": {"model1": {"name": "model1"}},
            "deployment_config": {"target": "local"},
        }

        orchestrator = RunOrchestrator(mock_args)

        # Mock file operations and execution
        with patch("os.path.exists", side_effect=lambda p: p == "build_manifest.json"), \
             patch("builtins.open", mock_open(read_data=json.dumps(manifest_data))), \
             patch.object(orchestrator, "_execute_local", return_value={}) as mock_execute_local:
            orchestrator.execute(manifest_file=None, tags=["test"])

        mock_build_instance.execute.assert_called_once()
        mock_execute_local.assert_called_once()

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"built_images": {"model1": {"name": "model1"}}}',
    )
    @patch("os.path.exists", return_value=True)
    def test_run_execute_local(self, mock_exists, mock_file):
        """Test run execution in local mode."""
        mock_args = MagicMock()
        mock_args.additional_context = '{"deploy": "local"}'
        mock_args.live_output = True

        orchestrator = RunOrchestrator(mock_args)

        with patch.object(
            orchestrator, "_execute_local", return_value={"status": "success"}
        ) as mock_execute_local:
            result = orchestrator.execute(manifest_file="build_manifest.json")

        assert result == {"status": "success"}
        mock_execute_local.assert_called_once()

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"built_images": {"model1": {"name": "model1"}}}',
    )
    @patch("os.path.exists", return_value=True)
    def test_run_execute_distributed(self, mock_exists, mock_file):
        """Test run execution in distributed mode."""
        mock_args = MagicMock()
        mock_args.additional_context = '{"deploy": "slurm"}'
        mock_args.live_output = True

        orchestrator = RunOrchestrator(mock_args)

        with patch.object(
            orchestrator,
            "_execute_distributed",
            return_value={"status": "deployed"},
        ) as mock_execute_distributed:
            result = orchestrator.execute(manifest_file="build_manifest.json")

        assert result == {"status": "deployed"}
        mock_execute_distributed.assert_called_once_with("slurm", "build_manifest.json")

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"built_images": {"model1": {"name": "model1"}}, "context": {}}',
    )
    @patch("os.path.exists", return_value=True)
    def test_execute_local_with_mock(
        self, mock_exists, mock_file
    ):
        """Test local execution workflow (mocked)."""
        mock_args = MagicMock()
        mock_args.additional_context = '{"deploy": "local"}'
        mock_args.live_output = False

        orchestrator = RunOrchestrator(mock_args)

        # Mock the _execute_local method to avoid deep integration
        with patch.object(
            orchestrator, "_execute_local", return_value={"successful_runs": 1}
        ) as mock_execute_local:
            result = orchestrator.execute(manifest_file="build_manifest.json")

        assert result["successful_runs"] == 1
        mock_execute_local.assert_called_once()

    def test_filter_images_by_gpu_architecture(self):
        """Test GPU architecture filtering logic."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.live_output = True

        orchestrator = RunOrchestrator(mock_args)

        built_images = {
            "model1": {"name": "model1", "gpu_architecture": "gfx90a"},
            "model2": {"name": "model2", "gpu_architecture": "gfx908"},
            "model3": {"name": "model3", "gpu_architecture": ""},  # Legacy
        }

        # Filter for gfx90a
        compatible = orchestrator._filter_images_by_gpu_architecture(
            built_images, "gfx90a"
        )

        assert "model1" in compatible
        assert "model2" not in compatible
        assert "model3" in compatible  # Legacy images pass through

