"""DEPRECATED: Test the distributed orchestrator module.

⚠️ DEPRECATED - DistributedOrchestrator has been removed ⚠️

This test file is DEPRECATED because DistributedOrchestrator class has been removed
from the codebase and replaced by BuildOrchestrator + RunOrchestrator.

See test_distributed_orchestrator.DEPRECATED.txt for migration guide.

Replacement tests:
- test_orchestration.py - Tests for BuildOrchestrator and RunOrchestrator
- test_mad_cli.py - Integration tests with new orchestrators

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

# Skip all tests in this file - DistributedOrchestrator has been removed
pytestmark = pytest.mark.skip(reason="DEPRECATED: DistributedOrchestrator removed, use test_orchestration.py instead")

# Import would fail since distributed_orchestrator.py has been deleted
# from madengine.tools.distributed_orchestrator import DistributedOrchestrator
from madengine.core.context import Context
from madengine.core.console import Console
from .fixtures.utils import BASE_DIR, MODEL_DIR


class TestDistributedOrchestrator:
    """Test the distributed orchestrator module."""

    @patch("madengine.tools.distributed_orchestrator.Context")
    def test_orchestrator_initialization(self, mock_context):
        """Test orchestrator initialization with minimal args."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.additional_context_file = None
        mock_args.data_config_file_name = "data.json"
        mock_args.force_mirror_local = False
        mock_args.live_output = True

        # Mock context instance
        mock_context_instance = MagicMock()
        mock_context.return_value = mock_context_instance

        with patch("os.path.exists", return_value=False):
            orchestrator = DistributedOrchestrator(mock_args)

        assert orchestrator.args == mock_args
        assert isinstance(orchestrator.console, Console)
        assert orchestrator.context == mock_context_instance
        assert orchestrator.data is None
        assert orchestrator.credentials is None

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"registry": "test", "token": "abc123"}',
    )
    @patch("os.path.exists")
    @patch("madengine.tools.distributed_orchestrator.Context")
    def test_orchestrator_with_credentials(self, mock_context, mock_exists, mock_file):
        """Test orchestrator initialization with credentials."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.additional_context_file = None
        mock_args.data_config_file_name = "data.json"
        mock_args.force_mirror_local = False
        mock_args.live_output = True

        # Mock context instance
        mock_context_instance = MagicMock()
        mock_context.return_value = mock_context_instance

        # Mock credential.json exists
        def exists_side_effect(path):
            return path == "credential.json"

        mock_exists.side_effect = exists_side_effect

        orchestrator = DistributedOrchestrator(mock_args)

        assert orchestrator.credentials == {"registry": "test", "token": "abc123"}

    @patch("madengine.tools.distributed_orchestrator.DiscoverModels")
    @patch("madengine.tools.distributed_orchestrator.DockerBuilder")
    @patch("madengine.tools.distributed_orchestrator.Context")
    def test_build_phase(
        self, mock_context_class, mock_docker_builder, mock_discover_models
    ):
        """Test the build phase functionality."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.additional_context_file = None
        mock_args.data_config_file_name = "data.json"
        mock_args.force_mirror_local = False
        mock_args.live_output = True

        # Mock context
        mock_context = MagicMock()
        mock_context_class.return_value = mock_context

        # Mock discover models
        mock_discover_instance = MagicMock()
        mock_discover_models.return_value = mock_discover_instance
        mock_discover_instance.run.return_value = [
            {"name": "model1", "dockerfile": "Dockerfile1"},
            {"name": "model2", "dockerfile": "Dockerfile2"},
        ]

        # Mock docker builder
        mock_builder_instance = MagicMock()
        mock_docker_builder.return_value = mock_builder_instance
        mock_builder_instance.build_all_models.return_value = {
            "successful_builds": ["model1", "model2"],
            "failed_builds": [],
            "total_build_time": 120.5,
        }

        with patch("os.path.exists", return_value=False):
            orchestrator = DistributedOrchestrator(mock_args)

        with patch.object(orchestrator, "_copy_scripts"):
            result = orchestrator.build_phase(
                registry="localhost:5000",
                clean_cache=True,
                manifest_output="test_manifest.json",
            )

        # Verify the flow
        mock_discover_models.assert_called_once_with(args=mock_args)
        mock_discover_instance.run.assert_called_once()
        mock_docker_builder.assert_called_once()
        mock_builder_instance.build_all_models.assert_called_once()
        mock_builder_instance.export_build_manifest.assert_called_once_with(
            "test_manifest.json", "localhost:5000", unittest.mock.ANY
        )

        assert result["successful_builds"] == ["model1", "model2"]
        assert result["failed_builds"] == []

    @patch("madengine.tools.distributed_orchestrator.ContainerRunner")
    @patch("madengine.tools.distributed_orchestrator.DiscoverModels")
    @patch("madengine.tools.distributed_orchestrator.Context")
    def test_run_phase(self, mock_context, mock_discover_models, mock_container_runner):
        """Test the run phase functionality."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.additional_context_file = None
        mock_args.data_config_file_name = "data.json"
        mock_args.force_mirror_local = False
        mock_args.live_output = True

        # Mock context instance
        mock_context_instance = MagicMock()
        mock_context.return_value = mock_context_instance

        # Mock discover models
        mock_discover_instance = MagicMock()
        mock_discover_models.return_value = mock_discover_instance
        mock_discover_instance.run.return_value = [
            {
                "name": "dummy",
                "dockerfile": "docker/dummy",
                "scripts": "scripts/dummy/run.sh",
            }
        ]

        # Mock container runner
        mock_runner_instance = MagicMock()
        mock_container_runner.return_value = mock_runner_instance
        mock_runner_instance.load_build_manifest.return_value = {
            "images": {"dummy": "localhost:5000/dummy:latest"}
        }
        mock_runner_instance.run_container.return_value = {
            "status": "completed",
            "test_duration": 120.5,
            "model": "dummy",
            "exit_code": 0,
        }
        mock_runner_instance.run_all_containers.return_value = {
            "successful_runs": ["dummy"],
            "failed_runs": [],
        }

        with patch("os.path.exists", return_value=False):
            orchestrator = DistributedOrchestrator(mock_args)

        # Mock manifest file existence and content
        manifest_content = '{"built_images": {"dummy": {"image": "localhost:5000/dummy:latest", "build_time": 120}}}'

        with patch.object(orchestrator, "_copy_scripts"), patch(
            "os.path.exists"
        ) as mock_exists, patch("builtins.open", mock_open(read_data=manifest_content)):

            # Mock manifest file exists but credential.json doesn't
            def exists_side_effect(path):
                return path == "manifest.json"

            mock_exists.side_effect = exists_side_effect

            result = orchestrator.run_phase(
                manifest_file="manifest.json",
                registry="localhost:5000",
                timeout=1800,
                keep_alive=False,
            )

        # Verify the flow
        mock_discover_models.assert_called_once_with(args=mock_args)
        mock_discover_instance.run.assert_called_once()
        mock_container_runner.assert_called_once()

        assert "successful_runs" in result
        assert "failed_runs" in result

    @patch("madengine.tools.distributed_orchestrator.DiscoverModels")
    @patch("madengine.tools.distributed_orchestrator.DockerBuilder")
    @patch("madengine.tools.distributed_orchestrator.ContainerRunner")
    @patch("madengine.tools.distributed_orchestrator.Context")
    def test_full_workflow(
        self,
        mock_context_class,
        mock_container_runner,
        mock_docker_builder,
        mock_discover_models,
    ):
        """Test the full workflow functionality."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.additional_context_file = None
        mock_args.data_config_file_name = "data.json"
        mock_args.force_mirror_local = False
        mock_args.live_output = True

        # Mock context
        mock_context = MagicMock()
        mock_context_class.return_value = mock_context

        # Mock discover models
        mock_discover_instance = MagicMock()
        mock_discover_models.return_value = mock_discover_instance
        mock_discover_instance.run.return_value = [{"name": "model1"}]

        # Mock docker builder
        mock_builder_instance = MagicMock()
        mock_docker_builder.return_value = mock_builder_instance
        mock_builder_instance.build_all_models.return_value = {
            "successful_builds": ["model1"],
            "failed_builds": [],
            "total_build_time": 120.5,
        }
        mock_builder_instance.get_build_manifest.return_value = {
            "images": {"model1": "ci-model1:latest"}
        }

        # Mock container runner
        mock_runner_instance = MagicMock()
        mock_container_runner.return_value = mock_runner_instance
        mock_runner_instance.run_container.return_value = {
            "status": "SUCCESS",
            "test_duration": 120.5,
            "model": "model1",
            "exit_code": 0,
        }
        mock_runner_instance.run_all_containers.return_value = {
            "successful_runs": ["model1"],
            "failed_runs": [],
        }

        with patch("os.path.exists", return_value=False):
            orchestrator = DistributedOrchestrator(mock_args)

        # Mock manifest file content for run phase
        manifest_content = """{"built_images": {"model1": {"docker_image": "ci-model1", "build_time": 120}}, "built_models": {"model1": {"name": "model1", "scripts": "scripts/model1/run.sh"}}}"""

        with patch.object(orchestrator, "_copy_scripts"), patch(
            "os.path.exists"
        ) as mock_exists, patch("builtins.open", mock_open(read_data=manifest_content)):

            # Mock build_manifest.json exists for run phase
            def exists_side_effect(path):
                return path == "build_manifest.json"

            mock_exists.side_effect = exists_side_effect

            result = orchestrator.full_workflow(
                registry="localhost:5000",
                clean_cache=True,
                timeout=3600,
                keep_alive=False,
            )

        # Verify the complete flow
        assert result["overall_success"] is True
        assert "build_phase" in result
        assert "run_phase" in result

    @patch("madengine.tools.distributed_orchestrator.Context")
    def test_copy_scripts_method(self, mock_context):
        """Test the _copy_scripts method."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.additional_context_file = None
        mock_args.data_config_file_name = "data.json"
        mock_args.force_mirror_local = False
        mock_args.live_output = True

        # Mock context instance
        mock_context_instance = MagicMock()
        mock_context.return_value = mock_context_instance

        with patch("os.path.exists", return_value=False):
            orchestrator = DistributedOrchestrator(mock_args)

        with patch.object(orchestrator.console, "sh") as mock_sh:
            with patch("os.path.exists", return_value=True):
                orchestrator._copy_scripts()
                mock_sh.assert_called_once()
