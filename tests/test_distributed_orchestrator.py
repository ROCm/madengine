"""Test the distributed orchestrator module.

This module tests the distributed orchestrator functionality.

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
from madengine.tools.distributed_orchestrator import DistributedOrchestrator
from madengine.core.context import Context
from madengine.core.console import Console
from .fixtures.utils import BASE_DIR, MODEL_DIR


class TestDistributedOrchestrator:
    """Test the distributed orchestrator module."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization with minimal args."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.additional_context_file = None
        mock_args.data_config_file_name = 'data.json'
        mock_args.force_mirror_local = False
        mock_args.live_output = True

        with patch('os.path.exists', return_value=False):
            orchestrator = DistributedOrchestrator(mock_args)
            
        assert orchestrator.args == mock_args
        assert isinstance(orchestrator.console, Console)
        assert isinstance(orchestrator.context, Context)
        assert orchestrator.data is None
        assert orchestrator.credentials is None

    @patch('builtins.open', new_callable=mock_open, read_data='{"registry": "test", "token": "abc123"}')
    @patch('os.path.exists')
    def test_orchestrator_with_credentials(self, mock_exists, mock_file):
        """Test orchestrator initialization with credentials."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.additional_context_file = None
        mock_args.data_config_file_name = 'data.json'
        mock_args.force_mirror_local = False
        mock_args.live_output = True

        # Mock credential.json exists
        def exists_side_effect(path):
            return path == "credential.json"
        
        mock_exists.side_effect = exists_side_effect

        orchestrator = DistributedOrchestrator(mock_args)
        
        assert orchestrator.credentials == {"registry": "test", "token": "abc123"}

    @patch('madengine.tools.distributed_orchestrator.DiscoverModels')
    @patch('madengine.tools.distributed_orchestrator.DockerBuilder')
    def test_build_phase(self, mock_docker_builder, mock_discover_models):
        """Test the build phase functionality."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.additional_context_file = None
        mock_args.data_config_file_name = 'data.json'
        mock_args.force_mirror_local = False
        mock_args.live_output = True

        # Mock discover models
        mock_discover_instance = MagicMock()
        mock_discover_models.return_value = mock_discover_instance
        mock_discover_instance.run.return_value = [
            {"name": "model1", "dockerfile": "Dockerfile1"},
            {"name": "model2", "dockerfile": "Dockerfile2"}
        ]

        # Mock docker builder
        mock_builder_instance = MagicMock()
        mock_docker_builder.return_value = mock_builder_instance
        mock_builder_instance.build_all_models.return_value = {
            "successful_builds": ["model1", "model2"],
            "failed_builds": [],
            "total_build_time": 120.5
        }

        with patch('os.path.exists', return_value=False):
            orchestrator = DistributedOrchestrator(mock_args)
            
        with patch.object(orchestrator, '_copy_scripts'):
            result = orchestrator.build_phase(
                registry="localhost:5000",
                clean_cache=True,
                manifest_output="test_manifest.json"
            )

        # Verify the flow
        mock_discover_models.assert_called_once_with(args=mock_args)
        mock_discover_instance.run.assert_called_once()
        mock_docker_builder.assert_called_once()
        mock_builder_instance.build_all_models.assert_called_once()
        mock_builder_instance.export_build_manifest.assert_called_once_with("test_manifest.json")
        
        assert result["successful_builds"] == ["model1", "model2"]
        assert result["failed_builds"] == []

    @patch('madengine.tools.distributed_orchestrator.ContainerRunner')
    @patch('madengine.tools.distributed_orchestrator.DiscoverModels')
    def test_run_phase(self, mock_discover_models, mock_container_runner):
        """Test the run phase functionality."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.additional_context_file = None
        mock_args.data_config_file_name = 'data.json'
        mock_args.force_mirror_local = False
        mock_args.live_output = True

        # Mock discover models
        mock_discover_instance = MagicMock()
        mock_discover_models.return_value = mock_discover_instance
        mock_discover_instance.run.return_value = [
            {"name": "dummy", "dockerfile": "docker/dummy", "scripts": "scripts/dummy/run.sh"}
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
            "exit_code": 0
        }
        mock_runner_instance.run_all_containers.return_value = {
            "successful_runs": ["dummy"],
            "failed_runs": []
        }

        with patch('os.path.exists', return_value=False):
            orchestrator = DistributedOrchestrator(mock_args)

        # Mock manifest file existence and content
        manifest_content = '{"built_images": {"dummy": {"image": "localhost:5000/dummy:latest", "build_time": 120}}}'
        
        with patch.object(orchestrator, '_copy_scripts'), \
             patch('os.path.exists') as mock_exists, \
             patch('builtins.open', mock_open(read_data=manifest_content)):
            
            # Mock manifest file exists but credential.json doesn't
            def exists_side_effect(path):
                return path == "manifest.json"
            mock_exists.side_effect = exists_side_effect
            
            result = orchestrator.run_phase(
                manifest_file="manifest.json",
                registry="localhost:5000",
                timeout=1800,
                keep_alive=False
            )

        # Verify the flow
        mock_discover_models.assert_called_once_with(args=mock_args)
        mock_discover_instance.run.assert_called_once()
        mock_container_runner.assert_called_once()
        
        assert "successful_runs" in result
        assert "failed_runs" in result

    @patch('madengine.tools.distributed_orchestrator.DiscoverModels')
    @patch('madengine.tools.distributed_orchestrator.DockerBuilder')
    @patch('madengine.tools.distributed_orchestrator.ContainerRunner')
    def test_full_workflow(self, mock_container_runner, mock_docker_builder, mock_discover_models):
        """Test the full workflow functionality."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.additional_context_file = None
        mock_args.data_config_file_name = 'data.json'
        mock_args.force_mirror_local = False
        mock_args.live_output = True

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
            "total_build_time": 120.5
        }
        mock_builder_instance.get_build_manifest.return_value = {
            "images": {"model1": "ci-model1:latest"}
        }

        # Mock container runner
        mock_runner_instance = MagicMock()
        mock_container_runner.return_value = mock_runner_instance
        mock_runner_instance.run_container.return_value = {
            "status": "completed",
            "test_duration": 120.5,
            "model": "model1",
            "exit_code": 0
        }
        mock_runner_instance.run_all_containers.return_value = {
            "successful_runs": ["model1"],
            "failed_runs": []
        }

        with patch('os.path.exists', return_value=False):
            orchestrator = DistributedOrchestrator(mock_args)

        # Mock manifest file content for run phase
        manifest_content = '{"built_images": {"model1": {"image": "localhost:5000/model1:latest", "build_time": 120}}}'
        
        with patch.object(orchestrator, '_copy_scripts'), \
             patch('os.path.exists') as mock_exists, \
             patch('builtins.open', mock_open(read_data=manifest_content)):
            
            # Mock build_manifest.json exists for run phase
            def exists_side_effect(path):
                return path == "build_manifest.json"
            mock_exists.side_effect = exists_side_effect
            
            result = orchestrator.full_workflow(
                registry="localhost:5000",
                clean_cache=True,
                timeout=3600,
                keep_alive=False
            )

        # Verify the complete flow
        assert result["overall_success"] is True
        assert "build_phase" in result
        assert "run_phase" in result

    def test_copy_scripts_method(self):
        """Test the _copy_scripts method."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.additional_context_file = None
        mock_args.data_config_file_name = 'data.json'
        mock_args.force_mirror_local = False
        mock_args.live_output = True

        with patch('os.path.exists', return_value=False):
            orchestrator = DistributedOrchestrator(mock_args)

        with patch.object(orchestrator.console, 'sh') as mock_sh:
            with patch('os.path.exists', return_value=True):
                orchestrator._copy_scripts()
                mock_sh.assert_called_once()

    def test_export_execution_config(self):
        """Test the export_execution_config method."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.additional_context_file = None
        mock_args.data_config_file_name = 'data.json'
        mock_args.force_mirror_local = False
        mock_args.live_output = True

        with patch('os.path.exists', return_value=False):
            orchestrator = DistributedOrchestrator(mock_args)

        # Mock models data
        test_models = [
            {"name": "model1", "cred": "test_cred"},
            {"name": "model2", "cred": ""}
        ]

        with patch('builtins.open', mock_open()) as mock_file:
            orchestrator.export_execution_config(test_models, "test_config.json")
            mock_file.assert_called_once_with("test_config.json", 'w')

    @patch('madengine.tools.distributed_orchestrator.create_ansible_playbook')
    def test_create_ansible_playbook_integration(self, mock_create_ansible):
        """Test create_ansible_playbook function call."""
        from madengine.tools.distributed_orchestrator import create_ansible_playbook
        
        create_ansible_playbook(
            manifest_file="test_manifest.json",
            execution_config="test_config.json",
            playbook_file="test_playbook.yml"
        )
        
        mock_create_ansible.assert_called_once_with(
            manifest_file="test_manifest.json",
            execution_config="test_config.json",
            playbook_file="test_playbook.yml"
        )

    @patch('madengine.tools.distributed_orchestrator.create_kubernetes_manifests')
    def test_create_kubernetes_manifests_integration(self, mock_create_k8s):
        """Test create_kubernetes_manifests function call."""
        from madengine.tools.distributed_orchestrator import create_kubernetes_manifests
        
        create_kubernetes_manifests(
            manifest_file="test_manifest.json",
            execution_config="test_config.json",
            namespace="test-namespace"
        )
        
        mock_create_k8s.assert_called_once_with(
            manifest_file="test_manifest.json",
            execution_config="test_config.json",
            namespace="test-namespace"
        )
