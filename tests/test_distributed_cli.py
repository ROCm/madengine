"""Test the distributed CLI module.

This module tests the distributed command-line interface functionality.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import os
import sys
import json
import tempfile
import subprocess
import unittest.mock
from unittest.mock import patch, MagicMock
# third-party modules
import pytest
# project modules
from madengine.tools import distributed_cli
from madengine.tools.distributed_orchestrator import DistributedOrchestrator
from .fixtures.utils import BASE_DIR, MODEL_DIR


class TestDistributedCLI:
    """Test the distributed CLI module."""

    def test_distributed_cli_help(self):
        """Test the distributed CLI --help command."""
        script_path = os.path.join(BASE_DIR, "src/madengine/tools", "distributed_cli.py")
        result = subprocess.run([sys.executable, script_path, "--help"], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert result.returncode == 0
        assert b"MADEngine Distributed" in result.stdout

    def test_build_command_help(self):
        """Test the build command --help."""
        script_path = os.path.join(BASE_DIR, "src/madengine/tools", "distributed_cli.py")
        result = subprocess.run([sys.executable, script_path, "build", "--help"], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert result.returncode == 0
        assert b"build" in result.stdout

    def test_run_command_help(self):
        """Test the run command --help."""
        script_path = os.path.join(BASE_DIR, "src/madengine/tools", "distributed_cli.py")
        result = subprocess.run([sys.executable, script_path, "run", "--help"], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert result.returncode == 0
        assert b"run" in result.stdout

    def test_generate_command_help(self):
        """Test the generate command --help."""
        script_path = os.path.join(BASE_DIR, "src/madengine/tools", "distributed_cli.py")
        result = subprocess.run([sys.executable, script_path, "generate", "--help"], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert result.returncode == 0
        assert b"generate" in result.stdout

    @patch('madengine.tools.distributed_cli.DistributedOrchestrator')
    def test_build_models_function(self, mock_orchestrator):
        """Test the build_models function."""
        # Mock args
        mock_args = MagicMock()
        mock_args.registry = "localhost:5000"
        mock_args.clean_docker_cache = True
        mock_args.manifest_output = "test_manifest.json"
        mock_args.summary_output = "test_summary.json"

        # Mock orchestrator instance and build phase
        mock_instance = MagicMock()
        mock_orchestrator.return_value = mock_instance
        mock_instance.build_phase.return_value = {
            "successful_builds": ["model1", "model2"],
            "failed_builds": []
        }

        # Test build command
        result = distributed_cli.build_models(mock_args)
        
        # Verify orchestrator was called correctly
        mock_orchestrator.assert_called_once_with(mock_args)
        mock_instance.build_phase.assert_called_once_with(
            registry="localhost:5000",
            clean_cache=True,
            manifest_output="test_manifest.json"
        )
        
        # Should return EXIT_SUCCESS for successful builds
        assert result == distributed_cli.EXIT_SUCCESS

    @patch('madengine.tools.distributed_cli.DistributedOrchestrator')
    def test_build_models_with_failures(self, mock_orchestrator):
        """Test the build_models function with build failures."""
        mock_args = MagicMock()
        mock_args.registry = None
        mock_args.clean_docker_cache = False
        mock_args.manifest_output = "manifest.json"
        mock_args.summary_output = None

        mock_instance = MagicMock()
        mock_orchestrator.return_value = mock_instance
        mock_instance.build_phase.return_value = {
            "successful_builds": ["model1"],
            "failed_builds": ["model2"]
        }

        result = distributed_cli.build_models(mock_args)
        
        # Should return EXIT_BUILD_FAILURE due to failures
        assert result == distributed_cli.EXIT_BUILD_FAILURE

    @patch('madengine.tools.distributed_cli.DistributedOrchestrator')
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

    @patch('madengine.tools.distributed_cli.DistributedOrchestrator')
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

    @patch('madengine.tools.distributed_cli.create_ansible_playbook')
    def test_generate_ansible_function(self, mock_create_ansible):
        """Test the generate_ansible function."""
        mock_args = MagicMock()
        mock_args.manifest_file = "manifest.json"
        mock_args.execution_config = "config.json"
        mock_args.output = "playbook.yml"

        result = distributed_cli.generate_ansible(mock_args)
        
        mock_create_ansible.assert_called_once_with(
            manifest_file="manifest.json",
            execution_config="config.json",
            playbook_file="playbook.yml"
        )
        
        assert result == distributed_cli.EXIT_SUCCESS

    @patch('madengine.tools.distributed_cli.create_kubernetes_manifests')
    def test_generate_k8s_function(self, mock_create_k8s):
        """Test the generate_k8s function."""
        mock_args = MagicMock()
        mock_args.manifest_file = "manifest.json"
        mock_args.execution_config = "config.json"
        mock_args.namespace = "madengine-test"

        result = distributed_cli.generate_k8s(mock_args)
        
        mock_create_k8s.assert_called_once_with(
            manifest_file="manifest.json",
            execution_config="config.json",
            namespace="madengine-test"
        )
        
        assert result == distributed_cli.EXIT_SUCCESS

    @patch('madengine.tools.distributed_cli.DistributedOrchestrator')
    @patch('madengine.tools.discover_models.DiscoverModels')
    def test_export_config_function(self, mock_discover_models, mock_orchestrator):
        """Test the export_config function."""
        mock_args = MagicMock()
        mock_args.output = "config.json"

        # Mock DiscoverModels to return a list of models
        mock_discover_instance = MagicMock()
        mock_discover_models.return_value = mock_discover_instance
        mock_discover_instance.run.return_value = ["model1", "model2"]

        mock_instance = MagicMock()
        mock_orchestrator.return_value = mock_instance
        mock_instance.export_execution_config.return_value = True

        result = distributed_cli.export_config(mock_args)
        
        mock_orchestrator.assert_called_once_with(mock_args)
        mock_discover_models.assert_called_once_with(args=mock_args)
        mock_discover_instance.run.assert_called_once()
        mock_instance.export_execution_config.assert_called_once_with(["model1", "model2"], "config.json")
        assert result == distributed_cli.EXIT_SUCCESS

    @patch('madengine.tools.distributed_cli.DistributedOrchestrator')
    @patch('madengine.tools.discover_models.DiscoverModels')
    def test_export_config_function_no_models(self, mock_discover_models, mock_orchestrator):
        """Test the export_config function when no models are discovered."""
        mock_args = MagicMock()
        mock_args.output = "config.json"

        # Mock DiscoverModels to return an empty list
        mock_discover_instance = MagicMock()
        mock_discover_models.return_value = mock_discover_instance
        mock_discover_instance.run.return_value = []

        mock_instance = MagicMock()
        mock_orchestrator.return_value = mock_instance
        mock_instance.export_execution_config.return_value = True

        result = distributed_cli.export_config(mock_args)
        
        mock_orchestrator.assert_called_once_with(mock_args)
        mock_discover_models.assert_called_once_with(args=mock_args)
        mock_discover_instance.run.assert_called_once()
        mock_instance.export_execution_config.assert_called_once_with([], "config.json")
        assert result == distributed_cli.EXIT_SUCCESS

    @patch('madengine.tools.distributed_cli.DistributedOrchestrator')
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

    @patch('madengine.tools.distributed_cli.DistributedOrchestrator')
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

    @patch('madengine.tools.distributed_cli.DistributedOrchestrator')
    def test_run_models_invalid_timeout(self, mock_orchestrator):
        """Test the run_models function with invalid timeout."""
        mock_args = MagicMock()
        mock_args.timeout = -5  # Invalid timeout
        mock_args.manifest_file = None

        result = distributed_cli.run_models(mock_args)
        
        # Should return EXIT_INVALID_ARGS without calling orchestrator
        assert result == distributed_cli.EXIT_INVALID_ARGS
        mock_orchestrator.assert_not_called()
