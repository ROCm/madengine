"""Integration tests for the distributed solution.

This module tests the complete distributed workflow including build and run phases.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import os
import json
import tempfile
import shutil
import unittest.mock
from unittest.mock import patch, MagicMock, mock_open
# third-party modules
import pytest
# project modules
from madengine.tools.distributed_orchestrator import DistributedOrchestrator
from madengine.tools.docker_builder import DockerBuilder
from madengine.tools.container_runner import ContainerRunner
from madengine.tools import distributed_cli
from .fixtures.utils import BASE_DIR, MODEL_DIR, clean_test_temp_files


class TestDistributedIntegration:
    """Integration tests for the distributed solution."""

    @pytest.mark.parametrize('clean_test_temp_files', [['test_manifest.json', 'test_summary.json']], indirect=True)
    def test_end_to_end_workflow_simulation(self, clean_test_temp_files):
        """Test complete end-to-end distributed workflow simulation."""
        # Mock args for orchestrator
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.additional_context_file = None
        mock_args.data_config_file_name = 'data.json'
        mock_args.force_mirror_local = False
        mock_args.live_output = True
        mock_args.tags = ['dummy_test']
        mock_args.models_config_file_name = 'models.json'

        # Test data
        test_models = [
            {
                "name": "test_model_1",
                "dockerfile": ["./docker/Dockerfile"],
                "dockercontext": "./docker"
            },
            {
                "name": "test_model_2", 
                "dockerfile": ["./docker/Dockerfile"],
                "dockercontext": "./docker"
            }
        ]

        # Mock manifest data with proper built_images structure
        test_manifest_for_run = {
            "built_images": {
                "ci-test_model_1_dockerfile": {
                    "docker_image": "ci-test_model_1_dockerfile",
                    "dockerfile": "./docker/Dockerfile",
                    "base_docker": "ubuntu:20.04",
                    "build_duration": 60.0,
                    "registry_image": "localhost:5000/ci-test_model_1:latest"
                },
                "ci-test_model_2_dockerfile": {
                    "docker_image": "ci-test_model_2_dockerfile", 
                    "dockerfile": "./docker/Dockerfile",
                    "base_docker": "ubuntu:20.04",
                    "build_duration": 60.5,
                    "registry_image": "localhost:5000/ci-test_model_2:latest"
                }
            },
            "context": {
                "docker_env_vars": {},
                "docker_mounts": {},
                "docker_build_arg": {}
            }
        }

        with patch('os.path.exists', return_value=False):
            orchestrator = DistributedOrchestrator(mock_args)

        # Mock all the dependencies
        with patch('madengine.tools.distributed_orchestrator.DiscoverModels') as mock_discover:
            with patch('madengine.tools.distributed_orchestrator.DockerBuilder') as mock_builder:
                with patch('madengine.tools.distributed_orchestrator.ContainerRunner') as mock_runner:
                    
                    # Setup discover models mock
                    mock_discover_instance = MagicMock()
                    mock_discover.return_value = mock_discover_instance
                    mock_discover_instance.run.return_value = test_models

                    # Setup docker builder mock
                    mock_builder_instance = MagicMock()
                    mock_builder.return_value = mock_builder_instance
                    mock_builder_instance.build_all_models.return_value = {
                        "successful_builds": ["test_model_1", "test_model_2"],
                        "failed_builds": [],
                        "total_build_time": 120.5
                    }
                    mock_builder_instance.get_build_manifest.return_value = test_manifest_for_run

                    # Setup container runner mock
                    mock_runner_instance = MagicMock()
                    mock_runner.return_value = mock_runner_instance
                    mock_runner_instance.load_build_manifest.return_value = test_manifest_for_run
                    
                    # Mock run_container to return proper dict structure
                    def mock_run_container(model_info, *args, **kwargs):
                        return {
                            "model": model_info["name"],
                            "status": "SUCCESS",
                            "test_duration": 30.0,
                            "performance": "100 fps",
                            "metric": "fps"
                        }
                    mock_runner_instance.run_container.side_effect = mock_run_container
                    
                    # Mock pull_image to return image name
                    mock_runner_instance.pull_image.return_value = "pulled_image_name"
                    
                    mock_runner_instance.run_all_containers.return_value = {
                        "successful_runs": ["test_model_1", "test_model_2"],
                        "failed_runs": []
                    }

                    # Mock script copying
                    with patch.object(orchestrator, '_copy_scripts'):
                        # Test build phase
                        build_result = orchestrator.build_phase(
                            registry="localhost:5000",
                            clean_cache=True,
                            manifest_output="test_manifest.json"
                        )
                        
                        # Verify build phase results
                        assert len(build_result["successful_builds"]) == 2
                        assert len(build_result["failed_builds"]) == 0

                        # Test run phase - mock file operations for manifest loading
                        with patch('os.path.exists', return_value=True):
                            with patch('builtins.open', mock_open(read_data=json.dumps(test_manifest_for_run))):
                                with patch('json.load', return_value=test_manifest_for_run):
                                    run_result = orchestrator.run_phase(
                                        manifest_file="test_manifest.json",
                                        registry="localhost:5000",
                                        timeout=1800
                                    )
                        
                        # Verify run phase results
                        assert len(run_result["successful_runs"]) == 2
                        assert len(run_result["failed_runs"]) == 0

                        # Test full workflow - mock file operations again
                        with patch('os.path.exists', return_value=True):
                            with patch('builtins.open', mock_open(read_data=json.dumps(test_manifest_for_run))):
                                with patch('json.load', return_value=test_manifest_for_run):
                                    full_result = orchestrator.full_workflow(
                                        registry="localhost:5000",
                                        clean_cache=True,
                                        timeout=3600
                                    )
                        
                        # Verify full workflow results
                        assert full_result["overall_success"] is True
                        assert "build_phase" in full_result
                        assert "run_phase" in full_result

    def test_cli_build_run_integration(self):
        """Test CLI build and run command integration."""
        # Mock args for build command
        build_args = MagicMock()
        build_args.registry = "localhost:5000"
        build_args.clean_docker_cache = True
        build_args.manifest_output = "integration_manifest.json"
        build_args.summary_output = "build_summary.json"
        build_args.additional_context = None
        build_args.additional_context_file = None
        build_args.data_config_file_name = 'data.json'
        build_args.force_mirror_local = False
        build_args.live_output = True

        # Mock args for run command
        run_args = MagicMock()
        run_args.manifest_file = "integration_manifest.json"
        run_args.registry = "localhost:5000"
        run_args.timeout = 1800
        run_args.keep_alive = False
        run_args.summary_output = "run_summary.json"
        run_args.additional_context = None
        run_args.additional_context_file = None
        run_args.data_config_file_name = 'data.json'
        run_args.force_mirror_local = False
        run_args.live_output = True

        with patch('madengine.tools.distributed_cli.DistributedOrchestrator') as mock_orchestrator:
            # Mock successful build
            mock_instance = MagicMock()
            mock_orchestrator.return_value = mock_instance
            mock_instance.build_phase.return_value = {
                "successful_builds": ["model1", "model2"],
                "failed_builds": []
            }
            
            with patch('builtins.open', mock_open()):
                with patch('json.dump'):
                    build_result = distributed_cli.build_models(build_args)
            
            assert build_result == distributed_cli.EXIT_SUCCESS

            # Mock successful run with existing manifest file
            mock_instance.run_phase.return_value = {
                "successful_runs": ["model1", "model2"],
                "failed_runs": []
            }
            
            with patch('os.path.exists', return_value=True):
                with patch('builtins.open', mock_open()):
                    with patch('json.dump'):
                        run_result = distributed_cli.run_models(run_args)
            
            assert run_result == distributed_cli.EXIT_SUCCESS

    def test_manifest_file_handling(self):
        """Test manifest file creation and loading."""
        # Test manifest data
        test_manifest = {
            "images": {
                "test_model": "localhost:5000/ci-test_model:latest"
            },
            "metadata": {
                "build_time": "2023-01-01T12:00:00Z",
                "registry": "localhost:5000"
            }
        }

        # Test DockerBuilder manifest export
        from madengine.core.context import Context
        context = Context()
        builder = DockerBuilder(context)
        builder.built_images = {
            "test_model": {
                "image_name": "ci-test_model",
                "registry_image": "localhost:5000/ci-test_model:latest"
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            temp_path = temp_file.name

        try:
            # Test export
            with patch('builtins.open', mock_open()) as mock_file:
                with patch('json.dump') as mock_json_dump:
                    builder.export_build_manifest(temp_path)
                    
                    # Verify file operations
                    mock_file.assert_called_once_with(temp_path, 'w')
                    mock_json_dump.assert_called_once()

            # Test ContainerRunner manifest loading
            runner = ContainerRunner()
            
            with patch('builtins.open', mock_open(read_data=json.dumps(test_manifest))):
                loaded_manifest = runner.load_build_manifest(temp_path)
                
                assert loaded_manifest == test_manifest
                assert "images" in loaded_manifest
                assert "test_model" in loaded_manifest["images"]

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_error_handling_integration(self):
        """Test error handling throughout the distributed workflow."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.additional_context_file = None
        mock_args.data_config_file_name = 'data.json'
        mock_args.force_mirror_local = False
        mock_args.live_output = True

        with patch('os.path.exists', return_value=False):
            orchestrator = DistributedOrchestrator(mock_args)

        # Test build phase with failures
        with patch('madengine.tools.distributed_orchestrator.DiscoverModels') as mock_discover:
            with patch('madengine.tools.distributed_orchestrator.DockerBuilder') as mock_builder:
                
                # Setup failing build
                mock_discover_instance = MagicMock()
                mock_discover.return_value = mock_discover_instance
                mock_discover_instance.run.return_value = [{"name": "failing_model"}]

                mock_builder_instance = MagicMock()
                mock_builder.return_value = mock_builder_instance
                mock_builder_instance.build_all_models.return_value = {
                    "successful_builds": [],
                    "failed_builds": ["failing_model"],
                    "total_build_time": 0.0
                }

                with patch.object(orchestrator, '_copy_scripts'):
                    result = orchestrator.build_phase()
                    
                    # Should handle build failures gracefully
                    assert len(result["failed_builds"]) == 1
                    assert len(result["successful_builds"]) == 0

        # Test run phase with missing manifest
        with patch('madengine.tools.distributed_orchestrator.ContainerRunner') as mock_runner:
            mock_runner_instance = MagicMock()
            mock_runner.return_value = mock_runner_instance
            mock_runner_instance.load_build_manifest.side_effect = FileNotFoundError("Manifest not found")

            with pytest.raises(FileNotFoundError):
                orchestrator.run_phase(manifest_file="nonexistent_manifest.json")

    def test_ansible_kubernetes_generation(self):
        """Test Ansible and Kubernetes manifest generation."""
        test_manifest = {
            "images": {"model1": "localhost:5000/model1:latest"},
            "metadata": {"registry": "localhost:5000"}
        }
        
        test_config = {
            "timeout": 3600,
            "gpu_requirements": {"model1": "1"}
        }

        # Test Ansible generation
        with patch('madengine.tools.distributed_cli.create_ansible_playbook') as mock_ansible:
            distributed_cli.generate_ansible(MagicMock(
                manifest_file="test_manifest.json",
                execution_config="test_config.json", 
                output="test_playbook.yml"
            ))
            
            mock_ansible.assert_called_once_with(
                manifest_file="test_manifest.json",
                execution_config="test_config.json",
                playbook_file="test_playbook.yml"
            )

        # Test Kubernetes generation
        with patch('madengine.tools.distributed_cli.create_kubernetes_manifests') as mock_k8s:
            distributed_cli.generate_k8s(MagicMock(
                manifest_file="test_manifest.json",
                execution_config="test_config.json",
                namespace="madengine-test"
            ))
            
            mock_k8s.assert_called_once_with(
                manifest_file="test_manifest.json",
                execution_config="test_config.json", 
                namespace="madengine-test"
            )

    def test_registry_integration(self):
        """Test registry push/pull integration."""
        from madengine.core.context import Context
        from madengine.core.console import Console
        
        # Mock the Context to avoid hardware-specific initialization issues
        with patch('madengine.core.context.Context.get_gpu_renderD_nodes', return_value=[]):
            context = Context()
        console = Console()
        
        # Test DockerBuilder with registry
        builder = DockerBuilder(context, console)
        
        model_info = {"name": "test_model"}
        dockerfile = "./docker/Dockerfile"
        registry = "localhost:5000"

        with patch.object(console, 'sh') as mock_sh:
            with patch.object(builder, 'get_build_arg', return_value=""):
                with patch.object(builder, 'get_context_path', return_value="./docker"):
                    mock_sh.return_value = "Success"
                    
                    # Test build image (without registry)
                    build_result = builder.build_image(model_info, dockerfile)
                    
                    # Test push to registry
                    registry_image = builder.push_image(build_result["docker_image"], registry)
                    
                    # Should have built and pushed to registry
                    build_calls = [call for call in mock_sh.call_args_list if 'docker build' in str(call)]
                    push_calls = [call for call in mock_sh.call_args_list if 'docker push' in str(call)]
                    
                    assert len(build_calls) >= 1
                    assert len(push_calls) >= 1
                    assert registry_image == f"{registry}/{build_result['docker_image']}"

        # Test ContainerRunner with registry pull
        runner = ContainerRunner(context)
        
        with patch.object(runner.console, 'sh') as mock_sh:
            mock_sh.return_value = "Pull successful"
            
            result = runner.pull_image("localhost:5000/test:latest", "local-test")
            
            assert result == "local-test"
            expected_calls = [
                unittest.mock.call("docker pull localhost:5000/test:latest"),
                unittest.mock.call("docker tag localhost:5000/test:latest local-test")
            ]
            mock_sh.assert_has_calls(expected_calls)

    def test_smart_run_command_integration(self):
        """Test the smart run command in both execution-only and complete workflow modes."""
        # Test execution-only mode (manifest file exists)
        run_args_execution_only = MagicMock()
        run_args_execution_only.manifest_file = "existing_manifest.json"
        run_args_execution_only.registry = "localhost:5000"
        run_args_execution_only.timeout = 1800
        run_args_execution_only.keep_alive = False
        run_args_execution_only.summary_output = None
        run_args_execution_only.additional_context = None
        run_args_execution_only.additional_context_file = None
        run_args_execution_only.data_config_file_name = 'data.json'
        run_args_execution_only.force_mirror_local = False
        run_args_execution_only.live_output = True

        with patch('madengine.tools.distributed_cli.DistributedOrchestrator') as mock_orchestrator:
            with patch('os.path.exists', return_value=True):  # Manifest exists
                mock_instance = MagicMock()
                mock_orchestrator.return_value = mock_instance
                mock_instance.run_phase.return_value = {
                    "successful_runs": ["model1"],
                    "failed_runs": []
                }
                
                with patch('builtins.open', mock_open()):
                    with patch('json.dump'):
                        result = distributed_cli.run_models(run_args_execution_only)
                
                assert result == distributed_cli.EXIT_SUCCESS
                # Only run phase should be called, not build phase
                mock_instance.run_phase.assert_called_once()
                mock_instance.build_phase.assert_not_called()

        # Test complete workflow mode (manifest file doesn't exist)
        run_args_complete = MagicMock()
        run_args_complete.manifest_file = None
        run_args_complete.registry = "localhost:5000" 
        run_args_complete.timeout = 1800
        run_args_complete.keep_alive = False
        run_args_complete.summary_output = None
        run_args_complete.manifest_output = "build_manifest.json"
        run_args_complete.additional_context = None
        run_args_complete.additional_context_file = None
        run_args_complete.data_config_file_name = 'data.json'
        run_args_complete.force_mirror_local = False
        run_args_complete.live_output = True

        with patch('madengine.tools.distributed_cli.DistributedOrchestrator') as mock_orchestrator:
            with patch('os.path.exists', return_value=False):  # Manifest doesn't exist
                mock_instance = MagicMock()
                mock_orchestrator.return_value = mock_instance
                mock_instance.build_phase.return_value = {
                    "successful_builds": ["model1"],
                    "failed_builds": []
                }
                mock_instance.run_phase.return_value = {
                    "successful_runs": ["model1"],
                    "failed_runs": []
                }
                
                with patch('builtins.open', mock_open()):
                    with patch('json.dump'):
                        result = distributed_cli.run_models(run_args_complete)
                
                assert result == distributed_cli.EXIT_SUCCESS
                # Both build and run phases should be called
                mock_instance.build_phase.assert_called_once()
                mock_instance.run_phase.assert_called_once()
