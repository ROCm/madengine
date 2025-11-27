"""Test the Docker builder module.

This module tests the Docker image building functionality for distributed execution.

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
from madengine.tools.docker_builder import DockerBuilder
from madengine.core.context import Context
from madengine.core.console import Console
from .fixtures.utils import BASE_DIR, MODEL_DIR


class TestDockerBuilder:
    """Test the Docker builder module."""

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    def test_docker_builder_initialization(
        self, mock_render, mock_docker_gpu, mock_hip, mock_arch, mock_ngpus, mock_vendor
    ):
        """Test DockerBuilder initialization."""
        context = Context()
        console = Console()

        builder = DockerBuilder(context, console)

        assert builder.context == context
        assert builder.console == console
        assert builder.built_images == {}

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    def test_docker_builder_initialization_without_console(
        self, mock_render, mock_docker_gpu, mock_hip, mock_arch, mock_ngpus, mock_vendor
    ):
        """Test DockerBuilder initialization without console."""
        context = Context()

        builder = DockerBuilder(context)

        assert builder.context == context
        assert isinstance(builder.console, Console)
        assert builder.built_images == {}

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    def test_get_context_path_with_dockercontext(
        self, mock_render, mock_docker_gpu, mock_hip, mock_arch, mock_ngpus, mock_vendor
    ):
        """Test get_context_path when dockercontext is specified."""
        context = Context()
        builder = DockerBuilder(context)

        info = {"dockercontext": "/custom/context"}
        result = builder.get_context_path(info)

        assert result == "/custom/context"

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    def test_get_context_path_without_dockercontext(
        self, mock_render, mock_docker_gpu, mock_hip, mock_arch, mock_ngpus, mock_vendor
    ):
        """Test get_context_path when dockercontext is not specified."""
        context = Context()
        builder = DockerBuilder(context)

        info = {}
        result = builder.get_context_path(info)

        assert result == "./docker"

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    def test_get_context_path_with_empty_dockercontext(
        self, mock_render, mock_docker_gpu, mock_hip, mock_arch, mock_ngpus, mock_vendor
    ):
        """Test get_context_path when dockercontext is empty."""
        context = Context()
        builder = DockerBuilder(context)

        info = {"dockercontext": ""}
        result = builder.get_context_path(info)

        assert result == "./docker"

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    def test_get_build_arg_no_args(
        self, mock_render, mock_docker_gpu, mock_hip, mock_arch, mock_ngpus, mock_vendor
    ):
        """Test get_build_arg with no additional runtime build arguments."""
        context = Context()
        builder = DockerBuilder(context)

        result = builder.get_build_arg()

        # Context automatically includes system GPU architecture
        assert "MAD_SYSTEM_GPU_ARCHITECTURE" in result
        assert "--build-arg" in result

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    def test_get_build_arg_with_context_args(
        self, mock_render, mock_docker_gpu, mock_hip, mock_arch, mock_ngpus, mock_vendor
    ):
        """Test get_build_arg with context build arguments."""
        context = Context()
        context.ctx = {"docker_build_arg": {"ARG1": "value1", "ARG2": "value2"}}
        builder = DockerBuilder(context)

        result = builder.get_build_arg()

        assert "--build-arg ARG1='value1'" in result
        assert "--build-arg ARG2='value2'" in result

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    def test_get_build_arg_with_run_args(
        self, mock_render, mock_docker_gpu, mock_hip, mock_arch, mock_ngpus, mock_vendor
    ):
        """Test get_build_arg with runtime build arguments."""
        context = Context()
        builder = DockerBuilder(context)

        run_build_arg = {"RUNTIME_ARG": "runtime_value"}
        result = builder.get_build_arg(run_build_arg)

        assert "--build-arg RUNTIME_ARG='runtime_value'" in result

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    def test_get_build_arg_with_both_args(
        self, mock_render, mock_docker_gpu, mock_hip, mock_arch, mock_ngpus, mock_vendor
    ):
        """Test get_build_arg with both context and runtime arguments."""
        context = Context()
        context.ctx = {"docker_build_arg": {"CONTEXT_ARG": "context_value"}}
        builder = DockerBuilder(context)

        run_build_arg = {"RUNTIME_ARG": "runtime_value"}
        result = builder.get_build_arg(run_build_arg)

        assert "--build-arg CONTEXT_ARG='context_value'" in result
        assert "--build-arg RUNTIME_ARG='runtime_value'" in result

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    @patch.object(Console, "sh")
    def test_build_image_success(
        self,
        mock_sh,
        mock_render,
        mock_docker_gpu,
        mock_hip,
        mock_arch,
        mock_ngpus,
        mock_vendor,
    ):
        """Test successful Docker image build."""
        context = Context()
        console = Console()
        builder = DockerBuilder(context, console)

        # Mock the console.sh calls
        mock_sh.return_value = "Build successful"

        model_info = {"name": "test/model", "dockercontext": "./docker"}
        dockerfile = "./docker/Dockerfile"

        with patch.object(builder, "get_build_arg", return_value=""):
            result = builder.build_image(model_info, dockerfile)

        # Verify the image name generation
        expected_image_name = "ci-test_model_Dockerfile"
        assert result["docker_image"] == expected_image_name
        assert "build_duration" in result

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    @patch.object(Console, "sh")
    def test_build_image_with_registry_push(
        self,
        mock_sh,
        mock_render,
        mock_docker_gpu,
        mock_hip,
        mock_arch,
        mock_ngpus,
        mock_vendor,
    ):
        """Test Docker image build with registry push."""
        context = Context()
        console = Console()
        builder = DockerBuilder(context, console)

        # Mock successful build and push
        mock_sh.return_value = "Success"

        model_info = {"name": "test_model"}
        dockerfile = "./docker/Dockerfile"
        registry = "localhost:5000"

        with patch.object(builder, "get_build_arg", return_value=""):
            with patch.object(builder, "get_context_path", return_value="./docker"):
                with patch.object(
                    builder, "push_image", return_value="localhost:5000/ci-test_model"
                ) as mock_push:
                    result = builder.build_image(model_info, dockerfile)
                    registry_image = builder.push_image(
                        result["docker_image"], registry
                    )

        # Should have called docker build
        build_calls = [
            call for call in mock_sh.call_args_list if "docker build" in str(call)
        ]
        assert len(build_calls) >= 1
        assert registry_image == "localhost:5000/ci-test_model"

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    @patch.object(Console, "sh")
    def test_build_image_failure(
        self,
        mock_sh,
        mock_render,
        mock_docker_gpu,
        mock_hip,
        mock_arch,
        mock_ngpus,
        mock_vendor,
    ):
        """Test Docker image build failure."""
        context = Context()
        console = Console()
        builder = DockerBuilder(context, console)

        # Mock build failure
        mock_sh.side_effect = RuntimeError("Build failed")

        model_info = {"name": "test_model"}
        dockerfile = "./docker/Dockerfile"

        with patch.object(builder, "get_build_arg", return_value=""):
            with patch.object(builder, "get_context_path", return_value="./docker"):
                # Test that the exception is raised
                with pytest.raises(RuntimeError, match="Build failed"):
                    builder.build_image(model_info, dockerfile)

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    def test_build_all_models(
        self, mock_render, mock_docker_gpu, mock_hip, mock_arch, mock_ngpus, mock_vendor
    ):
        """Test building all models."""
        context = Context()
        builder = DockerBuilder(context)

        models = [
            {"name": "model1", "dockerfile": "./docker/Dockerfile1"},
            {"name": "model2", "dockerfile": "./docker/Dockerfile2"},
        ]

        # Mock console.sh calls for dockerfile listing
        def mock_sh_side_effect(command, **kwargs):
            if "ls ./docker/Dockerfile1.*" in command:
                return "./docker/Dockerfile1"
            elif "ls ./docker/Dockerfile2.*" in command:
                return "./docker/Dockerfile2"
            elif "head -n5" in command:
                return "# CONTEXT AMD"
            else:
                return "success"

        # Mock context filter to return only the specific dockerfile for each model
        def mock_filter_side_effect(dockerfiles):
            # Return only the dockerfile that was requested for each model
            if "./docker/Dockerfile1" in dockerfiles:
                return {"./docker/Dockerfile1": "AMD"}
            elif "./docker/Dockerfile2" in dockerfiles:
                return {"./docker/Dockerfile2": "AMD"}
            return dockerfiles

        # Mock successful builds
        with patch.object(builder.console, "sh", side_effect=mock_sh_side_effect):
            with patch.object(context, "filter", side_effect=mock_filter_side_effect):
                with patch.object(builder, "build_image") as mock_build:
                    mock_build.return_value = {
                        "docker_image": "test_image",
                        "build_duration": 30.0,
                    }

                    result = builder.build_all_models(models)

        assert len(result["successful_builds"]) == 2
        assert len(result["failed_builds"]) == 0
        assert mock_build.call_count == 2

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    def test_build_all_models_with_failures(
        self, mock_render, mock_docker_gpu, mock_hip, mock_arch, mock_ngpus, mock_vendor
    ):
        """Test building all models with some failures."""
        context = Context()
        builder = DockerBuilder(context)

        models = [
            {"name": "model1", "dockerfile": "./docker/Dockerfile1"},
            {"name": "model2", "dockerfile": "./docker/Dockerfile2"},
        ]

        # Mock console.sh calls for dockerfile listing
        def mock_sh_side_effect(command, **kwargs):
            if "ls ./docker/Dockerfile1.*" in command:
                return "./docker/Dockerfile1"
            elif "ls ./docker/Dockerfile2.*" in command:
                return "./docker/Dockerfile2"
            elif "head -n5" in command:
                return "# CONTEXT AMD"
            else:
                return "success"

        # Mock context filter to return only the specific dockerfile for each model
        def mock_filter_side_effect(dockerfiles):
            # Return only the dockerfile that was requested for each model
            if "./docker/Dockerfile1" in dockerfiles:
                return {"./docker/Dockerfile1": "AMD"}
            elif "./docker/Dockerfile2" in dockerfiles:
                return {"./docker/Dockerfile2": "AMD"}
            return dockerfiles

        # Mock one success, one failure
        def mock_build_side_effect(model_info, dockerfile, *args, **kwargs):
            if model_info["name"] == "model1" and "Dockerfile1" in dockerfile:
                return {"docker_image": "model1_image", "build_duration": 30.0}
            else:
                raise RuntimeError("Build failed")

        with patch.object(builder.console, "sh", side_effect=mock_sh_side_effect):
            with patch.object(context, "filter", side_effect=mock_filter_side_effect):
                with patch.object(
                    builder, "build_image", side_effect=mock_build_side_effect
                ):
                    result = builder.build_all_models(models)

        assert len(result["successful_builds"]) == 1
        assert len(result["failed_builds"]) == 1  # 1 failure: model2/Dockerfile2

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    def test_export_build_manifest(
        self, mock_render, mock_docker_gpu, mock_hip, mock_arch, mock_ngpus, mock_vendor
    ):
        """Test exporting build manifest."""
        context = Context()
        builder = DockerBuilder(context)

        # Set up some built images (key should match real DockerBuilder output)
        builder.built_images = {
            "ci-model1": {"docker_image": "ci-model1", "dockerfile": "./docker/Dockerfile"}
        }

        with patch("builtins.open", mock_open()) as mock_file:
            with patch("json.dump") as mock_json_dump:
                builder.export_build_manifest("manifest.json")

        # Verify file was opened and JSON was written
        mock_file.assert_called_once_with("manifest.json", "w")
        mock_json_dump.assert_called_once()

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    @patch.object(Console, "sh")
    def test_build_image_with_credentials(
        self,
        mock_sh,
        mock_render,
        mock_docker_gpu,
        mock_hip,
        mock_arch,
        mock_ngpus,
        mock_vendor,
    ):
        """Test Docker image build with credentials."""
        context = Context()
        builder = DockerBuilder(context)

        mock_sh.return_value = "Success"

        model_info = {"name": "test_model", "cred": "testcred"}
        dockerfile = "./docker/Dockerfile"
        credentials = {"testcred": {"username": "testuser", "password": "testpass"}}

        with patch.object(builder, "get_build_arg") as mock_get_build_arg:
            with patch.object(builder, "get_context_path", return_value="./docker"):
                result = builder.build_image(
                    model_info, dockerfile, credentials=credentials
                )

        # Verify credentials were passed to build args
        mock_get_build_arg.assert_called_once()
        call_args = mock_get_build_arg.call_args[0][0]
        assert "testcred_USERNAME" in call_args
        assert "testcred_PASSWORD" in call_args

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    def test_clean_cache_option(
        self, mock_render, mock_docker_gpu, mock_hip, mock_arch, mock_ngpus, mock_vendor
    ):
        """Test clean cache option in build."""
        context = Context()
        builder = DockerBuilder(context)

        model_info = {"name": "test_model"}
        dockerfile = "./docker/Dockerfile"

        with patch.object(builder.console, "sh") as mock_sh:
            with patch.object(builder, "get_build_arg", return_value=""):
                with patch.object(builder, "get_context_path", return_value="./docker"):
                    builder.build_image(model_info, dockerfile, clean_cache=True)

        # Verify --no-cache was used
        build_calls = [
            call for call in mock_sh.call_args_list if "docker build" in str(call)
        ]
        assert any("--no-cache" in str(call) for call in build_calls)

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    @patch.object(Console, "sh")
    def test_push_image_dockerhub_with_repository(
        self,
        mock_sh,
        mock_render,
        mock_docker_gpu,
        mock_hip,
        mock_arch,
        mock_ngpus,
        mock_vendor,
    ):
        """Test pushing image to DockerHub with repository specified in credentials."""
        context = Context()
        console = Console()
        builder = DockerBuilder(context, console)

        docker_image = "ci-dummy_dummy.ubuntu.amd"
        registry = "dockerhub"
        credentials = {
            "dockerhub": {
                "repository": "your-repository",
                "username": "your-dockerhub-username",
                "password": "your-dockerhub-password-or-token",
            }
        }

        # Mock successful operations
        mock_sh.return_value = "Success"

        result = builder.push_image(docker_image, registry, credentials)

        # Verify the correct tag and push commands were called
        expected_tag = "your-repository:ci-dummy_dummy.ubuntu.amd"
        tag_calls = [
            call for call in mock_sh.call_args_list if "docker tag" in str(call)
        ]
        push_calls = [
            call for call in mock_sh.call_args_list if "docker push" in str(call)
        ]

        assert len(tag_calls) == 1
        assert expected_tag in str(tag_calls[0])
        assert len(push_calls) == 1
        assert expected_tag in str(push_calls[0])
        assert result == expected_tag

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    @patch.object(Console, "sh")
    def test_push_image_local_registry_with_repository(
        self,
        mock_sh,
        mock_render,
        mock_docker_gpu,
        mock_hip,
        mock_arch,
        mock_ngpus,
        mock_vendor,
    ):
        """Test pushing image to local registry with repository specified in credentials."""
        context = Context()
        console = Console()
        builder = DockerBuilder(context, console)

        docker_image = "ci-dummy_dummy.ubuntu.amd"
        registry = "localhost:5000"
        credentials = {
            "localhost:5000": {
                "repository": "your-repository",
                "username": "your-local-registry-username",
                "password": "your-local-registry-password",
            }
        }

        # Mock successful operations
        mock_sh.return_value = "Success"

        result = builder.push_image(docker_image, registry, credentials)

        # Verify the correct tag and push commands were called
        expected_tag = "localhost:5000/your-repository:ci-dummy_dummy.ubuntu.amd"
        tag_calls = [
            call for call in mock_sh.call_args_list if "docker tag" in str(call)
        ]
        push_calls = [
            call for call in mock_sh.call_args_list if "docker push" in str(call)
        ]

        assert len(tag_calls) == 1
        assert expected_tag in str(tag_calls[0])
        assert len(push_calls) == 1
        assert expected_tag in str(push_calls[0])
        assert result == expected_tag

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    @patch.object(Console, "sh")
    def test_push_image_dockerhub_no_repository(
        self,
        mock_sh,
        mock_render,
        mock_docker_gpu,
        mock_hip,
        mock_arch,
        mock_ngpus,
        mock_vendor,
    ):
        """Test pushing image to DockerHub without repository specified in credentials."""
        context = Context()
        console = Console()
        builder = DockerBuilder(context, console)

        docker_image = "ci-dummy_dummy.ubuntu.amd"
        registry = "dockerhub"
        credentials = {
            "dockerhub": {
                "username": "your-dockerhub-username",
                "password": "your-dockerhub-password-or-token",
            }
        }

        # Mock successful operations
        mock_sh.return_value = "Success"

        result = builder.push_image(docker_image, registry, credentials)

        # DockerHub without repository should just use the image name (no tagging needed)
        push_calls = [
            call for call in mock_sh.call_args_list if "docker push" in str(call)
        ]
        assert len(push_calls) == 1
        assert docker_image in str(push_calls[0])
        assert result == docker_image

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    @patch.object(Console, "sh")
    def test_push_image_local_registry_no_repository(
        self,
        mock_sh,
        mock_render,
        mock_docker_gpu,
        mock_hip,
        mock_arch,
        mock_ngpus,
        mock_vendor,
    ):
        """Test pushing image to local registry without repository specified in credentials."""
        context = Context()
        console = Console()
        builder = DockerBuilder(context, console)

        docker_image = "ci-dummy_dummy.ubuntu.amd"
        registry = "localhost:5000"
        credentials = {
            "localhost:5000": {
                "username": "your-local-registry-username",
                "password": "your-local-registry-password",
            }
        }

        # Mock successful operations
        mock_sh.return_value = "Success"

        result = builder.push_image(docker_image, registry, credentials)

        # Should fallback to registry/imagename format
        expected_tag = "localhost:5000/ci-dummy_dummy.ubuntu.amd"
        tag_calls = [
            call for call in mock_sh.call_args_list if "docker tag" in str(call)
        ]
        push_calls = [
            call for call in mock_sh.call_args_list if "docker push" in str(call)
        ]

        assert len(tag_calls) == 1
        assert expected_tag in str(tag_calls[0])
        assert len(push_calls) == 1
        assert expected_tag in str(push_calls[0])
        assert result == expected_tag

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    @patch.object(Console, "sh")
    def test_push_image_no_registry(
        self,
        mock_sh,
        mock_render,
        mock_docker_gpu,
        mock_hip,
        mock_arch,
        mock_ngpus,
        mock_vendor,
    ):
        """Test pushing image with no registry specified."""
        context = Context()
        console = Console()
        builder = DockerBuilder(context, console)

        docker_image = "ci-dummy_dummy.ubuntu.amd"

        result = builder.push_image(docker_image)

        # Should not call docker tag or push commands and return the original image name
        docker_calls = [
            call
            for call in mock_sh.call_args_list
            if "docker tag" in str(call) or "docker push" in str(call)
        ]
        assert len(docker_calls) == 0
        assert result == docker_image

    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx908")
    @patch.object(Context, "get_system_hip_version", return_value="5.4")
    @patch.object(Context, "get_docker_gpus", return_value="all")
    @patch.object(Context, "get_gpu_renderD_nodes", return_value=["renderD128"])
    @patch.object(Console, "sh")
    def test_build_manifest_with_tagged_image(
        self,
        mock_sh,
        mock_render,
        mock_docker_gpu,
        mock_hip,
        mock_arch,
        mock_ngpus,
        mock_vendor,
    ):
        """Test that build manifest includes registry_image when pushing to registry."""
        import tempfile
        import os

        # Mock successful operations BEFORE creating Context
        # to avoid MagicMock objects being stored during initialization
        mock_sh.return_value = "Success"

        context = Context()
        console = Console()
        builder = DockerBuilder(context, console)

        model_info = {"name": "test_model"}
        dockerfile = "./docker/Dockerfile"
        registry = "localhost:5000"
        credentials = {
            "localhost:5000": {
                "repository": "test-repository",
                "username": "test-user",
                "password": "test-password",
            }
        }

        with patch.object(builder, "get_build_arg", return_value=""):
            with patch.object(builder, "get_context_path", return_value="./docker"):
                # Build image
                build_info = builder.build_image(model_info, dockerfile, credentials)
                local_image = build_info["docker_image"]

                # Push to registry
                registry_image = builder.push_image(local_image, registry, credentials)

                # Update built_images with tagged image (simulating what build_all_models does)
                if local_image in builder.built_images:
                    builder.built_images[local_image]["registry_image"] = registry_image

                # Export manifest to temporary file
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as tmp_file:
                    builder.export_build_manifest(tmp_file.name, registry)

                    # Read and verify the manifest
                    with open(tmp_file.name, "r") as f:
                        import json

                        manifest = json.load(f)

                    # Clean up
                    os.unlink(tmp_file.name)

        # Verify the manifest contains the tagged image
        assert local_image in manifest["built_images"]
        assert "registry_image" in manifest["built_images"][local_image]
        assert manifest["built_images"][local_image]["registry_image"] == registry_image
        assert manifest["built_images"][local_image]["registry"] == registry

        # Verify the tagged image format is correct
        expected_tagged_image = f"localhost:5000/test-repository:{local_image}"
        assert registry_image == expected_tagged_image
