"""
Multi-platform integration tests for madengine.

Tests the complete build and run workflows across AMD GPU, NVIDIA GPU, and CPU platforms.
These tests focus on integration and end-to-end flows rather than isolated unit tests.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import pytest

from madengine.execution.docker_builder import DockerBuilder
from madengine.execution.dockerfile_utils import (
    parse_dockerfile_gpu_variables,
    normalize_architecture_name,
    is_target_arch_compatible_with_variable,
    is_compilation_arch_compatible,
)
from madengine.orchestration.build_orchestrator import BuildOrchestrator
from madengine.orchestration.run_orchestrator import RunOrchestrator
from madengine.core.errors import BuildError


# ============================================================================
# Multi-Platform Build Tests
# ============================================================================

class TestMultiPlatformBuild:
    """Test build orchestration across different platforms."""

    @pytest.mark.unit
    @pytest.mark.parametrize("platform", ["amd", "nvidia", "cpu"])
    def test_build_initialization_all_platforms(
        self, platform, multi_platform_context, mock_build_args
    ):
        """Test that BuildOrchestrator initializes correctly on all platforms."""
        with patch(
            "madengine.orchestration.build_orchestrator.Context",
            return_value=multi_platform_context,
        ):
            with patch("os.path.exists", return_value=False):
                orchestrator = BuildOrchestrator(mock_build_args)
                
                assert orchestrator.args == mock_build_args
                assert orchestrator.context == multi_platform_context
                assert orchestrator.credentials is None

    @pytest.mark.unit
    @pytest.mark.amd
    def test_build_amd_gpu_architecture_detection(self, amd_gpu_context, mock_build_args):
        """Test AMD GPU architecture is correctly detected and used."""
        with patch(
            "madengine.orchestration.build_orchestrator.Context",
            return_value=amd_gpu_context,
        ):
            with patch("os.path.exists", return_value=False):
                orchestrator = BuildOrchestrator(mock_build_args)
                
                assert orchestrator.context.get_gpu_vendor() == "AMD"
                assert orchestrator.context.get_system_gpu_architecture() == "gfx90a"

    @pytest.mark.unit
    @pytest.mark.nvidia
    def test_build_nvidia_gpu_architecture_detection(
        self, nvidia_gpu_context, mock_build_args
    ):
        """Test NVIDIA GPU architecture is correctly detected and used."""
        with patch(
            "madengine.orchestration.build_orchestrator.Context",
            return_value=nvidia_gpu_context,
        ):
            with patch("os.path.exists", return_value=False):
                orchestrator = BuildOrchestrator(mock_build_args)
                
                assert orchestrator.context.get_gpu_vendor() == "NVIDIA"
                assert orchestrator.context.get_system_gpu_architecture() == "sm_90"

    @pytest.mark.unit
    @pytest.mark.cpu
    def test_build_cpu_only_mode(self, cpu_context, mock_build_args):
        """Test CPU-only build mode works correctly."""
        with patch(
            "madengine.orchestration.build_orchestrator.Context",
            return_value=cpu_context,
        ):
            with patch("os.path.exists", return_value=False):
                orchestrator = BuildOrchestrator(mock_build_args)
                
                assert orchestrator.context.get_gpu_vendor() == "NONE"
                assert orchestrator.context.get_system_ngpus() == 0


# ============================================================================
# Error Handling and Resilience Tests
# ============================================================================

class TestBuildResilience:
    """Test build resilience and error handling."""

    @pytest.mark.unit
    def test_partial_build_failure_saves_manifest(
        self, mock_build_args, amd_gpu_context, sample_build_summary_partial
    ):
        """Test that partial failures still save the manifest with successful builds."""
        with patch(
            "madengine.orchestration.build_orchestrator.Context",
            return_value=amd_gpu_context,
        ):
            with patch("os.path.exists", return_value=False):
                with patch("pathlib.Path.exists", return_value=False):
                    with patch(
                        "madengine.orchestration.build_orchestrator.DiscoverModels"
                    ) as mock_discover:
                        with patch(
                            "madengine.orchestration.build_orchestrator.DockerBuilder"
                        ) as mock_builder:
                            # Setup mocks
                            mock_discover_instance = MagicMock()
                            mock_discover_instance.run.return_value = [
                                {"name": "model1", "tags": ["test"]},
                                {"name": "model2", "tags": ["test"]},
                            ]
                            mock_discover.return_value = mock_discover_instance

                            mock_builder_instance = MagicMock()
                            mock_builder_instance.build_all_models.return_value = (
                                sample_build_summary_partial
                            )
                            mock_builder.return_value = mock_builder_instance

                            # Execute
                            orchestrator = BuildOrchestrator(mock_build_args)
                            manifest_file = orchestrator.execute()

                            # Verify manifest was saved despite partial failure
                            assert manifest_file == "build_manifest.json"
                            mock_builder_instance.export_build_manifest.assert_called_once()

                            # Verify successful builds are available
                            summary = mock_builder_instance.build_all_models.return_value
                            assert len(summary["successful_builds"]) == 1
                            assert len(summary["failed_builds"]) == 1

    @pytest.mark.unit
    def test_all_builds_fail_raises_error(
        self, mock_build_args, amd_gpu_context, sample_build_summary_all_failed
    ):
        """Test that when ALL builds fail, BuildError is raised."""
        with patch(
            "madengine.orchestration.build_orchestrator.Context",
            return_value=amd_gpu_context,
        ):
            with patch("os.path.exists", return_value=False):
                with patch("pathlib.Path.exists", return_value=False):
                    with patch(
                        "madengine.orchestration.build_orchestrator.DiscoverModels"
                    ) as mock_discover:
                        with patch(
                            "madengine.orchestration.build_orchestrator.DockerBuilder"
                        ) as mock_builder:
                            # Setup mocks
                            mock_discover_instance = MagicMock()
                            mock_discover_instance.run.return_value = [
                                {"name": "model1", "tags": ["test"]},
                                {"name": "model2", "tags": ["test"]},
                            ]
                            mock_discover.return_value = mock_discover_instance

                            mock_builder_instance = MagicMock()
                            mock_builder_instance.build_all_models.return_value = (
                                sample_build_summary_all_failed
                            )
                            mock_builder.return_value = mock_builder_instance

                            # Execute and expect error
                            orchestrator = BuildOrchestrator(mock_build_args)

                            with pytest.raises(BuildError, match="All builds failed"):
                                orchestrator.execute()

    @pytest.mark.unit
    def test_multi_model_build_continues_on_single_failure(
        self, mock_build_args, amd_gpu_context
    ):
        """Test that multi-model build continues when one model fails."""
        with patch(
            "madengine.orchestration.build_orchestrator.Context",
            return_value=amd_gpu_context,
        ):
            with patch("os.path.exists", return_value=False):
                with patch("pathlib.Path.exists", return_value=False):
                    with patch(
                        "madengine.orchestration.build_orchestrator.DiscoverModels"
                    ) as mock_discover:
                        with patch(
                            "madengine.orchestration.build_orchestrator.DockerBuilder"
                        ) as mock_builder:
                            # Setup mocks
                            mock_discover_instance = MagicMock()
                            mock_discover_instance.run.return_value = [
                                {"name": "model1", "tags": ["test"]},
                                {"name": "model2", "tags": ["test"]},
                                {"name": "model3", "tags": ["test"]},
                            ]
                            mock_discover.return_value = mock_discover_instance

                            mock_builder_instance = MagicMock()
                            # 2 successes, 1 failure
                            mock_builder_instance.build_all_models.return_value = {
                                "successful_builds": [
                                    {
                                        "model": "model1",
                                        "docker_image": "ci-model1",
                                    },
                                    {
                                        "model": "model3",
                                        "docker_image": "ci-model3",
                                    },
                                ],
                                "failed_builds": [
                                    {
                                        "model": "model2",
                                        "error": "Build failed",
                                    },
                                ],
                                "total_build_time": 20.0,
                            }
                            mock_builder.return_value = mock_builder_instance

                            # Execute - should not raise
                            orchestrator = BuildOrchestrator(mock_build_args)
                            manifest_file = orchestrator.execute()

                            # Verify manifest saved and both successes are there
                            assert manifest_file == "build_manifest.json"
                            mock_builder_instance.export_build_manifest.assert_called_once()


# ============================================================================
# Multi-Architecture Build Tests
# ============================================================================

class TestMultiArchitectureBuild:
    """Test multi-architecture build scenarios."""

    @pytest.mark.unit
    @pytest.mark.amd
    def test_multi_arch_amd_builds(self, mock_build_args, amd_gpu_context):
        """Test building for multiple AMD GPU architectures."""
        mock_build_args.target_archs = ["gfx908", "gfx90a", "gfx942"]

        with patch(
            "madengine.orchestration.build_orchestrator.Context",
            return_value=amd_gpu_context,
        ):
            with patch("os.path.exists", return_value=False):
                with patch("pathlib.Path.exists", return_value=False):
                    with patch(
                        "madengine.orchestration.build_orchestrator.DiscoverModels"
                    ) as mock_discover:
                        with patch(
                            "madengine.orchestration.build_orchestrator.DockerBuilder"
                        ) as mock_builder:
                            # Setup mocks
                            mock_discover_instance = MagicMock()
                            mock_discover_instance.run.return_value = [
                                {"name": "model1", "tags": ["test"]},
                            ]
                            mock_discover.return_value = mock_discover_instance

                            mock_builder_instance = MagicMock()
                            # Build for each architecture
                            mock_builder_instance.build_all_models.return_value = {
                                "successful_builds": [
                                    {
                                        "model": "model1",
                                        "docker_image": "ci-model1_gfx908",
                                        "gpu_architecture": "gfx908",
                                    },
                                    {
                                        "model": "model1",
                                        "docker_image": "ci-model1_gfx90a",
                                        "gpu_architecture": "gfx90a",
                                    },
                                    {
                                        "model": "model1",
                                        "docker_image": "ci-model1_gfx942",
                                        "gpu_architecture": "gfx942",
                                    },
                                ],
                                "failed_builds": [],
                                "total_build_time": 45.0,
                            }
                            mock_builder.return_value = mock_builder_instance

                            # Execute
                            orchestrator = BuildOrchestrator(mock_build_args)
                            manifest_file = orchestrator.execute()

                            # Verify all architectures were built
                            summary = mock_builder_instance.build_all_models.return_value
                            assert len(summary["successful_builds"]) == 3
                            archs = [
                                b["gpu_architecture"]
                                for b in summary["successful_builds"]
                            ]
                            assert "gfx908" in archs
                            assert "gfx90a" in archs
                            assert "gfx942" in archs


# ============================================================================
# Run Orchestrator Multi-Platform Tests
# ============================================================================

class TestMultiPlatformRun:
    """Test run orchestration across different platforms."""

    @pytest.mark.unit
    def test_run_with_manifest_local_execution(
        self, mock_run_args, amd_gpu_context, temp_manifest_file
    ):
        """Test local execution from manifest file."""
        mock_run_args.manifest_file = temp_manifest_file

        with patch("os.path.exists", return_value=True):
            with patch(
                "builtins.open",
                mock_open(
                    read_data=json.dumps(
                        {
                            "built_images": {"ci-model1": {"name": "model1"}},
                            "deployment_config": {},
                        }
                    )
                ),
            ):
                orchestrator = RunOrchestrator(mock_run_args)

                with patch.object(
                    orchestrator,
                    "_execute_local",
                    return_value={"successful_runs": 1, "failed_runs": 0},
                ) as mock_execute_local:
                    result = orchestrator.execute(manifest_file=temp_manifest_file)

                    assert result["successful_runs"] == 1
                    mock_execute_local.assert_called_once()

    @pytest.mark.unit
    def test_run_multi_model_continues_on_failure(
        self, mock_run_args, amd_gpu_context, temp_manifest_file
    ):
        """Test that run continues when one model fails."""
        mock_run_args.manifest_file = temp_manifest_file

        with patch("os.path.exists", return_value=True):
            with patch(
                "builtins.open",
                mock_open(
                    read_data=json.dumps(
                        {
                            "built_images": {
                                "ci-model1": {"name": "model1"},
                                "ci-model2": {"name": "model2"},
                            },
                            "deployment_config": {},
                        }
                    )
                ),
            ):
                orchestrator = RunOrchestrator(mock_run_args)

                # Mock execution with 1 success, 1 failure
                with patch.object(
                    orchestrator,
                    "_execute_local",
                    return_value={
                        "successful_runs": [{"model": "model1"}],
                        "failed_runs": [{"model": "model2", "error": "Runtime error"}],
                        "total_runs": 2,
                    },
                ) as mock_execute_local:
                    result = orchestrator.execute(manifest_file=temp_manifest_file)

                    # Verify both were attempted
                    assert len(result["successful_runs"]) == 1
                    assert len(result["failed_runs"]) == 1
                    assert result["total_runs"] == 2


# ============================================================================
# Integration Tests (Full Flow)
# ============================================================================

class TestEndToEndIntegration:
    """Integration tests for complete build + run workflows."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_build_then_run_workflow(
        self, mock_build_args, mock_run_args, amd_gpu_context, temp_working_dir
    ):
        """Test complete workflow: build models, then run them."""
        # Phase 1: Build
        with patch(
            "madengine.orchestration.build_orchestrator.Context",
            return_value=amd_gpu_context,
        ):
            with patch("pathlib.Path.exists", return_value=False):
                with patch(
                    "madengine.orchestration.build_orchestrator.DiscoverModels"
                ) as mock_discover:
                    with patch(
                        "madengine.orchestration.build_orchestrator.DockerBuilder"
                    ) as mock_builder:
                        # Setup build mocks
                        mock_discover_instance = MagicMock()
                        mock_discover_instance.run.return_value = [
                            {"name": "model1", "tags": ["test"]},
                        ]
                        mock_discover.return_value = mock_discover_instance

                        mock_builder_instance = MagicMock()
                        mock_builder_instance.build_all_models.return_value = {
                            "successful_builds": [
                                {
                                    "model": "model1",
                                    "docker_image": "ci-model1",
                                },
                            ],
                            "failed_builds": [],
                            "total_build_time": 10.0,
                        }
                        mock_builder.return_value = mock_builder_instance

                        # Execute build
                        build_orchestrator = BuildOrchestrator(mock_build_args)
                        manifest_file = build_orchestrator.execute()

                        assert manifest_file == "build_manifest.json"
                        mock_builder_instance.export_build_manifest.assert_called_once()

        # Phase 2: Run (using manifest from build)
        manifest_data = {
            "built_images": {"ci-model1": {"docker_image": "ci-model1"}},
            "built_models": {"ci-model1": {"name": "model1"}},
            "deployment_config": {},
        }

        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=json.dumps(manifest_data))):
                run_orchestrator = RunOrchestrator(mock_run_args)

                with patch.object(
                    run_orchestrator,
                    "_execute_local",
                    return_value={
                        "successful_runs": [{"model": "model1"}],
                        "failed_runs": [],
                        "total_runs": 1,
                    },
                ):
                    result = run_orchestrator.execute(manifest_file="build_manifest.json")

                    assert len(result["successful_runs"]) == 1
                    assert len(result["failed_runs"]) == 0


# ============================================================================
# Platform-Specific Behavior Tests
# ============================================================================

class TestPlatformSpecificBehavior:
    """Test platform-specific behaviors and edge cases."""

    @pytest.mark.unit
    @pytest.mark.amd
    def test_amd_gpu_renderD_node_detection(self, amd_gpu_context, mock_run_args):
        """Test AMD GPU renderD node detection."""
        with patch(
            "madengine.orchestration.run_orchestrator.Context",
            return_value=amd_gpu_context,
        ):
            orchestrator = RunOrchestrator(mock_run_args)
            orchestrator._init_runtime_context()

            # Verify AMD-specific context
            assert orchestrator.context.get_gpu_vendor() == "AMD"
            assert orchestrator.context.get_gpu_renderD_nodes() == [
                "renderD128",
                "renderD129",
            ]

    @pytest.mark.unit
    @pytest.mark.nvidia
    def test_nvidia_gpu_cuda_detection(self, nvidia_gpu_context, mock_run_args):
        """Test NVIDIA GPU CUDA detection."""
        with patch(
            "madengine.orchestration.run_orchestrator.Context",
            return_value=nvidia_gpu_context,
        ):
            orchestrator = RunOrchestrator(mock_run_args)
            orchestrator._init_runtime_context()

            # Verify NVIDIA-specific context
            assert orchestrator.context.get_gpu_vendor() == "NVIDIA"
            assert orchestrator.context.get_system_cuda_version() == "12.1"

    @pytest.mark.unit
    @pytest.mark.cpu
    def test_cpu_only_execution(self, cpu_context, mock_run_args, temp_manifest_file):
        """Test CPU-only execution without GPU requirements."""
        with patch(
            "madengine.orchestration.run_orchestrator.Context", return_value=cpu_context
        ):
            with patch("os.path.exists", return_value=True):
                with patch(
                    "builtins.open",
                    mock_open(
                        read_data=json.dumps(
                            {
                                "built_images": {"ci-model1": {"name": "model1"}},
                                "deployment_config": {},
                            }
                        )
                    ),
                ):
                    orchestrator = RunOrchestrator(mock_run_args)

                    # CPU execution should not require GPU detection
                    with patch.object(
                        orchestrator,
                        "_execute_local",
                        return_value={
                            "successful_runs": [{"model": "model1"}],
                            "failed_runs": [],
                        },
                    ):
                        result = orchestrator.execute(manifest_file=temp_manifest_file)

                        assert len(result["successful_runs"]) == 1
                        # Context is initialized during execute, verify CPU mode
                        if orchestrator.context:
                            assert orchestrator.context.get_system_ngpus() == 0


# ============================================================================
# Multi-GPU architecture (Dockerfile parsing, normalization, image filtering)
# ============================================================================

class TestMultiGPUArch:
    """Multi-arch DockerBuilder logic, dockerfile_utils, and run-phase image filtering."""

    def setup_method(self):
        self.context = MagicMock()
        self.console = MagicMock()
        self.builder = DockerBuilder(self.context, self.console)
        mock_args = MagicMock()
        mock_args.additional_context = '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
        mock_args.additional_context_file = None
        mock_args.live_output = True
        mock_args.data_config_file_name = "data.json"
        mock_args.tags = []
        mock_args.target_archs = []
        mock_args.force_mirror_local = None
        self.orchestrator = BuildOrchestrator(mock_args)

    @patch.object(DockerBuilder, "_get_dockerfiles_for_model")
    @patch.object(DockerBuilder, "_check_dockerfile_has_gpu_variables")
    @patch.object(DockerBuilder, "build_image")
    def test_multi_arch_build_image_naming(self, mock_build_image, mock_check_gpu_vars, mock_get_dockerfiles):
        model_info = {"name": "dummy", "dockerfile": "docker/dummy.Dockerfile"}
        mock_get_dockerfiles.return_value = ["docker/dummy.Dockerfile"]
        mock_check_gpu_vars.return_value = (True, "docker/dummy.Dockerfile")
        mock_build_image.return_value = {"docker_image": "ci-dummy_dummy.ubuntu.amd_gfx908", "build_duration": 1.0}
        result = self.builder._build_model_for_arch(model_info, "gfx908", None, False, None, "", None)
        assert result[0]["docker_image"].endswith("_gfx908")
        mock_check_gpu_vars.return_value = (False, "docker/dummy.Dockerfile")
        mock_build_image.return_value = {"docker_image": "ci-dummy_dummy.ubuntu.amd", "build_duration": 1.0}
        result = self.builder._build_model_for_arch(model_info, "gfx908", None, False, None, "", None)
        assert not result[0]["docker_image"].endswith("_gfx908")

    @patch.object(DockerBuilder, "_get_dockerfiles_for_model")
    @patch.object(DockerBuilder, "_check_dockerfile_has_gpu_variables")
    @patch.object(DockerBuilder, "build_image")
    def test_multi_arch_manifest_fields(self, mock_build_image, mock_check_gpu_vars, mock_get_dockerfiles):
        model_info = {"name": "dummy", "dockerfile": "docker/dummy.Dockerfile"}
        mock_get_dockerfiles.return_value = ["docker/dummy.Dockerfile"]
        mock_check_gpu_vars.return_value = (True, "docker/dummy.Dockerfile")
        mock_build_image.return_value = {"docker_image": "ci-dummy_dummy.ubuntu.amd_gfx908", "build_duration": 1.0}
        result = self.builder._build_model_for_arch(model_info, "gfx908", None, False, None, "", None)
        assert result[0]["gpu_architecture"] == "gfx908"

    @patch.object(DockerBuilder, "_get_dockerfiles_for_model")
    @patch.object(DockerBuilder, "build_image")
    def test_legacy_single_arch_build(self, mock_build_image, mock_get_dockerfiles):
        model_info = {"name": "dummy", "dockerfile": "docker/dummy.Dockerfile"}
        mock_get_dockerfiles.return_value = ["docker/dummy.Dockerfile"]
        mock_build_image.return_value = {"docker_image": "ci-dummy_dummy.ubuntu.amd", "build_duration": 1.0}
        result = self.builder._build_model_single_arch(model_info, None, False, None, "", None)
        assert result[0]["docker_image"] == "ci-dummy_dummy.ubuntu.amd"

    @patch.object(DockerBuilder, "_build_model_single_arch")
    def test_additional_context_overrides_target_archs(self, mock_single_arch):
        self.context.ctx = {"docker_build_arg": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx908"}}
        model_info = {"name": "dummy", "dockerfile": "docker/dummy.Dockerfile"}
        mock_single_arch.return_value = [{"docker_image": "ci-dummy_dummy.ubuntu.amd", "build_duration": 1.0}]
        result = self.builder.build_all_models([model_info], target_archs=["gfx908", "gfx90a"])
        assert result["successful_builds"][0]["docker_image"] == "ci-dummy_dummy.ubuntu.amd"

    def test_parse_dockerfile_gpu_variables(self):
        content = """
        ARG MAD_SYSTEM_GPU_ARCHITECTURE=gfx908
        ENV PYTORCH_ROCM_ARCH=gfx908;gfx90a
        ARG GPU_TARGETS=gfx908,gfx942
        ENV GFX_COMPILATION_ARCH=gfx908
        ARG GPU_ARCHS=gfx908;gfx90a;gfx942
        """
        result = parse_dockerfile_gpu_variables(content)
        assert result["MAD_SYSTEM_GPU_ARCHITECTURE"] == ["gfx908"]
        assert result["PYTORCH_ROCM_ARCH"] == ["gfx908", "gfx90a"]
        assert result["GPU_TARGETS"] == ["gfx908", "gfx942"]
        assert result["GFX_COMPILATION_ARCH"] == ["gfx908"]
        assert result["GPU_ARCHS"] == ["gfx908", "gfx90a", "gfx942"]

    def test_parse_dockerfile_gpu_variables_env_delimiter(self):
        result = parse_dockerfile_gpu_variables("ENV PYTORCH_ROCM_ARCH = gfx908,gfx90a")
        assert result["PYTORCH_ROCM_ARCH"] == ["gfx908", "gfx90a"]

    def test_parse_malformed_dockerfile(self):
        result = parse_dockerfile_gpu_variables(
            "ENV BAD_LINE\nARG MAD_SYSTEM_GPU_ARCHITECTURE=\nENV PYTORCH_ROCM_ARCH=\n"
        )
        assert isinstance(result, dict)

    def test_normalize_architecture_name(self):
        cases = {
            "gfx908": "gfx908", "GFX908": "gfx908", "mi100": "gfx908", "mi-100": "gfx908",
            "mi200": "gfx90a", "mi-200": "gfx90a", "mi210": "gfx90a", "mi250": "gfx90a",
            "mi300": "gfx940", "mi-300": "gfx940", "mi300a": "gfx940",
            "mi300x": "gfx942", "mi-300x": "gfx942", "unknown": "unknown", "": None,
        }
        for inp, expected in cases.items():
            assert normalize_architecture_name(inp) == expected

    def test_is_target_arch_compatible_with_variable(self):
        assert is_target_arch_compatible_with_variable("MAD_SYSTEM_GPU_ARCHITECTURE", ["gfx908"], "gfx942")
        assert is_target_arch_compatible_with_variable("PYTORCH_ROCM_ARCH", ["gfx908", "gfx942"], "gfx942")
        assert not is_target_arch_compatible_with_variable("PYTORCH_ROCM_ARCH", ["gfx908"], "gfx942")
        assert is_target_arch_compatible_with_variable("GFX_COMPILATION_ARCH", ["gfx908"], "gfx908")
        assert not is_target_arch_compatible_with_variable("GFX_COMPILATION_ARCH", ["gfx908"], "gfx942")
        assert is_target_arch_compatible_with_variable("UNKNOWN_VAR", ["foo"], "bar")

    def test_is_compilation_arch_compatible(self):
        assert is_compilation_arch_compatible("gfx908", "gfx908")
        assert not is_compilation_arch_compatible("gfx908", "gfx942")
        assert is_compilation_arch_compatible("foo", "foo")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

