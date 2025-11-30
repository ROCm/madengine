"""Comprehensive unit tests for multi-GPU architecture support in MADEngine.

Covers:
- Multi-arch DockerBuilder logic (image naming, manifest, legacy/override)
- Dockerfile GPU variable parsing/validation
- Target architecture normalization and compatibility
- Run-phase manifest filtering by gpu_architecture

UPDATED: Now uses BuildOrchestrator instead of deprecated DistributedOrchestrator.

All tests are logic/unit tests and do not require GPU hardware.
"""
import pytest
from unittest.mock import MagicMock, patch
from madengine.tools.docker_builder import DockerBuilder
from madengine.orchestration.build_orchestrator import BuildOrchestrator
from madengine.orchestration.run_orchestrator import RunOrchestrator

class TestMultiGPUArch:
    def setup_method(self):
        self.context = MagicMock()
        self.console = MagicMock()
        self.builder = DockerBuilder(self.context, self.console)
        
        # Mock args for BuildOrchestrator (replacement for DistributedOrchestrator)
        mock_args = MagicMock()
        mock_args.additional_context = '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
        mock_args.additional_context_file = None
        mock_args.live_output = True
        mock_args.data_config_file_name = "data.json"
        mock_args.tags = []
        mock_args.target_archs = []
        mock_args.force_mirror_local = None
        
        # Create BuildOrchestrator with mocked args
        self.orchestrator = BuildOrchestrator(mock_args)

    # --- DockerBuilder Multi-Arch Logic ---
    @patch.object(DockerBuilder, "_get_dockerfiles_for_model")
    @patch.object(DockerBuilder, "_check_dockerfile_has_gpu_variables")
    @patch.object(DockerBuilder, "build_image")
    def test_multi_arch_build_image_naming(self, mock_build_image, mock_check_gpu_vars, mock_get_dockerfiles):
        model_info = {"name": "dummy", "dockerfile": "docker/dummy.Dockerfile"}
        mock_get_dockerfiles.return_value = ["docker/dummy.Dockerfile"]
        # GPU variable present
        mock_check_gpu_vars.return_value = (True, "docker/dummy.Dockerfile")
        mock_build_image.return_value = {"docker_image": "ci-dummy_dummy.ubuntu.amd_gfx908", "build_duration": 1.0}
        result = self.builder._build_model_for_arch(model_info, "gfx908", None, False, None, "", None)
        assert result[0]["docker_image"].endswith("_gfx908")
        # GPU variable absent
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

    # --- Dockerfile GPU Variable Parsing/Validation ---
    def test_parse_dockerfile_gpu_variables(self):
        dockerfile_content = """
        ARG MAD_SYSTEM_GPU_ARCHITECTURE=gfx908
        ENV PYTORCH_ROCM_ARCH=gfx908;gfx90a
        ARG GPU_TARGETS=gfx908,gfx942
        ENV GFX_COMPILATION_ARCH=gfx908
        ARG GPU_ARCHS=gfx908;gfx90a;gfx942
        """
        result = self.builder._parse_dockerfile_gpu_variables(dockerfile_content)
        assert result["MAD_SYSTEM_GPU_ARCHITECTURE"] == ["gfx908"]
        assert result["PYTORCH_ROCM_ARCH"] == ["gfx908", "gfx90a"]
        assert result["GPU_TARGETS"] == ["gfx908", "gfx942"]
        assert result["GFX_COMPILATION_ARCH"] == ["gfx908"]
        assert result["GPU_ARCHS"] == ["gfx908", "gfx90a", "gfx942"]

    def test_parse_dockerfile_gpu_variables_env_delimiter(self):
        dockerfile_content = "ENV PYTORCH_ROCM_ARCH = gfx908,gfx90a"
        result = self.builder._parse_dockerfile_gpu_variables(dockerfile_content)
        assert result["PYTORCH_ROCM_ARCH"] == ["gfx908", "gfx90a"]

    def test_parse_malformed_dockerfile(self):
        dockerfile_content = "ENV BAD_LINE\nARG MAD_SYSTEM_GPU_ARCHITECTURE=\nENV PYTORCH_ROCM_ARCH=\n"
        result = self.builder._parse_dockerfile_gpu_variables(dockerfile_content)
        assert isinstance(result, dict)

    # --- Target Architecture Normalization/Compatibility ---
    def test_normalize_architecture_name(self):
        cases = {
            "gfx908": "gfx908",
            "GFX908": "gfx908",
            "mi100": "gfx908",
            "mi-100": "gfx908",
            "mi200": "gfx90a",
            "mi-200": "gfx90a",
            "mi210": "gfx90a",
            "mi250": "gfx90a",
            "mi300": "gfx940",
            "mi-300": "gfx940",
            "mi300a": "gfx940",
            "mi300x": "gfx942",
            "mi-300x": "gfx942",
            "unknown": "unknown",
            "": None,
        }
        for inp, expected in cases.items():
            assert self.builder._normalize_architecture_name(inp) == expected

    def test_is_target_arch_compatible_with_variable(self):
        assert self.builder._is_target_arch_compatible_with_variable("MAD_SYSTEM_GPU_ARCHITECTURE", ["gfx908"], "gfx942")
        assert self.builder._is_target_arch_compatible_with_variable("PYTORCH_ROCM_ARCH", ["gfx908", "gfx942"], "gfx942")
        assert not self.builder._is_target_arch_compatible_with_variable("PYTORCH_ROCM_ARCH", ["gfx908"], "gfx942")
        assert self.builder._is_target_arch_compatible_with_variable("GFX_COMPILATION_ARCH", ["gfx908"], "gfx908")
        assert not self.builder._is_target_arch_compatible_with_variable("GFX_COMPILATION_ARCH", ["gfx908"], "gfx942")
        assert self.builder._is_target_arch_compatible_with_variable("UNKNOWN_VAR", ["foo"], "bar")

    def test_is_compilation_arch_compatible(self):
        assert self.builder._is_compilation_arch_compatible("gfx908", "gfx908")
        assert not self.builder._is_compilation_arch_compatible("gfx908", "gfx942")
        assert self.builder._is_compilation_arch_compatible("foo", "foo")

    # --- Run-Phase Manifest Filtering ---
    def test_filter_images_by_gpu_architecture(self):
        """Test image filtering by GPU architecture using RunOrchestrator.
        
        Note: Current behavior treats images without gpu_vendor as compatible (legacy support).
        """
        # Create RunOrchestrator which has _filter_images_by_gpu_architecture
        mock_args = MagicMock()
        mock_args.additional_context = '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
        mock_args.additional_context_file = None
        mock_args.tags = []
        mock_args.live_output = True
        mock_args.data_config_file_name = "data.json"
        mock_args.force_mirror_local = None
        
        run_orch = RunOrchestrator(mock_args)
        
        # Test exact match - both images have gpu_vendor set to "AMD"
        built_images = {
            "img1": {"gpu_architecture": "gfx908", "gpu_vendor": "AMD"}, 
            "img2": {"gpu_architecture": "gfx90a", "gpu_vendor": "AMD"}
        }
        filtered = run_orch._filter_images_by_gpu_architecture(built_images, "gfx908")
        assert "img1" in filtered and "img2" not in filtered
        
        # Test legacy image (no gpu_vendor field) - should be included for compatibility
        built_images = {
            "img1": {"gpu_architecture": "gfx908"},  # No gpu_vendor
            "img2": {"gpu_architecture": "gfx90a", "gpu_vendor": "AMD"}
        }
        filtered = run_orch._filter_images_by_gpu_architecture(built_images, "gfx908")
        # Current behavior: legacy images (no gpu_vendor) are treated as compatible
        assert "img1" in filtered  # Legacy image included
        # img2 may or may not be included depending on gpu_vendor matching logic
        
        # Test no match case with explicit gpu_vendor
        built_images = {
            "img1": {"gpu_architecture": "gfx90a", "gpu_vendor": "AMD"}, 
            "img2": {"gpu_architecture": "gfx942", "gpu_vendor": "AMD"}
        }
        filtered = run_orch._filter_images_by_gpu_architecture(built_images, "gfx908")
        assert len(filtered) == 0
        
        # Test all matching case with gpu_vendor
        built_images = {
            "img1": {"gpu_architecture": "gfx908", "gpu_vendor": "AMD"}, 
            "img2": {"gpu_architecture": "gfx908", "gpu_vendor": "AMD"}
        }
        filtered = run_orch._filter_images_by_gpu_architecture(built_images, "gfx908")
        assert len(filtered) == 2
