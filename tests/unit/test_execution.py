"""Unit tests for execution: container_runner_helpers and dockerfile_utils."""

import pytest

from madengine.execution.container_runner_helpers import (
    make_run_log_file_path,
    resolve_run_timeout,
)
from madengine.execution.dockerfile_utils import (
    GPU_ARCH_VARIABLES,
    is_compilation_arch_compatible,
    is_target_arch_compatible_with_variable,
    normalize_architecture_name,
    parse_dockerfile_gpu_variables,
)


# ---- container_runner_helpers ----

class TestResolveRunTimeout:
    """resolve_run_timeout behavior."""

    def test_model_timeout_used_when_cli_default(self):
        assert resolve_run_timeout({"timeout": 3600}, 7200) == 3600
        assert resolve_run_timeout({"timeout": 100}, 7200) == 100

    def test_cli_timeout_used_when_explicit(self):
        assert resolve_run_timeout({"timeout": 3600}, 6000) == 6000
        assert resolve_run_timeout({"timeout": 3600}, 100) == 100

    def test_cli_default_returned_when_no_model_timeout(self):
        assert resolve_run_timeout({}, 7200) == 7200
        assert resolve_run_timeout({"name": "x"}, 3600) == 3600

    @pytest.mark.parametrize("model_timeout", [None, 0])
    def test_falsy_model_timeout_ignored_uses_cli(self, model_timeout):
        assert resolve_run_timeout({"timeout": model_timeout}, 7200) == 7200

    def test_custom_default_cli(self):
        assert resolve_run_timeout({"timeout": 100}, 5000, default_cli_timeout=5000) == 100
        assert resolve_run_timeout({"timeout": 100}, 7200, default_cli_timeout=5000) == 7200


class TestMakeRunLogFilePath:
    """make_run_log_file_path behavior."""

    def test_basic_format(self):
        out = make_run_log_file_path(
            {"name": "org/model"}, "ci-org_model_ubuntu.22.04", "",
        )
        assert out == "org_model_ubuntu.22.04.live.log"

    def test_phase_suffix_appended(self):
        out = make_run_log_file_path({"name": "a/b"}, "ci-a_b_cuda", ".run")
        assert out == "a_b_cuda.run.live.log"

    def test_slashes_in_model_name_replaced(self):
        out = make_run_log_file_path(
            {"name": "foo/bar/baz"}, "ci-foo_bar_baz_ubuntu", "",
        )
        assert "/" not in out
        assert out.endswith(".live.log")

    def test_image_without_ci_prefix(self):
        out = make_run_log_file_path({"name": "x/y"}, "registry/x_y_tag", "")
        assert "registry_x_y_tag" in out or "x_y" in out
        assert out.endswith(".live.log")

    def test_no_model_prefix_in_image(self):
        out = make_run_log_file_path(
            {"name": "other/model"}, "ci-some_ubuntu_22", "",
        )
        assert out == "other_model_some_ubuntu_22.live.log"

    def test_full_registry_ref_matches_short_ci_tag(self):
        """Run log name must match build log base when image is registry/name:ci-…."""
        model = {"name": "primus_pretrain/torchtitan_MI300X_qwen3_4B-pretrain"}
        short = "ci-primus_pretrain_torchtitan_mi300x_qwen3_4b-pretrain_primus.ubuntu.amd"
        full = f"rocm/mad-private:{short}"
        assert make_run_log_file_path(model, short, ".run") == make_run_log_file_path(
            model, full, ".run"
        )
        assert make_run_log_file_path(model, short, ".run") == (
            "primus_pretrain_torchtitan_MI300X_qwen3_4B-pretrain_"
            "primus.ubuntu.amd.run.live.log"
        )


# ---- dockerfile_utils ----

class TestGpuArchVariables:
    def test_contains_expected_vars(self):
        assert "MAD_SYSTEM_GPU_ARCHITECTURE" in GPU_ARCH_VARIABLES
        assert "PYTORCH_ROCM_ARCH" in GPU_ARCH_VARIABLES
        assert "GFX_COMPILATION_ARCH" in GPU_ARCH_VARIABLES


class TestParseDockerfileGpuVariables:
    def test_empty_content(self):
        assert parse_dockerfile_gpu_variables("") == {}
        assert parse_dockerfile_gpu_variables("FROM ubuntu") == {}

    def test_arg_parsed(self):
        content = "ARG PYTORCH_ROCM_ARCH=gfx90a"
        out = parse_dockerfile_gpu_variables(content)
        assert "PYTORCH_ROCM_ARCH" in out
        assert out["PYTORCH_ROCM_ARCH"] == ["gfx90a"]

    def test_multi_arch_semicolon(self):
        content = "ARG GPU_TARGETS=gfx90a;gfx908"
        out = parse_dockerfile_gpu_variables(content)
        assert "GPU_TARGETS" in out
        assert set(out["GPU_TARGETS"]) == {"gfx90a", "gfx908"}

    def test_takes_last_definition(self):
        content = "ARG PYTORCH_ROCM_ARCH=gfx908\nARG PYTORCH_ROCM_ARCH=gfx90a"
        out = parse_dockerfile_gpu_variables(content)
        assert out["PYTORCH_ROCM_ARCH"] == ["gfx90a"]


class TestNormalizeArchitectureName:
    def test_gfx_passthrough(self):
        assert normalize_architecture_name("gfx90a") == "gfx90a"
        assert normalize_architecture_name("gfx942") == "gfx942"

    def test_mi_aliases(self):
        assert normalize_architecture_name("mi100") == "gfx908"
        assert normalize_architecture_name("mi-200") == "gfx90a"
        assert normalize_architecture_name("mi300x") == "gfx942"

    def test_empty_returns_none(self):
        assert normalize_architecture_name("") is None
        assert normalize_architecture_name("   ") is None


class TestIsTargetArchCompatibleWithVariable:
    def test_mad_system_always_compatible(self):
        assert is_target_arch_compatible_with_variable(
            "MAD_SYSTEM_GPU_ARCHITECTURE", ["gfx90a"], "gfx908"
        ) is True

    def test_multi_arch_target_in_list(self):
        assert is_target_arch_compatible_with_variable(
            "PYTORCH_ROCM_ARCH", ["gfx90a", "gfx908"], "gfx90a"
        ) is True
        assert is_target_arch_compatible_with_variable(
            "GPU_TARGETS", ["gfx90a"], "gfx908"
        ) is False

    def test_gfx_compilation_exact_match(self):
        assert is_target_arch_compatible_with_variable(
            "GFX_COMPILATION_ARCH", ["gfx90a"], "gfx90a"
        ) is True


class TestIsCompilationArchCompatible:
    def test_exact_match(self):
        assert is_compilation_arch_compatible("gfx90a", "gfx90a") is True
        assert is_compilation_arch_compatible("gfx942", "gfx942") is True

    def test_mismatch(self):
        assert is_compilation_arch_compatible("gfx90a", "gfx908") is False
        assert is_compilation_arch_compatible("gfx908", "gfx90a") is False

    def test_unknown_arch_treated_as_exact(self):
        assert is_compilation_arch_compatible("gfx999", "gfx999") is True
        assert is_compilation_arch_compatible("gfx999", "gfx90a") is False
