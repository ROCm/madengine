"""Unit tests for execution: container_runner_helpers and dockerfile_utils."""

import pytest

from madengine.core.timeout import (
    DEFAULT_RUN_TIMEOUT,
    Timeout,
    resolve_run_timeout,
    subprocess_timeout,
)
from madengine.execution.container_runner_helpers import (
    _docker_image_ref_for_log_naming,
    make_run_log_file_path,
)
from madengine.execution.dockerfile_utils import (
    GPU_ARCH_VARIABLES,
    is_compilation_arch_compatible,
    is_target_arch_compatible_with_variable,
    normalize_architecture_name,
    parse_dockerfile_gpu_variables,
)


# ---- Timeout ----

class TestTimeout:
    """Timeout context manager: None/0 must not arm signal.alarm."""

    def test_none_seconds_does_not_raise(self):
        with Timeout(None):
            pass  # must not crash

    def test_zero_seconds_does_not_raise(self):
        with Timeout(0):
            pass  # must not crash

    def test_positive_seconds_raises_on_expiry(self):
        with pytest.raises(TimeoutError):
            with Timeout(1):
                import time
                time.sleep(2)


# ---- container_runner_helpers ----

class TestResolveRunTimeout:
    """resolve_run_timeout: default < model card < explicit CLI (v1 precedence).

    Sentinel contract: -1 unspecified, 0 no timeout, >0 explicit seconds.
    """

    def test_default_when_nothing_specified(self):
        assert resolve_run_timeout({}, -1) == DEFAULT_RUN_TIMEOUT
        assert resolve_run_timeout({"timeout": -1}, -1) == DEFAULT_RUN_TIMEOUT
        assert resolve_run_timeout({"name": "x"}, -1) == DEFAULT_RUN_TIMEOUT

    def test_model_timeout_overrides_default(self):
        assert resolve_run_timeout({"timeout": 360}, -1) == 360
        assert resolve_run_timeout({"timeout": 100}, -1) == 100

    def test_cli_timeout_overrides_model(self):
        assert resolve_run_timeout({"timeout": 360}, 120) == 120
        assert resolve_run_timeout({"timeout": 3600}, 6000) == 6000

    def test_explicit_cli_equal_to_default_still_wins(self):
        # Regression: the old resolver detected "CLI is default" by comparing
        # against 7200, so an explicit --timeout 7200 silently lost to the model
        # card. With the -1 sentinel the two are distinguishable.
        assert resolve_run_timeout({"timeout": 360}, DEFAULT_RUN_TIMEOUT) == DEFAULT_RUN_TIMEOUT

    def test_zero_means_no_timeout_and_wins(self):
        # 0 is a real choice, not "unset" — it must beat the level below it.
        assert resolve_run_timeout({"timeout": 360}, 0) == 0
        assert resolve_run_timeout({}, 0) == 0
        assert resolve_run_timeout({"timeout": 0}, -1) == 0

    def test_legacy_none_in_manifest_treated_as_unset(self):
        # Manifests written by older builds store null for an absent timeout.
        assert resolve_run_timeout({"timeout": None}, -1) == DEFAULT_RUN_TIMEOUT
        assert resolve_run_timeout({"timeout": None}, 120) == 120

    def test_custom_default(self):
        assert resolve_run_timeout({}, -1, default_timeout=5000) == 5000
        assert resolve_run_timeout({"timeout": 100}, -1, default_timeout=5000) == 100

    def test_always_returns_int(self):
        # No None ever escapes the resolver — downstream consumers rely on this.
        for model, cli in (({}, -1), ({"timeout": None}, -1), ({"timeout": 0}, -1), ({}, 0)):
            assert isinstance(resolve_run_timeout(model, cli), int)


class TestSubprocessTimeout:
    """subprocess_timeout: sentinel -> subprocess/communicate semantics.

    subprocess treats timeout=0 as "expire immediately", not "no timeout", so
    both 0 and -1 must become None.
    """

    @pytest.mark.parametrize("value", [0, -1, None])
    def test_no_timeout_values_become_none(self, value):
        assert subprocess_timeout(value) is None

    @pytest.mark.parametrize("value", [1, 120, 7200])
    def test_positive_passes_through(self, value):
        assert subprocess_timeout(value) == value


class TestDockerImageRefForLogNaming:
    """_docker_image_ref_for_log_naming: CI tag extraction vs stable non-ci refs."""

    def test_ci_tag_from_registry_ref(self):
        assert (
            _docker_image_ref_for_log_naming("rocm/ns/img:ci-m_model_df")
            == "ci-m_model_df"
        )

    def test_non_ci_tag_sanitizes_full_ref(self):
        assert _docker_image_ref_for_log_naming("ubuntu:22.04") == "ubuntu_22.04"
        assert (
            _docker_image_ref_for_log_naming("registry/ns/myimg:latest")
            == "registry_ns_myimg_latest"
        )

    def test_short_ci_tag_unchanged(self):
        assert _docker_image_ref_for_log_naming("ci-model_ubuntu") == "ci-model_ubuntu"


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
