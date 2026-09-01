"""Unit tests for execution: container_runner_helpers and dockerfile_utils."""

import re

import pytest

from madengine.core.timeout import Timeout
from madengine.execution.container_runner_helpers import (
    _docker_image_ref_for_log_naming,
    container_name_from_image_ref,
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

    def test_no_timeout_sentinel_none_passthrough(self):
        # --timeout 0 is converted to None by the CLI; resolve_run_timeout must
        # pass None through unchanged (model timeout must NOT override "no timeout").
        assert resolve_run_timeout({"timeout": 3600}, None) is None
        assert resolve_run_timeout({}, None) is None


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

    def test_pinned_reference_names_same_as_untagged_reference(self):
        digest = "sha256:" + "df36ef7e" * 8
        assert _docker_image_ref_for_log_naming(
            f"registry/ns/myimg@{digest}"
        ) == _docker_image_ref_for_log_naming("registry/ns/myimg")

    def test_pinned_ci_reference_still_yields_tag(self):
        digest = "sha256:" + "df36ef7e" * 8
        assert (
            _docker_image_ref_for_log_naming(f"rocm/ns/img:ci-m_model_df@{digest}")
            == "ci-m_model_df"
        )


class TestContainerNameFromImageRef:
    """container_name_from_image_ref: names must satisfy Docker's charset."""

    # Docker: "only [a-zA-Z0-9][a-zA-Z0-9_.-] are allowed".
    DOCKER_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")

    def test_digest_pinned_ref_is_docker_legal(self):
        # Regression: require_pinned_image rewrites the run image to
        # repo@sha256:..., which previously produced a name containing "@" and
        # made `docker run` fail with "Invalid container name".
        digest = "sha256:" + "af99a16c" * 8
        name = container_name_from_image_ref(f"registry/ns/mad-private@{digest}")
        assert "@" not in name
        assert self.DOCKER_NAME_RE.match(name)
        assert name == "container_registry_ns_mad-private"

    def test_digest_pinned_ref_keeps_tag(self):
        digest = "sha256:" + "af99a16c" * 8
        assert (
            container_name_from_image_ref(f"registry/ns/img:ci-m_model_df@{digest}")
            == "container_registry_ns_img_ci-m_model_df"
        )

    @pytest.mark.parametrize(
        "image, expected",
        [
            # Historical names for non-pinned refs must not change.
            ("ci-model_ubuntu", "container_ci-model_ubuntu"),
            ("ubuntu:22.04", "container_ubuntu_22.04"),
            ("registry/ns/myimg:latest", "container_registry_ns_myimg_latest"),
            (
                "registry/ns/img:ci-m_model_df",
                "container_registry_ns_img_ci-m_model_df",
            ),
            ("localhost:5000/ns/img:tag", "container_localhost_5000_ns_img_tag"),
        ],
    )
    def test_unpinned_refs_keep_previous_names(self, image, expected):
        assert container_name_from_image_ref(image) == expected
        assert self.DOCKER_NAME_RE.match(expected)

    def test_pinned_and_unpinned_agree_on_same_repo_tag(self):
        digest = "sha256:" + "af99a16c" * 8
        assert container_name_from_image_ref(
            f"registry/ns/img:tag@{digest}"
        ) == container_name_from_image_ref("registry/ns/img:tag")


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
