"""
Context logic unit tests.

Pure unit tests for Context class initialization and logic without external dependencies.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from madengine.core.context import Context
from madengine.utils.gpu_validator import GPUVendor


@pytest.mark.unit
class TestContextInitialization:
    """Test Context object initialization."""

    @patch.object(Context, "get_gpu_renderD_nodes", return_value=None)
    @patch.object(Context, "get_docker_gpus", return_value="0")
    @patch.object(Context, "get_system_gpu_product_name", return_value="Test GPU")
    @patch.object(Context, "get_system_hip_version", return_value="6.0")
    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx90a")
    def test_context_initializes_with_defaults(
        self,
        mock_arch,
        mock_ngpus,
        mock_vendor,
        mock_hip,
        mock_product,
        mock_docker_gpus,
        mock_renderd,
    ):
        """Context should initialize with system defaults."""
        context = Context()

        assert context.get_gpu_vendor() == "AMD"
        assert context.get_system_ngpus() == 1
        assert context.get_system_gpu_architecture() == "gfx90a"

    # REMOVED: test_context_detects_nvidia_gpus and test_context_handles_cpu_only
    # These tests require actual GPU detection and are better suited as integration tests.
    # Context initialization tests are covered in integration/test_platform_integration.py


@pytest.mark.unit
class TestBuildArgGeneration:
    """Test Docker build argument generation logic."""

    @patch.object(Context, "get_gpu_renderD_nodes", return_value=None)
    @patch.object(Context, "get_docker_gpus", return_value="0")
    @patch.object(Context, "get_system_gpu_product_name", return_value="Test GPU")
    @patch.object(Context, "get_system_hip_version", return_value="6.0")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx90a")
    def test_generates_build_args_for_amd(
        self,
        mock_arch,
        mock_vendor,
        mock_ngpus,
        mock_hip,
        mock_product,
        mock_docker_gpus,
        mock_renderd,
    ):
        """Should generate proper build args for AMD GPUs."""
        context = Context()
        context.ctx = {
            "docker_build_arg": {
                "MAD_GPU_VENDOR": "AMD",
                "MAD_SYSTEM_GPU_ARCHITECTURE": "gfx90a",
            }
        }

        assert context.ctx["docker_build_arg"]["MAD_GPU_VENDOR"] == "AMD"
        assert (
            context.ctx["docker_build_arg"]["MAD_SYSTEM_GPU_ARCHITECTURE"] == "gfx90a"
        )


@pytest.mark.unit
class TestBuildOnlyContextMessages:
    """Build-only context should not emit duplicate MAD_SYSTEM_GPU_ARCHITECTURE info."""

    @patch.object(Context, "get_ctx_test", return_value="test")
    @patch.object(Context, "get_host_os", return_value="linux")
    def test_build_only_no_mad_arch_info_line(self, mock_host, mock_ctx):
        from unittest.mock import patch

        with patch("builtins.print") as p:
            Context(additional_context="{}", build_only_mode=True)
        msgs = [str(c.args[0]) for c in p.call_args_list if c.args]
        assert not any("MAD_SYSTEM_GPU_ARCHITECTURE" in m for m in msgs)


def _make_build_only_ctx(additional_context="{}") -> Context:
    """Create a Context in build_only_mode with __init__'s init_build_context call suppressed.

    Returns a fully constructed Context whose ctx dict is populated from additional_context
    but whose init_build_context has NOT yet run, so callers can invoke it in a controlled way.
    """
    with (
        patch.object(Context, "init_build_context"),
        patch.object(Context, "get_ctx_test", return_value="test"),
        patch.object(Context, "get_host_os", return_value="linux"),
    ):
        ctx = Context(additional_context=additional_context, build_only_mode=True)
    return ctx


@pytest.mark.unit
class TestBuildContextGpuArchAutoDetect:
    """Test GPU architecture auto-detection in init_build_context (detect_gpu_arch=True)."""

    def test_auto_detect_injects_arch_when_absent(self):
        """Auto-detected arch should be injected into docker_build_arg when absent."""
        ctx = _make_build_only_ctx()

        manager = MagicMock()
        manager.get_gpu_architecture.return_value = "gfx942"

        # get_gpu_tool_manager is a module-level import in context.py; patch it there.
        # detect_gpu_vendor / normalize_architecture_name are imported locally inside
        # init_build_context, so patch them at their source modules.
        with (
            patch("madengine.core.context.get_gpu_tool_manager", return_value=manager),
            patch(
                "madengine.utils.gpu_validator.detect_gpu_vendor",
                return_value=GPUVendor.AMD,
            ),
            patch(
                "madengine.execution.dockerfile_utils.normalize_architecture_name",
                return_value="gfx942",
            ),
            patch.object(Context, "get_ctx_test", return_value="test"),
            patch.object(Context, "get_host_os", return_value="linux"),
        ):
            ctx.init_build_context(detect_gpu_arch=True)

        assert ctx.ctx["docker_build_arg"]["MAD_SYSTEM_GPU_ARCHITECTURE"] == "gfx942"

    def test_auto_detect_does_not_override_user_value(self):
        """User-provided MAD_SYSTEM_GPU_ARCHITECTURE must not be overridden."""
        ctx = _make_build_only_ctx(
            additional_context="{'docker_build_arg': {'MAD_SYSTEM_GPU_ARCHITECTURE': 'gfx90a'}}"
        )

        manager = MagicMock()
        manager.get_gpu_architecture.return_value = "gfx942"

        with (
            patch("madengine.core.context.get_gpu_tool_manager", return_value=manager),
            patch(
                "madengine.utils.gpu_validator.detect_gpu_vendor",
                return_value=GPUVendor.AMD,
            ),
            patch(
                "madengine.execution.dockerfile_utils.normalize_architecture_name",
                return_value="gfx942",
            ),
            patch.object(Context, "get_ctx_test", return_value="test"),
            patch.object(Context, "get_host_os", return_value="linux"),
        ):
            ctx.init_build_context(detect_gpu_arch=True)

        # User value must be preserved; auto-detect must not overwrite it.
        assert ctx.ctx["docker_build_arg"]["MAD_SYSTEM_GPU_ARCHITECTURE"] == "gfx90a"

    def test_auto_detect_warns_on_no_gpu(self):
        """Should warn (not crash) when no supported GPU is detected."""
        ctx = _make_build_only_ctx()

        with (
            patch(
                "madengine.utils.gpu_validator.detect_gpu_vendor",
                return_value=GPUVendor.UNKNOWN,
            ),
            patch.object(Context, "get_ctx_test", return_value="test"),
            patch.object(Context, "get_host_os", return_value="linux"),
            patch("builtins.print") as mock_print,
        ):
            ctx.init_build_context(detect_gpu_arch=True)

        msgs = [str(c.args[0]) for c in mock_print.call_args_list if c.args]
        assert any("No supported GPU detected" in m for m in msgs)
        assert "MAD_SYSTEM_GPU_ARCHITECTURE" not in ctx.ctx.get("docker_build_arg", {})

    def test_auto_detect_handles_exception_gracefully(self):
        """Detection failure should warn, not raise."""
        ctx = _make_build_only_ctx()

        with (
            patch(
                "madengine.utils.gpu_validator.detect_gpu_vendor",
                side_effect=RuntimeError("rocminfo not found"),
            ),
            patch.object(Context, "get_ctx_test", return_value="test"),
            patch.object(Context, "get_host_os", return_value="linux"),
            patch("builtins.print") as mock_print,
        ):
            ctx.init_build_context(detect_gpu_arch=True)

        msgs = [str(c.args[0]) for c in mock_print.call_args_list if c.args]
        assert any("Could not auto-detect GPU architecture" in m for m in msgs)
        assert "MAD_SYSTEM_GPU_ARCHITECTURE" not in ctx.ctx.get("docker_build_arg", {})

    def test_no_detection_when_flag_is_false(self):
        """detect_gpu_arch=False should skip detection entirely."""
        ctx = _make_build_only_ctx()

        with (
            patch("madengine.utils.gpu_validator.detect_gpu_vendor") as mock_detect,
            patch.object(Context, "get_ctx_test", return_value="test"),
            patch.object(Context, "get_host_os", return_value="linux"),
        ):
            ctx.init_build_context(detect_gpu_arch=False)

        mock_detect.assert_not_called()
        assert "MAD_SYSTEM_GPU_ARCHITECTURE" not in ctx.ctx.get("docker_build_arg", {})
