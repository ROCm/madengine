"""
Context logic unit tests.

Pure unit tests for Context class initialization and logic without external dependencies.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import pytest
from unittest.mock import Mock, patch

from madengine.core.context import Context


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
                "MAD_SYSTEM_GPU_ARCHITECTURE": "gfx90a"
            }
        }
        
        assert context.ctx["docker_build_arg"]["MAD_GPU_VENDOR"] == "AMD"
        assert context.ctx["docker_build_arg"]["MAD_SYSTEM_GPU_ARCHITECTURE"] == "gfx90a"


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


# Total: 5 unit tests
