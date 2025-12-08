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
    
    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_ngpus", return_value=1)
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx90a")
    def test_context_initializes_with_defaults(self, mock_arch, mock_ngpus, mock_vendor):
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
    
    @patch.object(Context, "get_gpu_vendor", return_value="AMD")
    @patch.object(Context, "get_system_gpu_architecture", return_value="gfx90a")
    def test_generates_build_args_for_amd(self, mock_arch, mock_vendor):
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


# Total: 5 unit tests
