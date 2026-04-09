"""
Unit tests for ROCm path (ROCM_PATH / --rocm-path) support.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import pytest

from madengine.core.constants import get_rocm_path


@pytest.mark.unit
class TestGetRocmPath:
    """Test get_rocm_path() resolution."""

    def test_get_rocm_path_default(self, monkeypatch):
        """Without override or ROCM_PATH, returns default /opt/rocm (normalized)."""
        monkeypatch.delenv("ROCM_PATH", raising=False)
        path = get_rocm_path(None)
        assert path == "/opt/rocm"

    def test_get_rocm_path_override(self, monkeypatch):
        """Override argument takes precedence over env."""
        path = get_rocm_path("/custom/rocm")
        assert path == os.path.abspath("/custom/rocm").rstrip(os.sep)
        monkeypatch.setenv("ROCM_PATH", "/env/rocm")
        path_with_env = get_rocm_path("/cli/rocm")
        assert path_with_env == os.path.abspath("/cli/rocm").rstrip(os.sep)
        monkeypatch.delenv("ROCM_PATH", raising=False)

    def test_get_rocm_path_env(self, monkeypatch):
        """ROCM_PATH env is used when override is None."""
        monkeypatch.setenv("ROCM_PATH", "/env/rocm")
        try:
            path = get_rocm_path(None)
            assert path == os.path.abspath("/env/rocm").rstrip(os.sep)
        finally:
            monkeypatch.delenv("ROCM_PATH", raising=False)


@pytest.mark.unit
class TestContextRocmPath:
    """Test Context stores and uses rocm_path."""

    def test_context_build_only_stores_rocm_path(self):
        """Context with build_only_mode=True and rocm_path sets _rocm_path."""
        from madengine.core.context import Context

        ctx = Context(build_only_mode=True, rocm_path="/opt/rocm")
        assert ctx._rocm_path == "/opt/rocm"

    def test_context_runtime_includes_rocm_path_in_ctx(self):
        """Context in runtime mode includes rocm_path and ROCM_PATH in docker_env_vars."""
        from madengine.core.context import Context
        from unittest.mock import patch

        with patch.object(Context, "get_gpu_vendor", return_value="AMD"), \
             patch.object(Context, "get_system_ngpus", return_value=2), \
             patch.object(Context, "get_system_gpu_architecture", return_value="gfx90a"), \
             patch.object(Context, "get_system_gpu_product_name", return_value="MI250"), \
             patch.object(Context, "get_system_hip_version", return_value="5.4"), \
             patch.object(Context, "get_docker_gpus", return_value="0-1"), \
             patch.object(Context, "get_gpu_renderD_nodes", return_value=None):
            ctx = Context(rocm_path="/my/rocm")
            assert ctx.ctx.get("rocm_path") == "/my/rocm"
            assert ctx.ctx["docker_env_vars"].get("ROCM_PATH") == "/my/rocm"


@pytest.mark.unit
class TestRocmToolManagerRocmPath:
    """Test ROCmToolManager uses configurable rocm_path."""

    def test_rocm_tool_manager_paths_under_rocm_path(self):
        """ROCmToolManager(rocm_path=X) sets paths under X."""
        from madengine.utils.rocm_tool_manager import ROCmToolManager

        manager = ROCmToolManager(rocm_path="/custom/rocm")
        assert manager.rocm_path == "/custom/rocm"
        assert manager.AMD_SMI_PATH == "/custom/rocm/bin/amd-smi"
        assert manager.ROCM_SMI_PATH == "/custom/rocm/bin/rocm-smi"
        assert manager.ROCM_VERSION_FILE == "/custom/rocm/.info/version"


@pytest.mark.unit
class TestRunCommandRocmPath:
    """Test run command exposes --rocm-path."""

    def test_run_help_includes_rocm_path(self):
        """madengine run --help mentions --rocm-path."""
        from typer.testing import CliRunner
        from madengine.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--rocm-path" in result.output
