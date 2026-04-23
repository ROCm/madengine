"""
Unit tests for ROCm path (ROCM_PATH / --rocm-path) support.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import pytest

from madengine.core.constants import get_rocm_path
from madengine.utils import rocm_path_resolver as rpr
from madengine.utils.rocm_path_resolver import (
    MAD_ROCM_PATH,
    auto_detect_rocm_path,
    get_rocm_path_legacy,
    normalize_rocm_path,
    resolve_container_rocm_path,
    resolve_host_rocm_path,
)


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
        from madengine.utils.rocm_path_resolver import normalize_rocm_path

        with patch.object(Context, "get_gpu_vendor", return_value="AMD"), \
             patch.object(Context, "get_system_ngpus", return_value=2), \
             patch.object(Context, "get_system_gpu_architecture", return_value="gfx90a"), \
             patch.object(Context, "get_system_gpu_product_name", return_value="MI250"), \
             patch.object(Context, "get_system_hip_version", return_value="5.4"), \
             patch.object(Context, "get_docker_gpus", return_value="0-1"), \
             patch.object(Context, "get_gpu_renderD_nodes", return_value=None):
            ctx = Context(rocm_path="/my/rocm")
            exp = normalize_rocm_path("/my/rocm")
            assert ctx.ctx.get("rocm_path") == exp
            assert ctx.ctx["docker_env_vars"].get("ROCM_PATH") == exp

    def test_context_container_mad_overrides_mirror(self):
        """docker_env_vars MAD_ROCM_PATH sets in-container ROCM_PATH (not host path)."""
        from madengine.core.context import Context
        from unittest.mock import patch
        from madengine.utils.rocm_path_resolver import normalize_rocm_path

        ac = repr(
            {
                "docker_env_vars": {MAD_ROCM_PATH: "/in/image"},
            }
        )
        with patch.object(Context, "get_gpu_vendor", return_value="AMD"), \
             patch.object(Context, "get_system_ngpus", return_value=2), \
             patch.object(Context, "get_system_gpu_architecture", return_value="gfx90a"), \
             patch.object(Context, "get_system_gpu_product_name", return_value="MI250"), \
             patch.object(Context, "get_system_hip_version", return_value="5.4"), \
             patch.object(Context, "get_docker_gpus", return_value="0-1"), \
             patch.object(Context, "get_gpu_renderD_nodes", return_value=None):
            ctx = Context(
                additional_context=ac,
                rocm_path="/on/host",
            )
        assert ctx._rocm_path == normalize_rocm_path("/on/host")
        assert ctx.ctx["docker_env_vars"].get("ROCM_PATH") == normalize_rocm_path(
            "/in/image"
        )
        assert MAD_ROCM_PATH not in ctx.ctx["docker_env_vars"]


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


@pytest.mark.unit
class TestResolveHostContainerRocmPath:
    """Test madengine rocm_path_resolver (MAD_ROCM_PATH / auto)."""

    def test_resolve_host_mad_takes_precedence_over_cli(self):
        """Top-level MAD_ROCM_PATH in context wins over --rocm-path."""
        p = resolve_host_rocm_path({MAD_ROCM_PATH: "/from/context"}, "/from/cli")
        assert os.path.isabs(p)
        assert p == os.path.abspath("/from/context").rstrip(os.sep)

    def test_resolve_host_cli_when_no_mad(self):
        """--rocm-path (cli) is used when MAD_ROCM_PATH is absent."""
        p = resolve_host_rocm_path({}, "/only/cli")
        assert p == os.path.abspath("/only/cli").rstrip(os.sep)

    def test_resolve_host_legacy_when_auto_off(self, monkeypatch):
        """MAD_AUTO_ROCM_PATH=0 uses ROCM_PATH env then default."""
        monkeypatch.setenv("MAD_AUTO_ROCM_PATH", "0")
        monkeypatch.delenv("ROCM_PATH", raising=False)
        p = resolve_host_rocm_path({}, None)
        assert p == os.path.abspath("/opt/rocm").rstrip(os.sep)
        monkeypatch.setenv("ROCM_PATH", "/env/ro")
        p2 = resolve_host_rocm_path({}, None)
        assert p2 == os.path.abspath("/env/ro").rstrip(os.sep)

    def test_resolve_container_mad_consumes_key(self):
        d = {MAD_ROCM_PATH: "/in/container"}
        host = "/on/host"
        r = resolve_container_rocm_path(d, host)
        assert r == os.path.abspath("/in/container").rstrip(os.sep)
        assert d.get("ROCM_PATH") == r
        assert MAD_ROCM_PATH not in d

    def test_resolve_container_mirrors_host(self):
        d = {}
        r = resolve_container_rocm_path(d, "/hostpath")
        assert r == os.path.abspath("/hostpath").rstrip(os.sep)
        assert d["ROCM_PATH"] == r

    def test_get_rocm_path_legacy_alias(self, monkeypatch):
        monkeypatch.delenv("ROCM_PATH", raising=False)
        g = get_rocm_path_legacy(None)
        assert g == os.path.abspath("/opt/rocm").rstrip(os.sep)


@pytest.mark.unit
class TestTheRockVersionedContainerLayout:
    """
    TheRock / CI images often use /opt/rocm-7.x.y instead of /opt/rocm, with
    amd-smi and rocm-smi under that tree (or duplicate copies under /opt/python).
    """

    def test_looks_like_root_with_amd_smi_and_rocm_smi(self, tmp_path):
        from madengine.utils.rocm_path_resolver import _looks_like_rocm_root

        root = tmp_path / "rocm-7.13.0"
        (root / "bin").mkdir(parents=True)
        for n in ("amd-smi", "rocm-smi"):
            f = root / "bin" / n
            f.write_text("#!/bin/sh\necho\n", encoding="utf-8")
            f.chmod(0o755)
        assert _looks_like_rocm_root(root)

    def test_rocm_root_from_bin_tool_amd_smi(self, tmp_path):
        from madengine.utils.rocm_path_resolver import _rocm_root_from_bin_tool

        root = tmp_path / "rocm-7.13.0"
        (root / "bin").mkdir(parents=True)
        smi = root / "bin" / "amd-smi"
        smi.write_text("x", encoding="utf-8")
        smi.chmod(0o755)
        assert _rocm_root_from_bin_tool(str(smi.resolve())) == root.resolve()

    def test_auto_detect_finds_injected_versioned_opt_rocm(
        self, monkeypatch, tmp_path
    ):
        """Simulate /opt/rocm-7.13.0 without depending on the host /opt tree."""
        vroot = (tmp_path / "rocm-7.13.0").resolve()
        (vroot / "bin").mkdir(parents=True)
        for n in ("amd-smi", "rocm-smi"):
            f = vroot / "bin" / n
            f.write_text("#!/bin/sh\necho\n", encoding="utf-8")
            f.chmod(0o755)

        real_looks = rpr._looks_like_rocm_root

        def merged_looks(p):
            r = p.resolve()
            if r == vroot:
                return real_looks(r)
            # Suppress any real /opt/rocm installation on the host (including
            # when /opt/rocm is a symlink, e.g. /opt/rocm -> /opt/rocm-6.4.2,
            # so p.resolve() won't equal Path("/opt/rocm")).
            p_str = str(p)
            r_str = str(r)
            if p_str == "/opt/rocm" or r_str == "/opt/rocm":
                return False
            # Also suppress versioned host installs reachable via /opt/rocm symlink
            if p_str.startswith("/opt/rocm-") and r_str != str(vroot):
                return False
            return real_looks(p)

        monkeypatch.setattr(rpr, "_looks_like_rocm_root", merged_looks)
        monkeypatch.setattr(
            rpr, "_versioned_opt_rocm_dirs", lambda: [vroot]
        )
        out = auto_detect_rocm_path()
        assert out == normalize_rocm_path(str(vroot))

    def test_infer_root_from_path_tools_amd_smi(
        self, monkeypatch, tmp_path
    ):
        """`which(amd-smi)` → .../rocm-7.13.0/bin/amd-smi` yields root with both smi tools."""
        vroot = (tmp_path / "rocm-7.13.0").resolve()
        (vroot / "bin").mkdir(parents=True)
        for n in ("amd-smi", "rocm-smi"):
            f = vroot / "bin" / n
            f.write_text("#!/bin/sh\necho\n", encoding="utf-8")
            f.chmod(0o755)

        def fake_which(name, path=None):
            if name == "rocminfo":
                return None
            if name in ("amd-smi", "rocm-smi"):
                return str(vroot / "bin" / name)
            return None

        monkeypatch.setattr(rpr.shutil, "which", fake_which)
        from madengine.utils.rocm_path_resolver import _infer_root_from_path_tools

        assert _infer_root_from_path_tools() == normalize_rocm_path(str(vroot))
