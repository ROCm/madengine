"""Unit tests for deployment: base (create_jinja_env) and common (launcher, rocprof, profiling)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from madengine.deployment.base import create_jinja_env
from madengine.deployment.common import (
    VALID_LAUNCHERS,
    configure_multi_node_profiling,
    is_rocprofv3_available,
    normalize_launcher,
    tools_include_rocprof_family,
)


# ---- deployment.base (create_jinja_env) ----

class TestCreateJinjaEnv:
    """Test create_jinja_env helper."""

    def test_returns_environment_with_dirname_basename_filters(self):
        """create_jinja_env returns Environment with dirname and basename filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "test.j2").write_text("dir={{ path | dirname }} name={{ path | basename }}")
            env = create_jinja_env(p)
            template = env.get_template("test.j2")
            out = template.render(path="/foo/bar/baz.txt")
            assert "dir=/foo/bar" in out or "dir=foo/bar" in out
            assert "name=baz.txt" in out

    def test_template_dir_must_exist(self):
        """create_jinja_env works when template_dir exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = create_jinja_env(Path(tmpdir))
            assert env is not None
            assert env.filters.get("dirname") is not None
            assert env.filters.get("basename") is not None


# ---- deployment.common ----

class TestValidLaunchers:
    """VALID_LAUNCHERS constant."""

    def test_contains_expected_launchers(self):
        assert "torchrun" in VALID_LAUNCHERS
        assert "vllm" in VALID_LAUNCHERS
        assert "sglang-disagg" in VALID_LAUNCHERS


class TestNormalizeLauncher:
    """normalize_launcher behavior."""

    def test_valid_launcher_passthrough(self):
        for lt in VALID_LAUNCHERS:
            assert normalize_launcher(lt, "kubernetes") == lt
            assert normalize_launcher(lt, "slurm") == lt
            assert normalize_launcher(lt, "local") == lt

    @pytest.mark.parametrize("launcher", [None, "", "invalid"])
    def test_invalid_or_missing_launcher_kubernetes_returns_native(self, launcher):
        assert normalize_launcher(launcher, "kubernetes") == "native"

    @pytest.mark.parametrize("deployment", ["slurm", "local", "unknown"])
    def test_invalid_or_missing_launcher_non_k8s_returns_docker(self, deployment):
        assert normalize_launcher(None, deployment) == "docker"


class TestToolsIncludeRocprofFamily:
    """tools_include_rocprof_family."""

    def test_detects_rocprof_and_presets(self):
        assert tools_include_rocprof_family([{"name": "rocprof"}]) is True
        assert tools_include_rocprof_family([{"name": "rocprofv3_lightweight"}]) is True

    def test_false_for_rocm_trace_lite(self):
        assert tools_include_rocprof_family([{"name": "rocm_trace_lite"}]) is False


class TestIsRocprofv3Available:
    """is_rocprofv3_available (mocked subprocess)."""

    def test_returns_true_when_help_succeeds(self):
        with patch("madengine.deployment.common.subprocess.run") as m:
            m.return_value = MagicMock(returncode=0)
            assert is_rocprofv3_available() is True
            m.assert_called_once()
            assert m.call_args[0][0] == ["rocprofv3", "--help"]

    def test_returns_false_when_not_found(self):
        with patch("madengine.deployment.common.subprocess.run") as m:
            m.side_effect = FileNotFoundError()
            assert is_rocprofv3_available() is False

    def test_returns_false_on_timeout(self):
        import subprocess
        with patch("madengine.deployment.common.subprocess.run") as m:
            m.side_effect = subprocess.TimeoutExpired("rocprofv3", 5)
            assert is_rocprofv3_available() is False


class TestConfigureMultiNodeProfiling:
    """configure_multi_node_profiling with mocked is_rocprofv3_available."""

    def test_single_node_returns_tools_unchanged(self):
        logger = MagicMock()
        tools = [{"name": "rocprof"}]
        out = configure_multi_node_profiling(1, tools, logger)
        assert out["enabled"] is True
        assert out["mode"] == "single_node"
        assert out["tools"] is tools
        assert out["per_node_collection"] is False

    @patch("madengine.deployment.common.is_rocprofv3_available", return_value=False)
    def test_multi_node_no_rocprofv3_returns_disabled(self, _mock_avail):
        logger = MagicMock()
        out = configure_multi_node_profiling(2, [{"name": "rocprof"}], logger)
        assert out["enabled"] is False
        assert out["mode"] == "multi_node_unsupported"
        assert out["tools"] == []
        logger.warning.assert_called_once()

    @patch("madengine.deployment.common.is_rocprofv3_available", return_value=False)
    def test_multi_node_no_rocprofv3_keeps_rocm_trace_lite(self, _mock_avail):
        logger = MagicMock()
        tools = [{"name": "rocm_trace_lite"}]
        out = configure_multi_node_profiling(2, tools, logger)
        assert out["enabled"] is True
        assert out["mode"] == "multi_node"
        assert out["tools"] == tools
        assert out["per_node_collection"] is True
        logger.info.assert_called()

    @patch("madengine.deployment.common.is_rocprofv3_available", return_value=False)
    def test_multi_node_no_rocprofv3_filters_mixed_tools(self, _mock_avail):
        logger = MagicMock()
        out = configure_multi_node_profiling(
            2,
            [{"name": "rocprof"}, {"name": "rocm_trace_lite"}],
            logger,
        )
        assert out["enabled"] is True
        assert out["mode"] == "multi_node"
        assert out["tools"] == [{"name": "rocm_trace_lite"}]
        assert out["per_node_collection"] is True
        logger.warning.assert_called()

    @patch("madengine.deployment.common.is_rocprofv3_available", return_value=True)
    def test_multi_node_upgrades_rocprof_to_rocprofv3(self, _mock_avail):
        logger = MagicMock()
        tools = [{"name": "rocprof", "args": ["--trace"]}]
        out = configure_multi_node_profiling(2, tools, logger)
        assert out["enabled"] is True
        assert out["mode"] == "multi_node"
        assert out["per_node_collection"] is True
        assert len(out["tools"]) == 1
        assert out["tools"][0]["name"] == "rocprofv3"
        assert out["tools"][0]["args"] == ["--trace"]

    @patch("madengine.deployment.common.is_rocprofv3_available", return_value=True)
    def test_multi_node_other_tools_unchanged(self, _mock_avail):
        logger = MagicMock()
        tools = [{"name": "rccl_trace"}, {"name": "rocprof"}]
        out = configure_multi_node_profiling(2, tools, logger)
        assert out["tools"][0]["name"] == "rccl_trace"
        assert out["tools"][1]["name"] == "rocprofv3"
