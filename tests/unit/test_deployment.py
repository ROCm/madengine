"""Unit tests for deployment: base (create_jinja_env) and common (launcher, rocprof, profiling)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from madengine.deployment.base import BaseDeployment, DeploymentConfig, create_jinja_env
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
        assert tools_include_rocprof_family([{"name": "rocm_trace_lite_default"}]) is False


class TestIsRocprofv3Available:
    """is_rocprofv3_available (mocked subprocess)."""

    def setup_method(self):
        # Clear the lru_cache so each test starts with a fresh result
        is_rocprofv3_available.cache_clear()

    def teardown_method(self):
        # Restore clean cache state after each test
        is_rocprofv3_available.cache_clear()

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
    def test_multi_node_no_rocprofv3_keeps_rocm_trace_lite_default(self, _mock_avail):
        logger = MagicMock()
        tools = [{"name": "rocm_trace_lite_default"}]
        out = configure_multi_node_profiling(2, tools, logger)
        assert out["enabled"] is True
        assert out["mode"] == "multi_node"
        assert out["tools"] == tools
        assert out["per_node_collection"] is True

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


# ---- BaseDeployment._parse_performance_from_log ----

class _ConcreteDeployment(BaseDeployment):
    """Minimal concrete subclass to exercise BaseDeployment methods under test."""

    DEPLOYMENT_TYPE = "test"

    def validate(self): pass
    def prepare(self): pass
    def deploy(self): pass
    def monitor(self, deployment_id): pass
    def collect_results(self, deployment_id): pass
    def cleanup(self, deployment_id): pass


def _make_deployment():
    cfg = MagicMock(spec=DeploymentConfig)
    cfg.manifest_file = None
    with patch.object(BaseDeployment, "_load_manifest", return_value={}):
        return _ConcreteDeployment(cfg)


class TestParsePerformanceFromLog:
    """_parse_performance_from_log handles all accepted log formats consistently."""

    def setup_method(self):
        self.dep = _make_deployment()

    def _parse(self, log_content):
        return self.dep._parse_performance_from_log(log_content, "test_model")

    # --- baseline formats ---

    def test_basic_integer(self):
        result = self._parse("performance: 12345 samples_per_second")
        assert result["performance"] == 12345.0
        assert result["metric"] == "samples_per_second"

    def test_decimal(self):
        result = self._parse("performance: 100.5 samples_per_second")
        assert result["performance"] == 100.5
        assert result["metric"] == "samples_per_second"

    def test_scientific_lowercase_e(self):
        result = self._parse("performance: 1.23e+4 samples_per_second")
        assert result["performance"] == pytest.approx(12300.0)
        assert result["metric"] == "samples_per_second"

    def test_scientific_uppercase_e(self):
        result = self._parse("performance: 1.23E+4 samples_per_second")
        assert result["performance"] == pytest.approx(12300.0)
        assert result["metric"] == "samples_per_second"

    # --- new formats: unit suffix and/or comma after value ---

    def test_unit_suffix_slash_s(self):
        """Value followed by /s unit suffix: suffix is stripped."""
        result = self._parse("performance: 100.5/s samples_per_second")
        assert result["performance"] == 100.5
        assert result["metric"] == "samples_per_second"

    def test_unit_suffix_and_comma(self):
        """Suffix then comma: performance: 100.5/s, samples_per_second."""
        result = self._parse("performance: 100.5/s, samples_per_second")
        assert result["performance"] == 100.5
        assert result["metric"] == "samples_per_second"

    def test_comma_separator_no_suffix(self):
        """Comma only: performance: 100.5, samples_per_second."""
        result = self._parse("performance: 100.5, samples_per_second")
        assert result["performance"] == 100.5
        assert result["metric"] == "samples_per_second"

    def test_comma_before_suffix(self):
        """Comma immediately before suffix: performance: 100.5,/s samples_per_second."""
        result = self._parse("performance: 100.5,/s samples_per_second")
        assert result["performance"] == 100.5
        assert result["metric"] == "samples_per_second"

    def test_comma_space_before_suffix(self):
        """Comma then space then suffix: performance: 100.5, /s samples_per_second."""
        result = self._parse("performance: 100.5, /s samples_per_second")
        assert result["performance"] == 100.5
        assert result["metric"] == "samples_per_second"

    # --- slash-containing metric names (e.g. samples/sec, tokens/sec) ---

    def test_metric_samples_per_sec_slash(self):
        """samples/sec is recognised and value is parsed correctly."""
        result = self._parse("performance: 1234.5 samples/sec")
        assert result["performance"] == 1234.5
        assert result["metric"] == "samples/sec"

    def test_metric_tokens_per_sec_slash(self):
        """tokens/sec is recognised; _determine_aggregation_method uses this name."""
        result = self._parse("performance: 500.0 tokens/sec")
        assert result["performance"] == 500.0
        assert result["metric"] == "tokens/sec"

    # --- no match ---

    def test_no_match_returns_none(self):
        assert self._parse("throughput: 100.5 samples_per_second") is None

    def test_no_match_missing_metric_returns_none(self):
        assert self._parse("performance: 100.5") is None

    # --- auxiliary fields parsed alongside performance ---

    def test_node_id_and_local_gpus_parsed(self):
        log = "performance: 50.0 tokens_per_second\nnode_id: 2\nlocal_gpus: 4"
        result = self._parse(log)
        assert result["node_id"] == 2
        assert result["local_gpus"] == 4

    def test_model_name_propagated(self):
        result = self._parse("performance: 1.0 metric")
        assert result["model"] == "test_model"
