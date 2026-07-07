"""Unit tests for the bare-metal (conda) execution backend."""

import os
from unittest import mock

import pytest

from madengine.execution.conda_env import CondaEnvManager, resolve_conda_env_name
from madengine.execution.run_reporting import (
    determine_status,
    extract_performance_from_log,
)
from madengine.orchestration.run_orchestrator import RunOrchestrator

# ---- Deployment target inference ----


class TestBareMetalInference:
    """_infer_deployment_target routes bare_metal before slurm/k8s/local."""

    def _make_orch(self):
        args = mock.MagicMock()
        args.additional_context = "{}"
        args.output = "perf.csv"
        return RunOrchestrator(args)

    def test_bare_metal_key_selects_bare_metal(self):
        orch = self._make_orch()
        assert orch._infer_deployment_target({"bare_metal": {}}) == "bare_metal"

    def test_bare_metal_wins_over_slurm(self):
        orch = self._make_orch()
        assert (
            orch._infer_deployment_target({"bare_metal": {}, "slurm": {}})
            == "bare_metal"
        )

    def test_no_key_is_local(self):
        orch = self._make_orch()
        assert orch._infer_deployment_target({}) == "local"


# ---- Conda env name resolution ----


class TestResolveCondaEnvName:
    def test_config_overrides_model(self):
        assert resolve_conda_env_name({"conda_env": "m"}, {"conda_env": "c"}) == "c"

    def test_model_used_when_no_config(self):
        assert resolve_conda_env_name({"conda_env": "m"}, {}) == "m"

    def test_derived_from_name(self):
        assert resolve_conda_env_name({"name": "foo/bar"}, {}) == "mad_foo_bar"


# ---- CondaEnvManager command construction ----


class TestCondaEnvManager:
    def _mgr(self, bm_config=None):
        console = mock.MagicMock()
        mgr = CondaEnvManager(console=console, bm_config=bm_config or {})
        mgr._conda_bin = "/opt/conda/bin/conda"  # bypass PATH detection
        return mgr, console

    def test_conda_run_prefix(self):
        mgr, _ = self._mgr()
        prefix = mgr.conda_run_prefix("myenv")
        assert "run -n myenv" in prefix
        assert "--no-capture-output" in prefix

    def test_env_exists_matches_name(self):
        mgr, console = self._mgr()
        console.sh.return_value = (
            "# conda envs\nbase   *  /opt/conda\nmyenv     /opt/conda/envs/myenv"
        )
        assert mgr.env_exists("myenv") is True
        assert mgr.env_exists("absent") is False

    def test_create_with_python_version(self):
        mgr, console = self._mgr()
        # env does not exist
        console.sh.return_value = "# conda envs\nbase * /opt/conda"
        name = mgr.create_or_update(
            {"name": "m", "conda_env": "e", "python_version": "3.11"}
        )
        assert name == "e"
        # Find the create command among the calls.
        create_calls = [
            c.args[0] for c in console.sh.call_args_list if "create" in c.args[0]
        ]
        assert any("python=3.11" in cmd and "-n e" in cmd for cmd in create_calls)

    def test_reuse_existing_env_skips_create(self):
        mgr, console = self._mgr(bm_config={"reuse_env": True})
        console.sh.return_value = "myenv  /opt/conda/envs/myenv"
        mgr.create_or_update({"name": "m", "conda_env": "myenv"})
        create_calls = [
            c.args[0]
            for c in console.sh.call_args_list
            if "conda create" in c.args[0] or "env create" in c.args[0]
        ]
        assert create_calls == []


# ---- Performance extraction ----


class TestExtractPerformance:
    def test_canonical_pattern(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text("some output\nperformance: 123.5 samples_per_second\n")
        perf, metric = extract_performance_from_log(str(log))
        assert perf == "123.5"
        assert metric == "samples_per_second"

    def test_missing_file(self):
        perf, metric = extract_performance_from_log("/nonexistent/x.log")
        assert perf is None and metric is None


class TestDetermineStatus:
    def test_success_with_performance(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text("performance: 5 samples_per_second\n")
        assert determine_status(str(log), "5", {}, {}) == "SUCCESS"

    def test_failure_no_performance(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text("nothing useful\n")
        assert determine_status(str(log), None, {}, {}) == "FAILURE"

    def test_failure_on_error_pattern(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text("performance: 5 x\nTraceback (most recent call last)\n")
        assert determine_status(str(log), "5", {}, {}) == "FAILURE"
