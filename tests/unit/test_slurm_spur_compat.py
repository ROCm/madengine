"""SLURM-compatible schedulers that implement a subset of the CLI (spur).

Spur exposes SLURM_NODELIST already expanded and has no `scontrol show
hostnames`, rejects `sacct -X`, and can refuse sbatch during a Raft leader
election. Each case here keeps stock SLURM's behaviour and adds the spur one.
"""

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from madengine.deployment.base import DeploymentStatus
from madengine.deployment.slurm import SlurmDeployment

from .test_slurm_preset_env import _deployment, _wrapper


def _proc(rc=0, out="", err=""):
    return SimpleNamespace(returncode=rc, stdout=out, stderr=err)


class TestExpandNodelist:
    def test_comma_list_is_split_without_scontrol(self, monkeypatch):
        def no_scontrol(*a, **k):
            raise AssertionError("scontrol must not be called for an expanded list")

        monkeypatch.setattr(subprocess, "run", no_scontrol)
        assert SlurmDeployment._expand_nodelist("n1,n2,n3") == ["n1", "n2", "n3"]
        assert SlurmDeployment._expand_nodelist("solo") == ["solo"]
        assert SlurmDeployment._expand_nodelist("") == []

    def test_compressed_form_uses_scontrol(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda cmd, **k: _proc(out="n01\nn02\nn03\n"))
        assert SlurmDeployment._expand_nodelist("n[01-03]") == ["n01", "n02", "n03"]


class TestSacctWithoutX:
    def _check(self, tmp_path, monkeypatch, responses):
        calls = []

        def fake_run(cmd, **k):
            calls.append(cmd)
            return responses.pop(0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr("madengine.deployment.slurm.time.sleep", lambda s: None)
        d = _deployment(tmp_path)
        monkeypatch.setattr(d, "_show_log_summary", lambda *a, **k: None)
        return d._check_job_completion("42"), calls

    def test_stock_slurm_uses_X(self, tmp_path, monkeypatch):
        result, calls = self._check(tmp_path, monkeypatch, [_proc(out="COMPLETED\n")])
        assert result.status == DeploymentStatus.SUCCESS
        assert calls == [["sacct", "-j", "42", "-n", "-X", "-o", "State"]]

    def test_rejected_X_is_dropped_and_the_job_row_read(self, tmp_path, monkeypatch):
        result, calls = self._check(
            tmp_path,
            monkeypatch,
            [
                _proc(rc=2, err="error: unexpected argument '-X' found\n"),
                _proc(out="FAILED\n"),
            ],
        )
        assert calls[1] == ["sacct", "-j", "42", "-n", "-o", "State"]
        assert result.status == DeploymentStatus.FAILED

    def test_first_row_is_the_job_when_steps_are_listed(self, tmp_path, monkeypatch):
        result, _ = self._check(
            tmp_path,
            monkeypatch,
            [
                _proc(rc=2, err="error: unexpected argument '-X' found\n"),
                _proc(out="COMPLETED\nFAILED\n"),
            ],
        )
        assert result.status == DeploymentStatus.SUCCESS

    def test_empty_answer_is_unknown_not_failed(self, tmp_path, monkeypatch):
        result, calls = self._check(tmp_path, monkeypatch, [_proc(out="")] * 3)
        assert len(calls) == 3
        assert result.status == DeploymentStatus.UNKNOWN


class TestSbatchRetry:
    def _deploy(self, tmp_path, monkeypatch, responses):
        calls = []

        def fake_run(cmd, **k):
            if cmd and cmd[0] == "sbatch":
                calls.append(cmd)
                return responses.pop(0)
            return _proc()

        d = _deployment(tmp_path)
        _wrapper(d)
        d.slurm_config["enable_node_check"] = False
        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr("madengine.deployment.slurm.time.sleep", lambda s: None)
        return d.deploy(), calls

    def test_leader_election_is_retried(self, tmp_path, monkeypatch):
        result, calls = self._deploy(
            tmp_path,
            monkeypatch,
            [_proc(rc=1, err="Error: not the Raft leader"), _proc(out="Submitted batch job 7\n")],
        )
        assert len(calls) == 2
        assert result.status == DeploymentStatus.SUCCESS and result.deployment_id == "7"

    def test_other_errors_are_not_retried(self, tmp_path, monkeypatch):
        result, calls = self._deploy(
            tmp_path, monkeypatch, [_proc(rc=1, err="sbatch: error: invalid partition")]
        )
        assert len(calls) == 1
        assert result.status == DeploymentStatus.FAILED


class TestSlurmMultiQosAccount:
    def test_qos_and_account_reach_the_wrapper(self, tmp_path):
        d = _deployment(tmp_path)
        d.slurm_config.update({"qos": "burst", "account": "acct"})
        script = _wrapper(d)
        assert "#SBATCH --qos=burst" in script
        assert "#SBATCH --account=acct" in script

    def test_absent_by_default(self, tmp_path):
        script = _wrapper(_deployment(tmp_path))
        assert "--qos" not in script and "--account" not in script


class TestBuildOnComputeSlurmDirectives:
    """The --build-on-compute job is submitted under the run job's account and QoS, and
    requests GPUs only when build_gpus_per_node is set (Spur refuses a job without a GPU
    request; elsewhere a build should not wait for a GPU node)."""

    def _build_script(self, tmp_path, slurm):
        import os
        from madengine.orchestration.build_orchestrator import BuildOrchestrator

        orch = BuildOrchestrator.__new__(BuildOrchestrator)
        orch.args = MagicMock()
        orch.console = MagicMock()
        orch.rich_console = MagicMock()
        orch.context = MagicMock()
        orch.context.ctx = {}
        orch.additional_context = {"slurm": dict(slurm)}
        orch._original_user_slurm_keys = set()
        orch.credentials = {}
        model = {
            "name": "card",
            "dockerfile": "docker/card",
            "scripts": "scripts/card/run.slurm",
            "n_gpus": "-1",
            "tags": ["card"],
            "slurm": {"partition": "gpu", "time": "01:00:00"},
            "distributed": {"launcher": "slurm_multi"},
            "env_vars": {},
        }
        df = tmp_path / "docker" / "card.ubuntu.amd.Dockerfile"
        df.parent.mkdir(parents=True, exist_ok=True)
        df.write_text("FROM scratch\n")
        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with patch("madengine.orchestration.build_orchestrator.DiscoverModels") as dm:
                dm.return_value.run.return_value = [model]
                with patch("subprocess.run") as run:
                    run.return_value = MagicMock(returncode=0)
                    orch._execute_build_on_compute(
                        registry="localhost:5000/repo",
                        manifest_output=str(tmp_path / "build_manifest.json"),
                    )
            return (tmp_path / "madengine_build_job.sh").read_text()
        finally:
            os.chdir(cwd)

    def _header(self, script):
        return [l for l in script.splitlines() if l.startswith("#SBATCH")]

    def test_account_qos_and_gpus_reach_the_build_job(self, tmp_path):
        h = self._header(self._build_script(tmp_path, {
            "partition": "amd-spur", "account": "amd-aifw-dev", "qos": "amd-burst-qos",
            "build_gpus_per_node": 8,
        }))
        assert "#SBATCH --partition=amd-spur" in h
        assert "#SBATCH --account=amd-aifw-dev" in h
        assert "#SBATCH --qos=amd-burst-qos" in h
        assert "#SBATCH --gpus-per-node=8" in h

    def test_nothing_extra_by_default(self, tmp_path):
        h = self._header(self._build_script(tmp_path, {"partition": "gpu"}))
        assert not [l for l in h if any(k in l for k in ("--account", "--qos", "--gpus"))]

    def test_run_job_gpus_do_not_make_the_build_request_gpus(self, tmp_path):
        h = self._header(self._build_script(tmp_path, {"partition": "gpu", "gpus_per_node": 8}))
        assert not [l for l in h if "--gpus" in l]


class TestBuildOnComputeImageNames:
    """A namespaced card ("dir/card") must give valid Docker references: tags allow no "/".
    The name is flattened as the regular build does, ci-<dir>_<card>_<dockerfile>."""

    # Docker reference: [registry[:port]/]path[:tag]; tag = [A-Za-z0-9_][A-Za-z0-9_.-]{0,127}
    REF = r"^[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})?$"

    def _manifest(self, tmp_path, registry, name):
        import os, re
        from madengine.orchestration.build_orchestrator import BuildOrchestrator
        orch = BuildOrchestrator.__new__(BuildOrchestrator)
        orch.args = MagicMock(); orch.console = MagicMock(); orch.rich_console = MagicMock()
        orch.context = MagicMock(); orch.context.ctx = {}
        orch.additional_context = {"slurm": {"partition": "gpu"}}
        orch._original_user_slurm_keys = set(); orch.credentials = {}
        model = {"name": name, "dockerfile": "docker/card", "scripts": "scripts/x/run.slurm",
                 "n_gpus": "-1", "tags": ["card"], "slurm": {"partition": "gpu"},
                 "distributed": {"launcher": "slurm_multi"}, "env_vars": {}}
        df = tmp_path / "docker" / "card.ubuntu.amd.Dockerfile"
        df.parent.mkdir(parents=True, exist_ok=True); df.write_text("FROM scratch\n")
        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with patch("madengine.orchestration.build_orchestrator.DiscoverModels") as dm:
                dm.return_value.run.return_value = [model]
                with patch("subprocess.run") as run:
                    run.return_value = MagicMock(returncode=0)
                    orch._execute_build_on_compute(registry=registry,
                                                   manifest_output=str(tmp_path / "m.json"))
            script = (tmp_path / "madengine_build_job.sh").read_text()
        finally:
            os.chdir(cwd)
        manifest = json.loads((tmp_path / "m.json").read_text())
        return manifest, script

    def test_namespaced_card_gives_a_valid_tag(self, tmp_path):
        import re
        manifest, script = self._manifest(tmp_path, "localhost:5000/repo", "sglang_disagg/pyt_card-x")
        img = manifest["built_models"]["sglang_disagg/pyt_card-x"]["env_vars"]["DOCKER_IMAGE_NAME"]
        assert img == "localhost:5000/repo:ci-sglang_disagg_pyt_card-x_card"
        assert re.match(self.REF, img), img
        assert "docker push localhost:5000/repo:ci-sglang_disagg_pyt_card-x_card" in script

    def test_registry_without_path_also_valid(self, tmp_path):
        import re
        manifest, _ = self._manifest(tmp_path, "localhost:5000", "sglang_disagg/pyt_card-x")
        img = manifest["built_models"]["sglang_disagg/pyt_card-x"]["env_vars"]["DOCKER_IMAGE_NAME"]
        assert img == "localhost:5000/ci-sglang_disagg_pyt_card-x_card:latest"
        assert re.match(self.REF, img), img
