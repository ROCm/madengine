"""SLURM-compatible schedulers that implement a subset of the CLI (spur).

Spur exposes SLURM_NODELIST already expanded and has no `scontrol show
hostnames`, rejects `sacct -X`, and can refuse sbatch during a Raft leader
election. Each case here keeps stock SLURM's behaviour and adds the spur one.
"""

import subprocess
from types import SimpleNamespace

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
