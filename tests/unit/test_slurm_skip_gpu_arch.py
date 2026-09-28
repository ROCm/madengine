"""
skip_gpu_arch on the SLURM path.

It used to be enforced only by _execute_local, which reads the arch of the
machine madengine runs on. On SLURM that is a GPU-less login node, and the
SLURM path never checked at all, so a card declaring `skip_gpu_arch: gfx950`
was sbatched onto gfx950 nodes and failed there. These tests pin the
replacement: slurm.gpu_arch wins, else one srun rocminfo probe, and an unknown
arch never skips anything.

No test here reaches a cluster: every srun/squeue is a mocked subprocess.run.
"""

import csv
import io
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console as RichConsole

from madengine.deployment.slurm_node_selector import (
    SlurmNodeSelector,
    parse_gpu_arch,
)
from madengine.orchestration.run_orchestrator import RunOrchestrator

SELECTOR_RUN = "madengine.deployment.slurm_node_selector.subprocess.run"

ROCMINFO = """\
*******
Agent 1
*******
  Name:                    AMD EPYC 9654 96-Core Processor
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx950
  Marketing Name:          AMD Instinct MI355X
  Device Type:             GPU
"""


def _manifest(tmp_path, models):
    """Write a manifest with one image/model pair per (key, skip_gpu_arch)."""
    manifest = {
        "built_images": {k: {"docker_image": k} for k, _ in models},
        "built_models": {
            k: {"name": k, "skip_gpu_arch": skip, "distributed": {"launcher": "slurm_multi"}}
            for k, skip in models
        },
        "context": {},
    }
    path = tmp_path / "build_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    # update_perf_csv also drops a perf_entry.csv into the cwd.
    monkeypatch.chdir(tmp_path)


def _orchestrator(tmp_path, slurm=None, disable=False):
    args = SimpleNamespace(
        additional_context=None,
        live_output=False,
        output=str(tmp_path / "perf.csv"),
        disable_skip_gpu_arch=disable,
        require_pinned_image=False,
    )
    orch = RunOrchestrator(args, additional_context={"slurm": slurm or {"partition": "mi355"}})
    orch.rich_console = RichConsole(file=io.StringIO(), width=400)
    return orch


def _printed(orch):
    return orch.rich_console.file.getvalue()


def _perf_rows(tmp_path):
    path = tmp_path / "perf.csv"
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


class TestParseGpuArch:
    def test_skips_cpu_agents_and_returns_first_gpu(self):
        assert parse_gpu_arch(ROCMINFO) == "gfx950"

    def test_hex_suffix_is_kept(self):
        # gfx90a is a real target; a [0-9]+ pattern would report gfx90.
        assert parse_gpu_arch("  Name: gfx90a\n") == "gfx90a"

    @pytest.mark.parametrize("output", ["", None, "Name: AMD EPYC\n"])
    def test_no_gfx_is_unknown(self, output):
        assert parse_gpu_arch(output) is None


class TestProbeGpuArch:
    def test_probe_names_partition_and_parses_output(self):
        selector = SlurmNodeSelector(console=MagicMock(), reservation="resv", timeout=5)
        ok = subprocess.CompletedProcess([], 0, stdout=ROCMINFO, stderr="")
        with patch(SELECTOR_RUN, return_value=ok) as run:
            arch = selector.probe_gpu_arch(partition="mi355", constraint="mi355x")
        assert arch == "gfx950"
        srun = run.call_args_list[0].args[0]
        assert srun[0] == "srun"
        assert "--partition=mi355" in srun
        assert "--reservation=resv" in srun
        assert "--constraint=mi355x" in srun
        assert "--nodes=1" in srun

    def test_timeout_returns_none(self):
        selector = SlurmNodeSelector(console=MagicMock(), timeout=5)
        with patch(SELECTOR_RUN, side_effect=subprocess.TimeoutExpired("srun", 5)):
            assert selector.probe_gpu_arch(partition="mi355") is None


class TestSlurmSkipGpuArch:
    def test_explicit_arch_skips_matching_card(self, tmp_path):
        manifest = _manifest(tmp_path, [("m_skip", "gfx950"), ("m_run", "")])
        orch = _orchestrator(tmp_path, {"partition": "mi355", "gpu_arch": "gfx950"})

        with patch(SELECTOR_RUN) as run:
            remaining, skipped = orch._apply_skip_gpu_arch_for_slurm(str(manifest))

        run.assert_not_called()  # an explicit value needs no probe
        assert (remaining, skipped) == (1, ["m_skip"])
        data = json.loads(manifest.read_text())
        # Both maps: SlurmDeployment submits built_models' FIRST key, so a
        # skipped model left there would still be the one sbatched.
        assert list(data["built_images"]) == ["m_run"]
        assert list(data["built_models"]) == ["m_run"]
        rows = _perf_rows(tmp_path)
        assert [(r["model"], r["status"]) for r in rows] == [("m_skip", "SKIPPED")]
        assert rows[0]["gpu_architecture"] == "gfx950"

    def test_explicit_arch_keeps_non_matching_card(self, tmp_path):
        manifest = _manifest(tmp_path, [("m1", "gfx950")])
        orch = _orchestrator(tmp_path, {"partition": "mi300", "gpu_arch": "gfx942"})
        before = manifest.read_text()

        with patch(SELECTOR_RUN) as run:
            remaining, skipped = orch._apply_skip_gpu_arch_for_slurm(str(manifest))

        run.assert_not_called()
        assert remaining == 1
        assert manifest.read_text() == before
        assert _perf_rows(tmp_path) == []

    def test_probed_arch_skips_matching_card(self, tmp_path):
        manifest = _manifest(tmp_path, [("m1", "gfx942, gfx950")])
        orch = _orchestrator(tmp_path)
        ok = subprocess.CompletedProcess([], 0, stdout=ROCMINFO, stderr="")

        with patch(SELECTOR_RUN, return_value=ok) as run:
            remaining, skipped = orch._apply_skip_gpu_arch_for_slurm(str(manifest))

        assert remaining == 0
        assert "--partition=mi355" in run.call_args_list[0].args[0]
        assert [r["status"] for r in _perf_rows(tmp_path)] == ["SKIPPED"]

    def test_list_form_skip_gpu_arch_is_honoured(self, tmp_path):
        # The --use-image synthetic manifest stores the field as a list; str.split
        # on it used to raise and abort the whole SLURM run.
        manifest = _manifest(tmp_path, [("m1", ["gfx942", "gfx950"])])
        orch = _orchestrator(tmp_path, {"partition": "mi355", "gpu_arch": "gfx950"})
        assert orch._apply_skip_gpu_arch_for_slurm(str(manifest)) == (0, ["m1"])

    def test_probe_failure_keeps_card_and_warns(self, tmp_path):
        manifest = _manifest(tmp_path, [("m1", "gfx950")])
        orch = _orchestrator(tmp_path)
        before = manifest.read_text()
        failed = subprocess.CompletedProcess(
            [], 1, stdout="", stderr="srun: error: Invalid partition name specified"
        )

        with patch(SELECTOR_RUN, return_value=failed):
            remaining, skipped = orch._apply_skip_gpu_arch_for_slurm(str(manifest))

        assert remaining == 1
        assert manifest.read_text() == before
        assert _perf_rows(tmp_path) == []
        out = _printed(orch)
        assert "skip_gpu_arch could NOT be enforced" in out
        assert "m1" in out
        assert "slurm.gpu_arch" in out

    def test_probe_that_finds_no_gpu_keeps_card(self, tmp_path):
        manifest = _manifest(tmp_path, [("m1", "gfx950")])
        orch = _orchestrator(tmp_path)
        cpu_only = subprocess.CompletedProcess([], 0, stdout="Name: AMD EPYC\n", stderr="")

        with patch(SELECTOR_RUN, return_value=cpu_only):
            assert orch._apply_skip_gpu_arch_for_slurm(str(manifest)) == (1, [])
        assert "could NOT be enforced" in _printed(orch)

    def test_disable_flag_bypasses_check_and_probe(self, tmp_path):
        manifest = _manifest(tmp_path, [("m1", "gfx950")])
        orch = _orchestrator(tmp_path, {"partition": "mi355", "gpu_arch": "gfx950"}, disable=True)
        before = manifest.read_text()

        with patch(SELECTOR_RUN) as run:
            remaining, skipped = orch._apply_skip_gpu_arch_for_slurm(str(manifest))

        run.assert_not_called()
        assert remaining == 1
        assert manifest.read_text() == before
        assert _perf_rows(tmp_path) == []

    def test_no_restricted_card_spends_no_probe(self, tmp_path):
        manifest = _manifest(tmp_path, [("m1", "")])
        orch = _orchestrator(tmp_path)
        with patch(SELECTOR_RUN) as run:
            assert orch._apply_skip_gpu_arch_for_slurm(str(manifest)) == (1, [])
        run.assert_not_called()


class TestExecuteDistributedSlurm:
    def test_all_skipped_slurm_multi_card_is_never_submitted(self, tmp_path):
        manifest = _manifest(tmp_path, [("m1", "gfx950")])
        orch = _orchestrator(tmp_path, {"partition": "mi355", "gpu_arch": "gfx950"})

        with patch("madengine.deployment.factory.DeploymentFactory.create") as create:
            result = orch._execute_distributed("slurm", str(manifest))

        create.assert_not_called()
        assert result["failed_runs"] == []
        assert [(r["model"], r["status"]) for r in result["successful_runs"]] == [
            ("m1", "SKIPPED")
        ]
        assert [r["status"] for r in _perf_rows(tmp_path)] == ["SKIPPED"]

    def test_kept_card_is_submitted(self, tmp_path):
        manifest = _manifest(tmp_path, [("m1", "gfx950")])
        orch = _orchestrator(tmp_path, {"partition": "mi300", "gpu_arch": "gfx942"})
        deployment = MagicMock()
        deployment.execute.return_value = MagicMock(is_success=True, metrics=None)

        with patch(
            "madengine.deployment.factory.DeploymentFactory.create", return_value=deployment
        ) as create:
            orch._execute_distributed("slurm", str(manifest))

        create.assert_called_once()
        deployment.execute.assert_called_once()
        submitted = json.loads(manifest.read_text())
        assert list(submitted["built_models"]) == ["m1"]


class TestFailedDeploymentIsAFailure:
    """Builds 119, 121 and 129: SLURM said FAILED / TIMEOUT, madengine printed
    "Deployment to slurm failed", then "All model executions completed
    successfully" and exited 0, because failed_runs only held perf-CSV rows and
    the job died before writing any."""

    def _run(self, tmp_path, result):
        manifest = _manifest(tmp_path, [("m1", None)])
        orch = _orchestrator(tmp_path, {"partition": "mi300", "gpu_arch": "gfx942"})
        deployment = MagicMock()
        deployment.execute.return_value = result
        with patch(
            "madengine.deployment.factory.DeploymentFactory.create", return_value=deployment
        ):
            return orch._execute_distributed("slurm", str(manifest))

    def test_failed_job_without_perf_rows_fails_every_submitted_model(self, tmp_path):
        summary = self._run(
            tmp_path,
            MagicMock(is_success=False, metrics={"successful_runs": [], "failed_runs": []},
                      message="Job 441903 failed: FAILED", deployment_id="441903"),
        )
        assert [(r["model"], r["status"]) for r in summary["failed_runs"]] == [("m1", "FAILURE")]
        assert summary["failed_runs"][0]["error"] == "Job 441903 failed: FAILED"

    def test_failed_job_with_no_metrics_at_all_still_fails(self, tmp_path):
        summary = self._run(
            tmp_path,
            MagicMock(is_success=False, metrics=None, message="Job 441934 failed: TIMEOUT",
                      deployment_id="441934"),
        )
        assert [r["status"] for r in summary["failed_runs"]] == ["FAILURE"]

    def test_existing_failure_rows_are_not_duplicated(self, tmp_path):
        row = {"model": "m1", "status": "FAILURE", "performance": None, "duration": None}
        summary = self._run(
            tmp_path,
            MagicMock(is_success=False, metrics={"successful_runs": [], "failed_runs": [row]},
                      message="x", deployment_id="1"),
        )
        assert summary["failed_runs"] == [row]

    def test_successful_deployment_is_unchanged(self, tmp_path):
        ok = {"model": "m1", "status": "SUCCESS", "performance": "1", "duration": "2"}
        summary = self._run(
            tmp_path,
            MagicMock(is_success=True, metrics={"successful_runs": [ok], "failed_runs": []},
                      message="", deployment_id="1"),
        )
        assert summary == {"successful_runs": [ok], "failed_runs": []}


class TestOutputHoldsThisRun:
    """The pipeline runs `madengine run -o perf_oci-64.csv`. The SLURM deployment
    appended to a cumulative perf.csv instead, sliced it from a start row counted
    in the -o file (0, since it did not exist yet), and never wrote the -o file:
    no build's numbers were ever archived."""

    HEADER = "model,status,performance\n"

    def test_only_this_runs_rows_reach_output_and_start_row_is_the_working_files(self, tmp_path):
        from madengine.orchestration import run_orchestrator as ro

        (tmp_path / "perf.csv").write_text(self.HEADER + "old1,SUCCESS,1\nold2,SUCCESS,2\n")
        manifest = _manifest(tmp_path, [("m1", None)])
        out = tmp_path / "perf_oci-64.csv"
        orch = _orchestrator(tmp_path, {"partition": "mi300", "gpu_arch": "gfx942"})
        orch.args.output = str(out)

        seen = {}

        def execute():
            seen["start"] = orch.additional_context["session_start_row"]
            with open("perf.csv", "a") as f:
                f.write("m1,SUCCESS,42\n")
            return MagicMock(is_success=True, metrics=None, message="", deployment_id="1")

        deployment = MagicMock()
        deployment.execute.side_effect = execute
        with patch("madengine.deployment.factory.DeploymentFactory.create", return_value=deployment):
            orch._execute_distributed("slurm", str(manifest))

        assert seen["start"] == 2
        assert out.read_text() == self.HEADER + "m1,SUCCESS,42\n"
        assert ro._csv_data_row_count(str(tmp_path / "perf.csv")) == 3

    def test_rows_are_exported_even_when_the_deployment_raises(self, tmp_path):
        manifest = _manifest(tmp_path, [("m1", None)])
        out = tmp_path / "perf_oci-64.csv"
        orch = _orchestrator(tmp_path, {"partition": "mi300", "gpu_arch": "gfx942"})
        orch.args.output = str(out)

        def execute():
            with open("perf.csv", "w") as f:
                f.write(self.HEADER + "m1,FAILURE,\n")
            raise RuntimeError("monitor lost the job")

        deployment = MagicMock()
        deployment.execute.side_effect = execute
        with patch("madengine.deployment.factory.DeploymentFactory.create", return_value=deployment):
            with pytest.raises(RuntimeError):
                orch._execute_distributed("slurm", str(manifest))
        assert out.read_text() == self.HEADER + "m1,FAILURE,\n"

    def test_export_is_a_noop_when_output_is_the_working_file(self, tmp_path):
        from madengine.orchestration.run_orchestrator import _export_session_rows

        (tmp_path / "perf.csv").write_text(self.HEADER + "a,SUCCESS,1\n")
        assert _export_session_rows("perf.csv", 0, str(tmp_path / "perf.csv")) == 0
        assert _export_session_rows("perf.csv", 0, None) == 0

    def test_existing_output_keeps_its_columns(self, tmp_path):
        from madengine.orchestration.run_orchestrator import _export_session_rows

        (tmp_path / "perf.csv").write_text(self.HEADER + "a,SUCCESS,1\n")
        out = tmp_path / "o.csv"
        out.write_text("status,model\nSKIPPED,s\n")
        assert _export_session_rows("perf.csv", 0, str(out)) == 1
        assert out.read_text().splitlines() == ["status,model", "SKIPPED,s", "SUCCESS,a"]
