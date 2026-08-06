#!/usr/bin/env python3
"""
Tests for reporting the truth about a multi-node run.

A two-node run that ended with the scheduler recording exit code 3 used to finish with
"All model executions completed successfully!", because the only thing anyone looked at was
whether a number could be parsed out of a log. Three separate defects made that possible:
per-node exit codes were never collected, nothing distinguished a crashed workload from a
missing metric, and a failed deployment whose results could not be parsed produced an empty
failure list. These tests pin all three.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from madengine.deployment.slurm import SlurmDeployment


@pytest.fixture
def deployment():
    """A SLURM deployment with the scheduler and manifest stubbed out."""
    instance = object.__new__(SlurmDeployment)
    instance.nodes = 2
    instance.gpus_per_node = 8
    instance.console = MagicMock()
    return instance


def write_status(job_dir, node, exit_code, host="node-a"):
    """Write the marker a task script leaves behind."""
    node_dir = job_dir / f"node_{node}"
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "node.status").write_text(
        f"node={node}\nhost={host}\nexit_code={exit_code}\ntimestamp=2026-08-06T12:00:00+00:00\n"
    )
    return node_dir


class TestReadingNodeStatuses:
    """Per-node exit codes, which nothing collected before."""

    def test_statuses_are_read_in_node_order(self, deployment, tmp_path):
        """Node 10 sorts after node 2, unlike its directory name."""
        deployment.nodes = 11
        for node in (0, 2, 10):
            write_status(tmp_path, node, 0)

        statuses = deployment._read_node_statuses(tmp_path)

        assert [s["node"] for s in statuses] == [0, 2, 10]

    def test_exit_code_and_host_are_kept(self, deployment, tmp_path):
        """Which node failed matters as much as that one did."""
        write_status(tmp_path, 1, 137, host="node-b")

        status = deployment._read_node_statuses(tmp_path)[0]

        assert status["exit_code"] == 137
        assert status["host"] == "node-b"

    def test_missing_markers_are_not_invented(self, deployment, tmp_path):
        """A job from before these markers existed reports nothing, not failure."""
        assert deployment._read_node_statuses(tmp_path) == []

    def test_unparseable_marker_is_skipped(self, deployment, tmp_path):
        """A truncated marker must not take down result collection."""
        node_dir = tmp_path / "node_0"
        node_dir.mkdir()
        (node_dir / "node.status").write_text("exit_code=not-a-number\n")

        assert deployment._read_node_statuses(tmp_path) == []


class TestReportingNodeStatuses:
    """Turning markers into a verdict."""

    def test_all_zero_is_no_failure(self, deployment, tmp_path):
        """The happy path stays quiet."""
        write_status(tmp_path, 0, 0)
        write_status(tmp_path, 1, 0)

        assert deployment._report_node_statuses(deployment._read_node_statuses(tmp_path)) == []

    def test_non_zero_exit_is_a_failure(self, deployment, tmp_path):
        """The case that used to be reported as success."""
        write_status(tmp_path, 0, 0)
        write_status(tmp_path, 1, 3, host="node-b")

        failures = deployment._report_node_statuses(deployment._read_node_statuses(tmp_path))

        assert [f["node"] for f in failures] == [1]
        assert failures[0]["exit_code"] == 3

    def test_a_node_that_never_reported_is_a_failure(self, deployment, tmp_path):
        """A node killed mid-run writes no marker; silence is not success."""
        write_status(tmp_path, 0, 0)

        failures = deployment._report_node_statuses(deployment._read_node_statuses(tmp_path))

        assert [f["node"] for f in failures] == [1]
        assert failures[0]["missing"] is True

    def test_no_markers_at_all_means_no_verdict(self, deployment, tmp_path):
        """Without a single marker there is nothing to conclude from their absence."""
        assert deployment._report_node_statuses([]) == []


class TestIncompleteReason:
    """A verdict has to say what happened, not just that something did."""

    def test_a_failed_node_is_named_with_its_exit_code(self, deployment):
        """"Execution failed" alone sends the reader back into the logs."""
        reason = deployment._incomplete_reason(
            [{"node": 0, "host": "node-a", "exit_code": 3}], "FAILED"
        )

        assert reason == "node 0 exited 3"

    def test_a_lost_node_is_described_as_lost(self, deployment):
        """A node with no marker did not exit; it disappeared."""
        reason = deployment._incomplete_reason(
            [{"node": 1, "host": "", "exit_code": None, "missing": True}], None
        )

        assert reason == "node 1 reported no status (killed or lost)"

    def test_every_failing_node_is_listed(self, deployment):
        """Two nodes down is a different story from one."""
        reason = deployment._incomplete_reason(
            [
                {"node": 0, "host": "node-a", "exit_code": 3},
                {"node": 1, "host": "node-b", "exit_code": 137},
            ],
            "FAILED",
        )

        assert reason == "node 0 exited 3, node 1 exited 137"

    def test_without_markers_slurm_speaks(self, deployment):
        """The step the scheduler killed leaves nothing behind but its state."""
        assert deployment._incomplete_reason([], "TIMEOUT") == (
            "SLURM reports the job as TIMEOUT"
        )

    def test_with_no_evidence_at_all_the_verdict_stays_plain(self, deployment):
        """Nothing to quote, so claim nothing specific."""
        assert deployment._incomplete_reason([], None) == "the job did not complete"


class TestVerdictWhenANodeEndsBadly:
    """A metric outweighs an exit code, as it already does for a single node."""

    def test_a_measured_run_keeps_its_numbers_and_gains_a_warning(self, deployment):
        """The rank that reports no throughput exits non-zero on a healthy run."""
        results: dict = {}

        deployment._record_verdict(
            results, "llama", [{"node": 0, "host": "node-a", "exit_code": 3}], "FAILED", 8
        )

        assert "incomplete" not in results
        assert len(results["warnings"]) == 1
        assert "node 0 exited 3" in results["warnings"][0]
        assert "8 metric row" in results["warnings"][0]

    def test_a_run_with_nothing_measured_is_a_failure(self, deployment):
        """With no metric anywhere the node evidence is all there is."""
        results: dict = {}

        deployment._record_verdict(
            results, "llama", [{"node": 1, "host": "node-b", "exit_code": 137}], "FAILED", 0
        )

        assert "warnings" not in results
        assert results["incomplete"]["reason"] == "node 1 exited 137"
        assert results["incomplete"]["model"] == "llama"

    def test_a_lost_node_is_named_in_the_warning(self, deployment):
        """Whatever the verdict, the operator hears which node went missing."""
        results: dict = {}

        deployment._record_verdict(
            results, "llama", [{"node": 1, "exit_code": None, "missing": True}], None, 4
        )

        assert "reported no status" in results["warnings"][0]


class TestSummaryTheCliReportsOn:
    """What the orchestrator hands the CLI, for a job the scheduler calls failed."""

    @staticmethod
    def _summarise(metrics, is_success=False):
        from madengine.orchestration.run_orchestrator import RunOrchestrator

        orchestrator = object.__new__(RunOrchestrator)
        result = MagicMock()
        result.metrics = metrics
        result.is_success = is_success
        result.status.value = "failed"
        result.message = "Job 24505 failed: FAILED"
        result.deployment_id = "24505"
        return orchestrator._summarise_deployment(result, "slurm", "manifest.json")

    def test_measurements_survive_a_failed_job_state(self):
        """The scheduler's verdict follows from a rank that exits non-zero by design."""
        summary = self._summarise({"successful_runs": [{"model": "llama"}]})

        assert summary["failed_runs"] == []
        assert "measurement(s) came back" in summary["warnings"][0]

    def test_the_deployment_does_not_repeat_a_warning_already_made(self):
        """The node-level warning is the specific one; two say no more than one."""
        summary = self._summarise(
            {"successful_runs": [{"model": "llama"}], "warnings": ["node 0 exited 3"]}
        )

        assert summary["warnings"] == ["node 0 exited 3"]

    def test_a_failed_job_with_nothing_measured_is_still_a_failure(self):
        """Otherwise the caller sees an empty failure list and reports success."""
        summary = self._summarise({"successful_runs": []})

        assert len(summary["failed_runs"]) == 1
        assert summary["failed_runs"][0]["error"] == "Job 24505 failed: FAILED"

    def test_an_incomplete_run_names_the_model_that_did_not_finish(self):
        """The deployment layer's own verdict, carried through."""
        summary = self._summarise(
            {"successful_runs": [], "incomplete": {"model": "llama", "reason": "node 1 exited 9"}}
        )

        assert summary["incomplete"]["reason"] == "node 1 exited 9"
        assert [f["model"] for f in summary["failed_runs"]] == ["llama"]


class TestJobExitState:
    """SLURM's own verdict, for the failures that leave no marker."""

    def test_state_is_read_from_sacct(self, deployment):
        """The state, upper-cased, as sacct reports it."""
        with patch("madengine.deployment.slurm.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stdout="COMPLETED\n")

            assert deployment._job_exit_state("123") == "COMPLETED"

    def test_failed_state_is_reported(self, deployment):
        """A job the scheduler killed."""
        with patch("madengine.deployment.slurm.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stdout="CANCELLED by 1001\n")

            assert deployment._job_exit_state("123") == "CANCELLED BY 1001"

    def test_unavailable_sacct_claims_nothing(self, deployment):
        """No accounting database is not evidence either way."""
        with patch("madengine.deployment.slurm.subprocess.run") as run:
            run.return_value = MagicMock(returncode=1, stdout="")

            assert deployment._job_exit_state("123") is None

    def test_exception_claims_nothing(self, deployment):
        """Neither is a missing sacct binary."""
        with patch("madengine.deployment.slurm.subprocess.run", side_effect=OSError):
            assert deployment._job_exit_state("123") is None


class TestJobScriptRecordsOutcomes:
    """What the rendered job script does on the nodes."""

    @pytest.fixture
    def rendered(self, tmp_path):
        """The two-node job script, rendered exactly as prepare() renders it."""
        from tests.unit.test_slurm_job_template import (
            MODEL_ENTRY,
            _build_deployment,
        )

        deployment = _build_deployment(tmp_path)
        context = deployment._prepare_template_context(MODEL_ENTRY)
        return deployment.jinja_env.get_template("job.sh.j2").render(**context)

    def test_every_node_records_its_exit_code(self, rendered):
        """The marker the submit node reads back."""
        assert "exit_code=${TASK_EXIT}" in rendered
        assert "node.status" in rendered

    def test_the_step_is_not_torn_down_on_a_bad_exit_by_default(self, rendered):
        """The rank that collects nothing exits non-zero on a healthy multi-node run.

        Tearing the step down there would kill the rank that holds the numbers, so the
        teardown is something a workload opts into.
        """
        assert "--kill-on-bad-exit" not in rendered.split("srun ")[-1]

    def test_a_failing_run_still_reaches_the_collection_block(self, rendered):
        """Under errexit the script died on the madengine line, losing marker and artifacts."""
        madengine_call = rendered.index("$MAD_CLI_COMMAND run")
        assert rendered.rindex("set +e", 0, madengine_call) < madengine_call
        assert "TASK_EXIT=$?\nset -e" in rendered

    def test_teardown_can_be_turned_on(self, tmp_path):
        """A workload where every rank must exit zero can stop holding the allocation."""
        from tests.unit.test_slurm_job_template import (
            MODEL_ENTRY,
            _build_deployment,
        )

        deployment = _build_deployment(tmp_path, {"kill_on_bad_exit": True})
        context = deployment._prepare_template_context(MODEL_ENTRY)
        rendered = deployment.jinja_env.get_template("job.sh.j2").render(**context)

        assert "srun --kill-on-bad-exit=1" in rendered
