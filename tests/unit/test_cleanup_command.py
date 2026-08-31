"""Unit tests for the `madengine cleanup` command.

The command exists so a cancelled CI job can reap containers from an
``if: always()`` step, after the run that created them is already gone.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import importlib
import os
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from madengine.cli.app import app
from madengine.cli.constants import ExitCode
from madengine.core import lifecycle

# The package re-exports the command function under the module's own name, so
# reach for the module explicitly.
cleanup_module = importlib.import_module("madengine.cli.commands.cleanup")


runner = CliRunner()

DEAD_PID = 999999999
LIVE_PID = os.getpid()


def ps_output(*rows):
    """Build a fake `docker ps --format` payload.

    Args:
        *rows: ``(id, name, owner_pid, state)`` tuples.

    Returns:
        tuple: A ``lifecycle.docker`` style ``(returncode, output)`` pair.
    """
    return 0, "\n".join("\t".join(str(f) for f in row) for row in rows)


class TestListing:
    """Parsing docker's tab-separated output."""

    def test_parses_rows(self):
        with patch.object(
            lifecycle,
            "docker",
            return_value=ps_output(("abc123", "container_a", 42, "running")),
        ):
            containers = cleanup_module._list_madengine_containers()
        assert containers == [
            {"id": "abc123", "name": "container_a", "owner_pid": 42, "state": "running"}
        ]

    def test_tolerates_a_missing_owner_label(self):
        with patch.object(
            lifecycle,
            "docker",
            return_value=ps_output(("abc123", "container_a", "", "exited")),
        ):
            containers = cleanup_module._list_madengine_containers()
        assert containers[0]["owner_pid"] == 0

    def test_ignores_malformed_lines(self):
        with patch.object(lifecycle, "docker", return_value=(0, "garbage")):
            assert cleanup_module._list_madengine_containers() == []

    def test_silent_daemon_yields_nothing(self):
        with patch.object(lifecycle, "docker", return_value=(None, "timed out")):
            assert cleanup_module._list_madengine_containers() == []


class TestPidLiveness:
    """Deciding whether a container's owner is still around."""

    def test_own_pid_is_alive(self):
        assert cleanup_module._pid_alive(LIVE_PID) is True

    def test_absent_pid_is_dead(self):
        assert cleanup_module._pid_alive(DEAD_PID) is False


class TestCleanupCommand:
    """End-to-end behaviour of the command."""

    def _invoke(self, containers, args=None, reap_result=None):
        reaped = []

        def fake_reap(ref, cid=None, verbose=True):
            reaped.append(ref)
            return reap_result or {
                "removed": True,
                "wedged": False,
                "stuck_pids": [],
            }

        with patch.object(
            cleanup_module, "_list_madengine_containers", return_value=containers
        ), patch.object(lifecycle, "reap", fake_reap):
            result = runner.invoke(app, ["cleanup"] + (args or []))
        return result, reaped

    def test_nothing_to_do(self):
        result, reaped = self._invoke([])
        assert result.exit_code == ExitCode.SUCCESS
        assert reaped == []

    def test_reaps_containers_whose_owner_is_gone(self):
        result, reaped = self._invoke(
            [{"id": "abc", "name": "c1", "owner_pid": DEAD_PID, "state": "running"}]
        )
        assert result.exit_code == ExitCode.SUCCESS
        assert reaped == ["abc"]

    def test_leaves_containers_of_a_running_madengine_alone(self):
        # Cron and `if: always()` steps run while other benchmarks are in
        # flight; killing a live run's container would be worse than the leak.
        result, reaped = self._invoke(
            [{"id": "abc", "name": "c1", "owner_pid": LIVE_PID, "state": "running"}]
        )
        assert result.exit_code == ExitCode.SUCCESS
        assert reaped == []

    def test_all_overrides_the_liveness_check(self):
        result, reaped = self._invoke(
            [{"id": "abc", "name": "c1", "owner_pid": LIVE_PID, "state": "running"}],
            args=["--all"],
        )
        assert result.exit_code == ExitCode.SUCCESS
        assert reaped == ["abc"]

    def test_unlabelled_owner_is_treated_as_orphaned(self):
        result, reaped = self._invoke(
            [{"id": "abc", "name": "c1", "owner_pid": 0, "state": "exited"}]
        )
        assert reaped == ["abc"]

    def test_dry_run_removes_nothing(self):
        result, reaped = self._invoke(
            [{"id": "abcdef123456", "name": "c1", "owner_pid": DEAD_PID, "state": "running"}],
            args=["--dry-run"],
        )
        assert result.exit_code == ExitCode.SUCCESS
        assert reaped == []
        assert "would remove" in result.stdout

    def test_wedged_container_gets_a_distinct_exit_code(self):
        # The workflow keys off this to take the runner out of rotation.
        result, _ = self._invoke(
            [{"id": "abc", "name": "c1", "owner_pid": DEAD_PID, "state": "running"}],
            reap_result={"removed": False, "wedged": True, "stuck_pids": [7]},
        )
        assert result.exit_code == ExitCode.GPU_WEDGED
        assert result.exit_code != ExitCode.SUCCESS
        assert result.exit_code != ExitCode.FAILURE
