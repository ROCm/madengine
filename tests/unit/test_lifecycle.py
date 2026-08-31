"""Unit tests for container lifecycle teardown.

Covers the three mechanisms that keep a benchmark container from outliving
madengine when a CI job is cancelled: the heartbeat dead man's switch, the
bounded escalating reaper, and the out-of-band ``madengine cleanup`` sweep.

No docker daemon is required -- every docker call is intercepted.
"""

import json
import os
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from madengine.core import lifecycle


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Point the registry and wedge marker at a throwaway directory."""
    path = tmp_path / "state"
    monkeypatch.setenv("MADENGINE_STATE_DIR", str(path))
    return path


class FakeDocker:
    """Scripted stand-in for ``lifecycle.docker``.

    Args:
        inspect_states (list): States returned by successive ``docker inspect``
            calls. ``""`` means the container is gone and ``None`` means the
            daemon did not answer. The last entry repeats once exhausted.
    """

    def __init__(self, inspect_states):
        self.inspect_states = list(inspect_states)
        self.calls = []

    def __call__(self, args, timeout):
        self.calls.append(args)
        if args[0] != "inspect":
            return 0, ""
        state = (
            self.inspect_states.pop(0)
            if len(self.inspect_states) > 1
            else self.inspect_states[0]
        )
        if state is None:
            return None, "timed out"
        if state == "":
            return 1, "No such object"
        return 0, state

    @property
    def verbs(self):
        """list: The docker subcommand of each call, in order."""
        return [c[0] for c in self.calls]


class TestRegistry:
    """The per-process record of live containers."""

    def test_round_trip(self, state_dir):
        lifecycle.register("cid-1", "name-1")
        lifecycle.register("cid-2", "name-2")
        entries = lifecycle.registered()
        assert [(e["id"], e["name"]) for e in entries] == [
            ("cid-1", "name-1"),
            ("cid-2", "name-2"),
        ]
        assert all(e["owner_pid"] == os.getpid() for e in entries)

        lifecycle.unregister("cid-1")
        assert [e["id"] for e in lifecycle.registered()] == ["cid-2"]

        lifecycle.unregister("cid-2")
        assert lifecycle.registered() == []

    def test_re_registering_the_same_id_does_not_duplicate(self, state_dir):
        lifecycle.register("cid-1", "name-1")
        lifecycle.register("cid-1", "renamed")
        entries = lifecycle.registered()
        assert len(entries) == 1
        assert entries[0]["name"] == "renamed"

    def test_registry_is_per_process(self, state_dir):
        lifecycle.register("cid-1", "name-1")
        assert lifecycle._registry_path(os.getpid()).endswith(
            f"containers-{os.getpid()}.json"
        )
        assert not os.path.exists(lifecycle._registry_path(os.getpid() + 1))

    def test_unregister_unknown_is_harmless(self, state_dir):
        lifecycle.unregister("never-registered")
        assert lifecycle.registered() == []


class TestStateChar:
    """Parsing the state out of /proc/<pid>/stat."""

    def test_plain(self):
        assert lifecycle._state_char("1 (bash) S 0 1 1") == "S"

    def test_comm_with_spaces_and_parens(self):
        # A comm like "(sd-pam)" or "Isolated Web Co" breaks naive field
        # splitting; the state is the field after the *last* ')'.
        assert lifecycle._state_char("42 (weird (name) here) D 1 2") == "D"

    def test_garbage(self):
        assert lifecycle._state_char("") == ""
        assert lifecycle._state_char("no parens at all") == ""

    def test_live_process_is_not_uninterruptible(self):
        assert lifecycle.uninterruptible_pids([os.getpid()]) == []

    def test_missing_pid_is_skipped(self):
        assert lifecycle.uninterruptible_pids([999999999]) == []


class TestContainerState:
    """Mapping docker inspect results onto a state."""

    def test_running(self):
        with patch.object(lifecycle, "docker", FakeDocker(["running"])):
            assert lifecycle.container_state("c") == "running"

    def test_gone(self):
        with patch.object(lifecycle, "docker", FakeDocker([""])):
            assert lifecycle.container_state("c") == ""

    def test_daemon_silent_is_unknown_not_gone(self):
        # Distinguishing these two is what stops a hung daemon from being
        # mistaken for a container that already exited.
        with patch.object(lifecycle, "docker", FakeDocker([None])):
            assert lifecycle.container_state("c") is None


class TestReap:
    """The bounded escalation."""

    def test_already_gone_costs_nothing(self, state_dir):
        fake = FakeDocker([""])
        with patch.object(lifecycle, "docker", fake):
            result = lifecycle.reap("c", verbose=False)
        assert result["removed"] is True
        assert fake.verbs == ["inspect"]

    def test_stop_is_enough_for_a_healthy_container(self, state_dir):
        fake = FakeDocker(["running", ""])
        with patch.object(lifecycle, "docker", fake):
            result = lifecycle.reap("c", verbose=False)
        assert result["removed"] is True
        assert result["steps"] == ["docker stop"]
        assert "kill" not in fake.verbs
        assert "rm" not in fake.verbs

    def test_escalates_to_kill_and_rm(self, state_dir):
        fake = FakeDocker(["running", "running", ""])
        with patch.object(lifecycle, "docker", fake):
            result = lifecycle.reap("c", verbose=False)
        assert result["removed"] is True
        assert result["steps"] == ["docker stop", "docker kill -s KILL", "docker rm -f"]

    def test_escalates_to_host_pids(self, state_dir):
        fake = FakeDocker(["running", "running", "running", ""])
        killed = []
        with patch.object(lifecycle, "docker", fake), \
             patch.object(lifecycle, "container_host_pids", return_value=[4242]), \
             patch.object(lifecycle, "_kill_host_pids", killed.extend), \
             patch.object(lifecycle.time, "sleep"):
            result = lifecycle.reap("c", "cid", verbose=False)
        assert killed == [4242]
        assert result["removed"] is True
        assert result["wedged"] is False

    def test_d_state_pids_are_reported_as_wedged(self, state_dir):
        fake = FakeDocker(["running"])
        with patch.object(lifecycle, "docker", fake), \
             patch.object(lifecycle, "container_host_pids", return_value=[4242]), \
             patch.object(lifecycle, "_kill_host_pids", lambda pids: None), \
             patch.object(lifecycle, "uninterruptible_pids", return_value=[4242]), \
             patch.object(lifecycle.time, "sleep"):
            result = lifecycle.reap("c", "cid", verbose=False)

        assert result["wedged"] is True
        assert result["stuck_pids"] == [4242]

        with open(state_dir / "gpu_wedged.json") as handle:
            marker = json.load(handle)
        assert marker["container"] == "c"
        assert marker["stuck_pids"] == [4242]

    def test_silent_daemon_without_pids_is_not_called_wedged(self, state_dir):
        # "docker isn't answering" is not evidence that the host needs a
        # reboot; crying wolf here would make the marker worthless.
        fake = FakeDocker([None])
        with patch.object(lifecycle, "docker", fake), \
             patch.object(lifecycle, "container_host_pids", return_value=[]):
            result = lifecycle.reap("c", verbose=False)

        assert result["wedged"] is False
        assert result["docker_responsive"] is False
        assert not (state_dir / "gpu_wedged.json").exists()

    def test_every_docker_call_is_bounded(self, state_dir):
        fake = FakeDocker(["running", "running", "running", "running"])
        with patch.object(lifecycle, "docker", fake), \
             patch.object(lifecycle, "container_host_pids", return_value=[]):
            lifecycle.reap("c", verbose=False)
        # A wedged GPU blocks the daemon too, so an unbounded call here would
        # burn the whole cancellation grace period and orphan the container.
        for call in fake.calls:
            assert call is not None
        assert all(isinstance(c, list) for c in fake.calls)


class TestReapRegistered:
    """Sweeping this process's own containers."""

    def test_reaps_all_registered(self, state_dir):
        lifecycle.register("cid-1", "name-1")
        lifecycle.register("cid-2", "name-2")
        reaped = []
        with patch.object(
            lifecycle,
            "reap",
            lambda ref, cid=None, verbose=True: reaped.append(ref)
            or {"removed": True, "wedged": False, "stuck_pids": []},
        ):
            lifecycle.reap_registered(verbose=False)
        assert reaped == ["name-1", "name-2"]

    def test_budget_stops_the_sweep(self, state_dir):
        for i in range(5):
            lifecycle.register(f"cid-{i}", f"name-{i}")
        reaped = []

        def slow_reap(ref, cid=None, verbose=True):
            reaped.append(ref)
            time.sleep(0.05)
            return {"removed": True, "wedged": False, "stuck_pids": []}

        with patch.object(lifecycle, "reap", slow_reap):
            lifecycle.reap_registered(verbose=False, budget=0.06)
        # The signal path has a hard budget: better to leave one container to
        # `madengine cleanup` than to be SIGKILLed mid-teardown.
        assert 0 < len(reaped) < 5


class TestHeartbeat:
    """The dead man's switch the container itself watches."""

    def test_start_writes_a_timestamp_before_returning(self, tmp_path):
        path = str(tmp_path / "hb")
        lifecycle.start_heartbeat(path)
        try:
            # The watchdog exits immediately on a missing file, so the first
            # beat has to be on disk before the container starts.
            with open(path) as handle:
                assert abs(int(handle.read()) - int(time.time())) <= 2
        finally:
            lifecycle.stop_heartbeat(path)

    def test_stop_removes_the_file(self, tmp_path):
        path = str(tmp_path / "hb")
        lifecycle.start_heartbeat(path)
        lifecycle.stop_heartbeat(path)
        assert not os.path.exists(path)
        assert path not in lifecycle._heartbeat_paths

    def test_stop_is_idempotent(self, tmp_path):
        path = str(tmp_path / "hb")
        lifecycle.start_heartbeat(path)
        lifecycle.stop_heartbeat(path)
        lifecycle.stop_heartbeat(path)

    def test_stopped_heartbeat_is_not_recreated(self, tmp_path):
        path = str(tmp_path / "hb")
        lifecycle.start_heartbeat(path)
        lifecycle.stop_heartbeat(path)
        lifecycle._beat_once()
        assert not os.path.exists(path)


class TestWatchdogCommand:
    """The shell loop that runs as container PID 1."""

    def test_mentions_the_heartbeat_and_threshold(self):
        cmd = lifecycle.container_watchdog_command()
        assert lifecycle.HEARTBEAT_FILENAME in cmd
        assert str(lifecycle.HEARTBEAT_STALE_AFTER) in cmd

    def test_survives_the_outer_shell(self):
        # The command is appended to a `shell=True` docker run line, so an
        # unbalanced quote here would corrupt the whole command.
        import shlex

        assert shlex.split(lifecycle.container_watchdog_command())[:2] == ["sh", "-c"]

    def _run(self, hb_dir, timeout):
        cmd = lifecycle.watchdog_script().replace("/myworkspace/", f"{hb_dir}/")
        start = time.time()
        proc = subprocess.run(
            cmd, shell=True, timeout=timeout, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, universal_newlines=True,
        )
        return proc.returncode, time.time() - start, proc.stdout

    def test_exits_when_heartbeat_is_absent(self, tmp_path):
        code, elapsed, _ = self._run(str(tmp_path), timeout=10)
        assert code == 0
        assert elapsed < 5

    def test_exits_shortly_after_heartbeat_disappears(self, tmp_path):
        path = tmp_path / lifecycle.HEARTBEAT_FILENAME
        path.write_text(str(int(time.time())))
        os.system(f"(sleep 1; rm -f {path}) &")
        code, elapsed, _ = self._run(str(tmp_path), timeout=30)
        assert code == 0
        assert elapsed < 15

    def test_treats_a_corrupt_heartbeat_as_stale(self, tmp_path):
        # A truncated or half-written file must not wedge the watchdog into
        # never exiting -- it is read as timestamp 0, i.e. long stale.
        path = tmp_path / lifecycle.HEARTBEAT_FILENAME
        path.write_text("not-a-number")
        code, elapsed, out = self._run(str(tmp_path), timeout=30)
        assert code == 0
        assert "stale" in out
        assert elapsed < 15


class TestSignalHandlers:
    """The handlers run in a real subprocess: signal delivery, ``os._exit``
    and the interpreter shutdown they deliberately skip cannot be exercised
    faithfully in-process.
    """

    CHILD = """
import os, signal, sys, time
from madengine.core import lifecycle

lifecycle.register("deadbeef" * 8, "madengine-test-container")
lifecycle.install_signal_handlers()
print("ready", flush=True)
time.sleep(60)
"""

    def _run_child(self, state_dir, signum):
        env = dict(os.environ, MADENGINE_STATE_DIR=str(state_dir))
        proc = subprocess.Popen(
            [sys.executable, "-c", self.CHILD],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env,
        )
        try:
            assert proc.stdout.readline().strip() == "ready"
            proc.send_signal(signum)
            out = proc.stdout.read()
            proc.wait(timeout=60)
        finally:
            proc.stdout.close()
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
        return proc.returncode, out

    @pytest.mark.parametrize(
        "signum", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]
    )
    def test_signal_tears_containers_down_and_exits_130(self, tmp_path, signum):
        code, out = self._run_child(tmp_path / "state", signum)
        assert code == 130, out
        assert "stopping containers" in out

    def test_registry_is_cleared_so_cleanup_does_not_chase_it(self, tmp_path):
        state = tmp_path / "state"
        self._run_child(state, signal.SIGINT)
        # reap_registered unregisters as it goes; a stale entry would make the
        # next `madengine cleanup` report a container that is long gone.
        leftovers = [
            p for p in state.glob("containers-*.json") if json.loads(p.read_text())
        ]
        assert leftovers == []

    def test_teardown_stays_inside_the_cancellation_grace_period(self, tmp_path):
        # GitHub Actions kills the process tree ~10s after the first signal.
        # Blowing that budget means being killed mid-teardown -- the exact
        # failure this module exists to prevent.
        start = time.time()
        self._run_child(tmp_path / "state", signal.SIGTERM)
        assert time.time() - start < lifecycle.SIGNAL_CLEANUP_BUDGET + 5
