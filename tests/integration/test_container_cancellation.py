"""Integration tests for container teardown on cancellation.

These need a real docker daemon: the point is to prove that a container really
does stop itself when madengine goes away, which is exactly what no amount of
mocking can establish.

Run with::

    pytest tests/integration/test_container_cancellation.py -m slow

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
import uuid

# third-party modules
import pytest

# project modules
from madengine.core import lifecycle


pytestmark = [pytest.mark.slow, pytest.mark.requires_docker, pytest.mark.integration]

# Any image with a POSIX shell will do; the watchdog needs nothing else.
TEST_IMAGE = os.environ.get("MADENGINE_TEST_IMAGE", "busybox:latest")


def docker_available() -> bool:
    """Report whether a usable docker daemon is present.

    Returns:
        bool: True if the docker CLI exists and the daemon answers.
    """
    if shutil.which("docker") is None:
        return False
    code, _ = lifecycle.docker(["info", "--format", "{{.ServerVersion}}"], 15)
    return code == 0


requires_docker = pytest.mark.skipif(
    not docker_available(), reason="docker daemon not available"
)


@pytest.fixture
def container_name():
    """Yield a unique container name and remove it afterwards.

    Yields:
        str: The container name.
    """
    name = f"madengine-test-{uuid.uuid4().hex[:10]}"
    yield name
    lifecycle.docker(["rm", "-f", name], 30)


def wait_gone(name, timeout):
    """Wait for a container to stop running.

    Args:
        name (str): Container name.
        timeout (float): Seconds to wait.

    Returns:
        float: Seconds elapsed, or -1 if it was still running at the deadline.
    """
    start = time.time()
    while time.time() - start < timeout:
        if lifecycle.container_state(name) not in ("running", None):
            return time.time() - start
        time.sleep(0.5)
    return -1


@requires_docker
class TestDeadManSwitch:
    """The container stops itself when the heartbeat stops."""

    def _launch(self, workspace, name):
        """Start a watchdog container bind-mounting workspace."""
        code, out = lifecycle.docker(
            [
                "run", "-t", "-d",
                "--label", f"{lifecycle.LABEL_SESSION}={lifecycle.session_id()}",
                "--label", f"{lifecycle.LABEL_OWNER_PID}={os.getpid()}",
                "-v", f"{workspace}:/myworkspace/",
                "--name", name,
                TEST_IMAGE,
                "sh", "-c", lifecycle.watchdog_script(),
            ],
            60,
        )
        assert code == 0, out
        assert lifecycle.container_state(name) == "running"

    def test_container_survives_while_the_heartbeat_is_refreshed(
        self, tmp_path, container_name
    ):
        heartbeat = str(tmp_path / lifecycle.HEARTBEAT_FILENAME)
        lifecycle.start_heartbeat(heartbeat)
        try:
            self._launch(tmp_path, container_name)
            time.sleep(12)
            assert lifecycle.container_state(container_name) == "running"
        finally:
            lifecycle.stop_heartbeat(heartbeat)

    def test_container_stops_when_the_heartbeat_is_removed(
        self, tmp_path, container_name
    ):
        heartbeat = str(tmp_path / lifecycle.HEARTBEAT_FILENAME)
        lifecycle.start_heartbeat(heartbeat)
        self._launch(tmp_path, container_name)

        lifecycle.stop_heartbeat(heartbeat)
        elapsed = wait_gone(container_name, timeout=30)
        assert elapsed >= 0, "container outlived its heartbeat"

    def test_container_stops_when_madengine_is_sigkilled(
        self, tmp_path, container_name
    ):
        """The case that motivated all of this.

        SIGKILL leaves no chance to run cleanup code, so only something inside
        the container can save us -- and on a cancelled CI job the runner
        eventually does exactly this to the whole process tree.
        """
        script = textwrap.dedent(
            f"""
            import time
            from madengine.core import lifecycle
            lifecycle.start_heartbeat({str(tmp_path / lifecycle.HEARTBEAT_FILENAME)!r})
            print("ready", flush=True)
            time.sleep(600)
            """
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )
        try:
            assert proc.stdout.readline().strip() == "ready"
            self._launch(tmp_path, container_name)
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=10)

            # The heartbeat file is still on disk -- nobody removed it -- so
            # the container must notice it has gone stale on its own.
            assert os.path.exists(tmp_path / lifecycle.HEARTBEAT_FILENAME)
            elapsed = wait_gone(
                container_name, timeout=lifecycle.HEARTBEAT_STALE_AFTER + 30
            )
            assert elapsed >= 0, "container outlived a SIGKILLed madengine"
        finally:
            if proc.poll() is None:
                proc.kill()


@requires_docker
class TestReap:
    """The escalating reaper against a real daemon."""

    def test_reaps_a_running_container(self, container_name):
        code, out = lifecycle.docker(
            ["run", "-d", "--name", container_name, TEST_IMAGE, "sleep", "600"], 60
        )
        assert code == 0, out

        result = lifecycle.reap(container_name, verbose=False)
        assert result["removed"] is True
        assert result["wedged"] is False
        assert lifecycle.container_state(container_name) == ""

    def test_reaping_an_absent_container_is_a_no_op(self, container_name):
        result = lifecycle.reap(container_name, verbose=False)
        assert result["removed"] is True
        assert result["steps"] == []

    def test_host_pids_are_discoverable_from_the_cgroup(self, container_name):
        """The daemon-free path used when dockerd itself is blocked."""
        code, cid = lifecycle.docker(
            ["run", "-d", "--name", container_name, TEST_IMAGE, "sleep", "600"], 60
        )
        assert code == 0, cid

        pids = lifecycle.container_host_pids(cid.strip())
        if not pids:
            pytest.skip("cgroup layout not recognised on this host")
        assert all(os.path.exists(f"/proc/{pid}") for pid in pids)
        assert lifecycle.uninterruptible_pids(pids) == []


@requires_docker
class TestCleanupCommand:
    """`madengine cleanup` against a real daemon."""

    def test_removes_a_labelled_container_whose_owner_is_gone(self, container_name):
        code, out = lifecycle.docker(
            [
                "run", "-d",
                "--label", f"{lifecycle.LABEL_SESSION}={lifecycle.session_id()}",
                # A pid that cannot be running: cleanup must treat it as orphaned.
                "--label", f"{lifecycle.LABEL_OWNER_PID}=999999999",
                "--name", container_name,
                TEST_IMAGE, "sleep", "600",
            ],
            60,
        )
        assert code == 0, out

        proc = subprocess.run(
            [sys.executable, "-m", "madengine.cli.app", "cleanup"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stdout
        assert lifecycle.container_state(container_name) == ""

    def test_leaves_a_container_owned_by_a_live_process(self, container_name):
        code, out = lifecycle.docker(
            [
                "run", "-d",
                "--label", f"{lifecycle.LABEL_SESSION}={lifecycle.session_id()}",
                "--label", f"{lifecycle.LABEL_OWNER_PID}={os.getpid()}",
                "--name", container_name,
                TEST_IMAGE, "sleep", "600",
            ],
            60,
        )
        assert code == 0, out

        subprocess.run(
            [sys.executable, "-m", "madengine.cli.app", "cleanup"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, timeout=120,
        )
        assert lifecycle.container_state(container_name) == "running"
