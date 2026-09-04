"""Unit tests for the Docker class's cancellation safety.

The container is launched under a heartbeat watchdog so that it stops itself
when madengine goes away -- which is what happens on a cancelled CI job, where
the runner kills the whole process tree without our cleanup code ever running.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from madengine.core import lifecycle
from madengine.core.console import Console
from madengine.core.docker import Docker


CID = "a" * 64


@pytest.fixture
def quiet_state(tmp_path, monkeypatch):
    """Keep the registry and heartbeat out of the developer's home directory."""
    monkeypatch.setenv("MADENGINE_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def fake_docker(quiet_state):
    """Patch out everything that would talk to a docker daemon.

    Yields:
        SimpleNamespace-like MagicMock bundle with ``sh``, ``state`` and
        ``reap`` attached.
    """
    def default_sh(command, *args, **kwargs):
        # An empty answer to the pre-existence probe keeps the constructor on
        # the happy path; everything else just needs to be non-fatal.
        if "container ps" in command:
            return ""
        return "1000"

    sh = MagicMock(side_effect=default_sh)
    state = MagicMock(return_value="running")
    reap = MagicMock(return_value={"removed": True, "wedged": False, "stuck_pids": []})
    with patch.object(Console, "sh", sh), \
         patch.object(lifecycle, "container_state", state), \
         patch.object(lifecycle, "reap", reap), \
         patch.object(lifecycle, "docker", return_value=(0, CID)):
        bundle = MagicMock()
        bundle.sh, bundle.state, bundle.reap = sh, state, reap
        yield bundle


def run_line(sh):
    """Return the `docker run` command Console.sh was given.

    Args:
        sh (MagicMock): The patched Console.sh.

    Returns:
        str: The last docker run command, or "" if there was none.
    """
    runs = [c.args[0] for c in sh.call_args_list if "docker run" in str(c.args[:1])]
    return runs[-1] if runs else ""


class TestLaunch:
    """How the container is started."""

    def test_runs_the_watchdog_as_pid_one(self, fake_docker):
        Docker(image="img:tag", container_name="c1", dockerOpts="")
        assert run_line(fake_docker.sh).endswith(
            lifecycle.container_watchdog_command()
        )

    def test_stays_detached(self, fake_docker):
        # Detached is deliberate: the watchdog, not an attached stdin, is what
        # ties the container's life to ours.
        docker = Docker(image="img:tag", container_name="c1", dockerOpts="")
        assert run_line(fake_docker.sh).startswith("docker run -t -d ")
        docker.close()

    def test_heartbeat_exists_before_the_container_starts(self, fake_docker, quiet_state):
        seen = {}

        def record(command, *args, **kwargs):
            if "docker run" in command:
                seen["heartbeat"] = os.path.exists(
                    quiet_state / lifecycle.HEARTBEAT_FILENAME
                )
            return "1000"

        fake_docker.sh.side_effect = record
        Docker(image="img:tag", container_name="c1", dockerOpts="")
        # The watchdog exits immediately if the file is missing, so a late
        # first beat would make every container die on startup.
        assert seen["heartbeat"] is True

    def test_labels_the_container_for_out_of_band_cleanup(self, fake_docker):
        Docker(image="img:tag", container_name="c1", dockerOpts="")
        line = run_line(fake_docker.sh)
        assert f"--label {lifecycle.LABEL_SESSION}=" in line
        assert f"--label {lifecycle.LABEL_OWNER_PID}={os.getpid()}" in line

    def test_registers_the_container(self, fake_docker):
        # Held in a local: dropping the last reference runs __del__, which
        # tears the container down and unregisters it.
        docker = Docker(image="img:tag", container_name="c1", dockerOpts="")
        assert [e["id"] for e in lifecycle.registered()] == [CID]
        docker.close()

    def test_dropping_the_last_reference_tears_the_container_down(self, fake_docker):
        Docker(image="img:tag", container_name="c1", dockerOpts="")
        fake_docker.reap.assert_called_once_with("c1", CID)


class TestWatchdogFallback:
    """Images without a POSIX shell must still run."""

    def test_falls_back_to_cat(self, fake_docker, capsys):
        # First state check (watchdog container) fails, the fallback succeeds.
        fake_docker.state.side_effect = ["exited", "running"]
        Docker(image="img:tag", container_name="c1", dockerOpts="")

        runs = [c.args[0] for c in fake_docker.sh.call_args_list if "docker run" in c.args[0]]
        assert len(runs) == 2
        assert runs[1].endswith("cat ")
        # A silent fallback would quietly reintroduce the orphaned-container
        # bug this whole mechanism exists to fix.
        assert "will NOT stop by itself" in capsys.readouterr().out

    def test_drops_the_heartbeat_on_fallback(self, fake_docker, quiet_state):
        fake_docker.state.side_effect = ["exited", "running"]
        docker = Docker(image="img:tag", container_name="c1", dockerOpts="")
        assert docker.heartbeat_path is None
        assert not os.path.exists(quiet_state / lifecycle.HEARTBEAT_FILENAME)

    def test_raises_when_the_container_will_not_start_at_all(self, fake_docker):
        fake_docker.state.side_effect = ["exited", "exited"]
        with pytest.raises(RuntimeError, match="failed to start"):
            Docker(image="img:tag", container_name="c1", dockerOpts="")


class TestClose:
    """Tearing the container down."""

    def test_drops_heartbeat_then_reaps(self, fake_docker, quiet_state):
        docker = Docker(image="img:tag", container_name="c1", dockerOpts="")
        heartbeat = quiet_state / lifecycle.HEARTBEAT_FILENAME
        assert heartbeat.exists()

        docker.close()
        assert not heartbeat.exists()
        fake_docker.reap.assert_called_with("c1", CID)
        assert lifecycle.registered() == []

    def test_is_idempotent(self, fake_docker):
        docker = Docker(image="img:tag", container_name="c1", dockerOpts="")
        docker.close()
        docker.close()
        docker.close()
        assert fake_docker.reap.call_count == 1

    def test_context_manager_closes(self, fake_docker):
        with Docker(image="img:tag", container_name="c1", dockerOpts="") as docker:
            pass
        assert docker._closed is True
        assert fake_docker.reap.call_count == 1

    def test_half_built_instance_closes_safely(self):
        # __del__ reaches objects whose __init__ raised part-way through, and
        # that is precisely when a container may already be running.
        docker = Docker.__new__(Docker)
        docker.close()

    def test_keep_alive_leaves_the_container(self, fake_docker, quiet_state):
        docker = Docker(
            image="img:tag", container_name="c1", dockerOpts="", keep_alive=True
        )
        assert run_line(fake_docker.sh).endswith("cat ")
        assert not (quiet_state / lifecycle.HEARTBEAT_FILENAME).exists()

        docker.close()
        fake_docker.reap.assert_not_called()
        assert lifecycle.registered() == []
