"""mad_k8s_node.sh, the entry point of every pod of a slurm_multi card on Kubernetes.

Only the entry script's own status and signal handling are under test: the card's
script here calls none of the pod's tools.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

NODE_SH = Path(__file__).resolve().parents[2] / "src/madengine/deployment/k8s_shim/mad_k8s_node.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and process groups")


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill -0; it has stopped all the same.
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def _pod(tmp_path, card_script, index):
    workspace = tmp_path / "ws"
    (workspace / "scripts" / "common" / "dispatch").mkdir(parents=True, exist_ok=True)
    (workspace / "scripts" / "common" / "dispatch" / "dispatch.sh").write_text("")
    (workspace / "scripts" / "card").mkdir(parents=True, exist_ok=True)
    (workspace / "scripts" / "card" / "run.sh").write_text(card_script)
    env = dict(os.environ, MAD_K8S_WORKSPACE=str(workspace), MAD_K8S_STATE=str(tmp_path / "state"),
               MAD_K8S_RESULTS=str(tmp_path / "results"), MAD_K8S_NNODES="2",
               MAD_K8S_NODE_PREFIX="job", MAD_K8S_JOB_ID="7",
               MAD_K8S_SCRIPT="scripts/card/run.sh", JOB_COMPLETION_INDEX=str(index))
    return subprocess.Popen(["bash", str(NODE_SH)], env=env, cwd=tmp_path,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def test_term_reaches_the_launchers_children_and_pod0_still_publishes(tmp_path):
    pidfile = tmp_path / "child.pid"
    # No TERM trap: the script dies, and only a signal to its process group
    # reaches the process it started.
    proc = _pod(tmp_path, f"""echo 'model,performance' > perf.csv
sleep 300 &
echo $! > {pidfile}
wait
""", 0)
    child = None
    try:
        for _ in range(100):
            if pidfile.exists() and pidfile.read_text().strip():
                child = int(pidfile.read_text())
                break
            time.sleep(0.1)
        assert child, "the card's script never started"
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        if child and _alive(child):
            os.kill(child, signal.SIGKILL)
            pytest.fail("the launcher's child did not get the TERM")
    assert proc.returncode == 143, out
    results = tmp_path / "results" / "job-0"
    assert (results / "perf.csv").read_text() == "model,performance\n"
    assert (results / "exit_code").read_text() == "143\n"


def test_a_failed_nonleader_records_its_status_for_pod0_and_exits_0(tmp_path):
    proc = _pod(tmp_path, "exit 3\n", 1)
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 0, out
    assert (tmp_path / "results" / "job-1" / "exit_code").read_text() == "3\n"
    assert not (tmp_path / "results" / "job-1" / "perf.csv").exists()


def test_pod0s_status_is_the_jobs(tmp_path):
    proc = _pod(tmp_path, "exit 3\n", 0)
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 3, out
    assert (tmp_path / "results" / "job-0" / "exit_code").read_text() == "3\n"
