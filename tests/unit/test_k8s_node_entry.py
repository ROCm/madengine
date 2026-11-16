"""mad_k8s_node.sh, the pod entry point, when Kubernetes stops pod 0 with TERM.

The pod's tools (mad_k8s.py) are replaced by a stub: only the entry script's own
signal handling is under test.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

NODE_SH = Path(__file__).resolve().parents[2] / "src/madengine/deployment/k8s_shim/mad_k8s_node.sh"

STUB = """import sys, time
if sys.argv[1] == "agent":
    time.sleep(300)
"""

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


def test_term_reaches_the_launchers_children_and_results_are_still_published(tmp_path):
    workspace = tmp_path / "ws"
    (workspace / ".mad-k8s").mkdir(parents=True)
    (workspace / ".mad-k8s" / "mad_k8s.py").write_text(STUB)
    card = workspace / "scripts" / "card"
    card.mkdir(parents=True)
    pidfile = tmp_path / "child.pid"
    # No TERM trap: the script dies, and only a signal to its process group
    # reaches the process it started.
    (card / "run.sh").write_text(f"""echo 'model,performance' > perf.csv
sleep 300 &
echo $! > {pidfile}
wait
""")
    env = dict(os.environ, MAD_K8S_WORKSPACE=str(workspace), MAD_K8S_STATE=str(tmp_path / "state"),
               MAD_K8S_NNODES="1", MAD_K8S_NODE_PREFIX="job", MAD_K8S_JOB_ID="7",
               MAD_K8S_SCRIPT="scripts/card/run.sh", JOB_COMPLETION_INDEX="0")
    proc = subprocess.Popen(["bash", str(NODE_SH)], env=env, cwd=tmp_path,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
    assert "run.sh exited 143" in out          # the publish block ran after the TERM
