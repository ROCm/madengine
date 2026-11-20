"""The docker / scontrol stand-ins a slurm_multi card runs under on Kubernetes.

Each pod runs the card's script as its own node (k8s-indexed dispatch), so nothing
fans out from one pod to the others: there is no srun stand-in and no agent. The
last test drives the command shapes one node of MAD's launchers uses in a pod
(scontrol show hostnames, docker image inspect, docker run with -e/-v/--entrypoint,
docker stop/rm).
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

SHIM = Path(__file__).resolve().parents[2] / "src/madengine/deployment/k8s_shim/mad_k8s.py"
NODES = ["127.0.0.1", "127.0.0.2"]
IMAGE = "rocm/test:img"



@pytest.fixture
def cluster(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("docker", "scontrol"):
        (bin_dir / tool).write_text(f'#!/bin/bash\nexec "{sys.executable}" "{SHIM}" {tool} "$@"\n')
        (bin_dir / tool).chmod(0o755)
    state = tmp_path / "state"
    (state / "containers").mkdir(parents=True)
    base_env = {"PATH": os.environ["PATH"], "IMAGE_ONLY": "from-image", "MAD_K8S_IMAGE_WORKDIR": str(tmp_path)}
    (state / "base_env.json").write_text(json.dumps(base_env))
    env = dict(os.environ)
    env.update({
        "PATH": f"{bin_dir}:{os.environ['PATH']}", "MAD_K8S_STATE": str(state),
        "MAD_K8S_IMAGE": IMAGE, "SLURM_JOB_NODELIST": ",".join(NODES),
    })
    yield env, tmp_path


def _run(env, cmd, **kw):
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60, **kw)


def test_there_is_no_srun_or_agent(cluster):
    env, _ = cluster
    for tool in ("srun", "agent", "wait-peers"):
        r = _run(env, [sys.executable, str(SHIM), tool, "true"])
        assert r.returncode != 0 and "unknown tool" in r.stderr, tool


class TestScontrol:
    def test_show_hostnames(self, cluster):
        env, _ = cluster
        assert _run(env, ["scontrol", "show", "hostnames"]).stdout.split() == NODES
        assert _run(env, ["scontrol", "show", "hostnames", "n[01-03],m"]).stdout.split() == ["n01", "n02", "n03", "m"]

    def test_topology_is_unavailable_and_others_refused(self, cluster):
        env, _ = cluster
        assert _run(env, ["scontrol", "show", "topology"]).returncode == 1
        r = _run(env, ["scontrol", "update", "job"])
        assert r.returncode != 0 and "unsupported" in r.stderr


class TestDocker:
    def test_run_applies_env_volumes_entrypoint_on_a_fresh_environment(self, cluster):
        env, tmp = cluster
        src = tmp / "logs"
        src.mkdir()
        dst = tmp / "run_logs"
        r = _run({**env, "LAUNCHER_ONLY": "leak"}, [
            "docker", "run", "--rm", "--device", "/dev/kfd", "--network", "host", "--privileged",
            "--ipc", "host", "--shm-size", "64G", "--ulimit", "nofile=1:1", "-v", f"{src}:{dst}",
            "-e", "A=1", "--entrypoint", "/bin/bash", IMAGE, "-c",
            'echo "$A ${IMAGE_ONLY} [${LAUNCHER_ONLY:-}]"; echo hi > ' + str(dst) + "/f"])
        assert r.returncode == 0, r.stderr
        assert r.stdout == "1 from-image []\n"
        assert (src / "f").read_text() == "hi\n"

    def test_a_different_image_is_refused(self, cluster):
        env, _ = cluster
        r = _run(env, ["docker", "run", "--rm", "other/image:x", "true"])
        assert r.returncode == 125 and "cannot start a different image" in r.stderr

    def test_unsupported_option_is_refused(self, cluster):
        env, _ = cluster
        r = _run(env, ["docker", "run", "--mount", "type=bind", IMAGE, "true"])
        assert r.returncode == 125 and "unsupported option" in r.stderr

    def test_stop_ps_rm_track_named_containers(self, cluster):
        env, _ = cluster
        p = subprocess.Popen(["docker", "run", "--rm", "--name", "c1", "--entrypoint", "sleep", IMAGE, "60"], env=env)
        for _ in range(50):
            if _run(env, ["docker", "ps", "-q"]).stdout.split() == ["c1"]:
                break
            time.sleep(0.1)
        assert _run(env, ["docker", "ps", "-q"]).stdout.split() == ["c1"]
        assert _run(env, ["docker", "stop", "-t", "2", "c1"]).returncode == 0
        assert p.wait(timeout=10) != 0
        assert _run(env, ["docker", "ps", "-q"]).stdout.split() == []
        assert _run(env, ["docker", "rm", "-f", "c1"]).returncode == 0
        assert _run(env, ["docker", "stop", "nope"]).returncode == 1

    def test_image_inspect_and_pull_for_the_pod_image(self, cluster):
        env, _ = cluster
        r = _run(env, ["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE])
        assert r.returncode == 0 and r.stdout.startswith("sha256:")
        assert _run(env, ["docker", "image", "inspect", "other:x"]).returncode == 1
        assert _run(env, ["docker", "pull", IMAGE]).returncode == 0
        assert _run(env, ["docker", "pull", "other:x"]).returncode == 1


def test_one_nodes_launcher_steps_run_in_the_pod(cluster):
    """What one node of MAD's launchers does in a pod (per-pod dispatch), through the stand-ins."""
    env, tmp = cluster
    logs = tmp / "shared_logs"
    logs.mkdir()
    script = tmp / "node.sh"
    script.write_text(f"""#!/bin/bash
ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export DOCKER_CONT_NAME=card_$SLURM_JOB_ID
docker rm -f $DOCKER_CONT_NAME 2>/dev/null || true
_IMG=$(docker image inspect --format "{{{{.Id}}}}" "$DOCKER_IMAGE_NAME" 2>/dev/null) || exit 3
docker run --rm --network host --privileged -v {logs}:/tmp/mad-k8s-test-run_logs \\
    -e NODE_RANK=$SLURM_PROCID -e NODES="$(echo $ALL_NODES | tr ' ' ,)" --name $DOCKER_CONT_NAME \\
    --entrypoint /bin/bash $DOCKER_IMAGE_NAME \\
    -c "echo rank \\$NODE_RANK of \\$NODES > /tmp/mad-k8s-test-run_logs/node\\$NODE_RANK.log"
rc=$?
docker stop $DOCKER_CONT_NAME 2>/dev/null || true
exit $rc
""")
    try:
        for rank in (0, 1):
            r = _run({**env, "SLURM_JOB_ID": "42", "SLURM_PROCID": str(rank), "DOCKER_IMAGE_NAME": IMAGE},
                     ["bash", str(script)])
            assert r.returncode == 0, r.stdout + r.stderr
        assert (logs / "node0.log").read_text() == f"rank 0 of {','.join(NODES)}\n"
        assert (logs / "node1.log").read_text() == f"rank 1 of {','.join(NODES)}\n"
    finally:
        Path("/tmp/mad-k8s-test-run_logs").unlink(missing_ok=True)
