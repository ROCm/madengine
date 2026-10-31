"""The srun / docker / scontrol stand-ins a slurm_multi card runs under on Kubernetes.

Two real agents on two loopback addresses stand in for two pods; the tools are run
as the card's script would run them. The last test drives the exact command shapes
MAD's launchers use (scontrol show hostnames, srun hostname -I per node, srun fan-out
of docker run with -e/-v/--entrypoint, docker stop/rm, docker image inspect).
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

SHIM = Path(__file__).resolve().parents[2] / "src/madengine/deployment/k8s_shim/mad_k8s.py"
NODES = ["127.0.0.1", "127.0.0.2"]
IMAGE = "rocm/test:img"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def cluster(tmp_path):
    port = str(_free_port())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("srun", "docker", "scontrol"):
        (bin_dir / tool).write_text(f'#!/bin/bash\nexec "{sys.executable}" "{SHIM}" {tool} "$@"\n')
        (bin_dir / tool).chmod(0o755)
    state = tmp_path / "state"
    (state / "containers").mkdir(parents=True)
    base_env = {"PATH": os.environ["PATH"], "IMAGE_ONLY": "from-image", "MAD_K8S_IMAGE_WORKDIR": str(tmp_path)}
    (state / "base_env.json").write_text(json.dumps(base_env))
    env = dict(os.environ)
    env.update({
        "PATH": f"{bin_dir}:{os.environ['PATH']}", "MAD_K8S_AGENT_PORT": port, "MAD_K8S_STATE": str(state),
        "MAD_K8S_IMAGE": IMAGE, "SLURM_JOB_NODELIST": ",".join(NODES), "MAD_K8S_CONNECT_TIMEOUT": "20",
    })
    agents = []
    for node in NODES:
        agents.append(subprocess.Popen([sys.executable, str(SHIM), "agent"],
                                       env={**env, "MAD_K8S_AGENT_BIND": node}))
    try:
        subprocess.run(["srun", "true"], env=env, check=True, timeout=30)  # waits for both agents
        yield env, tmp_path
    finally:
        for a in agents:
            a.kill()
            a.wait()


def _run(env, cmd, **kw):
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60, **kw)


class TestSrun:
    def test_one_task_per_node_ranked_in_allocation_order(self, cluster):
        env, _ = cluster
        r = _run(env, ["srun", "--nodelist=127.0.0.2,127.0.0.1", "bash", "-c",
                       'echo "$SLURM_PROCID $SLURMD_NODENAME $SLURM_STEP_NUM_NODES"'])
        assert r.returncode == 0, r.stderr
        assert sorted(r.stdout.split("\n")[:2]) == ["0 127.0.0.1 2", "1 127.0.0.2 2"]

    def test_single_node_query_stdout_only(self, cluster):
        env, _ = cluster
        r = _run(env, ["srun", "--nodes=1", "--ntasks=1", "--time=00:20:00", "--nodelist=127.0.0.2",
                       "bash", "-c", "echo out; echo err >&2"])
        assert (r.returncode, r.stdout, r.stderr) == (0, "out\n", "err\n")

    def test_highest_exit_status_wins(self, cluster):
        env, _ = cluster
        r = _run(env, ["srun", "bash", "-c", '[ "$SLURM_PROCID" = 1 ] && exit 7; exit 0'])
        assert r.returncode == 7

    def test_caller_environment_reaches_the_task(self, cluster):
        env, _ = cluster
        r = _run({**env, "CARD_VAR": "x1"}, ["srun", "--nodelist=127.0.0.2", "bash", "-c", 'echo "$CARD_VAR"'])
        assert r.stdout == "x1\n"

    def test_unknown_node_and_option_are_refused(self, cluster):
        env, _ = cluster
        r = _run(env, ["srun", "--nodelist=10.9.9.9", "true"])
        assert r.returncode != 0 and "not in this allocation" in r.stderr
        r = _run(env, ["srun", "--distribution=cyclic", "true"])
        assert r.returncode != 0 and "unsupported option" in r.stderr
        r = _run(env, ["srun", "--ntasks-per-node=2", "true"])
        assert r.returncode != 0 and "one task per node" in r.stderr


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


def test_a_launcher_shaped_script_runs_end_to_end(cluster):
    """The command shapes of MAD's run_xPyD_models.slurm, through the stand-ins."""
    env, tmp = cluster
    logs = tmp / "shared_logs"
    script = tmp / "launcher.slurm"
    script.write_text(f"""#!/bin/bash
ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_NODE=$(echo "$ALL_NODES" | head -1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$MASTER_NODE" bash -c 'echo 10.0.0.$SLURM_NODEID')
export DOCKER_CONT_NAME=card_$SLURM_JOB_ID MASTER_ADDR
srun --nodelist="$(echo "$ALL_NODES" | paste -sd,)" bash -c '
docker ps -q | xargs --no-run-if-empty docker stop;
docker rm -f $DOCKER_CONT_NAME 2>/dev/null || true;
_IMG=$(docker image inspect --format "{{{{.Id}}}}" "$DOCKER_IMAGE_NAME" 2>/dev/null);
docker run --rm --network host --privileged -v {logs}:/tmp/mad-k8s-test-run_logs-$SLURM_PROCID \\
    -e NODE_RANK=$SLURM_PROCID -e MASTER_ADDR=$MASTER_ADDR --name $DOCKER_CONT_NAME-$SLURM_PROCID \\
    --entrypoint /bin/bash $DOCKER_IMAGE_NAME \\
    -c "echo rank \\$NODE_RANK master \\$MASTER_ADDR > /tmp/mad-k8s-test-run_logs-\\$NODE_RANK/node\\$NODE_RANK.log"
'
rc=$?
srun --nodelist="$(echo "$ALL_NODES" | paste -sd,)" bash -c 'docker stop $DOCKER_CONT_NAME 2>/dev/null || true'
exit $rc
""")
    r = _run({**env, "SLURM_JOB_ID": "42", "DOCKER_IMAGE_NAME": IMAGE}, ["bash", str(script)])
    try:
        assert r.returncode == 0, r.stdout + r.stderr
        assert (logs / "node0.log").read_text() == "rank 0 master 10.0.0.0\n"
        assert (logs / "node1.log").read_text() == "rank 1 master 10.0.0.0\n"
    finally:
        for i in (0, 1):
            Path(f"/tmp/mad-k8s-test-run_logs-{i}").unlink(missing_ok=True)
