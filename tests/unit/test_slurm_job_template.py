#!/usr/bin/env python3
"""
Unit tests for the generated SLURM job script (`job.sh.j2`).

Locks in the portability contract points that clusters keep re-discovering
downstream (see ROCm/rocm-systems#9055, which patched madengine's source
rather than filing them):

1. The job script puts madengine back on PATH itself instead of assuming the
   batch environment inherited the submitter's PATH.
2. The shared-filesystem probe recognizes `nfs4`, which is what `df -T`
   reports on most modern NFS mounts.
3. `slurm.skip_gpus_directive` removes `#SBATCH --gpus-per-node`, which a
   cluster advertising no GPU GRES rejects outright.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from madengine.deployment.base import DeploymentConfig
from madengine.deployment.slurm import SlurmDeployment


MODEL_ENTRY = {
    "name": "dummy_torchrun_multinode",
    "url": "",
    "dockerfile": "docker/dummy",
    "scripts": "scripts/dummy/run.sh",
    "n_gpus": "8",
    "owner": "mad.support@amd.com",
    "training_precision": "",
    "tags": ["pyt", "training"],
    "timeout": -1,
    "args": "",
}


def _build_deployment(
    tmp_path: Path,
    slurm_overrides: dict = None,
    distributed_overrides: dict = None,
) -> SlurmDeployment:
    """SlurmDeployment over a minimal torchrun manifest, output_dir under tmp_path."""
    manifest = {
        "built_images": {"dummy-image": {"docker_image": "dummy:latest"}},
        "built_models": {"dummy-image": MODEL_ENTRY},
        "context": {
            "docker_env_vars": {},
            "docker_mounts": {},
            "docker_build_arg": {},
            "gpu_vendor": "AMD",
            "guest_os": "UBUNTU",
            "docker_gpus": "all",
        },
    }
    manifest_path = tmp_path / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    slurm_config = {
        "partition": "test-partition",
        "nodes": 2,
        "gpus_per_node": 8,
        "time": "01:00:00",
        "output_dir": str(tmp_path / "slurm_output"),
        "exclusive": True,
    }
    slurm_config.update(slurm_overrides or {})

    distributed_config = {
        "launcher": "torchrun",
        "nnodes": 2,
        "nproc_per_node": 8,
        "backend": "nccl",
        "port": 29500,
    }
    distributed_config.update(distributed_overrides or {})

    cfg = DeploymentConfig(
        target="slurm",
        manifest_file=str(manifest_path),
        additional_context={
            "deploy": "slurm",
            "gpu_vendor": "AMD",
            "guest_os": "UBUNTU",
            "slurm": slurm_config,
            "distributed": distributed_config,
        },
    )
    return SlurmDeployment(cfg)


def _render(deployment: SlurmDeployment) -> str:
    """Render job.sh.j2 exactly as prepare() does, without submitting anything."""
    context = deployment._prepare_template_context(MODEL_ENTRY)
    return deployment.jinja_env.get_template("job.sh.j2").render(**context)


# ---------------------------------------------------------------------------
# 1. PATH is re-established inside the job

class TestJobScriptPath:
    """The job script must not depend on the submitter's PATH being inherited."""

    def test_user_bin_dir_is_prepended(self, tmp_path):
        script = _render(_build_deployment(tmp_path))
        assert 'export PATH="$HOME/.local/bin:$PATH"' in script

    def test_submission_bin_dir_is_prepended(self, tmp_path):
        with patch("madengine.deployment.slurm.shutil.which", return_value="/opt/venv/bin/madengine"):
            script = _render(_build_deployment(tmp_path))
        assert 'export PATH="/opt/venv/bin:$PATH"' in script

    def test_no_empty_export_when_cli_not_on_path(self, tmp_path):
        """madengine missing at submission time must not render an empty PATH entry."""
        with patch("madengine.deployment.slurm.shutil.which", return_value=None):
            script = _render(_build_deployment(tmp_path))
        assert 'export PATH=":$PATH"' not in script
        assert 'export PATH="$HOME/.local/bin:$PATH"' in script

    def test_path_is_set_before_madengine_is_looked_up(self, tmp_path):
        """The export is useless if it lands after `command -v madengine`."""
        with patch("madengine.deployment.slurm.shutil.which", return_value="/opt/venv/bin/madengine"):
            script = _render(_build_deployment(tmp_path))
        assert script.index('export PATH="/opt/venv/bin:$PATH"') < script.index("command -v madengine")


# ---------------------------------------------------------------------------
# 2. Shared-filesystem probe

class TestSharedFilesystemProbe:
    """`df -T` reports nfs4 on modern mounts; the probe must not miss it.

    And it must read the filesystem type, nothing else: the mount point travels on the
    same `df -T` line, so a local disk under a path such as /mnt/nfs-scratch used to answer
    yes and the job then trusted node-local storage to be visible from every node.
    """

    @staticmethod
    def _probe_pattern(script: str) -> str:
        match = re.search(r"SUBMIT_FSTYPE\"?\s*\|\s*grep -qE '([^']+)'", script)
        assert match, "shared-filesystem probe not found in rendered script"
        return match.group(1)

    @pytest.mark.parametrize("fstype,expected", [
        ("nfs", True),
        ("nfs3", True),
        ("nfs4", True),
        ("lustre", True),
        ("gpfs", True),
        ("ceph", True),
        ("beegfs", True),
        ("panfs", True),
        ("ext4", False),
        ("xfs", False),
        ("overlay", False),
        ("tmpfs", False),
    ])
    def test_probe_matches_shared_filesystems(self, tmp_path, fstype, expected):
        # The probe only exists on the single-node branch of the template.
        deployment = _build_deployment(tmp_path, {"nodes": 1}, {"nnodes": 1})
        pattern = self._probe_pattern(_render(deployment))
        assert bool(re.search(pattern, fstype)) is expected

    def test_the_probe_reads_the_fstype_column_only(self, tmp_path):
        script = _render(_build_deployment(tmp_path, {"nodes": 1}, {"nnodes": 1}))
        assert 'df --output=fstype "$SUBMIT_DIR"' in script
        assert 'df -T "$SUBMIT_DIR" 2>/dev/null | grep' not in script

    def test_a_mount_point_that_says_nfs_does_not_make_a_disk_shared(self, tmp_path):
        """/mnt/nfs-scratch on ext4 is local, whatever its name suggests."""
        script = _render(_build_deployment(tmp_path, {"nodes": 1}, {"nnodes": 1}))
        pattern = self._probe_pattern(script)
        df_line = "/dev/nvme0n1p2 ext4 104857600 50106368 54751232 48% /mnt/nfs-scratch"
        assert re.search(pattern, df_line) is None
        assert re.search(pattern, "ext4") is None

    def test_there_is_a_fallback_for_df_without_output(self, tmp_path):
        """--output is coreutils 8.21; older df still has to be read correctly."""
        script = _render(_build_deployment(tmp_path, {"nodes": 1}, {"nnodes": 1}))
        assert "awk 'NR > 1 { print $2; exit }'" in script


# ---------------------------------------------------------------------------
# 3. GPU GRES directive opt-out

class TestGpusPerNodeDirective:
    """A cluster with GresTypes=(null) rejects any job carrying --gpus-per-node."""

    def test_directive_present_by_default(self, tmp_path):
        script = _render(_build_deployment(tmp_path))
        assert "#SBATCH --gpus-per-node=8" in script

    def test_directive_omitted_when_opted_out(self, tmp_path):
        script = _render(_build_deployment(tmp_path, {"skip_gpus_directive": True}))
        assert "--gpus-per-node" not in script


# ---------------------------------------------------------------------------
# 4. SGLang disaggregated peer list resolves to routable addresses

class TestSglangDisaggNodeIps:
    """Peers must never be published as loopback.

    On Ubuntu /etc/hosts maps the local hostname to 127.0.1.1, so a plain
    `getent hosts` makes every node advertise itself as loopback and any
    all-nodes barrier hangs.
    """

    @staticmethod
    def _sglang_env(tmp_path) -> str:
        deployment = _build_deployment(
            tmp_path,
            {"nodes": 4},
            {"launcher": "sglang-disagg", "nnodes": 4},
        )
        return deployment._generate_sglang_disagg_command(
            nnodes=4, nproc_per_node=8, master_port=29500
        )

    @staticmethod
    def _code_lines(script: str) -> list:
        """Executable lines only — the comments name the rejected commands."""
        return [l for l in script.splitlines() if not l.lstrip().startswith("#")]

    def test_uses_ahostsv4_and_skips_loopback(self, tmp_path):
        script = self._sglang_env(tmp_path)
        assert "getent ahostsv4" in script
        assert "/^127\\./" in script
        # the plain lookup is what returned the 127.0.1.1 self-mapping
        assert not any("getent hosts" in l for l in self._code_lines(script))

    def test_fallback_is_restricted_to_the_local_node(self, tmp_path):
        """A peer that fails to resolve must not inherit this node's address."""
        out = self._run_resolution(tmp_path, ["node1", "node4"])
        assert out.returncode != 0, out.stdout + out.stderr
        assert "node4" in out.stderr
        # publishing our own address in the peer's slot is worse than no entry
        assert "10.0.0.2" not in out.stdout

    def test_local_node_is_identified_by_its_slurm_nodename(self, tmp_path):
        """The list holds NodeName, which need not equal the machine hostname.

        With NodeHostname configured the two differ, and matching on hostname
        alone would treat the local entry as an unresolvable peer and abort a
        job that has a perfectly good address to advertise.
        """
        out = self._run_resolution(tmp_path, ["node1", "node9"], nodename="node9")
        assert out.returncode == 0, out.stdout + out.stderr
        assert "RESULT=10.0.0.1,10.0.0.2" in out.stdout, out.stdout + out.stderr

    def _run_resolution(self, tmp_path, nodes, ifname="fenic0", nodename=None):
        """Execute the rendered resolution logic against stubbed system tools.

        The simulated machine answers to hostname ``node2`` and holds a docker
        bridge (172.17.0.1), a management address (192.168.1.5) and the cluster
        interface (10.0.0.2). ``node1`` resolves normally, ``node2``/``node4``/
        ``node9`` only to loopback and anything else not at all. ``nodename``
        sets SLURMD_NODENAME, i.e. the identity SLURM gives the local node.
        """
        script = self._sglang_env(tmp_path)
        start = script.index("# Address this node advertises")
        snippet = script[start : script.index("export SGLANG_NODE_IPS", start)]

        bin_dir = tmp_path / "stubbin"
        bin_dir.mkdir(exist_ok=True)
        (bin_dir / "ip").write_text(
            "#!/bin/bash\n"
            'case "$*" in\n'
            '  *"addr show dev fenic0"*) echo "3: fenic0 inet 10.0.0.2/24 scope global fenic0";;\n'
            '  *"addr show dev docker0"*) echo "4: docker0 inet 172.17.0.1/16 scope global docker0";;\n'
            '  *"route get"*) echo "1.1.1.1 via 192.168.1.1 dev mgmt0 src 192.168.1.5 uid 0";;\n'
            "  *) exit 1;;\n"
            "esac\n"
        )
        (bin_dir / "getent").write_text(
            '#!/bin/bash\ncase "$2" in\n  node1) echo "10.0.0.1 node1";;\n'
            '  node2|node4|node9) echo "127.0.1.1 $2";;\n  *) exit 2;;\nesac\n'
        )
        (bin_dir / "hostname").write_text(
            '#!/bin/bash\ncase "$1" in\n  -s) echo node2;;\n  *) echo node2;;\nesac\n'
        )
        (bin_dir / "scontrol").write_text(
            "#!/bin/bash\nprintf '%s\\n' " + " ".join(nodes) + "\n"
        )
        for f in bin_dir.iterdir():
            f.chmod(0o755)

        runner = tmp_path / "run.sh"
        runner.write_text(snippet + '\necho "RESULT=$SLURM_NODE_IPS"\n')
        env = {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "NCCL_SOCKET_IFNAME": ifname,
            "SLURM_JOB_NODELIST": "stub",
        }
        if nodename is not None:
            env["SLURMD_NODENAME"] = nodename
        return subprocess.run(
            ["bash", str(runner)], capture_output=True, text=True, env=env
        )

    def test_unresolvable_peer_fails_the_job(self, tmp_path):
        """Failing fast is what prevents another allocation-length hang."""
        out = self._run_resolution(tmp_path, ["node1", "node3"])
        assert out.returncode != 0, out.stdout + out.stderr
        # the operator needs to know which node could not be resolved
        assert "node3" in out.stderr
        assert "RESULT=" not in out.stdout

    def test_local_address_comes_from_the_cluster_interface(self, tmp_path):
        """`hostname -I` lists every interface unordered, so it is not used."""
        script = self._sglang_env(tmp_path)
        assert not any("hostname -I" in l for l in self._code_lines(script))
        assert "NCCL_SOCKET_IFNAME" in script

        out = self._run_resolution(tmp_path, ["node1", "node2"])
        assert out.returncode == 0, out.stdout + out.stderr
        assert "RESULT=10.0.0.1,10.0.0.2" in out.stdout, out.stdout + out.stderr
        # neither the docker bridge nor the management address may be published
        assert "172.17.0.1" not in out.stdout
        assert "192.168.1.5" not in out.stdout
