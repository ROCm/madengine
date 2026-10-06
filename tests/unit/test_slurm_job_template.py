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

from madengine.core.timeout import DEFAULT_RUN_TIMEOUT
from madengine.deployment.base import DeploymentConfig, DeploymentStatus
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
    timeout: int = None,
    cli_timeout: int = None,
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

    cfg_kwargs = {} if timeout is None else {"timeout": timeout}
    if cli_timeout is not None:
        cfg_kwargs["cli_timeout"] = cli_timeout
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
        **cfg_kwargs,
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
# 4. The --timeout the job script passes back to madengine

class TestTimeoutForwarding:
    """The rendered `madengine run --timeout N` must always carry a valid int.

    The template used `{{ timeout | default(3600) }}`, but Jinja's default filter
    only substitutes for *undefined* — a None slipped straight through and
    rendered the literal `--timeout None`, which Typer then rejected.
    """

    @staticmethod
    def _timeout_args(script: str) -> list:
        return re.findall(r"--timeout (\S+)", script)

    def test_no_timeout_renders_zero_not_none(self, tmp_path):
        # --timeout 0 (no timeout) is the case that used to render "None".
        script = _render(_build_deployment(tmp_path, cli_timeout=0))
        args = self._timeout_args(script)
        assert args, "job script does not forward --timeout at all"
        assert all(a == "0" for a in args), args
        assert "--timeout None" not in script

    def test_explicit_timeout_forwarded(self, tmp_path):
        script = _render(_build_deployment(tmp_path, cli_timeout=120))
        assert all(a == "120" for a in self._timeout_args(script))

    def test_unspecified_sentinel_forwarded_verbatim(self, tmp_path):
        # -1 must survive to the inner CLI so it can apply model-card precedence
        # there, rather than being flattened to a concrete default here.
        script = _render(_build_deployment(tmp_path, cli_timeout=-1))
        assert all(a == "-1" for a in self._timeout_args(script))

    def test_resolved_process_cap_does_not_leak_into_the_job(self, tmp_path):
        """config.timeout caps *this* process; only cli_timeout reaches the job.

        Regression: the template read config.timeout, so a default run rendered
        --timeout 7200 into the job script. The inner madengine cannot tell that
        from a user-supplied --timeout 7200, so it outranked the model card and
        a model declaring "timeout": 3600 silently ran with a 2h cap instead.
        """
        deployment = _build_deployment(
            tmp_path, timeout=DEFAULT_RUN_TIMEOUT, cli_timeout=-1
        )
        assert all(a == "-1" for a in self._timeout_args(_render(deployment)))

    def test_default_config_forwards_the_sentinel(self, tmp_path):
        # A config built without an explicit CLI timeout forwards "unspecified",
        # leaving the model card free to win inside the job.
        script = _render(_build_deployment(tmp_path))
        assert all(a == "-1" for a in self._timeout_args(script))


# ---------------------------------------------------------------------------
# 5. The timeout handed to subprocess on the in-allocation path

class TestInAllocationTimeout:
    """`_run_inside_existing_allocation` must not pass a sentinel to subprocess.

    Regression: the call site read `self.config.timeout if ... > 0 else None`,
    which raised TypeError once the CLI started sending None for "no timeout".
    subprocess spells "no timeout" as None and reads 0 as "expire now", so
    both sentinels have to be mapped, not compared inline.
    """

    def _invoke(self, tmp_path, timeout):
        deployment = _build_deployment(tmp_path)
        # Set on the config directly: None is one of the values under test, so
        # it cannot be routed through _build_deployment's "omit the kwarg" flag.
        deployment.config.timeout = timeout
        deployment.inside_allocation = False  # skip the allocation-size check
        deployment.script_path = tmp_path / "job.sh"
        deployment.script_path.write_text("#!/bin/bash\nexit 0\n")
        with patch(
            "madengine.deployment.slurm.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as mock_run:
            deployment._run_inside_existing_allocation()
        mock_run.assert_called_once()
        return mock_run.call_args.kwargs["timeout"]

    @pytest.mark.parametrize("timeout", [0, -1, None])
    def test_no_timeout_values_become_none(self, tmp_path, timeout):
        assert self._invoke(tmp_path, timeout) is None

    def test_positive_timeout_passed_through(self, tmp_path):
        assert self._invoke(tmp_path, 120) == 120

    def test_default_config_carries_the_shared_default(self, tmp_path):
        assert _build_deployment(tmp_path).config.timeout == DEFAULT_RUN_TIMEOUT


# ---------------------------------------------------------------------------
# 6. SGLang disaggregated peer list resolves to routable addresses

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
            "#!/bin/bash\nexit 1\n"
            if not nodes
            else "#!/bin/bash\nprintf '%s\\n' " + " ".join(nodes) + "\n"
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

    def test_empty_node_list_fails_the_job(self, tmp_path):
        """A failing `scontrol` yields "" through the pipeline, not a marker.

        Without an explicit check that falls straight past the UNRESOLVED case
        and launches with no peers, which hangs the barrier just as loopback did.
        """
        out = self._run_resolution(tmp_path, [])
        assert out.returncode != 0, out.stdout + out.stderr
        assert "empty node list" in out.stderr
        assert "RESULT=" not in out.stdout

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


# ---------------------------------------------------------------------------
# 4. Running inside an existing allocation

class TestInsideExistingAllocation:
    """A second run in one allocation must not land on the first one's paths.

    Everything the job script writes is keyed on the job id, which is unique per
    sbatch but shared by every run inside one `salloc`. These lock in that the
    tag carries a per-run discriminator there, and nowhere else: the sbatch
    script every cluster already depends on must render exactly as before.
    """

    @staticmethod
    def _inside(tmp_path, nodes=2, job_id="99999", **slurm_overrides):
        with patch.dict("os.environ", {"SLURM_JOB_ID": job_id}), patch.object(
            SlurmDeployment, "_get_allocation_node_count", return_value=nodes
        ):
            dep = _build_deployment(
                tmp_path, slurm_overrides={"nodes": nodes, **slurm_overrides}
            )
            return dep, _render(dep)

    def test_paths_are_untagged_outside_an_allocation(self, tmp_path):
        script = _render(_build_deployment(tmp_path))
        assert "madengine_task_${SLURM_JOB_ID}.sh" in script
        assert "WORKSPACE=$SLURM_TMPDIR/madengine_node_${SLURM_PROCID}\n" in script
        assert not re.search(r"\$\{SLURM_JOB_ID\}_\d{14}", script)

    def test_srun_is_unconstrained_outside_an_allocation(self, tmp_path):
        """The #SBATCH directives already shape the step; do not duplicate them."""
        assert 'srun bash "$TASK_SCRIPT"' in _render(_build_deployment(tmp_path))

    def test_every_written_path_is_run_scoped_inside_an_allocation(self, tmp_path):
        dep, script = self._inside(tmp_path)
        tag = dep.job_tag
        assert re.fullmatch(r"\$\{SLURM_JOB_ID\}_\d{14}_[0-9a-f]{6}", tag), tag
        for path in (
            f"madengine_task_{tag}.sh",
            f"WORKSPACE=/tmp/madengine_job_{tag}_node_",
            f"_{tag}_node_${{SLURM_PROCID}}.out",
            f"/dummy_torchrun_multinode/{tag}",
            # SLURM_TMPDIR is itself per-job, so this path carries no job id and
            # the suffix is the only thing separating two runs that share one.
            f"WORKSPACE=$SLURM_TMPDIR/madengine_node_${{SLURM_PROCID}}{dep.run_suffix}",
        ):
            assert path in script, path
        # No path may keep the bare id, or the two runs collide there instead.
        assert not re.search(r"madengine[-_][\w{}$]*\$\{SLURM_JOB_ID\}(?!_\d)", script)

    def test_run_id_matches_what_the_script_expands_to(self, tmp_path):
        """collect_results() looks under the expanded tag, so they must agree."""
        dep, _ = self._inside(tmp_path, job_id="12345")
        assert dep.run_id == dep.job_tag.replace("${SLURM_JOB_ID}", "12345")

    def test_two_runs_in_the_same_second_get_distinct_tags(self, tmp_path):
        with patch(
            "madengine.deployment.slurm.time.strftime", return_value="20261006120000"
        ):
            first, _ = self._inside(tmp_path)
            second, _ = self._inside(tmp_path)
        assert first.job_tag != second.job_tag

    def test_srun_is_held_to_the_requested_nodes_inside_an_allocation(self, tmp_path):
        """#SBATCH is inert here, so a bare srun would take the whole allocation."""
        _, script = self._inside(tmp_path, nodes=2)
        assert 'srun -N 2 -n 2 --nodelist "$STEP_NODELIST" bash "$TASK_SCRIPT"' in script

    @pytest.mark.parametrize(
        "overrides, expected",
        [({}, ["h1,h2", "h1"]), ({"nodelist": "h3,h4"}, ["h3,h4", "h3"])],
    )
    def test_rendezvous_comes_from_the_step_hosts(self, tmp_path, overrides, expected):
        """In a larger allocation the step must contain MASTER_ADDR, and
        WORLD_SIZE must match its task count rather than the allocation's. An
        explicit slurm.nodelist picks the hosts, as it does under sbatch."""
        _, script = self._inside(tmp_path, nodes=2, **overrides)
        lines = script.splitlines()
        start = next(i for i, l in enumerate(lines) if l.startswith("STEP_NODELIST="))
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        scontrol = bin_dir / "scontrol"
        # Expands whatever list it is given, as scontrol does.
        scontrol.write_text(
            "#!/bin/bash\n"
            'case "$3" in h3,h4) printf "h3\\nh4\\n" ;; '
            '*) printf "h1\\nh2\\nh3\\nh4\\n" ;; esac\n'
        )
        scontrol.chmod(0o755)
        snippet = "\n".join(
            ["export SLURM_NTASKS=4"]
            + lines[start : start + 3]
            + ['echo "$STEP_NODELIST $MASTER_ADDR $WORLD_SIZE"']
        )
        out = subprocess.run(
            ["bash", "-c", snippet],
            env={"PATH": f"{bin_dir}:/usr/bin:/bin", "SLURM_JOB_NODELIST": "h[1-4]"},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        assert out == expected + ["2"]

    def test_a_single_node_run_keeps_the_standard_script(self, tmp_path):
        """It submits its own sbatch, so its paths and the id deploy() returns
        must be the bare job id that collect_results() looks under."""
        dep, script = self._inside(tmp_path, nodes=1)
        assert dep.job_tag == "${SLURM_JOB_ID}"
        assert dep.run_id == "99999"
        assert script == _render(_build_deployment(tmp_path, slurm_overrides={"nodes": 1}))

    def test_the_tag_follows_a_node_count_resolved_later(self, tmp_path):
        """prepare() may still raise the node count from the model card."""
        dep, _ = self._inside(tmp_path, nodes=1)
        dep.nodes = 2
        assert dep.run_suffix and dep.job_tag.endswith(dep.run_suffix)

    @pytest.mark.parametrize(
        "generator, nnodes",
        [("_generate_deepspeed_command", 2), ("_generate_sglang_disagg_command", 3)],
    )
    def test_launchers_read_the_step_hosts_in_place(self, tmp_path, generator, nnodes):
        """They run inside the step; the allocation may hold more hosts."""
        dep, _ = self._inside(tmp_path, nodes=nnodes)
        cmd = getattr(dep, generator)(nnodes, 8, 29500)
        assert "SLURM_STEP_NODELIST" in cmd
        assert "SLURM_JOB_NODELIST" not in cmd

    @pytest.mark.parametrize(
        "generator, nnodes",
        [("_generate_deepspeed_command", 2), ("_generate_sglang_disagg_command", 3)],
    )
    def test_launchers_read_the_job_hosts_under_sbatch(self, tmp_path, generator, nnodes):
        dep = _build_deployment(tmp_path, slurm_overrides={"nodes": nnodes})
        cmd = getattr(dep, generator)(nnodes, 8, 29500)
        assert "SLURM_JOB_NODELIST" in cmd
        assert "SLURM_STEP_NODELIST" not in cmd

    def test_deepspeed_hostfile_is_run_scoped(self, tmp_path):
        """Two in-place runs on the same hosts would share /tmp otherwise."""
        dep, _ = self._inside(tmp_path)
        cmd = dep._generate_deepspeed_command(2, 8, 29500)
        assert cmd.count(f"/tmp/deepspeed_hostfile_{dep.job_tag}.txt") == 2

    def test_deepspeed_hostfile_keeps_the_job_id_under_sbatch(self, tmp_path):
        dep = _build_deployment(tmp_path, slurm_overrides={"nodes": 2})
        cmd = dep._generate_deepspeed_command(2, 8, 29500)
        assert cmd.count("/tmp/deepspeed_hostfile_${SLURM_JOB_ID}.txt") == 2

    def test_rendezvous_is_unchanged_outside_an_allocation(self, tmp_path):
        script = _render(_build_deployment(tmp_path))
        assert "STEP_NODELIST" not in script
        assert "export GPUS_PER_NODE=8\n\n# GPU visibility" in script

    @pytest.mark.parametrize("own_id", ["99999", "run_id"])
    def test_cleanup_leaves_the_callers_allocation_alone(self, tmp_path, own_id):
        """A failed in-place run reaches cleanup(); scancel on the allocation
        would end every later run that shares it."""
        dep, _ = self._inside(tmp_path)
        with patch("madengine.deployment.slurm.subprocess.run") as run:
            assert dep.cleanup(dep.run_id if own_id == "run_id" else own_id)
        run.assert_not_called()

    def test_cleanup_still_cancels_a_job_this_run_submitted(self, tmp_path):
        """A single-node run in an allocation submits its own sbatch."""
        dep, _ = self._inside(tmp_path)
        with patch("madengine.deployment.slurm.subprocess.run") as run:
            dep.cleanup("12345")
        assert run.call_args.args[0] == ["scancel", "12345"]

    def test_multi_node_runs_in_place(self, tmp_path):
        with patch.dict("os.environ", {"SLURM_JOB_ID": "99999"}), patch.object(
            SlurmDeployment, "_get_allocation_node_count", return_value=2
        ):
            dep = _build_deployment(tmp_path, slurm_overrides={"nodes": 2})
            dep.script_path = tmp_path / "job.sh"
            dep.script_path.write_text("#!/bin/bash\nexit 0\n")
            with patch.object(
                SlurmDeployment, "_run_inside_existing_allocation"
            ) as in_place:
                dep.deploy()
        in_place.assert_called_once()

    def test_single_node_still_submits_an_sbatch(self, tmp_path):
        """The single-node branch runs madengine inline, which under `bash` in an
        allocation means the submitting host -- usually a login node. Submitting
        a fresh sbatch is the right answer there."""
        submitted = subprocess.CompletedProcess(
            args=["sbatch"], returncode=0, stdout="Submitted batch job 12345\n", stderr=""
        )
        with patch.dict("os.environ", {"SLURM_JOB_ID": "99999"}), patch.object(
            SlurmDeployment, "_get_allocation_node_count", return_value=1
        ):
            dep = _build_deployment(
                tmp_path,
                # Preflight shells out to the scheduler; this test is about the
                # branch taken, not node selection.
                slurm_overrides={"nodes": 1, "enable_node_check": False},
                distributed_overrides={"nnodes": 1},
            )
            dep.script_path = tmp_path / "job.sh"
            dep.script_path.write_text("#!/bin/bash\nexit 0\n")
            with patch.object(
                SlurmDeployment, "_run_inside_existing_allocation"
            ) as in_place, patch(
                "madengine.deployment.slurm.subprocess.run", return_value=submitted
            ) as run:
                result = dep.deploy()

        in_place.assert_not_called()
        assert run.call_args.args[0] == ["sbatch", str(dep.script_path)]
        assert result.status is DeploymentStatus.SUCCESS
        assert result.deployment_id == "12345"
