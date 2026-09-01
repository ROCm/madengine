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
