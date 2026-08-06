#!/usr/bin/env python3
"""
Unit tests for the generated SLURM job script (`job.sh.j2`).

Locks in the portability contract points that clusters keep re-discovering
downstream (see ROCm/rocm-systems#9055, which patched madengine's source
rather than filing them):

1. The job script puts madengine back on PATH itself instead of assuming the
   batch environment inherited the submitter's PATH.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
from pathlib import Path
from unittest.mock import patch

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


def _build_deployment(tmp_path: Path, slurm_overrides: dict = None) -> SlurmDeployment:
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

    cfg = DeploymentConfig(
        target="slurm",
        manifest_file=str(manifest_path),
        additional_context={
            "deploy": "slurm",
            "gpu_vendor": "AMD",
            "guest_os": "UBUNTU",
            "slurm": slurm_config,
            "distributed": {
                "launcher": "torchrun",
                "nnodes": 2,
                "nproc_per_node": 8,
                "backend": "nccl",
                "port": 29500,
            },
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
