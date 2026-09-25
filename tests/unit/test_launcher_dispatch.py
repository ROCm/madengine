#!/usr/bin/env python3
"""
Unit tests locking in that every launcher in ``VALID_LAUNCHERS`` actually reaches
a dispatch arm on the backends that claim to support it.

The defect these exist to prevent: ``megatron-lm`` — the documented spelling, used
in all four shipped example configs — reached no dispatch arm on *either* backend.
Both chains compared against the bare literal ``"megatron"``, so the documented name
fell through to the "unknown launcher" default. SLURM printed a warning and ran with
no distributed setup; Kubernetes produced no launcher command, no headless service,
and no rank env vars. Both reported SUCCESS with a wrong benchmark number.

Nothing caught it because no test asserted "backend X has an arm for launcher Y".
That assertion is what this file is.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from madengine.deployment.base import DeploymentConfig
from madengine.deployment.common import VALID_LAUNCHERS
from madengine.deployment.slurm import SlurmDeployment


# slurm_multi bypasses the templated launcher chain entirely: it runs the model's own
# .slurm script on the head node and orchestrates containers via srun itself. It is an
# escape hatch, not a peer of the templated launchers, so it has no dispatch arm on
# either backend. Writing the exception down is the point — an *unexplained* absence
# is exactly what the megatron bug looked like.
SELF_MANAGED = {"slurm_multi"}

TEMPLATED_LAUNCHERS = [lt for lt in VALID_LAUNCHERS if lt not in SELF_MANAGED]

MODEL_ENTRY = {
    "name": "dummy_multinode",
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

MANIFEST_CONTEXT = {
    "docker_env_vars": {},
    "docker_mounts": {},
    "docker_build_arg": {},
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU",
    "docker_gpus": "all",
}


def _write_manifest(tmp_path: Path, distributed: dict) -> Path:
    manifest = {
        "built_images": {"dummy-image": {"docker_image": "dummy:latest"}},
        "built_models": {"dummy-image": MODEL_ENTRY},
        "context": MANIFEST_CONTEXT,
        "deployment_config": {"distributed": distributed},
    }
    manifest_path = tmp_path / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


# ---------------------------------------------------------------------------
# SLURM

def _slurm_deployment(tmp_path: Path, launcher: str, nnodes: int = 3) -> SlurmDeployment:
    distributed = {
        "launcher": launcher,
        "nnodes": nnodes,
        "nproc_per_node": 8,
        "backend": "nccl",
        "port": 29500,
    }
    cfg = DeploymentConfig(
        target="slurm",
        manifest_file=str(_write_manifest(tmp_path, distributed)),
        additional_context={
            "deploy": "slurm",
            "gpu_vendor": "AMD",
            "guest_os": "UBUNTU",
            "slurm": {
                "partition": "test-partition",
                "nodes": nnodes,
                "gpus_per_node": 8,
                "time": "01:00:00",
                "output_dir": str(tmp_path / "slurm_output"),
            },
            "distributed": distributed,
        },
    )
    return SlurmDeployment(cfg)


class TestSlurmLauncherDispatch:
    """Every templated launcher must reach its own arm, not the unknown-launcher default."""

    @pytest.mark.parametrize("launcher", TEMPLATED_LAUNCHERS)
    def test_every_valid_launcher_has_an_arm(self, tmp_path, launcher):
        deployment = _slurm_deployment(tmp_path, launcher)
        with patch.object(
            SlurmDeployment, "_generate_basic_env_command", side_effect=AssertionError(
                f"launcher '{launcher}' fell through to the unknown-launcher default"
            )
        ):
            command = deployment._generate_launcher_command(
                launcher_type=launcher,
                nnodes=3,
                nproc_per_node=8,
                master_port=29500,
                model_name=MODEL_ENTRY["name"],
            )
        assert command is not None

    def test_the_documented_megatron_spelling_reaches_megatron(self, tmp_path):
        """The original bug: this returned the basic-env fallback, silently."""
        deployment = _slurm_deployment(tmp_path, "megatron-lm")
        command = deployment._generate_launcher_command(
            launcher_type="megatron-lm",
            nnodes=3,
            nproc_per_node=8,
            master_port=29500,
        )
        assert "MAD_MULTI_NODE_RUNNER" in command

    def test_an_unknown_launcher_still_reaches_the_default(self, tmp_path):
        """Sanity: the fallback arm is real, so the test above proves something."""
        deployment = _slurm_deployment(tmp_path, "torchrun")
        with patch.object(
            SlurmDeployment, "_generate_basic_env_command", return_value="basic"
        ) as basic:
            deployment._generate_launcher_command(
                launcher_type="not-a-launcher",
                nnodes=3,
                nproc_per_node=8,
                master_port=29500,
            )
        basic.assert_called_once()


class TestSlurmRayGpuVisibility:
    """Ray-based launchers must not export HIP+ROCR+CUDA together.

    Ray fails with "Inconsistent values found" when more than one visibility
    variable is set. sglang-disagg was missing from the guard, so it took the
    else-branch that exports all three.
    """

    @staticmethod
    def _render(deployment: SlurmDeployment) -> str:
        context = deployment._prepare_template_context(MODEL_ENTRY)
        return deployment.jinja_env.get_template("job.sh.j2").render(**context)

    @pytest.mark.parametrize("launcher", ["vllm", "sglang", "sglang-disagg"])
    def test_ray_launchers_take_the_single_variable_branch(self, tmp_path, launcher):
        script = self._render(_slurm_deployment(tmp_path, launcher))
        assert "unset RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES" in script
        assert 'unset CUDA_VISIBLE_DEVICES  # Unset to avoid "Inconsistent values" error' in script

    def test_non_ray_launchers_do_not(self, tmp_path):
        script = self._render(_slurm_deployment(tmp_path, "torchrun"))
        assert "unset RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES" not in script


# ---------------------------------------------------------------------------
# Kubernetes

def _k8s_deployment(tmp_path: Path, launcher: str, nnodes: int = 3):
    """KubernetesDeployment over a minimal manifest, with cluster access stubbed.

    ``__init__`` loads kubeconfig and builds API clients, both of which raise
    without a cluster. Patch the real class rather than a mixin stub, so the
    dispatch chain under test is the one that actually ships.
    """
    from madengine.deployment import kubernetes as k8s_module

    distributed = {
        "launcher": launcher,
        "nnodes": nnodes,
        "nproc_per_node": 8,
        "backend": "nccl",
    }
    cfg = DeploymentConfig(
        target="kubernetes",
        manifest_file=str(_write_manifest(tmp_path, distributed)),
        additional_context={
            "deploy": "kubernetes",
            "gpu_vendor": "AMD",
            "guest_os": "UBUNTU",
            "k8s": {"namespace": "default"},
            "distributed": distributed,
        },
    )
    with patch.object(k8s_module, "k8s_config"), patch.object(k8s_module, "client"):
        deployment = k8s_module.KubernetesDeployment(cfg)
    # Set in deploy(), which the render path below bypasses.
    deployment.job_name = "test-job"
    deployment.job_label = "test-job"
    deployment.service_name = "test-svc"
    deployment.main_container_name = "test-svc"
    deployment.configmap_name = "test-cm"
    return deployment


IMAGE_INFO = {
    "docker_image": "dummy:latest",
    "registry_image": "registry.local/dummy:latest",
}


class TestKubernetesLauncherDispatch:
    """The K8s dispatch chain must produce a launcher command for every valid launcher."""

    @pytest.mark.parametrize("launcher", TEMPLATED_LAUNCHERS)
    def test_every_valid_launcher_produces_a_launcher_command(self, tmp_path, launcher):
        deployment = _k8s_deployment(tmp_path, launcher)
        context = deployment._prepare_template_context(MODEL_ENTRY, IMAGE_INFO)
        assert context["launcher_type"] == launcher
        assert context["launcher_command"], (
            f"launcher '{launcher}' produced no launcher command — it reached no "
            f"dispatch arm in the K8s chain"
        )

    def test_the_documented_megatron_spelling_reaches_megatron(self, tmp_path):
        """The original bug: this produced no launcher command and no headless service."""
        deployment = _k8s_deployment(tmp_path, "megatron-lm")
        context = deployment._prepare_template_context(MODEL_ENTRY, IMAGE_INFO)
        assert context["launcher_command"]
        assert context["create_headless_service"] is True

    @pytest.mark.parametrize(
        "launcher", ["torchrun", "deepspeed", "torchtitan", "megatron-lm", "primus"]
    )
    def test_pytorch_native_launchers_get_a_pod_subdomain(self, tmp_path, launcher):
        """Multi-node PyTorch-native launchers need DNS for rank discovery."""
        deployment = _k8s_deployment(tmp_path, launcher)
        context = deployment._prepare_template_context(MODEL_ENTRY, IMAGE_INFO)
        assert context["subdomain"] == deployment.service_name


# ---------------------------------------------------------------------------
# Cross-backend parity

def test_slurm_and_kubernetes_support_the_same_templated_launchers(tmp_path):
    """A launcher valid on one backend must be valid on the other, or be a known exception."""
    for launcher in TEMPLATED_LAUNCHERS:
        slurm = _slurm_deployment(tmp_path, launcher)
        with patch.object(
            SlurmDeployment, "_generate_basic_env_command",
            side_effect=AssertionError(f"SLURM has no arm for '{launcher}'"),
        ):
            slurm._generate_launcher_command(
                launcher_type=launcher, nnodes=3, nproc_per_node=8, master_port=29500,
                model_name=MODEL_ENTRY["name"],
            )
        k8s = _k8s_deployment(tmp_path, launcher)
        context = k8s._prepare_template_context(MODEL_ENTRY, IMAGE_INFO)
        assert context["launcher_command"], f"K8s has no arm for '{launcher}'"


# ---------------------------------------------------------------------------
# The config-boundary chokepoint

class TestBaseDeploymentValidatesLaunchers:
    """Launchers are validated in ``BaseDeployment.__init__``, deliberately.

    ``execute()`` catches bare ``Exception`` and returns a FAILED result without
    re-raising, so a ConfigurationError raised any later never reaches the CLI's
    handler. ``__init__`` runs under DeploymentFactory.create(), which re-raises.
    """

    def test_bad_launcher_in_additional_context_raises(self, tmp_path):
        from madengine.core.errors import ConfigurationError

        with pytest.raises(ConfigurationError) as exc_info:
            _slurm_deployment(tmp_path, "megatron")
        assert "megatron-lm" in " ".join(exc_info.value.suggestions)

    def test_bad_launcher_in_the_manifest_raises(self, tmp_path):
        """The manifest is a separate source; the CLI validator never sees it."""
        from madengine.core.errors import ConfigurationError

        manifest_path = _write_manifest(tmp_path, {"launcher": "megatron"})
        cfg = DeploymentConfig(
            target="slurm",
            manifest_file=str(manifest_path),
            additional_context={
                "gpu_vendor": "AMD",
                "guest_os": "UBUNTU",
                "slurm": {"partition": "p", "output_dir": str(tmp_path)},
            },
        )
        with pytest.raises(ConfigurationError):
            SlurmDeployment(cfg)

    def test_bad_launcher_on_a_model_card_raises(self, tmp_path):
        from madengine.core.errors import ConfigurationError

        manifest = {
            "built_images": {"dummy-image": {"docker_image": "dummy:latest"}},
            "built_models": {
                "dummy-image": {**MODEL_ENTRY, "distributed": {"launcher": "megatron"}}
            },
            "context": MANIFEST_CONTEXT,
        }
        manifest_path = tmp_path / "build_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        cfg = DeploymentConfig(
            target="slurm",
            manifest_file=str(manifest_path),
            additional_context={
                "gpu_vendor": "AMD",
                "guest_os": "UBUNTU",
                "slurm": {"partition": "p", "output_dir": str(tmp_path)},
            },
        )
        with pytest.raises(ConfigurationError) as exc_info:
            SlurmDeployment(cfg)
        assert "dummy-image" in str(exc_info.value)

    def test_the_documented_alias_is_canonicalized_in_place(self, tmp_path):
        deployment = _slurm_deployment(tmp_path, "slurm-multi")
        assert deployment.config.additional_context["distributed"]["launcher"] == "slurm_multi"
