#!/usr/bin/env python3
"""
Regression guards for the llm-d integration.

These tests contain no llm-d assertions. Their only job is to pin down the
behaviour of the shared code paths that the llm-d work touches, so that a
config *without* an ``llm_d`` key keeps taking the exact same branch it took
before. They were written and made green before any llm-d code existed.

Guarded paths:

1. ``RunOrchestrator._infer_deployment_target`` — run-time target selection.
2. ``BuildOrchestrator._save_deployment_config`` — build-time target written
   into ``build_manifest.json``.
3. ``ConfigLoader.infer_and_validate_deploy_type`` — build-time validation that
   every k8s and slurm user already goes through.
4. ``DeploymentFactory`` registration — a broken optional deployment module must
   never deregister slurm/k8s.
5. The rendered Kubernetes Job for an ordinary k8s config.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import builtins
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from madengine.deployment.base import DeploymentConfig
from madengine.deployment.config_loader import ConfigLoader
from madengine.deployment.factory import (
    DeploymentFactory,
    register_default_deployments,
)
from madengine.orchestration.build_orchestrator import BuildOrchestrator
from madengine.orchestration.run_orchestrator import RunOrchestrator

MODEL_ENTRY = {
    "name": "dummy",
    "url": "",
    "dockerfile": "docker/dummy",
    "scripts": "scripts/dummy/run.sh",
    "n_gpus": "1",
    "owner": "mad.support@amd.com",
    "training_precision": "",
    "tags": ["pyt", "training"],
    "timeout": -1,
    "args": "",
}


# ---------------------------------------------------------------------------
# 1. Run-time target inference
# ---------------------------------------------------------------------------


class TestRunTargetInferenceUnchanged:
    """_infer_deployment_target must be untouched for llm_d-free configs."""

    @pytest.mark.parametrize(
        "config,expected",
        [
            ({}, "local"),
            ({"env_vars": {"A": "1"}}, "local"),
            ({"k8s": {"gpu_count": 1}}, "k8s"),
            ({"kubernetes": {"gpu_count": 1}}, "k8s"),
            ({"slurm": {"nodes": 2}}, "slurm"),
            # k8s wins over slurm today; lock the precedence in, not just the values.
            ({"k8s": {}, "slurm": {}}, "k8s"),
            ({"distributed": {"launcher": "torchrun"}}, "local"),
        ],
    )
    def test_target_for_configs_without_llm_d(self, config, expected):
        assert RunOrchestrator._infer_deployment_target(None, config) == expected


# ---------------------------------------------------------------------------
# 2. Build-time target written into the manifest
# ---------------------------------------------------------------------------


def _run_save_deployment_config(tmp_path: Path, additional_context: dict) -> dict:
    """Invoke _save_deployment_config against a minimal manifest, return the result."""
    manifest_file = tmp_path / "build_manifest.json"
    manifest_file.write_text(
        json.dumps({"built_images": {}, "built_models": {}, "context": {}})
    )

    # No spec: the method reaches for rich_console, which a spec'd mock refuses.
    orch = MagicMock()
    orch.additional_context = additional_context
    BuildOrchestrator._save_deployment_config(orch, str(manifest_file))

    return json.loads(manifest_file.read_text()).get("deployment_config", {})


class TestBuildTargetInferenceUnchanged:
    """_save_deployment_config must record the same target as before."""

    @pytest.mark.parametrize(
        "config,expected",
        [
            ({"k8s": {"gpu_count": 1}}, "k8s"),
            ({"kubernetes": {"gpu_count": 1}}, "k8s"),
            ({"slurm": {"nodes": 2}}, "slurm"),
            # slurm is checked first here, unlike the run-time helper.
            ({"k8s": {}, "slurm": {"nodes": 1}}, "slurm"),
            ({"deploy": "local", "k8s": {"gpu_count": 1}}, "local"),
            ({"env_vars": {"A": "1"}}, "local"),
        ],
    )
    def test_target_for_configs_without_llm_d(self, tmp_path, config, expected):
        assert _run_save_deployment_config(tmp_path, config)["target"] == expected

    def test_empty_context_writes_no_deployment_config(self, tmp_path):
        """An empty additional_context returns early, leaving the manifest alone."""
        assert _run_save_deployment_config(tmp_path, {}) == {}

    def test_none_valued_keys_are_stripped(self, tmp_path):
        """Only keys with real values land in deployment_config."""
        result = _run_save_deployment_config(tmp_path, {"k8s": {"gpu_count": 1}})

        assert "slurm" not in result
        assert "kubernetes" not in result
        assert result["k8s"] == {"gpu_count": 1}


# ---------------------------------------------------------------------------
# 3. Build-time deploy-type validation
# ---------------------------------------------------------------------------


class TestInferAndValidateDeployTypeUnchanged:
    """This runs at build time for every user; it must not gain new behaviour."""

    @pytest.mark.parametrize(
        "config,expected",
        [
            ({}, "local"),
            ({"k8s": {}}, "k8s"),
            ({"kubernetes": {}}, "k8s"),
            ({"slurm": {}}, "slurm"),
            ({"deploy": "k8s", "k8s": {}}, "k8s"),
            ({"deploy": "slurm", "slurm": {}}, "slurm"),
            ({"deploy": "local"}, "local"),
        ],
    )
    def test_accepted_configs(self, config, expected):
        assert ConfigLoader.infer_and_validate_deploy_type(config) == expected

    @pytest.mark.parametrize(
        "config",
        [
            {"k8s": {}, "slurm": {}},
            {"deploy": "k8s"},
            {"deploy": "slurm"},
            {"deploy": "local", "k8s": {}},
        ],
    )
    def test_rejected_configs_still_raise(self, config):
        with pytest.raises(ValueError):
            ConfigLoader.infer_and_validate_deploy_type(config)


# ---------------------------------------------------------------------------
# 4. Factory registration resilience
# ---------------------------------------------------------------------------


class TestFactoryRegistrationResilience:
    """A broken optional deployment module must not take the others down."""

    def test_core_targets_registered(self):
        available = DeploymentFactory.available_deployments()

        assert "slurm" in available
        assert "k8s" in available
        assert "kubernetes" in available

    def test_core_targets_survive_a_failing_llm_d_import(self):
        """Force the optional llm_d module import to fail.

        register_default_deployments() must still leave slurm and k8s
        registered, so adding a new optional target can never regress the
        working ones.
        """
        real_import = builtins.__import__

        def exploding_import(name, globals=None, locals=None, fromlist=(), level=0):
            # Relative imports from the factory arrive as level>0, name=="llm_d".
            if name == "llm_d" or name.endswith("deployment.llm_d"):
                raise ImportError("simulated failure importing llm_d")
            return real_import(name, globals, locals, fromlist, level)

        with patch.object(builtins, "__import__", side_effect=exploding_import):
            try:
                register_default_deployments()
            except ImportError:
                pytest.fail(
                    "register_default_deployments() let an optional import failure escape"
                )

        available = DeploymentFactory.available_deployments()
        assert "slurm" in available
        assert "k8s" in available
        assert "kubernetes" in available


# ---------------------------------------------------------------------------
# 5. Golden render of the Kubernetes Job
# ---------------------------------------------------------------------------


def _build_k8s_deployment(tmp_path: Path, additional_context: dict):
    """A real KubernetesDeployment over a minimal manifest, no cluster contact."""
    from madengine.deployment.kubernetes import KubernetesDeployment

    (tmp_path / "scripts" / "dummy").mkdir(parents=True)
    (tmp_path / "scripts" / "dummy" / "run.sh").write_text("#!/bin/bash\necho hi\n")

    manifest_file = tmp_path / "build_manifest.json"
    manifest_file.write_text(
        json.dumps(
            {
                "built_images": {"dummy": {"docker_image": "dummy:latest"}},
                "built_models": {"dummy": MODEL_ENTRY},
                "context": {
                    "docker_env_vars": {},
                    "docker_mounts": {},
                    "docker_build_arg": {},
                    "gpu_vendor": "AMD",
                    "guest_os": "UBUNTU",
                    "docker_gpus": "all",
                },
            }
        )
    )

    config = DeploymentConfig(
        target="k8s",
        manifest_file=str(manifest_file),
        additional_context=dict(additional_context),
    )

    with (
        patch("madengine.deployment.kubernetes.k8s_config"),
        patch("madengine.deployment.kubernetes.client"),
    ):
        return KubernetesDeployment(config)


class TestKubernetesJobRenderUnchanged:
    """Pin the rendered Job for an ordinary k8s config.

    Not a byte-for-byte golden file (the image tag and script bundling move with
    unrelated changes); instead the structural contract that the llm-d work must
    not perturb.
    """

    def test_single_node_job_structure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        deployment = _build_k8s_deployment(tmp_path, {"k8s": {"gpu_count": 1}})

        context = deployment._prepare_template_context(
            MODEL_ENTRY,
            {"docker_image": "dummy:latest", "registry_image": "dummy:latest"},
        )
        rendered = deployment.jinja_env.get_template("job.yaml.j2").render(**context)

        assert "kind: Job" in rendered
        assert "restartPolicy: Never" in rendered
        assert 'amd.com/gpu: "1"' in rendered
        # Single node: no Indexed completion mode, no headless Service.
        assert "completionMode" not in rendered
        assert context["completions"] == 1
        assert context["parallelism"] == 1
        assert context["create_headless_service"] is False
        assert context["subdomain"] is None
        # The container reports itself as a kubernetes deployment; the execution
        # layer branches on this value (container_runner.py).
        assert "MAD_DEPLOYMENT_TYPE=kubernetes" in rendered

    def test_multi_node_torchrun_job_structure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        deployment = _build_k8s_deployment(
            tmp_path,
            {
                "k8s": {"gpu_count": 8},
                "launcher": {"type": "torchrun", "nnodes": 2},
            },
        )

        context = deployment._prepare_template_context(
            MODEL_ENTRY,
            {"docker_image": "dummy:latest", "registry_image": "dummy:latest"},
        )
        rendered = deployment.jinja_env.get_template("job.yaml.j2").render(**context)

        assert "completionMode: Indexed" in rendered
        assert context["completions"] == 2
        assert context["parallelism"] == 2
        assert context["create_headless_service"] is True
        # torchrun is pytorch-native, so pods get a subdomain for stable DNS.
        assert context["subdomain"] == deployment.service_name
