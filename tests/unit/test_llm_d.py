#!/usr/bin/env python3
"""
Unit tests for the llm-d deployment target.

Follows the pattern in test_slurm_job_template.py: build a real deployment over
a minimal build_manifest.json under tmp_path, then assert on the merged config
and the rendered Kubernetes Job. No cluster is contacted.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from madengine.deployment.base import DeploymentConfig, DeploymentStatus
from madengine.deployment.config_loader import ConfigLoader
from madengine.deployment.factory import DeploymentFactory

MODEL_ENTRY = {
    "name": "dummy_llm_d",
    "url": "",
    "dockerfile": "docker/dummy_llm_d",
    "scripts": "scripts/dummy_llm_d/run.sh",
    "n_gpus": "0",
    "owner": "mad.support@amd.com",
    "training_precision": "",
    "tags": ["llm_d", "dummy_llm_d"],
    "timeout": -1,
    "args": "",
}

ATTACH_CONTEXT = {
    "k8s": {"gpu_count": 0, "namespace": "llm-d-bench"},
    "llm_d": {
        "endpoint_url": "http://llm-d-gw.example.com",
        "model": {"name": "Qwen3-32B"},
    },
}


def _build_deployment(tmp_path: Path, additional_context: dict):
    """A real LlmdDeployment over a minimal manifest, with the k8s client mocked."""
    scripts_dir = tmp_path / "scripts" / "dummy_llm_d"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "run.sh").write_text("#!/bin/bash\necho hi\n")

    manifest_file = tmp_path / "build_manifest.json"
    manifest_file.write_text(
        json.dumps(
            {
                "built_images": {"dummy_llm_d": {"docker_image": "client:latest"}},
                "built_models": {"dummy_llm_d": MODEL_ENTRY},
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
        target="llm-d",
        manifest_file=str(manifest_file),
        additional_context=json.loads(json.dumps(additional_context)),
    )

    with (
        patch("madengine.deployment.kubernetes.k8s_config"),
        patch("madengine.deployment.kubernetes.client"),
    ):
        from madengine.deployment.llm_d import LlmdDeployment

        return LlmdDeployment(config)


def _render_job(deployment) -> str:
    """Render job.yaml.j2 exactly as prepare() does, without touching a cluster."""
    context = deployment._prepare_template_context(
        MODEL_ENTRY,
        {"docker_image": "client:latest", "registry_image": "client:latest"},
    )
    return deployment.jinja_env.get_template("job.yaml.j2").render(**context)


# ---------------------------------------------------------------------------
# Registration and target inference
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_llm_d_is_registered(self):
        available = DeploymentFactory.available_deployments()
        assert "llm-d" in available
        assert "llm_d" in available

    def test_run_target_inference_prefers_llm_d_over_k8s(self):
        """An llm-d config carries a k8s block too; llm-d must win."""
        from madengine.orchestration.run_orchestrator import RunOrchestrator

        assert RunOrchestrator._infer_deployment_target(None, ATTACH_CONTEXT) == "llm-d"

    def test_build_target_inference_records_llm_d(self, tmp_path):
        from madengine.orchestration.build_orchestrator import BuildOrchestrator

        manifest_file = tmp_path / "build_manifest.json"
        manifest_file.write_text(
            json.dumps({"built_images": {}, "built_models": {}, "context": {}})
        )

        # No spec: the method reaches for rich_console.
        orch = MagicMock()
        orch.additional_context = dict(ATTACH_CONTEXT)
        BuildOrchestrator._save_deployment_config(orch, str(manifest_file))

        saved = json.loads(manifest_file.read_text())["deployment_config"]
        assert saved["target"] == "llm-d"
        # The llm_d block must survive into the manifest, or run-time inference
        # would fall back to plain k8s.
        assert saved["llm_d"]["endpoint_url"] == "http://llm-d-gw.example.com"

    def test_llm_d_survives_the_manifest_round_trip(self):
        """The build->run key lists must both carry llm_d, or it is dropped."""
        import inspect

        from madengine.orchestration import run_orchestrator

        source = inspect.getsource(run_orchestrator)
        # Both merge loops enumerate their keys literally; neither may omit llm_d.
        assert source.count('"llm_d"') >= 2

    def test_validator_accepts_an_llm_d_object(self):
        from madengine.cli.validators import validate_additional_context_structure

        # Must not raise.
        validate_additional_context_structure(dict(ATTACH_CONTEXT))

    def test_validator_rejects_a_non_object_llm_d(self):
        import typer

        from madengine.cli.constants import ExitCode
        from madengine.cli.validators import validate_additional_context_structure

        with pytest.raises(typer.Exit) as excinfo:
            validate_additional_context_structure({"llm_d": "yes please"})

        assert excinfo.value.exit_code == ExitCode.INVALID_ARGS


ROOT_MODELS_JSON = Path(__file__).resolve().parents[2] / "models.json"


@pytest.mark.skipif(
    not ROOT_MODELS_JSON.exists(),
    reason="root models.json is a gitignored dev fixture; absent on a clean checkout",
)
class TestReferenceModelTags:
    """dummy_llm_d needs an llm-d cluster, so no shared tag may reach it."""

    SHARED_TAGS = {"dummies", "inference", "dummy_distributed", "pyt", "training"}

    @staticmethod
    def _models():
        return json.loads(ROOT_MODELS_JSON.read_text())

    def test_reference_model_is_registered(self):
        assert any(m["name"] == "dummy_llm_d" for m in self._models())

    def test_reference_model_carries_no_shared_tag(self):
        """Otherwise an existing sweep silently grows a run needing an llm-d stack."""
        entry = next(m for m in self._models() if m["name"] == "dummy_llm_d")

        assert self.SHARED_TAGS.isdisjoint(entry["tags"]), (
            f"dummy_llm_d must not share tags with existing models: "
            f"{sorted(self.SHARED_TAGS.intersection(entry['tags']))}"
        )

    def test_no_other_model_carries_the_llm_d_tag(self):
        """--tags llm_d must select exactly the llm-d reference model."""
        tagged = [m["name"] for m in self._models() if "llm_d" in m.get("tags", [])]

        assert tagged == ["dummy_llm_d"]


# ---------------------------------------------------------------------------
# Configuration merging
# ---------------------------------------------------------------------------


class TestConfigMerge:
    def test_defaults_are_applied(self, tmp_path):
        deployment = _build_deployment(tmp_path, ATTACH_CONTEXT)

        assert deployment.llmd_config["release_prefix"] == "madengine"
        assert deployment.llmd_config["gateway"] == "agentgateway"
        assert deployment.llmd_config["teardown"] is True
        assert deployment.llmd_config["prefill"]["replicas"] == 1

    def test_user_values_beat_defaults(self, tmp_path):
        deployment = _build_deployment(
            tmp_path,
            {
                **ATTACH_CONTEXT,
                "llm_d": {
                    **ATTACH_CONTEXT["llm_d"],
                    "gateway": "istio",
                    "decode": {"replicas": 4},
                },
            },
        )

        assert deployment.llmd_config["gateway"] == "istio"
        assert deployment.llmd_config["decode"]["replicas"] == 4
        # A partial override must not wipe its siblings.
        assert deployment.llmd_config["decode"]["tensor_parallel"] == 1

    def test_k8s_presets_still_apply(self, tmp_path):
        """The client Job is an ordinary k8s Job; it must get the k8s defaults."""
        deployment = _build_deployment(tmp_path, ATTACH_CONTEXT)

        assert deployment.k8s_config["namespace"] == "llm-d-bench"
        assert deployment.gpu_resource_name == "amd.com/gpu"
        # Sanity: the parent read the same dict the llm-d loader produced.
        assert deployment.k8s_config is deployment.config.additional_context.get("k8s")

    def test_chart_versions_are_unpinned_by_default(self, tmp_path):
        """Deliberate: managed mode must refuse to run until they are pinned."""
        deployment = _build_deployment(tmp_path, ATTACH_CONTEXT)

        for name, spec in deployment.llmd_config["charts"].items():
            assert spec["version"] is None, f"{name} should ship unpinned"

    def test_load_llmd_config_leaves_a_plain_k8s_config_alone(self):
        """Guard: the new loader must agree with load_k8s_config where it overlaps."""
        user = {"k8s": {"gpu_count": 2}}
        k8s_only = ConfigLoader.load_k8s_config(user)
        llmd = ConfigLoader.load_llmd_config(user)

        assert llmd["k8s"] == k8s_only["k8s"]


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------


class TestModes:
    def test_endpoint_url_selects_attach_mode(self, tmp_path):
        deployment = _build_deployment(tmp_path, ATTACH_CONTEXT)

        assert deployment.is_attach_mode is True

    def test_attach_mode_never_tears_down(self, tmp_path):
        """madengine must not be able to destroy a stack it did not create."""
        deployment = _build_deployment(
            tmp_path,
            {**ATTACH_CONTEXT, "llm_d": {**ATTACH_CONTEXT["llm_d"], "teardown": True}},
        )

        assert deployment.should_teardown is False

    def test_no_endpoint_url_selects_managed_mode(self, tmp_path):
        deployment = _build_deployment(
            tmp_path,
            {"k8s": {"gpu_count": 0}, "llm_d": {"model": {"name": "Qwen3-32B"}}},
        )

        assert deployment.is_attach_mode is False
        assert deployment.should_teardown is True


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


class TestValidate:
    def test_slurm_alongside_llm_d_is_rejected(self, tmp_path):
        deployment = _build_deployment(
            tmp_path, {**ATTACH_CONTEXT, "slurm": {"nodes": 2}}
        )

        assert deployment.validate() is False

    def test_missing_model_name_is_rejected(self, tmp_path):
        deployment = _build_deployment(
            tmp_path, {"k8s": {}, "llm_d": {"endpoint_url": "http://gw"}}
        )

        assert deployment.validate() is False

    def test_attach_mode_does_not_require_helm(self, tmp_path):
        deployment = _build_deployment(tmp_path, ATTACH_CONTEXT)

        with (
            patch(
                "madengine.deployment.kubernetes.KubernetesDeployment.validate",
                return_value=True,
            ),
            patch(
                "madengine.deployment.llm_d.shutil.which", return_value=None
            ) as which,
        ):
            assert deployment.validate() is True

        which.assert_not_called()

    def test_managed_mode_requires_helm(self, tmp_path):
        deployment = _build_deployment(
            tmp_path,
            {"k8s": {}, "llm_d": {"model": {"name": "Qwen3-32B"}}},
        )

        with (
            patch(
                "madengine.deployment.kubernetes.KubernetesDeployment.validate",
                return_value=True,
            ),
            patch("madengine.deployment.llm_d.shutil.which", return_value=None),
        ):
            assert deployment.validate() is False

    def test_managed_mode_rejects_unpinned_charts(self, tmp_path):
        deployment = _build_deployment(
            tmp_path,
            {"k8s": {}, "llm_d": {"model": {"name": "Qwen3-32B"}}},
        )

        with (
            patch(
                "madengine.deployment.kubernetes.KubernetesDeployment.validate",
                return_value=True,
            ),
            patch(
                "madengine.deployment.llm_d.shutil.which", return_value="/usr/bin/helm"
            ),
            patch.object(deployment, "_validate_crds", return_value=True) as crds,
        ):
            assert deployment.validate() is False

        # Rejected on the version pins, before ever reaching the cluster.
        crds.assert_not_called()

    def test_managed_mode_is_not_implemented_yet(self, tmp_path):
        """Every real prerequisite passes; standup itself is still missing."""
        deployment = _build_deployment(
            tmp_path,
            {
                "k8s": {},
                "llm_d": {
                    "model": {"name": "Qwen3-32B"},
                    "charts": {
                        "infra": {"version": "1.0.0"},
                        "gaie": {"version": "1.0.0"},
                        "modelservice": {"version": "1.0.0"},
                    },
                },
            },
        )

        with (
            patch(
                "madengine.deployment.kubernetes.KubernetesDeployment.validate",
                return_value=True,
            ),
            patch(
                "madengine.deployment.llm_d.shutil.which", return_value="/usr/bin/helm"
            ),
            patch.object(deployment, "_validate_crds", return_value=True),
        ):
            assert deployment.validate() is False

    def test_underscore_comment_keys_are_not_mistaken_for_charts(self, tmp_path):
        """_comment_* keys live alongside the charts and must be skipped."""
        deployment = _build_deployment(
            tmp_path,
            {
                "k8s": {},
                "llm_d": {
                    "model": {"name": "Qwen3-32B", "uri": "hf://Qwen/Qwen3-32B"},
                    "charts": {
                        "_comment": "not a chart",
                        "infra": {"version": "1.0.0"},
                        "gaie": {"version": "1.0.0"},
                        "modelservice": {"version": "1.0.0"},
                    },
                },
            },
        )

        with (
            patch(
                "madengine.deployment.kubernetes.KubernetesDeployment.validate",
                return_value=True,
            ),
            patch(
                "madengine.deployment.llm_d.shutil.which", return_value="/usr/bin/helm"
            ),
            patch.object(deployment, "_validate_crds", return_value=True) as crds,
        ):
            deployment.validate()

        # Got past the pin check, so "_comment" was not treated as an unpinned chart.
        crds.assert_called_once()


# ---------------------------------------------------------------------------
# deploy() guard
# ---------------------------------------------------------------------------


class TestDeployGuard:
    def test_deploy_without_a_resolved_endpoint_raises(self, tmp_path):
        """Backstop for a deploy() that skipped prepare(): never run blind.

        A benchmark client with no endpoint would fail inside the pod after
        spending a Job scheduling round-trip; failing here is cheaper and the
        message names the two config keys that fix it.
        """
        from madengine.core.errors import ConfigurationError

        deployment = _build_deployment(
            tmp_path, {"k8s": {}, "llm_d": {"model": {"name": "Qwen3-32B"}}}
        )

        with pytest.raises(ConfigurationError, match="No llm-d endpoint resolved"):
            deployment.deploy()

    def test_attach_deploy_delegates_to_kubernetes(self, tmp_path):
        deployment = _build_deployment(tmp_path, ATTACH_CONTEXT)

        with patch(
            "madengine.deployment.kubernetes.KubernetesDeployment.deploy"
        ) as parent_deploy:
            deployment.deploy()

        parent_deploy.assert_called_once()


# ---------------------------------------------------------------------------
# Rendered client Job
# ---------------------------------------------------------------------------


class TestRenderedJob:
    def test_endpoint_and_model_reach_the_container(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rendered = _render_job(_build_deployment(tmp_path, ATTACH_CONTEXT))

        assert "MAD_LLM_D_ENDPOINT" in rendered
        assert "http://llm-d-gw.example.com" in rendered
        assert "MAD_LLM_D_MODEL" in rendered
        assert "Qwen3-32B" in rendered

    def test_topology_reaches_the_container(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        deployment = _build_deployment(
            tmp_path,
            {
                **ATTACH_CONTEXT,
                "llm_d": {
                    **ATTACH_CONTEXT["llm_d"],
                    "prefill": {"replicas": 2},
                    "decode": {"replicas": 3, "tensor_parallel": 8},
                },
            },
        )
        env = deployment._llmd_env_vars()

        assert env["MAD_LLM_D_PREFILL_REPLICAS"] == "2"
        assert env["MAD_LLM_D_DECODE_REPLICAS"] == "3"
        assert env["MAD_LLM_D_TP"] == "8"
        assert env["MAD_LLM_D_NAMESPACE"] == "llm-d-bench"

    def test_client_job_is_a_single_cpu_only_pod(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        deployment = _build_deployment(tmp_path, ATTACH_CONTEXT)
        context = deployment._prepare_template_context(
            MODEL_ENTRY,
            {"docker_image": "client:latest", "registry_image": "client:latest"},
        )
        rendered = deployment.jinja_env.get_template("job.yaml.j2").render(**context)

        assert context["completions"] == 1
        assert context["parallelism"] == 1
        assert context["gpu_count"] == 0
        assert 'amd.com/gpu: "0"' in rendered
        assert "completionMode" not in rendered

    def test_container_still_reports_itself_as_kubernetes(self, tmp_path, monkeypatch):
        """container_runner.py branches on this; llm-d must not introduce a value."""
        monkeypatch.chdir(tmp_path)
        rendered = _render_job(_build_deployment(tmp_path, ATTACH_CONTEXT))

        assert "MAD_DEPLOYMENT_TYPE=kubernetes" in rendered

    def test_rocenv_collection_is_off_by_default(self, tmp_path, monkeypatch):
        """No GPU in the client pod, so there is no ROCm environment to record."""
        monkeypatch.chdir(tmp_path)
        rendered = _render_job(_build_deployment(tmp_path, ATTACH_CONTEXT))

        assert "run_rocenv_tool.sh" not in rendered

    def test_rocenv_collection_can_be_switched_back_on(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rendered = _render_job(
            _build_deployment(
                tmp_path, {**ATTACH_CONTEXT, "generate_sys_env_details": True}
            )
        )

        assert "run_rocenv_tool.sh" in rendered

    def test_base_env_vars_are_preserved(self, tmp_path):
        """The llm-d additions must not clobber the user's own env_vars."""
        deployment = _build_deployment(
            tmp_path, {**ATTACH_CONTEXT, "env_vars": {"MY_KNOB": "7"}}
        )
        env = deployment._prepare_env_vars(MODEL_ENTRY)

        assert env["MY_KNOB"] == "7"
        assert env["MAD_LLM_D_ENDPOINT"] == "http://llm-d-gw.example.com"

    def test_attach_mode_advertises_no_release_prefix(self, tmp_path):
        """There are no helm releases to correlate with when attaching."""
        deployment = _build_deployment(tmp_path, ATTACH_CONTEXT)

        assert "MAD_LLM_D_RELEASE_PREFIX" not in deployment._llmd_env_vars()


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


class TestTeardown:
    def test_teardown_runs_after_a_successful_execute(self, tmp_path):
        """BaseDeployment.execute() never calls cleanup() on success."""
        deployment = _build_deployment(tmp_path, ATTACH_CONTEXT)

        with (
            patch("madengine.deployment.base.BaseDeployment.execute") as parent_execute,
            patch.object(deployment, "_teardown_stack") as teardown,
        ):
            parent_execute.return_value = MagicMock(status=DeploymentStatus.SUCCESS)
            deployment.execute()

        teardown.assert_called_once()

    def test_teardown_runs_after_a_failing_execute(self, tmp_path):
        deployment = _build_deployment(tmp_path, ATTACH_CONTEXT)

        with (
            patch(
                "madengine.deployment.base.BaseDeployment.execute",
                side_effect=RuntimeError("boom"),
            ),
            patch.object(deployment, "_teardown_stack") as teardown,
        ):
            with pytest.raises(RuntimeError):
                deployment.execute()

        teardown.assert_called_once()

    def test_teardown_is_a_noop_when_nothing_was_installed(self, tmp_path):
        deployment = _build_deployment(tmp_path, ATTACH_CONTEXT)

        with patch.object(deployment, "_uninstall_release") as uninstall:
            deployment._teardown_stack()

        uninstall.assert_not_called()

    def test_teardown_uninstalls_newest_first(self, tmp_path):
        deployment = _build_deployment(
            tmp_path, {"k8s": {}, "llm_d": {"model": {"name": "m"}}}
        )
        deployment._installed_releases = ["infra-x", "gaie-x", "ms-x"]

        with patch.object(deployment, "_uninstall_release") as uninstall:
            deployment._teardown_stack()

        assert [c.args[0] for c in uninstall.call_args_list] == [
            "ms-x",
            "gaie-x",
            "infra-x",
        ]

    def test_a_failing_uninstall_does_not_mask_the_result(self, tmp_path):
        """A teardown error must never propagate over the benchmark outcome."""
        deployment = _build_deployment(
            tmp_path, {"k8s": {}, "llm_d": {"model": {"name": "m"}}}
        )
        deployment._installed_releases = ["infra-x", "ms-x"]

        with patch.object(
            deployment, "_uninstall_release", side_effect=RuntimeError("helm exploded")
        ) as uninstall:
            deployment._teardown_stack()  # must not raise

        # Both are still attempted; one failure does not abandon the rest.
        assert uninstall.call_count == 2

    def test_teardown_false_leaves_the_stack_up(self, tmp_path):
        deployment = _build_deployment(
            tmp_path,
            {"k8s": {}, "llm_d": {"model": {"name": "m"}, "teardown": False}},
        )
        deployment._installed_releases = ["ms-x"]

        with patch.object(deployment, "_uninstall_release") as uninstall:
            deployment._teardown_stack()

        uninstall.assert_not_called()
