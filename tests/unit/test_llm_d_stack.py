#!/usr/bin/env python3
"""
Unit tests for llm-d managed standup (Phase 2).

Covers the helm mechanics in llm_d_stack.py and the standup/readiness/teardown
orchestration in LlmdDeployment. No cluster is contacted and no helm binary is
invoked: the shell runner is a mock and every assertion is on the command
strings and values dicts madengine produces.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from madengine.core.errors import ConfigurationError
from madengine.deployment.base import DeploymentConfig, DeploymentStatus
from madengine.deployment.llm_d_stack import (
    COMPONENTS,
    MODEL_SERVER_PORT,
    LlmdStack,
    LlmdStackError,
)

from .test_llm_d import MODEL_ENTRY

PINNED_CHARTS = {
    "infra": {"ref": "oci://example.test/infra", "version": "1.2.3"},
    "gaie": {"ref": "oci://example.test/gaie", "version": "4.5.6"},
    "modelservice": {"ref": "oci://example.test/ms", "version": "7.8.9"},
}

MANAGED_LLMD = {
    "model": {"name": "Qwen3-32B", "uri": "hf://Qwen/Qwen3-32B"},
    "gateway": "agentgateway",
    "prefill": {"replicas": 2, "tensor_parallel": 8, "gpu_count": 8},
    "decode": {"replicas": 1, "tensor_parallel": 8, "gpu_count": 8},
    "charts": PINNED_CHARTS,
}


def _stack(llmd_overrides=None, **kwargs):
    """An LlmdStack over MANAGED_LLMD with a mock shell."""
    config = json.loads(json.dumps(MANAGED_LLMD))
    config.update(llmd_overrides or {})
    shell = MagicMock()
    shell.sh.return_value = ""
    params = {
        "llmd_config": config,
        "namespace": "llm-d-bench",
        "release_prefix": "madengine-dummy-llm-d",
        "shell": shell,
    }
    params.update(kwargs)
    return LlmdStack(**params)


def _managed_deployment(tmp_path: Path, llmd_overrides=None, k8s=None):
    """A real LlmdDeployment in managed mode, with the k8s client mocked."""
    llmd = json.loads(json.dumps(MANAGED_LLMD))
    llmd.update(llmd_overrides or {})

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
        additional_context={
            "k8s": k8s if k8s is not None else {"output_dir": str(tmp_path / "out")},
            "llm_d": llmd,
        },
    )

    with (
        patch("madengine.deployment.kubernetes.k8s_config"),
        patch("madengine.deployment.kubernetes.client"),
    ):
        from madengine.deployment.llm_d import LlmdDeployment

        deployment = LlmdDeployment(config)

    deployment.shell = MagicMock()
    deployment.shell.sh.return_value = ""
    return deployment


def _gateway(name="llm-d-gw", instance=None, addresses=None, port=80):
    """A Gateway resource as CustomObjectsApi returns it."""
    labels = {"app.kubernetes.io/instance": instance} if instance else {}
    return {
        "metadata": {"name": name, "labels": labels},
        "spec": {"listeners": [{"port": port}]},
        "status": {"addresses": addresses if addresses is not None else []},
    }


def _deployment_obj(name, replicas, ready):
    obj = MagicMock()
    obj.metadata.name = name
    obj.spec.replicas = replicas
    obj.status.ready_replicas = ready
    return obj


# ---------------------------------------------------------------------------
# Release naming
# ---------------------------------------------------------------------------


class TestReleaseNaming:
    def test_release_names_are_prefixed_per_component(self):
        stack = _stack(release_prefix="madengine-qwen")
        assert stack.release_names == [
            "madengine-qwen-infra",
            "madengine-qwen-gaie",
            "madengine-qwen-modelservice",
        ]

    def test_install_order_puts_modelservice_last(self):
        """Model servers reference the InferencePool the gaie release creates."""
        assert COMPONENTS == ("infra", "gaie", "modelservice")

    def test_prefix_leaves_room_for_the_53_char_helm_limit(self, tmp_path):
        """A long model name must not produce an over-length release name."""
        deployment = _managed_deployment(tmp_path)
        deployment.manifest["built_models"]["dummy_llm_d"]["name"] = "x" * 200

        for release in deployment.stack.release_names:
            assert len(release) <= 53, release

    def test_prefix_is_a_dns_label(self, tmp_path):
        """Helm release names may not contain dots."""
        deployment = _managed_deployment(tmp_path)
        deployment.manifest["built_models"]["dummy_llm_d"]["name"] = "org/model.v1.5"

        for release in deployment.stack.release_names:
            assert "." not in release
            assert "/" not in release

    def test_the_client_learns_the_release_prefix(self, tmp_path):
        """So a benchmark script can correlate its run with 'helm list'."""
        deployment = _managed_deployment(tmp_path)

        assert (
            deployment._llmd_env_vars()["MAD_LLM_D_RELEASE_PREFIX"]
            == "madengine-dummy-llm-d"
        )


# ---------------------------------------------------------------------------
# Chart values
# ---------------------------------------------------------------------------


class TestModelserviceValues:
    def test_model_uri_and_name_are_wired_through(self):
        values = _stack().values("modelservice")
        assert values["modelArtifacts"]["uri"] == "hf://Qwen/Qwen3-32B"
        assert values["routing"]["modelName"] == "Qwen3-32B"
        assert values["routing"]["servicePort"] == MODEL_SERVER_PORT

    def test_replicas_and_tensor_parallel_reach_both_roles(self):
        values = _stack().values("modelservice")
        assert values["prefill"]["replicas"] == 2
        assert values["decode"]["replicas"] == 1
        for role in ("prefill", "decode"):
            args = values[role]["containers"][0]["args"]
            assert args[args.index("--tensor-parallel-size") + 1] == "8"

    def test_gpu_resource_name_follows_the_k8s_config(self):
        values = _stack(gpu_resource_name="nvidia.com/gpu").values("modelservice")
        limits = values["decode"]["containers"][0]["resources"]["limits"]
        assert limits == {"nvidia.com/gpu": "8"}

    def test_zero_prefill_replicas_disables_the_role(self):
        """Aggregated serving: do not create a prefill Deployment at all."""
        values = _stack({"prefill": {"replicas": 0}}).values("modelservice")
        assert values["prefill"] == {"create": False}
        assert values["decode"]["create"] is True

    def test_modelservice_does_not_create_a_second_inferencepool(self):
        values = _stack().values("modelservice")
        pool = values["routing"]["inferencePool"]
        assert pool["create"] is False
        assert pool["name"] == "madengine-dummy-llm-d-gaie"

    def test_hf_token_is_referenced_by_secret_name_only(self):
        """The token itself must never reach a values file."""
        values = _stack(
            {"model": dict(MANAGED_LLMD["model"], hf_token_secret="hf-tok")}
        )
        rendered = yaml.safe_dump(values.values("modelservice"))
        assert "hf-tok" in rendered
        assert values.values("modelservice")["modelArtifacts"]["authSecretName"] == (
            "hf-tok"
        )

    def test_auth_secret_is_omitted_when_unset(self):
        assert "authSecretName" not in _stack().values("modelservice")["modelArtifacts"]

    def test_role_image_override(self):
        values = _stack(
            {"decode": dict(MANAGED_LLMD["decode"], image="rocm/vllm:latest")}
        ).values("modelservice")
        assert values["decode"]["containers"][0]["image"] == "rocm/vllm:latest"


class TestInfraAndGaieValues:
    def test_gateway_class_flows_to_infra_and_provider(self):
        stack = _stack({"gateway": "istio"})
        assert stack.values("infra")["gateway"]["gatewayClassName"] == "istio"
        assert stack.values("gaie")["provider"]["name"] == "istio"

    def test_inferencepool_targets_the_model_server_port(self):
        pool = _stack().values("gaie")["inferencePool"]
        assert pool["targetPortNumber"] == MODEL_SERVER_PORT
        assert pool["modelServers"]["matchLabels"] == {
            "llm-d.ai/inferenceServing": "true"
        }

    def test_unknown_component_is_rejected(self):
        with pytest.raises(LlmdStackError, match="Unknown llm-d component"):
            _stack().values("epp")


class TestExtraValues:
    def test_bare_dict_targets_modelservice(self):
        stack = _stack({"extra_values": {"decode": {"replicas": 99}}})
        assert stack.values("modelservice")["decode"]["replicas"] == 99

    def test_component_keyed_dict_targets_each_chart(self):
        stack = _stack(
            {
                "extra_values": {
                    "gaie": {"inferenceExtension": {"replicas": 3}},
                    "modelservice": {"routing": {"servicePort": 9000}},
                }
            }
        )
        assert stack.values("gaie")["inferenceExtension"]["replicas"] == 3
        assert stack.values("modelservice")["routing"]["servicePort"] == 9000
        # Untargeted components keep their generated values.
        assert stack.values("infra")["gateway"]["enabled"] is True

    def test_extra_values_deep_merge_rather_than_replace(self):
        stack = _stack({"extra_values": {"routing": {"servicePort": 9000}}})
        routing = stack.values("modelservice")["routing"]
        assert routing["servicePort"] == 9000
        assert routing["modelName"] == "Qwen3-32B"  # sibling survived

    def test_extra_values_win_over_generated_values(self):
        """The escape hatch is only useful if it is applied last."""
        stack = _stack({"extra_values": {"modelArtifacts": {"uri": "pvc://local"}}})
        assert stack.values("modelservice")["modelArtifacts"]["uri"] == "pvc://local"


class TestValuesFiles:
    def test_write_values_emits_one_file_per_component(self, tmp_path):
        paths = _stack().write_values(tmp_path / "out")
        assert set(paths) == set(COMPONENTS)
        for component, path in paths.items():
            assert path.exists()
            assert yaml.safe_load(path.read_text())


# ---------------------------------------------------------------------------
# helm commands
# ---------------------------------------------------------------------------


class TestHelmCommands:
    def test_install_is_idempotent_and_waits(self, tmp_path):
        stack = _stack()
        stack.install("infra", tmp_path / "v.yaml", timeout=900)

        command = stack.shell.sh.call_args.args[0]
        assert command.startswith("helm upgrade --install madengine-dummy-llm-d-infra")
        assert "oci://example.test/infra --version 1.2.3" in command
        assert "--namespace llm-d-bench" in command
        assert "--wait --timeout 900s" in command

    def test_install_shell_timeout_exceeds_the_helm_timeout(self, tmp_path):
        """Let helm hit its own deadline and report; do not kill it first."""
        stack = _stack()
        stack.install("infra", tmp_path / "v.yaml", timeout=900)
        assert stack.shell.sh.call_args.kwargs["timeout"] > 900

    def test_install_returns_the_release_name(self, tmp_path):
        stack = _stack()
        assert stack.install("gaie", tmp_path / "v.yaml", 60) == (
            "madengine-dummy-llm-d-gaie"
        )

    def test_install_failure_raises_llmd_stack_error(self, tmp_path):
        stack = _stack()
        stack.shell.sh.side_effect = RuntimeError("exit 1")
        with pytest.raises(LlmdStackError, match="helm install of"):
            stack.install("infra", tmp_path / "v.yaml", 60)

    def test_unpinned_version_is_refused_at_the_command_layer(self, tmp_path):
        stack = _stack({"charts": {"infra": {"ref": "oci://example.test/infra"}}})
        with pytest.raises(LlmdStackError, match="not pinned"):
            stack.install("infra", tmp_path / "v.yaml", 60)
        stack.shell.sh.assert_not_called()

    def test_missing_ref_is_refused(self, tmp_path):
        stack = _stack({"charts": {"infra": {"version": "1.0.0"}}})
        with pytest.raises(LlmdStackError, match="ref is not set"):
            stack.install("infra", tmp_path / "v.yaml", 60)

    def test_template_contacts_no_cluster(self, tmp_path):
        stack = _stack()
        stack.shell.sh.return_value = "kind: Deployment"
        assert stack.template("modelservice", tmp_path / "v.yaml") == "kind: Deployment"
        command = stack.shell.sh.call_args.args[0]
        assert command.startswith("helm template ")
        assert "--wait" not in command

    def test_uninstall_tolerates_an_already_gone_release(self):
        stack = _stack()
        stack.uninstall("madengine-dummy-llm-d-infra")
        command = stack.shell.sh.call_args.args[0]
        assert command.startswith("helm uninstall madengine-dummy-llm-d-infra")
        assert "--ignore-not-found" in command
        assert "--namespace llm-d-bench" in command

    def test_uninstall_failure_raises_llmd_stack_error(self):
        stack = _stack()
        stack.shell.sh.side_effect = RuntimeError("exit 1")
        with pytest.raises(LlmdStackError, match="helm uninstall of"):
            stack.uninstall("madengine-dummy-llm-d-infra")


# ---------------------------------------------------------------------------
# validate() in managed mode
# ---------------------------------------------------------------------------


class TestManagedValidate:
    def _validate(self, deployment):
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
            return deployment.validate()

    def test_managed_mode_passes_with_a_complete_config(self, tmp_path):
        assert self._validate(_managed_deployment(tmp_path)) is True

    def test_managed_mode_requires_a_model_uri(self, tmp_path):
        deployment = _managed_deployment(tmp_path, {"model": {"name": "Qwen3-32B"}})
        assert self._validate(deployment) is False


# ---------------------------------------------------------------------------
# Standup, unwind, readiness
# ---------------------------------------------------------------------------


class TestStandup:
    def test_prepare_stands_the_stack_up_before_rendering_the_job(self, tmp_path):
        """The Job carries MAD_LLM_D_ENDPOINT, so standup must precede render."""
        deployment = _managed_deployment(tmp_path)
        order = []

        with (
            patch.object(deployment, "_standup", side_effect=lambda: order.append("s")),
            patch(
                "madengine.deployment.kubernetes.KubernetesDeployment.prepare",
                side_effect=lambda: order.append("p") or True,
            ),
        ):
            assert deployment.prepare() is True

        assert order == ["s", "p"]

    def test_attach_mode_prepare_skips_standup(self, tmp_path):
        deployment = _managed_deployment(
            tmp_path, {"endpoint_url": "http://gw.example"}
        )

        with (
            patch.object(deployment, "_standup") as standup,
            patch(
                "madengine.deployment.kubernetes.KubernetesDeployment.prepare",
                return_value=True,
            ),
        ):
            deployment.prepare()

        standup.assert_not_called()

    def test_standup_installs_all_three_in_order(self, tmp_path):
        deployment = _managed_deployment(tmp_path)

        with (
            patch.object(deployment, "_wait_for_model_servers"),
            patch.object(deployment, "_resolve_endpoint", return_value="http://gw:80"),
            patch.object(
                deployment.stack, "install", side_effect=lambda c, v, t: f"rel-{c}"
            ) as install,
        ):
            deployment._standup()

        assert [c.args[0] for c in install.call_args_list] == list(COMPONENTS)
        assert deployment._installed_releases == [
            deployment.stack.release_name(c) for c in COMPONENTS
        ]
        assert deployment.endpoint_url == "http://gw:80"

    def test_standup_records_a_release_before_helm_runs(self, tmp_path):
        """The release helm was mid-way through must still be torn down.

        helm can create a release and then fail — or be interrupted — so the
        unwind list is recorded up-front. Naming a release that was never
        created is free: _uninstall_release passes --ignore-not-found.
        """
        deployment = _managed_deployment(tmp_path)

        def install(component, values, timeout):
            if component == "gaie":
                raise LlmdStackError("chart not found")
            return f"rel-{component}"

        with (
            patch.object(deployment.stack, "install", side_effect=install),
            pytest.raises(LlmdStackError),
        ):
            deployment._standup()

        assert deployment._installed_releases == [
            deployment.stack.release_name("infra"),
            deployment.stack.release_name("gaie"),
        ]

    def test_a_failed_standup_unwinds_and_fails_prepare(self, tmp_path):
        deployment = _managed_deployment(tmp_path)
        deployment._installed_releases = []

        def standup():
            deployment._installed_releases.extend(["rel-infra", "rel-gaie"])
            raise LlmdStackError("modelservice exploded")

        with (
            patch.object(deployment, "_standup", side_effect=standup),
            patch.object(deployment, "_uninstall_release") as uninstall,
            patch(
                "madengine.deployment.kubernetes.KubernetesDeployment.prepare"
            ) as parent_prepare,
        ):
            assert deployment.prepare() is False

        # Unwound newest-first, and the client Job was never rendered.
        assert [c.args[0] for c in uninstall.call_args_list] == [
            "rel-gaie",
            "rel-infra",
        ]
        parent_prepare.assert_not_called()

    def test_teardown_false_leaves_a_failed_standup_in_place(self, tmp_path):
        """Debugging a broken standup requires the wreckage to still be there."""
        deployment = _managed_deployment(tmp_path, {"teardown": False})
        deployment._installed_releases = ["rel-infra"]

        with patch.object(deployment, "_uninstall_release") as uninstall:
            deployment._unwind()

        uninstall.assert_not_called()

    def test_standup_timeout_is_passed_to_helm(self, tmp_path):
        deployment = _managed_deployment(tmp_path, {"standup_timeout": 42})

        with (
            patch.object(deployment, "_wait_for_model_servers"),
            patch.object(deployment, "_resolve_endpoint", return_value="http://gw"),
            patch.object(deployment.stack, "install", return_value="r") as install,
        ):
            deployment._standup()

        assert all(c.args[2] == 42 for c in install.call_args_list)


class TestReadiness:
    def _api(self, deployment, pages):
        """Patch AppsV1Api so successive calls return successive pages."""
        api = MagicMock()
        api.list_namespaced_deployment.side_effect = [
            MagicMock(items=page) for page in pages
        ]
        client = MagicMock()
        client.AppsV1Api.return_value = api
        return patch.dict("sys.modules", {}), api, client

    def test_returns_when_every_replica_is_ready(self, tmp_path):
        deployment = _managed_deployment(tmp_path)
        _, api, client = self._api(deployment, [[_deployment_obj("decode", 1, 1)]])

        with patch("kubernetes.client", client):
            deployment._wait_for_model_servers()

        selector = api.list_namespaced_deployment.call_args.kwargs["label_selector"]
        assert selector == (
            "app.kubernetes.io/instance=madengine-dummy-llm-d-modelservice"
        )

    def test_polls_until_ready(self, tmp_path):
        deployment = _managed_deployment(tmp_path)
        _, api, client = self._api(
            deployment,
            [[_deployment_obj("decode", 2, 0)], [_deployment_obj("decode", 2, 2)]],
        )

        with (
            patch("kubernetes.client", client),
            patch("madengine.deployment.llm_d.time.sleep") as sleep,
        ):
            deployment._wait_for_model_servers()

        assert api.list_namespaced_deployment.call_count == 2
        sleep.assert_called_once()

    def test_no_matching_deployments_is_not_fatal(self, tmp_path):
        """helm --wait already gated readiness; do not block on a label guess."""
        deployment = _managed_deployment(tmp_path)
        _, api, client = self._api(deployment, [[]])

        with patch("kubernetes.client", client):
            deployment._wait_for_model_servers()  # must not raise

        assert api.list_namespaced_deployment.call_count == 1

    def test_timeout_names_the_pending_deployments(self, tmp_path):
        deployment = _managed_deployment(tmp_path, {"readiness_timeout": 0})
        _, api, client = self._api(deployment, [[_deployment_obj("decode", 2, 1)]])

        with patch("kubernetes.client", client):
            with pytest.raises(ConfigurationError, match="decode"):
                deployment._wait_for_model_servers()


class TestEndpointResolution:
    def _client(self, gateways):
        api = MagicMock()
        api.list_namespaced_custom_object.return_value = {"items": gateways}
        client = MagicMock()
        client.CustomObjectsApi.return_value = api
        return client, api

    def test_reads_the_address_off_the_owned_gateway(self, tmp_path):
        deployment = _managed_deployment(tmp_path)
        client, _ = self._client(
            [
                _gateway(name="other", instance="somebody-else"),
                _gateway(
                    name="ours",
                    instance="madengine-dummy-llm-d-infra",
                    addresses=[{"value": "10.0.0.5"}],
                    port=8080,
                ),
            ]
        )

        with patch("kubernetes.client", client):
            assert deployment._resolve_endpoint() == "http://10.0.0.5:8080"

    def test_falls_back_to_a_lone_unlabelled_gateway(self, tmp_path):
        """Some providers label the Gateway after the class, not the release."""
        deployment = _managed_deployment(tmp_path)
        client, _ = self._client([_gateway(addresses=[{"value": "gw.svc"}])])

        with patch("kubernetes.client", client):
            assert deployment._resolve_endpoint() == "http://gw.svc:80"

    def test_ambiguous_unlabelled_gateways_are_refused(self, tmp_path):
        """Guessing between two gateways would benchmark the wrong stack."""
        deployment = _managed_deployment(tmp_path)
        client, _ = self._client([_gateway(name="a"), _gateway(name="b")])

        with patch("kubernetes.client", client):
            with pytest.raises(ConfigurationError, match="endpoint_url"):
                deployment._resolve_endpoint()

    def test_a_gateway_with_no_address_reports_the_controller(self, tmp_path):
        deployment = _managed_deployment(tmp_path)
        client, _ = self._client([_gateway(addresses=[])])

        with patch("kubernetes.client", client):
            with pytest.raises(ConfigurationError, match="published an address"):
                deployment._resolve_endpoint()

    def test_a_listing_failure_points_at_the_manual_override(self, tmp_path):
        deployment = _managed_deployment(tmp_path)
        client = MagicMock()
        client.CustomObjectsApi.return_value.list_namespaced_custom_object.side_effect = RuntimeError(
            "forbidden"
        )

        with patch("kubernetes.client", client):
            with pytest.raises(ConfigurationError, match="llm_d.endpoint_url"):
                deployment._resolve_endpoint()


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


class TestManagedTeardown:
    def test_uninstall_release_shells_out_to_helm(self, tmp_path):
        deployment = _managed_deployment(tmp_path)
        deployment._uninstall_release("madengine-dummy-llm-d-infra")

        command = deployment.shell.sh.call_args.args[0]
        assert command.startswith("helm uninstall madengine-dummy-llm-d-infra")

    def test_teardown_after_a_successful_run_removes_every_release(self, tmp_path):
        """BaseDeployment.execute() never calls cleanup() on success."""
        deployment = _managed_deployment(tmp_path)
        deployment._installed_releases = ["a-infra", "a-gaie", "a-ms"]

        with patch(
            "madengine.deployment.base.BaseDeployment.execute",
            return_value=MagicMock(status=DeploymentStatus.SUCCESS),
        ):
            deployment.execute()

        commands = [c.args[0] for c in deployment.shell.sh.call_args_list]
        assert [c.split()[2] for c in commands] == ["a-ms", "a-gaie", "a-infra"]

    def test_a_helm_teardown_failure_does_not_mask_the_result(self, tmp_path):
        deployment = _managed_deployment(tmp_path)
        deployment._installed_releases = ["a-infra"]
        deployment.shell.sh.side_effect = RuntimeError("helm exploded")

        result = MagicMock(status=DeploymentStatus.SUCCESS)
        with patch(
            "madengine.deployment.base.BaseDeployment.execute", return_value=result
        ):
            assert deployment.execute() is result


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


class TestDryRun:
    @pytest.fixture(autouse=True)
    def _helm_on_path(self):
        """A dry run shells out to 'helm template'; pretend helm is installed."""
        with patch(
            "madengine.deployment.llm_d.shutil.which", return_value="/usr/bin/helm"
        ):
            yield

    def test_dry_run_renders_everything_and_deploys_nothing(self, tmp_path):
        deployment = _managed_deployment(tmp_path, {"dry_run": True})
        deployment.shell.sh.return_value = "kind: Deployment\n"

        with (
            patch.object(deployment, "validate") as validate,
            patch.object(deployment, "prepare") as prepare,
            patch.object(deployment, "deploy") as deploy,
        ):
            result = deployment.execute()

        assert result.status == DeploymentStatus.SUCCESS
        validate.assert_not_called()
        prepare.assert_not_called()
        deploy.assert_not_called()

        out = tmp_path / "out"
        for component in COMPONENTS:
            assert (out / f"llm-d-{component}-values.yaml").exists()
            assert (out / f"llm-d-{component}-manifests.yaml").exists()

    def test_dry_run_only_runs_helm_template(self, tmp_path):
        deployment = _managed_deployment(tmp_path, {"dry_run": True})
        deployment.execute()

        commands = [c.args[0] for c in deployment.shell.sh.call_args_list]
        assert commands and all(c.startswith("helm template ") for c in commands)

    def test_dry_run_installs_nothing_to_tear_down(self, tmp_path):
        deployment = _managed_deployment(tmp_path, {"dry_run": True})
        deployment.execute()
        assert deployment._installed_releases == []

    def test_dry_run_surfaces_an_unpinned_chart(self, tmp_path):
        deployment = _managed_deployment(
            tmp_path,
            {"dry_run": True, "charts": {"infra": {"ref": "oci://example.test/infra"}}},
        )
        result = deployment.execute()

        assert result.status == DeploymentStatus.FAILED
        assert "not pinned" in result.message

    def test_dry_run_without_helm_says_so(self, tmp_path):
        """execute() short-circuits before validate(), so its helm check is skipped."""
        deployment = _managed_deployment(tmp_path, {"dry_run": True})

        with patch("madengine.deployment.llm_d.shutil.which", return_value=None):
            result = deployment.execute()

        assert result.status == DeploymentStatus.FAILED
        assert "helm" in result.message
        deployment.shell.sh.assert_not_called()

    def test_dry_run_in_attach_mode_is_rejected(self, tmp_path):
        """Nothing to render: attach mode installs no charts."""
        deployment = _managed_deployment(
            tmp_path, {"dry_run": True, "endpoint_url": "http://gw.example"}
        )
        result = deployment.execute()

        assert result.status == DeploymentStatus.FAILED
        assert "attach mode" in result.message
