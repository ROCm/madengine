"""
Kubernetes-related unit tests (secrets/config helpers, name sanitization, PVC → pod).

Keep new K8s-focused unit tests here to avoid many small `test_k8s_*.py` files.
Integration/e2e tests stay in their own modules.
"""

import json
import re
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from madengine.core.timeout import DEFAULT_RUN_TIMEOUT
from madengine.deployment.base import DeploymentConfig, create_jinja_env
from madengine.deployment.k8s_names import (
    sanitize_k8s_container_name,
    sanitize_k8s_label_value,
    sanitize_k8s_object_name,
)
from madengine.deployment.k8s_secrets import (
    CONFIGMAP_MAX_BYTES,
    SECRETS_STRATEGY_EXISTING,
    SECRETS_STRATEGY_FROM_LOCAL,
    SECRETS_STRATEGY_OMIT,
    estimate_configmap_payload_bytes,
    merge_secrets_config,
    resolve_image_pull_secret_refs,
    resolve_runtime_secret_name,
    build_registry_secret_data,
)
from madengine.deployment.k8s_results import (
    collector_pod_name,
    decode_pod_log,
    extract_perf_csv_blocks,
    materialize_perf_csvs_from_logs,
    perf_csv_artifact_sources,
)
from madengine.deployment.kubernetes import (
    KubernetesDeployment,
    _pod_job_name_label_selector,
    assign_pvc_subdirs_to_pods,
    match_pvc_subdir_to_k8s_pod,
)
from madengine.core.errors import ConfigurationError
from madengine.deployment.base import DeploymentConfig


def test_merge_secrets_config_defaults():
    merged = merge_secrets_config({})
    assert merged["strategy"] == SECRETS_STRATEGY_FROM_LOCAL
    assert merged["image_pull_secret_names"] == ["dockerhub-rocm"]


def test_resolve_image_pull_from_local_with_preview():
    refs = resolve_image_pull_secret_refs(
        SECRETS_STRATEGY_FROM_LOCAL,
        {"image_pull_secret_names": ["extra"]},
        ["job-reg"],
    )
    assert refs == [{"name": "job-reg"}, {"name": "extra"}]


def test_resolve_image_pull_from_local_includes_dockerhub_rocm():
    refs = resolve_image_pull_secret_refs(
        SECRETS_STRATEGY_FROM_LOCAL,
        merge_secrets_config({}),
        ["job-reg"],
    )
    assert refs == [{"name": "job-reg"}, {"name": "dockerhub-rocm"}]


def test_results_layout_auto_uses_per_pod_when_rwx_class_is_missing():
    from madengine.deployment.k8s_pvc import resolve_results_layout

    assert resolve_results_layout("auto", 2, False) == "per_pod"
    assert resolve_results_layout("auto", 2, True) == "shared"
    assert resolve_results_layout("shared", 2, False) == "shared"
    assert resolve_results_layout("per_pod", 2, True) == "per_pod"
    assert resolve_results_layout("auto", 1, False) == "shared"


def test_shared_filesystem_provisioners_support_read_write_many():
    from madengine.deployment.k8s_pvc import provisioner_supports_rwx

    assert provisioner_supports_rwx("nfs.csi.k8s.io")
    assert provisioner_supports_rwx("cluster.local/nfs-subdir-external-provisioner")
    assert provisioner_supports_rwx("cephfs.csi.ceph.com")
    assert provisioner_supports_rwx("efs.csi.aws.com")
    assert provisioner_supports_rwx("file.csi.azure.com")
    assert provisioner_supports_rwx("filestore.csi.storage.gke.io")
    assert provisioner_supports_rwx("nfs-client")


def test_block_and_local_provisioners_are_not_read_write_many():
    from madengine.deployment.k8s_pvc import provisioner_supports_rwx

    assert not provisioner_supports_rwx("rancher.io/local-path")
    assert not provisioner_supports_rwx("kubernetes.io/no-provisioner")
    assert not provisioner_supports_rwx("rbd.csi.ceph.com")
    assert not provisioner_supports_rwx("")
    assert not provisioner_supports_rwx(None)


def test_data_layout_uses_a_local_disk_unless_read_write_many_is_usable():
    from madengine.deployment.k8s_pvc import resolve_data_layout

    assert resolve_data_layout("auto", False, None) == "per_pod"
    assert resolve_data_layout("auto", True, None) == "shared"
    assert resolve_data_layout("auto", False, True) == "shared"
    assert resolve_data_layout("auto", True, False) == "per_pod"
    assert resolve_data_layout("shared", False, False) == "shared"
    assert resolve_data_layout("per_pod", True, True) == "per_pod"


def _layout_mixin(provisioner, volumes=None, pvc=None, pvc_status=None):
    """PVC mixin with a fake API.

    ``pvc_status`` is the error status for a missing claim.
    """
    from types import SimpleNamespace

    from kubernetes.client.rest import ApiException

    from madengine.deployment.k8s_pvc import KubernetesPVCMixin

    mixin = KubernetesPVCMixin()
    mixin.namespace = "default"
    mixin.k8s_config = {
        "nfs_storage_class": "nfs-banff",
        "data_storage_class": "nfs-banff",
        "local_path_storage_class": "local-path",
        "results_layout": "auto",
        "data_layout": "auto",
    }
    mixin.storage_v1 = MagicMock()
    mixin.storage_v1.read_storage_class.return_value = SimpleNamespace(
        provisioner=provisioner
    )
    mixin.core_v1 = MagicMock()
    mixin.core_v1.list_persistent_volume.return_value = SimpleNamespace(
        items=volumes or []
    )
    if pvc_status is not None:
        mixin.core_v1.read_namespaced_persistent_volume_claim.side_effect = (
            ApiException(status=pvc_status)
        )
    else:
        mixin.core_v1.read_namespaced_persistent_volume_claim.return_value = pvc
    return mixin


def test_local_path_class_does_not_select_a_shared_results_volume():
    mixin = _layout_mixin("rancher.io/local-path", pvc_status=404)
    assert mixin._select_results_layout(2) == "per_pod"


def test_existing_read_write_many_volume_selects_shared_results():
    from types import SimpleNamespace

    volume = SimpleNamespace(spec=SimpleNamespace(access_modes=["ReadWriteMany"]))
    mixin = _layout_mixin("example.com/custom-fs", volumes=[volume], pvc_status=404)
    assert mixin._select_results_layout(2) == "shared"


def test_non_rwx_data_claim_is_not_selected_for_mounting():
    from types import SimpleNamespace

    claim = SimpleNamespace(spec=SimpleNamespace(access_modes=["ReadWriteOnce"]))
    mixin = _layout_mixin("nfs.csi.k8s.io", pvc=claim)
    assert mixin._select_data_layout() == "per_pod"


def test_parallel_job_is_not_finished_when_only_some_pods_succeed():
    from types import SimpleNamespace

    from madengine.deployment.kubernetes import (
        job_finished_failed,
        job_finished_successfully,
    )

    def job(succeeded, completions, conditions=None):
        return SimpleNamespace(
            spec=SimpleNamespace(completions=completions),
            status=SimpleNamespace(succeeded=succeeded, conditions=conditions or []),
        )

    partial = job(2, 4)
    assert job_finished_successfully(partial) is False
    assert job_finished_failed(partial) is False

    done = job(
        4,
        4,
        [SimpleNamespace(type="Complete", status="True")],
    )
    assert job_finished_successfully(done) is True

    one_pod_failed = job(1, 4, [SimpleNamespace(type="Failed", status="False")])
    assert job_finished_failed(one_pod_failed) is False
    given_up = job(1, 4, [SimpleNamespace(type="Failed", status="True")])
    assert job_finished_failed(given_up) is True


def test_single_node_results_pvc_uses_local_path():
    from madengine.deployment.config_loader import ConfigLoader
    from madengine.deployment.k8s_pvc import KubernetesPVCMixin

    cfg = ConfigLoader.load_k8s_config({"k8s": {"gpu_count": 1}})
    mixin = KubernetesPVCMixin()
    mixin.k8s_config = cfg["k8s"]
    assert mixin._k8s_results_storage_class(1) == "local-path"


def test_k8s_preset_defaults_to_dockerhub_rocm():
    from madengine.deployment.config_loader import ConfigLoader

    cfg = ConfigLoader.load_k8s_config({"k8s": {"gpu_count": 1}})
    assert cfg["k8s"]["secrets"]["image_pull_secret_names"] == ["dockerhub-rocm"]


def test_missing_registry_image_is_a_configuration_error(tmp_path):
    with pytest.raises(ConfigurationError, match="registry image"):
        _k8s_template_context(tmp_path=tmp_path, image_info={"push_failed": True})


def test_default_job_yaml_references_dockerhub_rocm(tmp_path):
    ctx = _k8s_template_context(tmp_path=tmp_path)
    assert "dockerhub-rocm" in [s["name"] for s in ctx["image_pull_secrets"]]

    template_dir = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "madengine"
        / "deployment"
        / "templates"
        / "kubernetes"
    )
    rendered = (
        create_jinja_env(template_dir).get_template("job.yaml.j2").render(**ctx)
    )
    job = list(yaml.safe_load_all(rendered))[0]
    pull_names = [
        s["name"]
        for s in job["spec"]["template"]["spec"]["imagePullSecrets"]
    ]
    assert pull_names[-1] == "dockerhub-rocm"


def test_per_pod_data_volume_is_a_local_claim(tmp_path):
    ctx = _k8s_template_context(tmp_path=tmp_path)
    ctx["data_per_pod"] = True
    ctx["data_pvc"] = None
    ctx["data_storage_class"] = "local-path"
    ctx["data_storage_size"] = "100Gi"
    template_dir = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "madengine"
        / "deployment"
        / "templates"
        / "kubernetes"
    )
    rendered = create_jinja_env(template_dir).get_template("job.yaml.j2").render(**ctx)
    job = list(yaml.safe_load_all(rendered))[0]
    volumes = job["spec"]["template"]["spec"]["volumes"]
    data = next(volume for volume in volumes if volume["name"] == "data")
    claim = data["ephemeral"]["volumeClaimTemplate"]["spec"]
    assert claim["accessModes"] == ["ReadWriteOnce"]
    assert claim["storageClassName"] == "local-path"
    assert "madengine-shared-data" not in rendered


def test_resolve_image_pull_existing():
    refs = resolve_image_pull_secret_refs(
        SECRETS_STRATEGY_EXISTING,
        {"image_pull_secret_names": ["precreated"]},
        [],
    )
    assert refs == [{"name": "precreated"}]


def test_resolve_image_pull_omit_extra_only():
    refs = resolve_image_pull_secret_refs(
        SECRETS_STRATEGY_OMIT,
        {"image_pull_secret_names": ["pull"]},
        [],
    )
    assert refs == [{"name": "pull"}]


def test_dockerhub_registry_payload():
    creds = {"dockerhub": {"username": "u", "password": "p"}}
    assert build_registry_secret_data(creds) is not None


def test_estimate_configmap_payload_bytes():
    ctx = {
        "manifest_content": "x" * 100,
        "include_credential_in_configmap": True,
        "credential_content": "{}",
        "model_scripts_contents": {},
        "common_script_contents": {},
    }
    assert estimate_configmap_payload_bytes(ctx) < CONFIGMAP_MAX_BYTES


def test_resolve_runtime_secret_name_from_local():
    assert (
        resolve_runtime_secret_name(
            SECRETS_STRATEGY_FROM_LOCAL,
            {},
            "job-runtime",
        )
        == "job-runtime"
    )


def test_resolve_runtime_secret_name_existing():
    assert (
        resolve_runtime_secret_name(
            SECRETS_STRATEGY_EXISTING,
            {"runtime_secret_name": "precreated"},
            None,
        )
        == "precreated"
    )


def test_resolve_runtime_secret_name_omit_optional():
    assert (
        resolve_runtime_secret_name(SECRETS_STRATEGY_OMIT, {}, None) is None
    )


def test_estimate_skips_credential_when_not_in_configmap():
    ctx = {
        "manifest_content": "a",
        "include_credential_in_configmap": False,
        "credential_content": "x" * 999999,
        "model_scripts_contents": {},
        "common_script_contents": {},
    }
    assert estimate_configmap_payload_bytes(ctx) < 100


# --- PVC /results subdir → pod name (kubernetes.collect_results) ------------


def test_pvc_match_exact():
    assigned: set = set()
    assert match_pvc_subdir_to_k8s_pod("my-pod", ["my-pod", "my-pod-0-abc"], assigned) == "my-pod"
    assigned.add("my-pod")
    assert match_pvc_subdir_to_k8s_pod("my-pod", ["my-pod", "my-pod-0-abc"], assigned) == "my-pod-0-abc"


def test_pvc_match_prefix_indexed_job():
    assigned: set = set()
    pods = ["madengine-dummy-torchrun-0-fz7th", "madengine-dummy-torchrun-1-88hw6"]
    assert (
        match_pvc_subdir_to_k8s_pod("madengine-dummy-torchrun-0", pods, assigned)
        == "madengine-dummy-torchrun-0-fz7th"
    )
    assigned.add("madengine-dummy-torchrun-0-fz7th")
    assert (
        match_pvc_subdir_to_k8s_pod("madengine-dummy-torchrun-1", pods, assigned)
        == "madengine-dummy-torchrun-1-88hw6"
    )


def test_pvc_assign_longest_subdir_first():
    pod_names = ["madengine-dummy-torchrun-0-fz7th", "madengine-dummy-torchrun-1-88hw6"]
    mapping = assign_pvc_subdirs_to_pods(
        ["madengine-dummy-torchrun-0", "madengine-dummy-torchrun-1"],
        pod_names,
    )
    assert mapping["madengine-dummy-torchrun-0"] == "madengine-dummy-torchrun-0-fz7th"
    assert mapping["madengine-dummy-torchrun-1"] == "madengine-dummy-torchrun-1-88hw6"


def test_pvc_assign_no_duplicate_pods():
    pods = ["a-x", "a-y"]
    m = assign_pvc_subdirs_to_pods(["a"], pods)
    assert len(m) == 1
    assert m["a"] in pods


def test_pvc_assign_empty_dirs():
    assert assign_pvc_subdirs_to_pods([], ["p"]) == {}
    assert assign_pvc_subdirs_to_pods(["  ", ""], ["p"]) == {}


# --- Object / label / container name sanitization (k8s_names) ----------------


@pytest.mark.unit
class TestSanitizeK8sObjectName:
    def test_slash_in_model_name(self):
        name = sanitize_k8s_object_name(
            "madengine", "primus_pretrain/torchtitan_MI300X_qwen3_1.7B-pretrain"
        )
        assert "/" not in name
        assert name.startswith("madengine-")
        assert name == "madengine-primus-pretrain-torchtitan-mi300x-qwen3-1.7b-pretrain"

    def test_uppercase_and_underscore(self):
        n = sanitize_k8s_object_name("madengine", "My_Model_NAME")
        assert n == "madengine-my-model-name"

    def test_max_length_stable_hash(self):
        long_name = "a" * 400
        n = sanitize_k8s_object_name("madengine", long_name)
        assert len(n) <= 253
        assert "/" not in n
        n2 = sanitize_k8s_object_name("madengine", long_name)
        assert n == n2

    def test_empty_body_uses_model(self):
        n = sanitize_k8s_object_name("madengine", "///")
        assert "madengine" in n
        assert "/" not in n


@pytest.mark.unit
def test_pod_job_name_label_selector_matches_sanitized_job_name():
    """Pods use job-name label value = sanitize_k8s_label_value(Job metadata name); list queries must match."""
    jid = sanitize_k8s_object_name("madengine", "z" * 400)
    sel = _pod_job_name_label_selector(jid)
    assert sel == f"job-name={sanitize_k8s_label_value(jid)}"
    assert len(sel.split("=", 1)[1]) <= 63


@pytest.mark.unit
class TestSanitizeK8sLabelValue:
    def test_slash_and_length(self):
        raw = "primus_pretrain/torchtitan_MI300X_qwen3_1.7B-pretrain"
        v = sanitize_k8s_label_value(raw)
        assert len(v) <= 63
        assert "/" not in v

    def test_long_value_truncated(self):
        raw = "x" * 200
        v = sanitize_k8s_label_value(raw)
        assert len(v) <= 63


@pytest.mark.unit
class TestSanitizeK8sContainerName:
    def test_dots_from_version_become_hyphens(self):
        job = "madengine-primus-pretrain-torchtitan-mi300x-qwen3-1.7b-pretrain"
        c = sanitize_k8s_container_name(job)
        assert "." not in c
        assert "1-7b" in c or "17" in c

    def test_max_63_chars(self):
        long_hint = "a" * 200
        c = sanitize_k8s_container_name(long_hint)
        assert len(c) <= 63


class TestGatherSystemEnvDetailsK8sRocenvMode:
    """K8s gather_system_env_details passes rocenv_mode to run_rocenv_tool.sh args."""

    def _make_mixin(self):
        from unittest.mock import MagicMock
        from madengine.deployment.k8s_scripts import KubernetesScriptsMixin

        mixin = KubernetesScriptsMixin()
        mixin.console = MagicMock()
        return mixin

    def test_default_mode_is_lite(self):
        mixin = self._make_mixin()
        pre_scripts = []
        mixin.gather_system_env_details(pre_scripts, "my_model")
        assert pre_scripts[0]["args"] == "my_model_env lite UBUNTU"

    def test_full_mode(self):
        mixin = self._make_mixin()
        pre_scripts = []
        mixin.gather_system_env_details(pre_scripts, "org/my_model", rocenv_mode="full")
        assert pre_scripts[0]["args"] == "org_my_model_env full UBUNTU"

    def test_explicit_lite_mode(self):
        mixin = self._make_mixin()
        pre_scripts = []
        mixin.gather_system_env_details(pre_scripts, "my_model", rocenv_mode="lite")
        assert pre_scripts[0]["args"] == "my_model_env lite UBUNTU"

    def test_guest_os_centos(self):
        mixin = self._make_mixin()
        pre_scripts = []
        mixin.gather_system_env_details(
            pre_scripts, "my_model", rocenv_mode="lite", guest_os="centos"
        )
        assert pre_scripts[0]["args"] == "my_model_env lite CENTOS"

    def test_invalid_mode_falls_back_to_lite(self):
        mixin = self._make_mixin()
        pre_scripts = []
        mixin.gather_system_env_details(pre_scripts, "my_model", rocenv_mode="bogus")
        assert pre_scripts[0]["args"] == "my_model_env lite UBUNTU"


# ---------------------------------------------------------------------------
# Run timeout on the K8s path


def _k8s_template_context(
    model_timeout=None,
    cli_timeout=-1,
    tmp_path=None,
    launcher_type=None,
    image_info=None,
):
    """Template context for a minimal single-node job, without touching a cluster.

    Builds the context off the same mixin the deployment uses, so the timeout
    the template sees is the one a real render would get. Pass ``launcher_type``
    (e.g. ``"torchrun"``) to exercise the launcher branch of the job template
    instead of the direct-script branch.
    """
    from madengine.deployment.k8s_pvc import KubernetesPVCMixin
    from madengine.deployment.k8s_scripts import KubernetesScriptsMixin
    from madengine.deployment.k8s_template_context import (
        KubernetesTemplateContextMixin,
    )
    from madengine.deployment.kubernetes_launcher_mixin import KubernetesLauncherMixin

    class _Harness(
        KubernetesTemplateContextMixin,
        KubernetesScriptsMixin,
        KubernetesLauncherMixin,
        KubernetesPVCMixin,
    ):
        pass

    model_info = {
        "name": "dummy",
        "scripts": "scripts/dummy/run.sh",
        "args": "",
        "n_gpus": "1",
    }
    if model_timeout is not None:
        model_info["timeout"] = model_timeout

    manifest = {
        "built_images": {"dummy": {"docker_image": "dummy:latest"}},
        "built_models": {"dummy": model_info},
        "context": {"gpu_vendor": "AMD", "guest_os": "UBUNTU"},
    }
    manifest_path = tmp_path / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    k8s_config = {"namespace": "ns"}
    additional_context = {"k8s": k8s_config}
    if launcher_type is not None:
        additional_context["launcher"] = {
            "type": launcher_type,
            "nnodes": 1,
            "nproc_per_node": 1,
        }
    harness = _Harness()
    harness.config = DeploymentConfig(
        target="k8s",
        manifest_file=str(manifest_path),
        additional_context=additional_context,
        cli_timeout=cli_timeout,
    )
    harness.k8s_config = k8s_config
    harness.console = MagicMock()
    harness.manifest = manifest
    harness.namespace = "ns"
    harness.job_name = "j"
    harness.job_label = "j"
    harness.main_container_name = "c"
    harness.configmap_name = "cm"
    harness.service_name = "s"
    harness.gpu_resource_name = "amd.com/gpu"
    harness.data = None

    return harness._prepare_template_context(
        model_info, image_info or {"registry_image": "dummy:latest"}
    )


def _render_k8s_job_script(context):
    """Render job.yaml.j2 and return the main container's shell script."""
    template_dir = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "madengine"
        / "deployment"
        / "templates"
        / "kubernetes"
    )
    rendered = (
        create_jinja_env(template_dir).get_template("job.yaml.j2").render(**context)
    )
    job = list(yaml.safe_load_all(rendered))[0]
    return job["spec"]["template"]["spec"]["containers"][0]["args"][0]


class TestK8sRunTimeoutResolution:
    """The model card's timeout must reach the K8s job, following v1 precedence.

    Unlike SLURM, no inner madengine re-resolves inside the pod, so the card has
    to be applied at render time -- config.timeout only bounds the submitting
    process's wait on the Job and never saw the card.
    """

    def test_default_when_neither_card_nor_cli_specifies(self, tmp_path):
        ctx = _k8s_template_context(tmp_path=tmp_path)
        assert ctx["timeout"] == DEFAULT_RUN_TIMEOUT

    def test_model_card_overrides_default(self, tmp_path):
        ctx = _k8s_template_context(model_timeout=360, tmp_path=tmp_path)
        assert ctx["timeout"] == 360

    def test_cli_overrides_model_card(self, tmp_path):
        ctx = _k8s_template_context(
            model_timeout=360, cli_timeout=120, tmp_path=tmp_path
        )
        assert ctx["timeout"] == 120

    @pytest.mark.parametrize("card_timeout", [0, -1])
    def test_model_card_can_ask_for_no_timeout(self, card_timeout, tmp_path):
        ctx = _k8s_template_context(model_timeout=card_timeout, tmp_path=tmp_path)
        assert ctx["timeout"] == card_timeout

    def test_cli_zero_disables_a_model_card_timeout(self, tmp_path):
        ctx = _k8s_template_context(model_timeout=360, cli_timeout=0, tmp_path=tmp_path)
        assert ctx["timeout"] == 0


class TestK8sJobScriptTimeout:
    """The rendered job script must actually enforce the resolved timeout."""

    def test_model_script_is_wrapped_in_timeout(self, tmp_path):
        ctx = _k8s_template_context(model_timeout=360, tmp_path=tmp_path)
        script = _render_k8s_job_script(ctx)
        assert "timeout 360 bash /tmp/run_model.sh" in script

    def test_non_positive_timeout_runs_unbounded(self, tmp_path):
        ctx = _k8s_template_context(model_timeout=0, tmp_path=tmp_path)
        script = _render_k8s_job_script(ctx)
        assert "timeout 0 " not in script
        assert "No timeout set" in script
        assert "bash /tmp/run_model.sh" in script

    def test_rendered_script_is_valid_bash(self, tmp_path):
        """The heredoc terminator must land in column 0 after YAML dedent."""
        for card_timeout in (360, 0):
            ctx = _k8s_template_context(model_timeout=card_timeout, tmp_path=tmp_path)
            script_path = tmp_path / f"job_{card_timeout}.sh"
            script_path.write_text(_render_k8s_job_script(ctx))
            result = subprocess.run(
                ["bash", "-n", str(script_path)], capture_output=True, text=True
            )
            assert result.returncode == 0, result.stderr


class TestK8sJobScriptTimeoutLauncherBranch:
    """Same guarantees as TestK8sJobScriptTimeout, but for the launcher branch.

    torchrun/deepspeed-style jobs render through the `launcher_command` branch of
    job.yaml.j2, not the direct-script branch -- the timeout wrapper and exit-code
    capture are templated separately in each, so each needs its own coverage.
    """

    def test_model_script_is_wrapped_in_timeout(self, tmp_path):
        ctx = _k8s_template_context(
            model_timeout=360, launcher_type="torchrun", tmp_path=tmp_path
        )
        assert ctx["launcher_command"] is not None
        script = _render_k8s_job_script(ctx)
        assert "timeout 360 bash /tmp/run_model.sh" in script

    def test_non_positive_timeout_runs_unbounded(self, tmp_path):
        ctx = _k8s_template_context(
            model_timeout=0, launcher_type="torchrun", tmp_path=tmp_path
        )
        script = _render_k8s_job_script(ctx)
        assert "timeout 0 " not in script
        assert "No timeout set" in script
        assert "bash /tmp/run_model.sh" in script

    def test_rendered_script_is_valid_bash(self, tmp_path):
        for card_timeout in (360, 0):
            ctx = _k8s_template_context(
                model_timeout=card_timeout, launcher_type="torchrun", tmp_path=tmp_path
            )
            script_path = tmp_path / f"job_launcher_{card_timeout}.sh"
            script_path.write_text(_render_k8s_job_script(ctx))
            result = subprocess.run(
                ["bash", "-n", str(script_path)], capture_output=True, text=True
            )
            assert result.returncode == 0, result.stderr

    @pytest.mark.parametrize("model_exit_code", [0, 1, 124])
    def test_execution_continues_past_the_model(self, model_exit_code, tmp_path):
        """Same set -e hazard as the direct-script branch: exit code must be
        captured, not acted on, so post-scripts and artifact copy still run."""
        ctx = _k8s_template_context(
            model_timeout=360, launcher_type="torchrun", tmp_path=tmp_path
        )
        script = _render_k8s_job_script(ctx)
        result = _run_model_invocation_block(
            script, model_exit_code, tmp_path, f"launcher_{model_exit_code}"
        )
        assert (
            f"REACHED_ARTIFACT_COPY exit={model_exit_code}" in result.stdout
        ), f"aborted early: rc={result.returncode} out={result.stdout!r} err={result.stderr!r}"


def _run_model_invocation_block(script, model_exit_code, tmp_path, name):
    """Execute just the model-invocation block of a rendered job script.

    Takes the lines from MODEL_START_TIME to MODEL_END_TIME verbatim, runs them
    under `set -e` against a stub model script exiting with `model_exit_code`,
    and reports whether execution reached the end of the block (i.e. whether the
    container would go on to run post-scripts and copy artifacts).
    """
    lines = script.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith("MODEL_START_TIME="))
    end = next(i for i, l in enumerate(lines) if l.strip().startswith("MODEL_END_TIME="))
    block = "\n".join(l.strip() for l in lines[start:end])

    stub = tmp_path / f"run_model_{name}.sh"
    stub.write_text(f"#!/bin/bash\nexit {model_exit_code}\n")
    harness = tmp_path / f"harness_{name}.sh"
    harness.write_text(
        "set -e\n"
        f"cp {stub} /tmp/run_model.sh\n"
        f"{block}\n"
        'echo "REACHED_ARTIFACT_COPY exit=$MODEL_EXIT_CODE"\n'
    )
    return subprocess.run(
        ["bash", str(harness)], capture_output=True, text=True, timeout=60
    )


class TestK8sJobScriptPublishesResultsOnFailure:
    """A failed or timed-out model must not abort the container early.

    The script runs under `set -e` and copies artifacts to the results PVC only
    after the model returns, so a bare invocation (or an early `exit`) would
    throw away perf.csv and the logs for exactly the runs worth diagnosing.
    Both branches capture the exit code and defer to the single exit at the end.
    """

    @pytest.mark.parametrize("model_timeout", [360, 0])
    @pytest.mark.parametrize("model_exit_code", [0, 1, 124])
    def test_execution_continues_past_the_model(
        self, model_timeout, model_exit_code, tmp_path
    ):
        ctx = _k8s_template_context(model_timeout=model_timeout, tmp_path=tmp_path)
        script = _render_k8s_job_script(ctx)
        result = _run_model_invocation_block(
            script, model_exit_code, tmp_path, f"{model_timeout}_{model_exit_code}"
        )
        assert (
            f"REACHED_ARTIFACT_COPY exit={model_exit_code}" in result.stdout
        ), f"aborted early: rc={result.returncode} out={result.stdout!r} err={result.stderr!r}"

    def test_timeout_exit_code_is_reported(self, tmp_path):
        """A real `timeout` kill (124) must be labelled, not just propagated."""
        ctx = _k8s_template_context(model_timeout=1, tmp_path=tmp_path)
        script = _render_k8s_job_script(ctx)
        lines = script.splitlines()
        start = next(
            i for i, l in enumerate(lines) if l.strip().startswith("MODEL_START_TIME=")
        )
        end = next(
            i for i, l in enumerate(lines) if l.strip().startswith("MODEL_END_TIME=")
        )
        block = "\n".join(l.strip() for l in lines[start:end])

        harness = tmp_path / "harness_real_timeout.sh"
        harness.write_text(
            "set -e\n"
            'printf "#!/bin/bash\\nsleep 30\\n" > /tmp/run_model.sh\n'
            f"{block}\n"
            'echo "REACHED_ARTIFACT_COPY exit=$MODEL_EXIT_CODE"\n'
        )
        result = subprocess.run(
            ["bash", str(harness)], capture_output=True, text=True, timeout=60
        )
        assert "model script timed out after 1s" in result.stdout
        assert "REACHED_ARTIFACT_COPY exit=124" in result.stdout


class TestK8sRequirePinnedImage:
    """The generated pod spec image field honours require_pinned_image."""

    DIGEST = "sha256:" + "df36ef7e" * 8

    def _template_context(self, tmp_path, monkeypatch, require_pinned, image_digest):
        """Build a real template context, the way prepare() does.

        _prepare_template_context reads the manifest and the model's scripts
        directory from the current working directory, so the test runs inside
        tmp_path with a minimal model tree.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / "scripts" / "dummy").mkdir(parents=True)
        (tmp_path / "scripts" / "dummy" / "run.sh").write_text("#!/bin/bash\necho hi\n")

        image_info = {"registry_image": "myorg/ci:m"}
        if image_digest:
            image_info["image_digest"] = image_digest
        model_info = {
            "name": "m",
            "tags": ["t"],
            "n_gpus": "1",
            "args": "",
            "scripts": "scripts/dummy/run.sh",
            "dockerfile": "docker/dummy",
        }
        manifest = {
            "built_images": {"img1": image_info},
            "built_models": {"img1": model_info},
            "context": {},
        }
        (tmp_path / "build_manifest.json").write_text(json.dumps(manifest))

        additional_context = {
            "k8s": {"namespace": "default"},
            "gpu_vendor": "AMD",
            "guest_os": "UBUNTU",
        }
        if require_pinned:
            additional_context["require_pinned_image"] = True

        cfg = DeploymentConfig(
            target="k8s",
            manifest_file="build_manifest.json",
            additional_context=additional_context,
        )
        deployment = KubernetesDeployment(cfg)
        return deployment._prepare_template_context(model_info, image_info)

    def test_default_uses_tag(self, tmp_path, monkeypatch):
        ctx = self._template_context(
            tmp_path, monkeypatch, require_pinned=False, image_digest=self.DIGEST
        )
        assert ctx["image"] == "myorg/ci:m"

    def test_enabled_uses_pinned_reference(self, tmp_path, monkeypatch):
        ctx = self._template_context(
            tmp_path, monkeypatch, require_pinned=True, image_digest=self.DIGEST
        )
        assert ctx["image"] == f"myorg/ci@{self.DIGEST}"

    def test_enabled_without_digest_raises(self, tmp_path, monkeypatch):
        with pytest.raises(ConfigurationError):
            self._template_context(
                tmp_path, monkeypatch, require_pinned=True, image_digest=None
            )


def test_shared_volume_csvs_are_used_instead_of_log_copies(tmp_path):
    """nfs-banff keeps one ReadWriteMany results volume. That copy wins."""
    pvc = tmp_path / "pvc"
    pvc.mkdir()
    (pvc / "perf_Qwen3-8B.csv").write_text(
        "model,performance,metric\nQwen3-8B,10,throughput_tot\n"
    )
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    (log_dir / "perf_Qwen3-8B.csv").write_text(
        "model,performance,metric\nQwen3-8B,999,throughput_tot\n"
    )
    chosen = perf_csv_artifact_sources(
        [
            {"type": "pvc_collection", "local_path": str(pvc)},
            {"type": "log_csv", "local_path": str(log_dir)},
        ]
    )
    assert [art["type"] for art in chosen] == ["pvc_collection"]

    local_only = perf_csv_artifact_sources(
        [{"type": "log_csv", "local_path": str(log_dir)}]
    )
    assert [art["type"] for art in local_only] == ["log_csv"]


def test_multiline_log_is_not_decoded_as_a_bytes_repr():
    text = "b'not a kubernetes bytes repr\nsecond line'"
    assert decode_pod_log(text) == text


def test_bytes_repr_pod_log_is_decoded_before_csv_recovery(tmp_path):
    """The k8s client returns str(log_bytes), which hides every newline."""
    csv_body = "model,performance,metric\nQwen3-8B,158.33,throughput_tot\n"
    text = (
        "serving done\n"
        "MADENGINE_PERF_CSV_BEGIN perf_Qwen3-8B.csv\n"
        f"{csv_body}"
        "MADENGINE_PERF_CSV_END perf_Qwen3-8B.csv\n"
    )
    wrapped = str(text.encode("utf-8"))
    assert "\n" not in wrapped
    decoded = decode_pod_log(wrapped)
    assert decoded == text
    assert decode_pod_log(text.encode("utf-8")) == text

    results = {"logs": [{"pod": "job-0", "log": decoded}], "artifacts": []}
    assert materialize_perf_csvs_from_logs(tmp_path, results) == 1
    saved = (tmp_path / "job-0" / "log_csv" / "perf_Qwen3-8B.csv").read_text()
    assert "158.33" in saved


def test_collector_pod_name_is_a_dns_label():
    """A sliced job name used to end with a hyphen and Kubernetes rejected it."""
    name = collector_pod_name("madengine-vllm-pyt-vllm-qwen3-8b")
    assert name == "collector-madengine-vllm-pyt-vllm-qwen3-8b"
    assert re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", name)


def test_perf_csvs_in_pod_logs_are_recovered_without_a_shared_volume(tmp_path):
    """local-path pods delete their disks on exit; the log is the durable copy."""
    csv_body = "model,performance,metric\nQwen3-8B,158.33,tokens_per_second\n"
    log = (
        "serving done\n"
        "MADENGINE_PERF_CSV_BEGIN perf_Qwen3-8B.csv\n"
        f"{csv_body}"
        "MADENGINE_PERF_CSV_END perf_Qwen3-8B.csv\n"
    )
    assert extract_perf_csv_blocks(log) == [("perf_Qwen3-8B.csv", csv_body.rstrip("\n"))]

    results = {
        "logs": [
            {"pod": "job-0-abc", "log": log},
            {"pod": "job-1-def", "log": log.replace("158.33", "160.00")},
        ],
        "artifacts": [],
    }
    written = materialize_perf_csvs_from_logs(tmp_path, results)
    assert written == 2
    assert (tmp_path / "job-0-abc" / "log_csv" / "perf_Qwen3-8B.csv").is_file()
    assert (tmp_path / "job-1-def" / "log_csv" / "perf_Qwen3-8B.csv").read_text().startswith(
        "model,performance,metric"
    )
    assert results["artifacts"][0]["type"] == "log_csv"

    # A shared-volume copy already on disk is left alone.
    again = materialize_perf_csvs_from_logs(tmp_path, results)
    assert again == 0
    assert len(results["artifacts"]) == 2
