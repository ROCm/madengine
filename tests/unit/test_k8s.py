"""
Kubernetes-related unit tests (secrets/config helpers, name sanitization, PVC → pod).

Keep new K8s-focused unit tests here to avoid many small `test_k8s_*.py` files.
Integration/e2e tests stay in their own modules.
"""

import json
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
from madengine.deployment.kubernetes import (
    _pod_job_name_label_selector,
    assign_pvc_subdirs_to_pods,
    match_pvc_subdir_to_k8s_pod,
)


def test_merge_secrets_config_defaults():
    merged = merge_secrets_config({})
    assert merged["strategy"] == SECRETS_STRATEGY_FROM_LOCAL
    assert merged["image_pull_secret_names"] == []


def test_resolve_image_pull_from_local_with_preview():
    refs = resolve_image_pull_secret_refs(
        SECRETS_STRATEGY_FROM_LOCAL,
        {"image_pull_secret_names": ["extra"]},
        ["job-reg"],
    )
    assert refs == [{"name": "job-reg"}, {"name": "extra"}]


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


def _k8s_template_context(model_timeout=None, cli_timeout=-1, tmp_path=None):
    """Template context for a minimal single-node job, without touching a cluster.

    Builds the context off the same mixin the deployment uses, so the timeout
    the template sees is the one a real render would get.
    """
    from madengine.deployment.k8s_scripts import KubernetesScriptsMixin
    from madengine.deployment.k8s_template_context import (
        KubernetesTemplateContextMixin,
    )

    class _Harness(KubernetesTemplateContextMixin, KubernetesScriptsMixin):
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
    harness = _Harness()
    harness.config = DeploymentConfig(
        target="k8s",
        manifest_file=str(manifest_path),
        additional_context={"k8s": k8s_config},
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
        model_info, {"registry_image": "dummy:latest"}
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
