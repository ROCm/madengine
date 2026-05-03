"""
Kubernetes-related unit tests (secrets/config helpers, name sanitization, PVC → pod).

Keep new K8s-focused unit tests here to avoid many small `test_k8s_*.py` files.
Integration/e2e tests stay in their own modules.
"""

import pytest

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
    build_registry_secret_data,
    estimate_configmap_payload_bytes,
    merge_secrets_config,
    resolve_image_pull_secret_refs,
    resolve_runtime_secret_name,
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
    assert resolve_runtime_secret_name(SECRETS_STRATEGY_OMIT, {}, None) is None


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
    assert (
        match_pvc_subdir_to_k8s_pod("my-pod", ["my-pod", "my-pod-0-abc"], assigned)
        == "my-pod"
    )
    assigned.add("my-pod")
    assert (
        match_pvc_subdir_to_k8s_pod("my-pod", ["my-pod", "my-pod-0-abc"], assigned)
        == "my-pod-0-abc"
    )


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
