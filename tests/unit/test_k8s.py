"""
Kubernetes-related unit tests (secrets/config helpers, PVC → pod mapping).

Keep new K8s-focused unit tests here to avoid many small `test_k8s_*.py` files.
Integration/e2e tests stay in their own modules.
"""

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
