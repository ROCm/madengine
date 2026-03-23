"""Unit tests for Kubernetes Secret helpers and ConfigMap size estimate."""

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
