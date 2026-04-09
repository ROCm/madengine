#!/usr/bin/env python3
"""
Kubernetes Secret helpers for madengine deployment: registry pull + runtime credentials.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from kubernetes import client
    from kubernetes.client.rest import ApiException
except ImportError:
    client = None  # type: ignore
    ApiException = Exception  # type: ignore

# Values for k8s.secrets.strategy (merged under config "k8s")
SECRETS_STRATEGY_FROM_LOCAL = "from_local_credentials"
SECRETS_STRATEGY_EXISTING = "existing"
SECRETS_STRATEGY_OMIT = "omit"


def default_secrets_config() -> Dict[str, Any]:
    return {
        "strategy": SECRETS_STRATEGY_FROM_LOCAL,
        "image_pull_secret_names": [],
        "runtime_secret_name": None,
    }


def merge_secrets_config(k8s_config: Dict[str, Any]) -> Dict[str, Any]:
    base = default_secrets_config()
    merged = {**base, **(k8s_config.get("secrets") or {})}
    return merged


def _dockerconfig_json_for_dockerhub(username: str, password: str) -> str:
    """Build .dockerconfigjson body for Docker Hub (index.docker.io)."""
    auth = base64.b64encode(f"{username}:{password}".encode()).decode()
    cfg = {
        "auths": {
            "https://index.docker.io/v1/": {
                "username": username,
                "password": password,
                "auth": auth,
            }
        }
    }
    return json.dumps(cfg)


def build_registry_secret_data(creds: Dict[str, Any]) -> Optional[bytes]:
    """Return dockerconfigjson bytes if credentials contain usable Docker Hub auth.

    Public API for callers that need to preview or validate registry auth (e.g. dry-run
    manifests) without creating Kubernetes Secrets.
    """
    dh = creds.get("dockerhub")
    if not isinstance(dh, dict):
        return None
    user = dh.get("username")
    pw = dh.get("password")
    if not user or not pw:
        return None
    return _dockerconfig_json_for_dockerhub(str(user), str(pw)).encode("utf-8")


def replace_namespaced_secret(
    core_v1: Any,
    namespace: str,
    name: str,
    body: Any,
) -> None:
    """Create or replace a Secret in the namespace."""
    try:
        core_v1.create_namespaced_secret(namespace=namespace, body=body)
    except ApiException as e:
        if e.status == 409:
            core_v1.replace_namespaced_secret(name=name, namespace=namespace, body=body)
        else:
            raise


def create_or_update_secrets_from_credentials(
    core_v1: Any,
    namespace: str,
    job_name: str,
    credential_path: Path,
) -> Tuple[List[str], Optional[str]]:
    """
    Create registry and runtime Secrets from a local credential.json file.

    Returns:
        (image_pull_secret_names, runtime_secret_name)
        image_pull_secret_names may be empty if no registry auth was present.
        runtime_secret_name is always set when the file was readable.
    """
    if client is None:
        raise ImportError("kubernetes package is required to create Secrets")
    raw = credential_path.read_text(encoding="utf-8")
    creds = json.loads(raw)

    pull_names: List[str] = []
    reg_data = build_registry_secret_data(creds)
    if reg_data:
        reg_name = f"{job_name}-registry-pull"
        reg_body = client.V1Secret(
            metadata=client.V1ObjectMeta(name=reg_name, namespace=namespace),
            type="kubernetes.io/dockerconfigjson",
            data={".dockerconfigjson": base64.b64encode(reg_data).decode("ascii")},
        )
        replace_namespaced_secret(core_v1, namespace, reg_name, reg_body)
        pull_names.append(reg_name)

    runtime_name = f"{job_name}-runtime"
    runtime_body = client.V1Secret(
        metadata=client.V1ObjectMeta(name=runtime_name, namespace=namespace),
        type="Opaque",
        string_data={"credential.json": raw},
    )
    replace_namespaced_secret(core_v1, namespace, runtime_name, runtime_body)

    return pull_names, runtime_name


def delete_job_secrets_if_exist(
    core_v1: Any,
    namespace: str,
    job_name: str,
) -> None:
    """Best-effort delete of Secrets created by create_or_update_secrets_from_credentials."""
    for name in (f"{job_name}-registry-pull", f"{job_name}-runtime"):
        try:
            core_v1.delete_namespaced_secret(name=name, namespace=namespace)
        except ApiException as e:
            if getattr(e, "status", None) == 404:
                continue
            raise


def resolve_image_pull_secret_refs(
    strategy: str,
    merged_secrets: Dict[str, Any],
    created_pull_names: List[str],
) -> List[Dict[str, str]]:
    """
    Build pod.spec.imagePullSecrets list: [{\"name\": \"...\"}, ...].
    """
    names: List[str] = []
    if strategy == SECRETS_STRATEGY_EXISTING:
        extra = merged_secrets.get("image_pull_secret_names") or []
        names.extend(str(n) for n in extra if n)
    elif strategy == SECRETS_STRATEGY_FROM_LOCAL:
        names.extend(created_pull_names)
        extra = merged_secrets.get("image_pull_secret_names") or []
        for n in extra:
            s = str(n)
            if s and s not in names:
                names.append(s)
    elif strategy == SECRETS_STRATEGY_OMIT:
        extra = merged_secrets.get("image_pull_secret_names") or []
        names.extend(str(n) for n in extra if n)
    return [{"name": n} for n in names]


def resolve_runtime_secret_name(
    strategy: str,
    merged_secrets: Dict[str, Any],
    created_runtime_name: Optional[str],
) -> Optional[str]:
    if strategy == SECRETS_STRATEGY_EXISTING:
        return merged_secrets.get("runtime_secret_name")
    if strategy == SECRETS_STRATEGY_FROM_LOCAL:
        return created_runtime_name
    if strategy == SECRETS_STRATEGY_OMIT:
        return merged_secrets.get("runtime_secret_name")
    return None


def estimate_configmap_payload_bytes(context: Dict[str, Any]) -> int:
    """Rough size of ConfigMap data payload for preflight (1 MiB limit)."""
    total = 0
    total += len((context.get("manifest_content") or "").encode("utf-8"))
    if context.get("include_credential_in_configmap"):
        total += len((context.get("credential_content") or "").encode("utf-8"))
    dj = context.get("data_json_content")
    if dj:
        total += len(dj.encode("utf-8"))
    for _k, v in (context.get("model_scripts_contents") or {}).items():
        total += len(v.encode("utf-8"))
    for _k, v in (context.get("common_script_contents") or {}).items():
        total += len(v.encode("utf-8"))
    dps = context.get("data_provider_script_content")
    if dps:
        total += len(dps.encode("utf-8"))
    return total


CONFIGMAP_MAX_BYTES = 1024 * 1024 - 25600  # leave headroom below 1 MiB etcd limit
