#!/usr/bin/env python3
"""
Infer Primus ``BACKEND`` from madengine model names.

Convention (see ``scripts/primus_pretrain/get_models_json.py`` in MAD-internal):
``primus_pretrain/<launcher>_<arch>_<config_stem>``, e.g.
``primus_pretrain/torchtitan_MI300X_qwen3_4B-pretrain`` → launcher ``torchtitan``.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Primus ``run_pretrain.sh`` / ``run.sh`` expect ``MaxText`` capitalization for JAX;
# other backends use lowercase tokens.
_BACKEND_BY_LAUNCHER_TOKEN = {
    "maxtext": "MaxText",
    "torchtitan": "torchtitan",
    "megatron": "megatron",
    "megatron_bridge": "megatron_bridge",
    "moe_package": "moe_package",
}


def merged_primus_config(
    manifest: Optional[Dict[str, Any]], additional_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge ``distributed.primus`` from build manifest ``deployment_config`` and
    runtime ``additional_context`` (runtime wins on key conflicts).
    """
    out: Dict[str, Any] = {}
    if manifest:
        dist = (manifest.get("deployment_config") or {}).get("distributed") or {}
        prim = dist.get("primus") or {}
        if isinstance(prim, dict):
            out.update(prim)
    ac_dist = additional_context.get("distributed") or {}
    ac_prim = ac_dist.get("primus") or {}
    if isinstance(ac_prim, dict):
        out.update(ac_prim)
    return out


def infer_primus_examples_overlay_subdirs(
    config_path: str,
    *,
    backend_hint: str = "",
    model_name: str = "",
) -> List[str]:
    """
    Which ``scripts/Primus/examples/<subdir>/`` trees to bundle for Kubernetes.

    Container images often install Primus from upstream GitHub without every experiment
    YAML that exists in a local ``scripts/Primus`` checkout. Madengine bundles the
    matching ``examples/<backend>/`` subtree from the project directory into the
    ConfigMap and overlays it onto ``PRIMUS_ROOT`` before running Primus.

    Resolution order: path heuristics → explicit ``distributed.primus.backend`` →
    ``primus_pretrain/...`` model name → ``torchtitan``.
    """
    cp = (config_path or "").replace("\\", "/").lower()
    if "/megatron_bridge/" in cp or cp.startswith("examples/megatron_bridge/"):
        return ["megatron_bridge"]
    if "/moe_package/" in cp or cp.startswith("examples/moe_package/"):
        return ["moe_package"]
    if "/maxtext/" in cp or cp.startswith("examples/maxtext/"):
        return ["maxtext"]
    if "/torchtitan/" in cp or cp.startswith("examples/torchtitan/"):
        return ["torchtitan"]
    if "/megatron/" in cp or cp.startswith("examples/megatron/"):
        return ["megatron"]

    bh = (backend_hint or "").strip().lower()
    if bh == "maxtext":
        return ["maxtext"]
    if bh in ("megatron_bridge", "moe_package", "torchtitan", "megatron"):
        return [bh]

    fb = infer_primus_backend_from_model_name(model_name) if model_name else None
    if fb == "MaxText":
        return ["maxtext"]
    if fb in ("megatron_bridge", "moe_package", "torchtitan", "megatron"):
        return [fb]

    return ["torchtitan"]


def infer_primus_backend_from_model_name(model_name: str) -> Optional[str]:
    """
    Return the ``BACKEND`` value to export when ``distributed.primus.backend`` is omitted.

    Only handles names with the ``primus_pretrain/`` prefix and a launcher token
    before the first ``_`` (e.g. ``torchtitan`` in ``primus_pretrain/torchtitan_MI300X_...``).

    Returns:
        The string to pass to ``export BACKEND=...``, or ``None`` if unknown / not applicable.
    """
    if not model_name or not model_name.strip():
        return None
    raw = model_name.strip()
    prefix = "primus_pretrain/"
    if not raw.startswith(prefix):
        return None
    suffix = raw[len(prefix) :].strip()
    if not suffix:
        return None
    launcher_token = suffix.split("_", 1)[0].lower()
    return _BACKEND_BY_LAUNCHER_TOKEN.get(launcher_token)
