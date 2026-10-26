#!/usr/bin/env python3
"""
Shared deployment utilities used by both SLURM and Kubernetes deployments.

Provides launcher validation, ROCm profiling checks, and multi-node
profiling configuration so logic is not duplicated across deployment modules.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import difflib
import functools
import os
import subprocess
from typing import Any, Dict, List, Optional

from madengine.core.errors import ConfigurationError, create_error_context

# Valid distributed launchers. Each has exactly one accepted spelling: a value
# that is not in this list is an error, not something to guess at.
VALID_LAUNCHERS = [
    "torchrun",
    "torchtitan",
    "deepspeed",
    "megatron-lm",
    "primus",
    "vllm",
    "sglang",
    "sglang-disagg",
    "vllm-disagg",
    "slurm_multi",
]

# Named aliases only. A blanket "-" → "_" rewrite would silently accept a
# hyphen variant of every launcher. slurm-multi is advertised in
# docs/launchers.md. sglang_disagg is the underscore spelling the config
# translator already accepts.
_LAUNCHER_ALIASES: Dict[str, str] = {
    "slurm-multi": "slurm_multi",
    "sglang_disagg": "sglang-disagg",
    "vllm_disagg": "vllm-disagg",
}

# Where each contract may run. ``local`` is this machine (Docker or bare metal).
# ``slurm_multi`` is also accepted on ``local`` here: a card that names it and
# has no scheduler block is promoted to a SLURM deployment at run time.
_SELF_MANAGED_TARGETS = frozenset({"slurm", "spur", "local", "docker"})
_LOCAL_TARGETS = frozenset({"local", "docker", "bare_metal"})
_CLUSTER_TARGETS = frozenset({"slurm", "spur", "k8s", "kubernetes", "llm-d"})
DISAGG_LAUNCHERS = frozenset({"sglang-disagg", "vllm-disagg"})

# Deployment-mode sentinels meaning "no distributed launcher". They are produced
# by launcher_for_reporting() for the perf.csv ``launcher`` column and are never
# read back as config, so validate_launcher rejects them: a user who writes
# ``launcher: docker`` means "no launcher" and should say so by omitting the key,
# not by naming a value that has no dispatch arm.
_LAUNCHER_SENTINELS = frozenset({"docker", "native"})


def validate_launcher(launcher: Optional[str], *, source: str) -> Optional[str]:
    """Validate a user-supplied launcher and return its canonical spelling.

    ``None`` and blank strings mean "no launcher configured" and return None.
    Every other value must be a recognized launcher: unknown values raise rather
    than falling back, because a silently-defaulted launcher runs the model as a
    plain single-process job and still reports SUCCESS. Other falsy values (0,
    False, []) are misconfiguration, not absence, and raise.

    Args:
        launcher: Raw launcher value from config, model card, or environment.
        source: Where the value came from, e.g. ``additional_context.distributed``.
            Included in the error so the user knows which file to edit.

    Returns:
        The canonical launcher name, or None when nothing was configured.

    Raises:
        ConfigurationError: If the value is not a recognized launcher.
    """
    if launcher is None:
        return None
    if not isinstance(launcher, str):
        raise ConfigurationError(
            f"Invalid launcher in {source}: expected a string, got "
            f"{type(launcher).__name__} ({launcher!r})",
            context=create_error_context(
                operation="validate_launcher",
                component="deployment.common",
                additional_info={"launcher": launcher, "source": source},
            ),
            suggestions=[f"Supported launchers: {', '.join(VALID_LAUNCHERS)}"],
        )

    normalized = launcher.strip().lower()
    if not normalized:
        return None
    normalized = _LAUNCHER_ALIASES.get(normalized, normalized)
    if normalized in VALID_LAUNCHERS:
        return normalized

    suggestions = []
    if normalized in _LAUNCHER_SENTINELS:
        # Emitted into perf.csv by launcher_for_reporting(); has no dispatch arm.
        suggestions.append(
            f"'{normalized}' is a reporting value for runs with no distributed "
            "launcher — omit the launcher key instead of setting it"
        )
    else:
        close = difflib.get_close_matches(normalized, VALID_LAUNCHERS, n=1, cutoff=0.6)
        if close:
            suggestions.append(f"Did you mean '{close[0]}'?")
    suggestions.append(f"Supported launchers: {', '.join(VALID_LAUNCHERS)}")
    raise ConfigurationError(
        f"Unknown launcher '{launcher}' in {source}",
        context=create_error_context(
            operation="validate_launcher",
            component="deployment.common",
            additional_info={"launcher": launcher, "source": source},
        ),
        suggestions=suggestions,
    )


def launcher_deployment_type(config: Optional[Dict[str, Any]]) -> str:
    """Return the deployment a launcher check should use.

    Matches the keys that change which launcher contract is legal. A missing
    scheduler block is ``local``. ``slurm_multi`` with no block is still
    promoted to SLURM later; this function does not do that promotion.
    """
    config = config or {}
    if "llm_d" in config:
        return "llm-d"
    if "k8s" in config or "kubernetes" in config:
        return "k8s"
    if "slurm" in config:
        slurm_config = config.get("slurm") or {}
        if not isinstance(slurm_config, dict):
            slurm_config = {}
        scheduler = str(slurm_config.get("scheduler", "")).lower()
        if scheduler == "spur" or str(config.get("deploy", "")).lower() == "spur":
            return "spur"
        return "slurm"
    platform = config.get("platform", "docker")
    if isinstance(platform, str) and platform.strip().lower() == "bare_metal":
        return "bare_metal"
    return "local"


def ensure_launcher_allowed(
    launcher: Optional[str], deployment_type: str, *, source: str
) -> None:
    """Reject a launcher that has no dispatch arm on this deployment.

    Three contracts share ``distributed.launcher``:

    * Templated names (``torchrun``, ``vllm``, ``vllm-disagg``, …) are rendered
      by the SLURM, Spur, and Kubernetes templates. Disaggregated names need a
      cluster. ``vllm-disagg`` is also the llm-d serving name.
    * ``slurm_multi`` runs the model's own script. Kubernetes and llm-d have
      no arm for it.
    * ``primus`` is a local single-node runner. Cluster templates must not
      claim to start it.
    """
    if not launcher:
        return

    def _reject(message: str, suggestions: List[str]) -> None:
        raise ConfigurationError(
            message,
            context=create_error_context(
                operation="ensure_launcher_allowed",
                component="deployment.common",
                additional_info={
                    "launcher": launcher,
                    "deployment_type": deployment_type,
                    "source": source,
                },
            ),
            suggestions=suggestions,
        )

    if launcher == "slurm_multi" and deployment_type not in _SELF_MANAGED_TARGETS:
        _reject(
            f"Launcher 'slurm_multi' in {source} only runs on SLURM or Spur",
            [
                "Drop the k8s/kubernetes/llm_d block so this job is submitted to SLURM",
                "Or pick a templated launcher such as torchrun, megatron-lm, "
                "vllm, or sglang",
            ],
        )
    if launcher == "primus" and deployment_type not in _LOCAL_TARGETS:
        _reject(
            f"Launcher 'primus' in {source} runs on a local single node",
            [
                "Remove the slurm, k8s, or llm_d block for a local Primus run",
                "Or pick a templated training launcher such as torchrun, "
                "megatron-lm, or torchtitan",
            ],
        )
    if launcher in DISAGG_LAUNCHERS and deployment_type not in _CLUSTER_TARGETS:
        _reject(
            f"Launcher '{launcher}' in {source} needs a SLURM, Spur, or "
            "Kubernetes deployment",
            [
                "Add a slurm or k8s block. Disaggregated serving needs at least "
                "2 nodes on SLURM or Spur, and 3 on Kubernetes",
                "A local Docker run is a single node and cannot host prefill "
                "and decode",
            ],
        )
    if launcher == "sglang-disagg" and deployment_type == "llm-d":
        _reject(
            f"Launcher 'sglang-disagg' in {source} is not an llm-d serving stack",
            [
                "llm-d disaggregated vLLM uses launcher 'vllm-disagg'",
                "Or drop the llm_d block and deploy SGLang on SLURM or Kubernetes",
            ],
        )


def resolve_disagg_topology(
    nnodes: int,
    prefill_nodes: Optional[int],
    decode_nodes: Optional[int],
    *,
    min_nodes: int,
    allow_colocated_proxy: bool,
    launcher_name: str,
) -> tuple:
    """Return ``(prefill_nodes, decode_nodes)`` for a disaggregated launcher.

    SLURM and Spur allow a co-located proxy (``prefill + decode == nnodes``)
    or a dedicated proxy node (``prefill + decode + 1 == nnodes``), with a
    minimum of 2 nodes. Kubernetes requires a dedicated proxy and at least
    3 nodes. The default split keeps one proxy aside and divides the workers
    about 40/60 between prefill and decode. Two nodes with a co-located proxy
    are 1 prefill and 1 decode. The general formula would give that case no
    decode nodes.
    """
    if not isinstance(nnodes, int) or nnodes < min_nodes:
        raise ValueError(
            f"{launcher_name} requires minimum {min_nodes} nodes, got {nnodes}"
        )
    if prefill_nodes is not None and decode_nodes is not None:
        if prefill_nodes < 1 or decode_nodes < 1:
            raise ValueError(
                f"{launcher_name} requires at least 1 prefill and 1 decode node, "
                f"got prefill={prefill_nodes}, decode={decode_nodes}"
            )
        worker_sum = prefill_nodes + decode_nodes
        if allow_colocated_proxy:
            if worker_sum != nnodes and worker_sum + 1 != nnodes:
                raise ValueError(
                    f"Custom split validation failed: prefill_nodes ({prefill_nodes}) + "
                    f"decode_nodes ({decode_nodes}) = {worker_sum} must equal "
                    f"nnodes ({nnodes}) for a co-located proxy, or nnodes-1 "
                    f"({nnodes - 1}) for a dedicated proxy node"
                )
        elif worker_sum + 1 != nnodes:
            raise ValueError(
                f"Custom split validation failed: "
                f"prefill_nodes ({prefill_nodes}) + decode_nodes ({decode_nodes}) + 1 proxy "
                f"must equal nnodes ({nnodes}), but got {worker_sum + 1}"
            )
        return prefill_nodes, decode_nodes
    if allow_colocated_proxy and nnodes == 2:
        return 1, 1
    prefill = max(1, (nnodes - 1) * 2 // 5)
    decode = nnodes - 1 - prefill
    if decode < 1:
        raise ValueError(
            f"{launcher_name} default split produced no decode nodes for nnodes={nnodes}"
        )
    return prefill, decode


_DISAGG_BLOCK_KEYS = {
    "vllm-disagg": "vllm_disagg",
    "sglang-disagg": "sglang_disagg",
}


def sglang_disagg_block(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the SGLang disagg split the templated generator should publish.

    ``distributed.sglang_disagg`` wins when it names a prefill or decode count.
    A top-level ``sglang_disagg`` object is the older place and is used only
    when the nested block does not name a split.
    """
    context = context or {}
    distributed = context.get("distributed") or {}
    nested: Dict[str, Any] = {}
    if isinstance(distributed, dict):
        raw = distributed.get("sglang_disagg")
        if isinstance(raw, dict) and _block_names_a_split(raw):
            nested = raw
    if nested:
        return nested
    top = context.get("sglang_disagg")
    if isinstance(top, dict):
        return top
    if isinstance(distributed, dict) and isinstance(
        distributed.get("sglang_disagg"), dict
    ):
        return distributed["sglang_disagg"]
    return {}


def _block_names_a_split(block: Dict[str, Any]) -> bool:
    return (
        block.get("prefill_nodes") is not None or block.get("decode_nodes") is not None
    )


def _canonical_launcher_name(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    return canonicalize_distributed_launcher(value.strip()) or value.strip()


def _context_launcher_names(context: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    distributed = context.get("distributed")
    launcher_obj = context.get("launcher")
    if isinstance(distributed, dict):
        name = _canonical_launcher_name(distributed.get("launcher"))
        if name:
            names.append(name)
    if isinstance(launcher_obj, dict):
        name = _canonical_launcher_name(launcher_obj.get("type"))
        if name:
            names.append(name)
    return names


def _shared_fields_differ(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return any(
            _shared_fields_differ(left[key], right[key])
            for key in left.keys() & right.keys()
        )
    return left != right


def _reject_context(message: str) -> None:
    raise ConfigurationError(
        message,
        context=create_error_context(
            operation="check_context_block",
            component="deployment.common",
        ),
        suggestions=[message],
    )


def check_context_block(context: Optional[Dict[str, Any]]) -> List[str]:
    """Reject two disagreeing copies of one fact. Return warnings.

    ``slurm_multi`` skips the launcher-copy checks. A file that sets only one
    copy of ``port``, ``launcher.type``, or a top-level ``sglang_disagg``
    block still passes.
    """
    context = context or {}
    warnings: List[str] = []
    k8s = context.get("k8s")
    kubernetes = context.get("kubernetes")
    if (
        isinstance(k8s, dict)
        and isinstance(kubernetes, dict)
        and _shared_fields_differ(k8s, kubernetes)
    ):
        _reject_context("k8s and kubernetes disagree")

    distributed = context.get("distributed")
    if not isinstance(distributed, dict):
        distributed = {}
    launcher_obj = context.get("launcher")
    if not isinstance(launcher_obj, dict):
        launcher_obj = {}

    names = _context_launcher_names(context)
    if "llm_d" in context and names:
        _reject_context(
            "llm_d does not take distributed.launcher; scale lives in "
            "llm_d.prefill and llm_d.decode"
        )

    self_managed = "slurm_multi" in names
    if not self_managed:
        if len(set(names)) > 1:
            _reject_context(
                "launcher.type and distributed.launcher disagree: " + ", ".join(names)
            )
        node_counts = [
            value
            for value in (distributed.get("nnodes"), launcher_obj.get("nnodes"))
            if value is not None
        ]
        if len(set(node_counts)) > 1:
            _reject_context(
                "launcher.nnodes and distributed.nnodes disagree: "
                + ", ".join(str(value) for value in node_counts)
            )
        ports = [
            value
            for value in (
                distributed.get("port"),
                distributed.get("master_port"),
                launcher_obj.get("master_port"),
            )
            if value is not None
        ]
        if len(set(ports)) > 1:
            _reject_context(
                "port and master_port disagree: "
                + ", ".join(str(value) for value in ports)
            )
        nested = distributed.get("sglang_disagg")
        top = context.get("sglang_disagg")
        if (
            isinstance(nested, dict)
            and isinstance(top, dict)
            and _block_names_a_split(nested)
            and _block_names_a_split(top)
            and (
                nested.get("prefill_nodes") != top.get("prefill_nodes")
                or nested.get("decode_nodes") != top.get("decode_nodes")
            )
        ):
            _reject_context("sglang_disagg and distributed.sglang_disagg disagree")
        launcher = names[0] if names else None
        if launcher in DISAGG_LAUNCHERS:
            deployment = launcher_deployment_type(context)
            if deployment in ("slurm", "spur", "k8s", "kubernetes"):
                _check_disagg_node_count(
                    context, distributed, launcher_obj, launcher, deployment
                )

    warnings.extend(_context_scale_warnings(context, distributed))
    return warnings


def _check_disagg_node_count(
    context: Dict[str, Any],
    distributed: Dict[str, Any],
    launcher_obj: Dict[str, Any],
    launcher: str,
    deployment: str,
) -> None:
    nnodes = distributed.get("nnodes", launcher_obj.get("nnodes", 1))
    if not isinstance(nnodes, int):
        return
    key = _DISAGG_BLOCK_KEYS[launcher]
    block = distributed.get(key)
    if launcher == "sglang-disagg":
        block = sglang_disagg_block(context)
    if not isinstance(block, dict):
        block = {}
    minimum = 3 if deployment in ("k8s", "kubernetes") else 2
    try:
        resolve_disagg_topology(
            nnodes,
            block.get("prefill_nodes"),
            block.get("decode_nodes"),
            min_nodes=minimum,
            allow_colocated_proxy=deployment in ("slurm", "spur"),
            launcher_name=launcher,
        )
    except ValueError as exc:
        _reject_context(str(exc))


def _context_scale_warnings(
    context: Dict[str, Any], distributed: Dict[str, Any]
) -> List[str]:
    warnings: List[str] = []
    nproc = distributed.get("nproc_per_node")
    slurm = context.get("slurm")
    if isinstance(slurm, dict):
        gpus = slurm.get("gpus_per_node")
        if nproc is not None and gpus is not None and nproc != gpus:
            warnings.append(
                f"nproc_per_node={nproc} differs from slurm.gpus_per_node={gpus}"
            )
        nodes = slurm.get("nodes")
        nnodes = distributed.get("nnodes")
        if nodes is not None and nnodes is not None and nodes != nnodes:
            warnings.append(
                f"slurm.nodes={nodes} conflicts with distributed.nnodes={nnodes}"
            )
    k8s = context.get("k8s")
    if not isinstance(k8s, dict):
        k8s = context.get("kubernetes")
    if isinstance(k8s, dict):
        gpu_count = k8s.get("gpu_count")
        if nproc is not None and gpu_count is not None and nproc != gpu_count:
            warnings.append(
                f"nproc_per_node={nproc} differs from k8s.gpu_count={gpu_count}"
            )
    return warnings


def canonicalize_distributed_launcher(launcher: Optional[str]) -> Optional[str]:
    """Normalize alternate launcher spellings to their canonical form.

    Resolves aliases (e.g. ``sglang_disagg`` → ``sglang-disagg``). Unknown or
    empty values are returned unchanged; callers do their own validation.
    """
    if not launcher:
        return launcher
    return _LAUNCHER_ALIASES.get(launcher, launcher)


def launcher_for_reporting(launcher_type: Optional[str], deployment_type: str) -> str:
    """Return the launcher to record in perf results. Never raises.

    Values reaching here have already passed validate_launcher at the config
    boundary, so this only supplies the sentinel for the "no launcher
    configured" case: ``native`` on Kubernetes (the pod is the container),
    ``docker`` elsewhere. Reporting must not raise — BaseDeployment.execute()
    drops metrics on exception, which would lose results from a successful run.
    """
    if launcher_type:
        return launcher_type
    return "native" if deployment_type == "kubernetes" else "docker"


# Tool names that use rocprof / rocprofv3 wrapping and need MPI-aware rocprofv3 on multi-node.
_ROCPROF_FAMILY_TOOL_NAMES = frozenset(
    {
        "rocprof",
        "rocprof_hip_only",
        "rocprof_sys",
        "rocprofv3",
        "rocprofv3_compute",
        "rocprofv3_memory",
        "rocprofv3_communication",
        "rocprofv3_full",
        "rocprofv3_lightweight",
        "rocprofv3_agent",
        "rocprofv3_agent_counter",
        "rocprofv3_perfetto",
        "rocprofv3_api_overhead",
        "rocprofv3_pc_sampling",
    }
)


def tools_include_rocprof_family(tools_config: List[Dict]) -> bool:
    """Return True if any tool in the list is a rocprof/rocprofv3 family entry."""
    for tool in tools_config:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if name in _ROCPROF_FAMILY_TOOL_NAMES:
            return True
    return False


# Launchers that run one independent replica per node, so a node-local metric is
# a per-replica figure and job throughput is that figure scaled by nnodes.
#
# Deliberately NOT the same set as the Ray-based launchers guarded in the job
# templates: sglang-disagg is Ray-based but its nodes are heterogeneous roles
# (proxy + prefill + decode) cooperating on a single endpoint, not replicas.
# Its throughput is reported once, by the proxy — scaling that by nnodes would
# multiply a whole-cluster number.
PER_REPLICA_LAUNCHERS: frozenset = frozenset({"vllm", "sglang"})


def is_per_replica_launcher(launcher_type: Optional[str]) -> bool:
    """Return True if each node reports its own replica's metric, not the job's."""
    if not launcher_type or not isinstance(launcher_type, str):
        return False
    return launcher_type.strip().lower() in PER_REPLICA_LAUNCHERS


def _first_set(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _process_count_from_model(n_gpus: Any) -> int:
    """Process count implied by a model card's ``n_gpus``.

    ``-1`` on a model card means every GPU on the node, not a process count.
    Using it as ``nproc_per_node`` produced a negative value and skipped the
    launcher that would have rejected it.
    """
    try:
        count = int(n_gpus)
    except (TypeError, ValueError):
        return 1
    return count if count >= 1 else 1


def resolve_distributed_launch(
    additional_context: Optional[Dict[str, Any]],
    manifest: Optional[Dict[str, Any]] = None,
    model_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve the launcher a Kubernetes job should run.

    ``madengine run --additional-context`` puts the launcher in
    ``distributed`` (``{"launcher": "vllm", "nnodes": 4}``). An older shape
    uses ``launcher`` (``{"type": "vllm"}``). A manifest saved at build time
    stores the same fields under ``deployment_config.distributed``.

    Runtime context wins over the manifest, field by field, so a run can
    change the launcher without rebuilding. Within the runtime context,
    ``launcher`` wins over ``distributed``.
    """
    additional_context = additional_context or {}
    manifest = manifest or {}
    model_info = model_info or {}

    runtime = additional_context.get("distributed") or {}
    if not isinstance(runtime, dict):
        runtime = {}
    launcher_obj = additional_context.get("launcher") or {}
    if not isinstance(launcher_obj, dict):
        launcher_obj = {}
    manifest_dist = (manifest.get("deployment_config") or {}).get("distributed") or {}
    if not isinstance(manifest_dist, dict):
        manifest_dist = {}

    nproc = _first_set(
        launcher_obj.get("nproc_per_node"),
        runtime.get("nproc_per_node"),
        manifest_dist.get("nproc_per_node"),
    )
    if nproc is None:
        nproc = _process_count_from_model(model_info.get("n_gpus", 1))

    return {
        "launcher": _first_set(
            launcher_obj.get("type"),
            runtime.get("launcher"),
            manifest_dist.get("launcher"),
        ),
        "nnodes": _first_set(
            launcher_obj.get("nnodes"),
            runtime.get("nnodes"),
            manifest_dist.get("nnodes"),
            1,
        ),
        "nproc_per_node": nproc,
        "master_port": _first_set(
            launcher_obj.get("master_port"),
            runtime.get("master_port"),
            runtime.get("port"),
            manifest_dist.get("master_port"),
            manifest_dist.get("port"),
            29500,
        ),
    }


_SELF_MANAGED_LAUNCHERS: frozenset = frozenset({"slurm_multi"})


def is_self_managed_launcher(launcher_type: Optional[str]) -> bool:
    """Return True if the launcher manages its own per-node containers.

    Self-managed launchers (e.g. slurm_multi) run the model's own .slurm script
    directly on the head node and orchestrate Docker containers via srun internally.
    They bypass the standard sbatch template entirely and are an escape hatch — not
    peers of the templated launchers (torchrun, vllm, sglang, etc.).

    Callers pass raw model-card values here during early "should we take the
    self-managed path?" peeks, before validation has run, so this normalizes
    inline and never raises.
    """
    if not launcher_type or not isinstance(launcher_type, str):
        return False
    normalized = launcher_type.strip().lower()
    normalized = _LAUNCHER_ALIASES.get(normalized, normalized)
    return normalized in _SELF_MANAGED_LAUNCHERS


# Reports madengine appends to on every run.
_CUMULATIVE_REPORTS: frozenset = frozenset(
    {"perf.csv", "perf_super.csv", "perf_super.json"}
)


def is_cumulative_report(results_file: str, perf_csv: str = "perf.csv") -> bool:
    """Return True if *results_file* names a report kept across runs.

    A model card may declare one of madengine's own reports as its
    ``multiple_results``; deleting it as a stale leftover would wipe earlier
    runs' rows. Compared by basename, so a same-named file in a subdirectory
    is spared too -- a missed cleanup is cheaper than lost history.

    Args:
        results_file: The model card's ``multiple_results``.
        perf_csv: The run's ``--output`` CSV.
    """
    name = os.path.basename(os.path.normpath(results_file))
    return name in _CUMULATIVE_REPORTS or name == os.path.basename(perf_csv)


def resolve_launcher_from_sources(
    deployment_launcher: Optional[str],
    model_launcher: Optional[str],
    default: str = "torchrun",
) -> str:
    """Resolve the effective launcher from the deployment config and the model card.

    Precedence is deployment config first, model card second. That matches how
    BuildOrchestrator builds the manifest: it copies the model card's launcher into
    ``deployment_config.distributed`` only when the key is absent, so a value present
    there is already the user's explicit choice.

    Every dispatch site must use this helper. Resolving the launcher differently in
    two places is what allowed ``prepare()`` to pick the self-managed path from the
    model card while ``_prepare_template_context()`` simultaneously emitted a
    ``torchrun`` env block from the deployment config.
    """
    return deployment_launcher or model_launcher or default


def resolve_node_count(
    configured_nodes: int,
    nnodes: Optional[Any],
    nodes_explicitly_set: bool,
) -> tuple:
    """Reconcile ``slurm.nodes`` with ``distributed.nnodes`` into one node count.

    ``slurm.nodes`` sizes the allocation (``#SBATCH --nodes``) while
    ``distributed.nnodes`` sizes the topology the launcher builds on top of it.
    Nothing used to reconcile them, so a model card declaring only ``nnodes`` was
    submitted against the ``slurm.nodes`` default of 1 — a multi-node topology
    squeezed onto a single node.

    Args:
        configured_nodes: ``slurm.nodes`` after ConfigLoader applied its defaults.
        nnodes: ``distributed.nnodes``, if declared.
        nodes_explicitly_set: True when ``slurm.nodes`` came from the user or the
            model card rather than from a preset default.

    Returns:
        ``(nodes, note)`` where ``note`` is a human-readable string to log, or None.
    """
    try:
        if isinstance(nnodes, bool) or (
            isinstance(nnodes, float) and not nnodes.is_integer()
        ):
            raise ValueError
        requested = int(nnodes) if nnodes is not None else None
    except (TypeError, ValueError, OverflowError):
        return configured_nodes, (
            f"Ignoring non-numeric distributed.nnodes={nnodes!r}; "
            f"using slurm.nodes={configured_nodes}."
        )

    if requested is not None and requested < 1:
        # An allocation of zero or fewer nodes is not a thing sbatch can schedule;
        # passing it through would fail at submit time with a scheduler error that
        # says nothing about the model card that caused it.
        return configured_nodes, (
            f"Ignoring distributed.nnodes={requested} (must be >= 1); "
            f"using slurm.nodes={configured_nodes}."
        )

    if requested is None or requested == configured_nodes:
        return configured_nodes, None

    if nodes_explicitly_set:
        # Both were set and they disagree. The explicit allocation size wins —
        # overriding it could request nodes the user did not ask to be billed for —
        # but the mismatch is almost always a config error, so say so.
        return configured_nodes, (
            f"slurm.nodes={configured_nodes} conflicts with "
            f"distributed.nnodes={requested}; using slurm.nodes={configured_nodes}. "
            "Set them to the same value to silence this."
        )

    return requested, (
        f"Sizing allocation from distributed.nnodes={requested} "
        f"(slurm.nodes was not set explicitly)."
    )


@functools.lru_cache(maxsize=None)
def is_rocprofv3_available() -> bool:
    """
    Check if rocprofv3 is available on the system.

    rocprofv3 is required for multi-node profiling with MPI support.
    It's part of rocprofiler-sdk package in ROCm >= 6.4.1.

    Returns:
        True if rocprofv3 is available and executable, False otherwise
    """
    try:
        result = subprocess.run(
            ["rocprofv3", "--help"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def configure_multi_node_profiling(
    nnodes: int, tools_config: List[Dict], logger: Any
) -> Dict[str, Any]:
    """
    Configure profiling for multi-node runs with rocprofv3 support.

    Industry best practice for multi-node profiling:
    - Profile ALL nodes to detect stragglers, load imbalances, and communication bottlenecks
    - Use rocprofv3 (MPI-aware) for distributed profiling
    - Collect per-node outputs for detailed analysis

    Logic:
    1. Single node (nnodes == 1): Use existing tool behavior
    2. Multi-node (nnodes > 1):
       a. If rocprofv3 is available: enable per-node profiling; upgrade "rocprof" to "rocprofv3"
       b. If rocprofv3 is unavailable and the list includes only non-rocprof tools
          (e.g. rocm_trace_lite, library traces): keep those tools with per-node collection
       c. If rocprofv3 is unavailable and the list includes rocprof/rocprofv3-family tools:
          drop only those entries; if nothing remains, disable profiling and log the
          same guidance as before (install rocprofiler-sdk)

    Args:
        nnodes: Number of nodes in the deployment
        tools_config: List of tool configurations from user
        logger: Logger instance for messages

    Returns:
        Dictionary with profiling configuration:
        - enabled: bool
        - mode: str - "single_node", "multi_node", or "multi_node_unsupported"
        - tools: List[Dict]
        - per_node_collection: bool
    """
    if nnodes == 1:
        return {
            "enabled": True,
            "mode": "single_node",
            "tools": tools_config,
            "per_node_collection": False,
        }

    if not is_rocprofv3_available():
        if tools_include_rocprof_family(tools_config):
            filtered_tools: List[Dict] = [
                t
                for t in tools_config
                if isinstance(t, dict)
                and t.get("name") not in _ROCPROF_FAMILY_TOOL_NAMES
            ]
            if filtered_tools:
                logger.warning(
                    "Multi-node: rocprofv3 not found on submission host; "
                    "omitting rocprof/rocprofv3-family tools. Remaining tools: "
                    + ", ".join(
                        str(x.get("name"))
                        for x in filtered_tools
                        if isinstance(x, dict) and x.get("name")
                    )
                )
                return {
                    "enabled": True,
                    "mode": "multi_node",
                    "tools": filtered_tools,
                    "per_node_collection": True,
                }
            logger.warning(
                "╔════════════════════════════════════════════════════════════════════════════╗\n"
                "║ Multi-Node Profiling Requirements Not Met                                 ║\n"
                "╠════════════════════════════════════════════════════════════════════════════╣\n"
                "║ Multi-node profiling requires rocprofv3 (MPI-aware profiling support).    ║\n"
                "║                                                                            ║\n"
                "║ Current Status: rocprofv3 NOT FOUND on system                             ║\n"
                "║                                                                            ║\n"
                "║ rocprof/rocprofv3-family profiling will be SKIPPED for this multi-node    ║\n"
                "║ run.                                                                      ║\n"
                "║                                                                            ║\n"
                "║ To enable multi-node rocprof profiling:                                   ║\n"
                "║   • Install rocprofiler-sdk package (ROCm >= 6.4.1)                       ║\n"
                "║   • Command: apt install rocprofiler-sdk                                  ║\n"
                "║   • Or upgrade to ROCm 6.4.1 or later                                     ║\n"
                "║                                                                            ║\n"
                "║ Note: Single-node profiling uses rocprof (no rocprofv3 required)          ║\n"
                "╚════════════════════════════════════════════════════════════════════════════╝"
            )
            return {
                "enabled": False,
                "mode": "multi_node_unsupported",
                "tools": [],
                "per_node_collection": False,
            }
        logger.info(
            "Multi-node: rocprofv3 not found on submission host; keeping non-rocprof tools "
            "(rocprof/rocprofv3 family requires rocprofv3 for MPI-aware capture)."
        )
        return {
            "enabled": True,
            "mode": "multi_node",
            "tools": tools_config,
            "per_node_collection": True,
        }

    logger.info(
        f"✓ Multi-node profiling enabled for {nnodes} nodes (rocprofv3 detected)"
    )

    upgraded_tools: List[Dict] = []
    for tool in tools_config:
        if not isinstance(tool, dict):
            upgraded_tools.append(tool)
            continue
        tool_name = tool.get("name")
        if tool_name == "rocprof":
            logger.info(
                "  → Upgrading 'rocprof' to 'rocprofv3' for multi-node MPI compatibility"
            )
            upgraded_tool = tool.copy()
            upgraded_tool["name"] = "rocprofv3"
            upgraded_tools.append(upgraded_tool)
        else:
            upgraded_tools.append(tool)

    if upgraded_tools:
        tool_names = [
            t.get("name") if isinstance(t, dict) else str(t) for t in upgraded_tools
        ]
        logger.info(
            f"  → Multi-node profiling tools: {', '.join(filter(None, tool_names))}"
        )
        if "rccl_trace" in tool_names:
            logger.info(
                "  → ✓ rccl_trace enabled (critical for multi-node communication profiling)"
            )

    return {
        "enabled": True,
        "mode": "multi_node",
        "tools": upgraded_tools,
        "per_node_collection": True,
        "profiler": "rocprofv3",
        "wrapper_mode": "launcher",
    }
