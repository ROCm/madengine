#!/usr/bin/env python3
"""
Context-variable schema for the madengine setup page.

This module is the single source of truth for every dimension exposed by the
PyTorch-style setup picker. Each entry in :data:`CONTEXT_SCHEMA` describes one
selectable variable; the generator and HTML template are fully data-driven from
this list, so adding a new context key is a one-line change here.

Descriptor fields:
    key          Context key. Dotted keys map to nested ``--additional-context``
                 JSON (e.g. ``slurm.partition`` -> ``{"slurm": {"partition": ...}}``).
    label        Human-friendly label shown in the UI.
    section      Section id (see :data:`SECTIONS`) used to group rows.
    type         One of ``enum``, ``int``, ``str``, ``bool``, ``json``.
    choices      List of allowed values for ``enum`` types.
    default      Default value. Values equal to the default (or empty) are
                 omitted from the generated command to keep it minimal.
    description  Short help text.
    deploy_scope When to include the key: ``always``, ``local``, ``k8s``,
                 ``slurm`` or ``distributed``. The picker only emits keys whose
                 scope matches the selected deployment target.
    flag         For CLI-flag dimensions (not context keys): the flag name, e.g.
                 ``--verbose``. Present only in the ``run`` section.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

from typing import Any, Dict, List


# Mirrors ``madengine.cli.constants.VALID_GPU_VENDORS`` / ``VALID_GUEST_OS``.
# Duplicated here (instead of imported) to avoid a circular import: importing
# the ``madengine.cli`` package eagerly loads the CLI app and every command,
# including the one that renders this schema.
VALID_GPU_VENDORS = ["AMD", "NVIDIA"]
VALID_GUEST_OS = ["UBUNTU", "CENTOS"]


# Ordered sections. ``id`` is referenced by each schema entry's ``section``.
SECTIONS: List[Dict[str, str]] = [
    {"id": "general", "label": "General"},
    {"id": "deploy", "label": "Deployment target"},
    {"id": "container", "label": "Container"},
    {"id": "distributed", "label": "Distributed"},
    {"id": "k8s", "label": "Kubernetes"},
    {"id": "slurm", "label": "SLURM"},
    {"id": "tools", "label": "Tools & scripts"},
    {"id": "logging", "label": "Logging"},
    {"id": "run", "label": "Run options (CLI flags)"},
]


# Known GPU architectures offered as build-arg choices. Free text is also
# allowed via the "Custom..." affordance in the UI.
GPU_ARCHITECTURES: List[str] = [
    "gfx90a",
    "gfx942",
    "gfx950",
    "gfx1100",
    "gfx908",
    "A100",
    "H100",
]

# Path (relative to the model run directory) of the agnostic workload entrypoint
# dispatcher shipped in ``src/madengine/scripts/common/entrypoints/``. It is
# copied into ``scripts/common/`` at run time, so from the model directory it is
# reached via ``../scripts/common/entrypoints/``.
WORKLOAD_ENTRYPOINT = "../scripts/common/entrypoints/run_workload.sh"


def workload_encapsulate_script(workload_id: str) -> str:
    """Build the ``encapsulate_script`` value that runs a given workload.

    Returns an empty string for the default workload (no wrapping).
    """
    if not workload_id or workload_id == "default":
        return ""
    return f"bash {WORKLOAD_ENTRYPOINT} --workload {workload_id} --"


# Selectable real-world workloads. A workload wraps the model's normal run
# command via the agnostic entrypoint dispatcher; model repos may also provide a
# ``<workload>.sh`` convention script that the dispatcher runs instead.
WORKLOADS: List[Dict[str, str]] = [
    {
        "id": "default",
        "label": "Default (model script)",
        "description": "Run the model's own script as-is (no workload wrapper).",
    },
    {
        "id": "train",
        "label": "Training",
        "description": "Run a training workload (MAD_WORKLOAD=train).",
    },
    {
        "id": "inference",
        "label": "Inference",
        "description": "Run an inference workload (MAD_WORKLOAD=inference).",
    },
    {
        "id": "finetune",
        "label": "Finetuning",
        "description": "Run a finetuning workload (MAD_WORKLOAD=finetune).",
    },
    {
        "id": "serve",
        "label": "Serving",
        "description": "Serve the model (MAD_WORKLOAD=serve).",
    },
    {
        "id": "benchmark",
        "label": "Benchmark",
        "description": "Run a benchmarking workload (MAD_WORKLOAD=benchmark).",
    },
]


# Distributed launchers recognized by the deployment layer.
LAUNCHERS: List[str] = [
    "torchrun",
    "torchtitan",
    "deepspeed",
    "megatron-lm",
    "primus",
    "vllm",
    "sglang",
    "sglang-disagg",
    "slurm_multi",
]


CONTEXT_SCHEMA: List[Dict[str, Any]] = [
    # ----------------------------------------------------------------- General
    {
        "key": "gpu_vendor",
        "label": "GPU vendor",
        "section": "general",
        "type": "enum",
        "choices": list(VALID_GPU_VENDORS),
        "default": "AMD",
        "description": "GPU vendor to target.",
        "deploy_scope": "always",
    },
    {
        "key": "guest_os",
        "label": "Guest OS",
        "section": "general",
        "type": "enum",
        "choices": list(VALID_GUEST_OS),
        "default": "UBUNTU",
        "description": "Container guest OS used for Dockerfile filtering.",
        "deploy_scope": "always",
    },
    {
        "key": "n_gpus",
        "label": "GPU count",
        "section": "general",
        "type": "int",
        "default": "",
        "description": "Number of GPUs to use (top-level override; blank = model default).",
        "deploy_scope": "always",
    },
    {
        "key": "timeout",
        "label": "Timeout (s)",
        "section": "general",
        "type": "int",
        "default": "",
        "description": "Per-run timeout in seconds (blank = madengine default).",
        "deploy_scope": "always",
    },
    # --------------------------------------------------------------- Container
    {
        "key": "docker_build_arg.MAD_SYSTEM_GPU_ARCHITECTURE",
        "label": "GPU architecture (build arg)",
        "section": "container",
        "type": "enum",
        "choices": GPU_ARCHITECTURES,
        "default": "",
        "description": "Override the GPU architecture build arg (useful on CPU build nodes).",
        "deploy_scope": "always",
    },
    {
        "key": "docker_gpus",
        "label": "Docker GPUs",
        "section": "container",
        "type": "str",
        "default": "",
        "description": "GPU index range/list for docker, e.g. '0-7' or '0,1,2,3'.",
        "deploy_scope": "local",
    },
    {
        "key": "docker_cpus",
        "label": "Docker CPUs",
        "section": "container",
        "type": "str",
        "default": "",
        "description": "CPU set for --cpuset-cpus, e.g. '0-15'.",
        "deploy_scope": "local",
    },
    {
        "key": "docker_env_vars",
        "label": "Docker env vars",
        "section": "container",
        "type": "json",
        "default": "",
        "description": 'Extra container env vars, e.g. {"HSA_ENABLE_SDMA": "0"}.',
        "deploy_scope": "always",
    },
    {
        "key": "docker_mounts",
        "label": "Docker mounts",
        "section": "container",
        "type": "json",
        "default": "",
        "description": 'Volume mounts {container_path: host_path}.',
        "deploy_scope": "local",
    },
    {
        "key": "MAD_CONTAINER_IMAGE",
        "label": "Pre-built image",
        "section": "container",
        "type": "str",
        "default": "",
        "description": "Use a pre-built image and skip the build, e.g. 'myregistry/model:latest'.",
        "deploy_scope": "always",
    },
    {
        "key": "MAD_ROCM_PATH",
        "label": "ROCm path",
        "section": "container",
        "type": "str",
        "default": "",
        "description": "Override the host ROCm root path.",
        "deploy_scope": "always",
    },
    # ------------------------------------------------------------- Distributed
    {
        "key": "distributed.launcher",
        "label": "Launcher",
        "section": "distributed",
        "type": "enum",
        "choices": LAUNCHERS,
        "default": "",
        "description": "Distributed launcher (leave blank for single-process).",
        "deploy_scope": "distributed",
    },
    {
        "key": "distributed.nnodes",
        "label": "Nodes (nnodes)",
        "section": "distributed",
        "type": "int",
        "default": "",
        "description": "Number of nodes for the distributed job.",
        "deploy_scope": "distributed",
    },
    {
        "key": "distributed.nproc_per_node",
        "label": "Procs per node",
        "section": "distributed",
        "type": "int",
        "default": "",
        "description": "Processes/GPUs per node.",
        "deploy_scope": "distributed",
    },
    {
        "key": "distributed.master_port",
        "label": "Master port",
        "section": "distributed",
        "type": "int",
        "default": "",
        "description": "Master port for the distributed launcher (default 29500).",
        "deploy_scope": "distributed",
    },
    {
        "key": "distributed.backend",
        "label": "Backend",
        "section": "distributed",
        "type": "str",
        "default": "",
        "description": "Distributed backend (default 'nccl').",
        "deploy_scope": "distributed",
    },
    # -------------------------------------------------------------- Kubernetes
    {
        "key": "k8s.namespace",
        "label": "Namespace",
        "section": "k8s",
        "type": "str",
        "default": "",
        "description": "Kubernetes namespace (default 'default').",
        "deploy_scope": "k8s",
    },
    {
        "key": "k8s.gpu_resource_name",
        "label": "GPU resource name",
        "section": "k8s",
        "type": "enum",
        "choices": ["amd.com/gpu", "nvidia.com/gpu"],
        "default": "",
        "description": "Extended GPU resource name.",
        "deploy_scope": "k8s",
    },
    {
        "key": "k8s.gpu_count",
        "label": "GPUs per pod",
        "section": "k8s",
        "type": "int",
        "default": "",
        "description": "GPUs requested per pod.",
        "deploy_scope": "k8s",
    },
    {
        "key": "k8s.memory",
        "label": "Memory request",
        "section": "k8s",
        "type": "str",
        "default": "",
        "description": "Pod memory request, e.g. '128Gi'.",
        "deploy_scope": "k8s",
    },
    {
        "key": "k8s.memory_limit",
        "label": "Memory limit",
        "section": "k8s",
        "type": "str",
        "default": "",
        "description": "Pod memory limit, e.g. '256Gi'.",
        "deploy_scope": "k8s",
    },
    {
        "key": "k8s.cpu",
        "label": "CPU request",
        "section": "k8s",
        "type": "str",
        "default": "",
        "description": "Pod CPU request, e.g. '32'.",
        "deploy_scope": "k8s",
    },
    {
        "key": "k8s.image_pull_policy",
        "label": "Image pull policy",
        "section": "k8s",
        "type": "enum",
        "choices": ["Always", "IfNotPresent", "Never"],
        "default": "",
        "description": "Pod image pull policy.",
        "deploy_scope": "k8s",
    },
    {
        "key": "k8s.backoff_limit",
        "label": "Backoff limit",
        "section": "k8s",
        "type": "int",
        "default": "",
        "description": "Job backoff limit (default 3).",
        "deploy_scope": "k8s",
    },
    # ------------------------------------------------------------------- SLURM
    {
        "key": "slurm.partition",
        "label": "Partition",
        "section": "slurm",
        "type": "str",
        "default": "",
        "description": "SLURM partition (default 'gpu').",
        "deploy_scope": "slurm",
    },
    {
        "key": "slurm.nodes",
        "label": "Nodes",
        "section": "slurm",
        "type": "int",
        "default": "",
        "description": "Number of nodes (default 1).",
        "deploy_scope": "slurm",
    },
    {
        "key": "slurm.gpus_per_node",
        "label": "GPUs per node",
        "section": "slurm",
        "type": "int",
        "default": "",
        "description": "GPUs per node (default 8).",
        "deploy_scope": "slurm",
    },
    {
        "key": "slurm.time",
        "label": "Wall time",
        "section": "slurm",
        "type": "str",
        "default": "",
        "description": "Wall time limit, e.g. '24:00:00'.",
        "deploy_scope": "slurm",
    },
    {
        "key": "slurm.account",
        "label": "Account",
        "section": "slurm",
        "type": "str",
        "default": "",
        "description": "SLURM account.",
        "deploy_scope": "slurm",
    },
    {
        "key": "slurm.reservation",
        "label": "Reservation",
        "section": "slurm",
        "type": "str",
        "default": "",
        "description": "SLURM reservation name.",
        "deploy_scope": "slurm",
    },
    {
        "key": "slurm.exclusive",
        "label": "Exclusive nodes",
        "section": "slurm",
        "type": "bool",
        "default": "",
        "description": "Request exclusive node allocation.",
        "deploy_scope": "slurm",
    },
    {
        "key": "slurm.constraint",
        "label": "Constraint",
        "section": "slurm",
        "type": "str",
        "default": "",
        "description": "Node feature constraint.",
        "deploy_scope": "slurm",
    },
    {
        "key": "slurm.nodelist",
        "label": "Node list",
        "section": "slurm",
        "type": "str",
        "default": "",
        "description": "Explicit list of nodes to use.",
        "deploy_scope": "slurm",
    },
    {
        "key": "slurm.modules",
        "label": "Modules",
        "section": "slurm",
        "type": "json",
        "default": "",
        "description": 'Environment modules to load, e.g. ["rocm/6.4"].',
        "deploy_scope": "slurm",
    },
    # ------------------------------------------------------------ Tools/scripts
    {
        "key": "tools",
        "label": "Profiling tools",
        "section": "tools",
        "type": "json",
        "default": "",
        "description": 'Profiling/tool wrappers, e.g. [{"name": "rocprofv3"}].',
        "deploy_scope": "always",
    },
    {
        "key": "pre_scripts",
        "label": "Pre-scripts",
        "section": "tools",
        "type": "json",
        "default": "",
        "description": 'Scripts to run before the model, e.g. [{"path": "...", "args": "..."}].',
        "deploy_scope": "always",
    },
    {
        "key": "post_scripts",
        "label": "Post-scripts",
        "section": "tools",
        "type": "json",
        "default": "",
        "description": "Scripts to run after the model.",
        "deploy_scope": "always",
    },
    {
        "key": "rocenv_mode",
        "label": "rocEnv mode",
        "section": "tools",
        "type": "enum",
        "choices": ["lite", "full"],
        "default": "",
        "description": "rocEnvTool collection mode (default 'lite').",
        "deploy_scope": "always",
    },
    # ----------------------------------------------------------------- Logging
    {
        "key": "log_error_pattern_scan",
        "label": "Log error scan",
        "section": "logging",
        "type": "bool",
        "default": "",
        "description": "Enable/disable post-run log error pattern scanning.",
        "deploy_scope": "always",
    },
    {
        "key": "log_error_benign_patterns",
        "label": "Benign log patterns",
        "section": "logging",
        "type": "json",
        "default": "",
        "description": 'Extra benign substrings to ignore, e.g. ["warning: x"].',
        "deploy_scope": "always",
    },
    # ------------------------------------------------------- Run options (CLI)
    {
        "key": "verbose",
        "label": "Verbose",
        "section": "run",
        "type": "bool",
        "default": "",
        "description": "Enable verbose logging.",
        "deploy_scope": "always",
        "flag": "--verbose",
    },
    {
        "key": "live_output",
        "label": "Live output",
        "section": "run",
        "type": "bool",
        "default": "",
        "description": "Stream output in real time.",
        "deploy_scope": "always",
        "flag": "--live-output",
    },
    {
        "key": "keep_alive",
        "label": "Keep alive",
        "section": "run",
        "type": "bool",
        "default": "",
        "description": "Keep containers alive after the run (local only).",
        "deploy_scope": "local",
        "flag": "--keep-alive",
    },
    {
        "key": "skip_model_run",
        "label": "Skip model run",
        "section": "run",
        "type": "bool",
        "default": "",
        "description": "Skip the model script; still run container + pre_scripts (local only).",
        "deploy_scope": "local",
        "flag": "--skip-model-run",
    },
    {
        "key": "disable_skip_gpu_arch",
        "label": "Disable skip-gpu-arch",
        "section": "run",
        "type": "bool",
        "default": "",
        "description": "Do not skip models based on GPU architecture.",
        "deploy_scope": "always",
        "flag": "--disable-skip-gpu-arch",
    },
    {
        "key": "output",
        "label": "Output CSV",
        "section": "run",
        "type": "str",
        "default": "",
        "description": "Performance output file (default 'perf.csv').",
        "deploy_scope": "always",
        "flag": "--output",
    },
    {
        "key": "tools_config",
        "label": "Tools config",
        "section": "run",
        "type": "str",
        "default": "",
        "description": "Custom tools JSON config path.",
        "deploy_scope": "always",
        "flag": "--tools-config",
    },
]


def validate_schema(schema: List[Dict[str, Any]] = CONTEXT_SCHEMA) -> None:
    """Validate schema integrity. Raises ``ValueError`` on any problem.

    Checks that keys are unique, sections are known, types are valid, and that
    ``enum`` entries provide non-empty ``choices``.
    """
    valid_types = {"enum", "int", "str", "bool", "json"}
    valid_scopes = {"always", "local", "k8s", "slurm", "distributed"}
    section_ids = {section["id"] for section in SECTIONS}

    seen_keys = set()
    for entry in schema:
        key = entry.get("key")
        if not key:
            raise ValueError(f"Schema entry missing 'key': {entry}")
        if key in seen_keys:
            raise ValueError(f"Duplicate schema key: {key}")
        seen_keys.add(key)

        if entry.get("section") not in section_ids:
            raise ValueError(f"Unknown section '{entry.get('section')}' for key {key}")
        if entry.get("type") not in valid_types:
            raise ValueError(f"Invalid type '{entry.get('type')}' for key {key}")
        if entry.get("deploy_scope") not in valid_scopes:
            raise ValueError(
                f"Invalid deploy_scope '{entry.get('deploy_scope')}' for key {key}"
            )
        if entry.get("type") == "enum" and not entry.get("choices"):
            raise ValueError(f"enum key {key} must define non-empty 'choices'")
