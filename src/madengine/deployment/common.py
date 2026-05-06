#!/usr/bin/env python3
"""
Shared deployment utilities used by both SLURM and Kubernetes deployments.

Provides launcher normalization, ROCm profiling checks, and multi-node
profiling configuration so logic is not duplicated across deployment modules.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import subprocess
from typing import Any, Dict, List, Optional

# Valid distributed launchers (used by normalize_launcher).
# NOTE: keep this in sync with the launcher branches in
# deployment/{slurm.py,kubernetes.py}, the templates/{slurm,kubernetes}/*.j2
# Jinja conditions, and examples/**/launcher fields. Currently the rest of
# the codebase (and shipped examples) use the bare "megatron" / "primus"
# strings, while "megatron-lm" is the docs-friendly synonym.
VALID_LAUNCHERS = [
    "torchrun",
    "torchtitan",
    "deepspeed",
    "megatron",
    "megatron-lm",
    "primus",
    "vllm",
    "sglang",
    "slurm_multi",
]


def normalize_launcher(launcher_type: Optional[str], deployment_type: str) -> str:
    """
    Normalize launcher field based on deployment type and launcher value.

    Logic:
    - If launcher is in VALID_LAUNCHERS: keep as-is
    - If launcher is None/empty/invalid:
        * local → "docker" (runs in Docker container)
        * slurm → "docker" (typically uses containers on compute nodes)
        * kubernetes → "native" (pod itself is the container)

    Args:
        launcher_type: Raw launcher type from config (may be None)
        deployment_type: "local", "slurm", or "kubernetes"

    Returns:
        Normalized launcher string
    """
    if launcher_type and launcher_type in VALID_LAUNCHERS:
        return launcher_type
    # Normalize hyphen variant: slurm-multi -> slurm_multi
    if launcher_type and launcher_type.replace("-", "_") in VALID_LAUNCHERS:
        return launcher_type.replace("-", "_")
    if deployment_type == "local":
        return "docker"
    if deployment_type == "slurm":
        return "docker"
    if deployment_type == "kubernetes":
        return "native"
    return "docker"


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
            ["rocprofv3", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def configure_multi_node_profiling(
    nnodes: int,
    tools_config: List[Dict],
    logger: Any
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
       a. Check if rocprofv3 is available
       b. If available: Enable per-node profiling, upgrade "rocprof" to "rocprofv3"
       c. If not available: Log warning and skip profiling

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
            "per_node_collection": False
        }

    if not is_rocprofv3_available():
        # Only `rocprof` requires rocprofv3 (MPI-aware) for multi-node runs.
        # Other tools (rccl_trace, rocblas_trace, etc.) work independently and
        # must not be silently dropped just because rocprofv3 is missing.
        non_rocprof_tools = [
            t for t in tools_config
            if not (isinstance(t, dict) and t.get("name") == "rocprof")
        ]
        dropped = len(tools_config) - len(non_rocprof_tools)
        if dropped:
            logger.warning(
                "╔════════════════════════════════════════════════════════════════════════════╗\n"
                "║ Multi-Node rocprof Skipped                                                ║\n"
                "╠════════════════════════════════════════════════════════════════════════════╣\n"
                "║ rocprofv3 (MPI-aware) NOT FOUND. The 'rocprof' tool will be SKIPPED for   ║\n"
                "║ this multi-node run; other tools (rccl_trace, rocblas_trace, ...) keep    ║\n"
                "║ running. To enable multi-node rocprof: install rocprofiler-sdk            ║\n"
                "║ (ROCm >= 6.4.1, e.g. 'apt install rocprofiler-sdk').                      ║\n"
                "╚════════════════════════════════════════════════════════════════════════════╝"
            )
        return {
            "enabled": bool(non_rocprof_tools),
            "mode": "multi_node_no_rocprofv3" if dropped else "multi_node",
            "tools": non_rocprof_tools,
            "per_node_collection": bool(non_rocprof_tools),
        }

    logger.info(f"✓ Multi-node profiling enabled for {nnodes} nodes (rocprofv3 detected)")

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
        logger.info(f"  → Multi-node profiling tools: {', '.join(filter(None, tool_names))}")
        if "rccl_trace" in tool_names:
            logger.info("  → ✓ rccl_trace enabled (critical for multi-node communication profiling)")

    return {
        "enabled": True,
        "mode": "multi_node",
        "tools": upgraded_tools,
        "per_node_collection": True,
        "profiler": "rocprofv3",
        "wrapper_mode": "launcher"
    }
