#!/usr/bin/env python3
"""
Shared deployment utilities used by both SLURM and Kubernetes deployments.

Provides launcher normalization, ROCm profiling checks, and multi-node
profiling configuration so logic is not duplicated across deployment modules.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import subprocess
from typing import Any, Dict, List, Optional

# Valid distributed launchers (used by normalize_launcher)
VALID_LAUNCHERS = [
    "torchrun",
    "torchtitan",
    "deepspeed",
    "megatron-lm",
    "vllm",
    "sglang",
    "sglang-disagg"
]

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
            "per_node_collection": False
        }

    if not is_rocprofv3_available():
        if tools_include_rocprof_family(tools_config):
            filtered_tools: List[Dict] = [
                t
                for t in tools_config
                if isinstance(t, dict) and t.get("name") not in _ROCPROF_FAMILY_TOOL_NAMES
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
                "per_node_collection": False
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
