#!/usr/bin/env python3
"""
GPU Memory Gate

Pre-run check that blocks local model execution until all GPUs report
free memory, to avoid contending with memory already held by another
process or a leaked container.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import time

from madengine.core.errors import ExecutionError, create_error_context
from madengine.utils.gpu_tool_manager import BaseGPUToolManager

DEFAULT_THRESHOLD_MB = 512
DEFAULT_POLL_INTERVAL_S = 1
DEFAULT_WAIT_TIMEOUT_S = 30


def _busy_gpus(tool_manager: BaseGPUToolManager, threshold_mb: int) -> list:
    """Return GPUs whose used memory exceeds the threshold."""
    usage = tool_manager.get_gpu_memory_usage_mb()
    return [entry for entry in usage if entry["used_mb"] > threshold_mb]


def check_gpu_memory_free(
    tool_manager: BaseGPUToolManager,
    timeout_s: int = DEFAULT_WAIT_TIMEOUT_S,
) -> None:
    """Verify all GPUs have free memory before allowing a run to proceed.

    Args:
        tool_manager: Vendor-specific GPU tool manager (ROCm or NVIDIA).
        timeout_s: Max seconds to poll for GPUs to free up. 0 checks once
            and raises immediately if any GPU is busy.

    Raises:
        ExecutionError: If any GPU remains over threshold (fail-fast, or
            after the wait timeout elapses).
    """
    busy = _busy_gpus(tool_manager, DEFAULT_THRESHOLD_MB)

    elapsed = 0
    while busy and elapsed < timeout_s:
        time.sleep(DEFAULT_POLL_INTERVAL_S)
        elapsed += DEFAULT_POLL_INTERVAL_S
        busy = _busy_gpus(tool_manager, DEFAULT_THRESHOLD_MB)

    if busy:
        raise ExecutionError(
            f"Refusing to run: {len(busy)} GPU(s) have memory in use "
            f"above the {DEFAULT_THRESHOLD_MB} MB threshold: "
            f"{[entry['gpu'] for entry in busy]}",
            context=create_error_context(
                operation="check_gpu_memory_free",
                component="gpu_memory_gate",
            ),
            suggestions=[
                "Free the GPU memory held by other processes/containers",
                "Pass --wait-for-gpu-memory with a timeout in seconds to poll "
                "until memory frees up instead of failing immediately",
            ],
        )
