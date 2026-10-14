#!/usr/bin/env python3
"""
GPU Memory Gate

Pre-run check that blocks local model execution until every GPU reports
used memory at or below a threshold. A follow-up run, especially one that
allocates most of the GPU, fails if the driver is still releasing memory
from the previous process or container.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import logging
import time
from typing import Dict, List

from madengine.core.errors import ExecutionError, create_error_context
from madengine.utils.gpu_tool_manager import BaseGPUToolManager

logger = logging.getLogger(__name__)

# Idle ROCm driver footprint on MI300-class GPUs is a few hundred MB.
# Anything above this is treated as memory still held by another job.
DEFAULT_THRESHOLD_MB = 512
DEFAULT_POLL_INTERVAL_S = 1


def _error(message: str, suggestions: List[str]) -> ExecutionError:
    return ExecutionError(
        message,
        context=create_error_context(
            operation="check_gpu_memory_free",
            component="gpu_memory_gate",
        ),
        suggestions=suggestions,
    )


def _describe(busy: List[Dict[str, int]]) -> str:
    return ", ".join(
        f"GPU {entry['gpu']} ({entry['used_mb']} MB used)" for entry in busy
    )


def _read_usage(
    tool_manager: BaseGPUToolManager,
) -> List[Dict[str, int]]:
    """Read per-GPU memory, converting tool failures into ExecutionError."""
    try:
        usage = tool_manager.get_gpu_memory_usage_mb()
    except ExecutionError:
        raise
    except Exception as exc:
        raise _error(
            "Refusing to run: unable to read GPU memory usage.",
            [
                "Confirm amd-smi or rocm-smi works on this host "
                "(nvidia-smi on NVIDIA).",
                "Re-run with --verbose to see the underlying tool error.",
            ],
        ) from exc

    if not usage:
        raise _error(
            "Refusing to run: GPU memory query returned no devices.",
            [
                "Confirm the SMI tool lists the GPUs this run will use.",
                "Re-run with --verbose to see the underlying tool error.",
            ],
        )
    return usage


def _busy_gpus(
    tool_manager: BaseGPUToolManager, threshold_mb: int
) -> List[Dict[str, int]]:
    """Return GPUs whose used memory exceeds the threshold."""
    usage = _read_usage(tool_manager)
    return [entry for entry in usage if entry["used_mb"] > threshold_mb]


def check_gpu_memory_free(
    tool_manager: BaseGPUToolManager,
    timeout_s: int,
    threshold_mb: int = DEFAULT_THRESHOLD_MB,
) -> None:
    """Verify every GPU is at or below the used-memory threshold.

    Args:
        tool_manager: Vendor-specific GPU tool manager (ROCm or NVIDIA).
        timeout_s: Max seconds to poll. 0 checks once and raises immediately
            if any GPU is over the threshold. The deadline is wall-clock
            time, so a slow SMI query counts against the budget.
        threshold_mb: Used memory, in MB, at or below which a GPU is free.

    Raises:
        ExecutionError: If timeout_s is not a non-negative integer, the
            memory query fails, or any GPU is still over the threshold
            when the deadline passes.
    """
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, int) or timeout_s < 0:
        raise _error(
            "Refusing to run: --wait-for-gpu-memory must be a "
            f"non-negative integer number of seconds, got {timeout_s!r}.",
            [
                "Pass 0 to check once, or a positive number of seconds to poll.",
            ],
        )

    logger.info(
        "Checking GPU memory before run (threshold %s MB, timeout %ss)",
        threshold_mb,
        timeout_s,
    )
    busy = _busy_gpus(tool_manager, threshold_mb)
    deadline = time.monotonic() + timeout_s

    while busy:
        now = time.monotonic()
        if now >= deadline:
            break
        remaining = deadline - now
        logger.info(
            "GPU memory still above %s MB (%s); waiting up to %.0fs",
            threshold_mb,
            _describe(busy),
            remaining,
        )
        time.sleep(min(DEFAULT_POLL_INTERVAL_S, remaining))
        busy = _busy_gpus(tool_manager, threshold_mb)

    if not busy:
        return

    detail = _describe(busy)
    if timeout_s == 0:
        suggestions = [
            "Free GPU memory held by other processes or containers, "
            "including a container left up by a previous --keep-alive run.",
            "Pass --wait-for-gpu-memory with a positive number of seconds "
            f"to poll until used memory drops to {threshold_mb} MB.",
        ]
    else:
        suggestions = [
            "Free GPU memory held by other processes or containers, "
            "including a container left up by a previous --keep-alive run.",
            "Raise --wait-for-gpu-memory above "
            f"{timeout_s} if the driver is still releasing memory.",
        ]
    raise _error(
        f"Refusing to run: {len(busy)} GPU(s) have memory in use "
        f"above the {threshold_mb} MB threshold: {detail}",
        suggestions,
    )
