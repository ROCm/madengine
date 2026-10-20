#!/usr/bin/env python3
"""
GPU Tool Manager Factory

Provides factory pattern for creating vendor-specific GPU tool managers with
singleton management.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import logging
from typing import Dict, Optional

from madengine.core.constants import get_rocm_path
from madengine.utils.gpu_tool_manager import BaseGPUToolManager
from madengine.utils.gpu_validator import GPUVendor, detect_gpu_vendor

logger = logging.getLogger(__name__)

# Singleton instances: key = (vendor, rocm_path) for AMD, (vendor, "") for NVIDIA
_manager_instances: Dict[tuple, BaseGPUToolManager] = {}


def get_gpu_tool_manager(
    vendor: Optional[GPUVendor] = None,
    rocm_path: Optional[str] = None,
) -> BaseGPUToolManager:
    """Get GPU tool manager for the specified vendor.

    This function implements the singleton pattern - only one manager instance
    is created per (vendor, rocm_path) and reused across all calls.

    Args:
        vendor: GPU vendor (AMD, NVIDIA, etc.). If None, auto-detects.
        rocm_path: Optional ROCm root path for AMD (default: ROCM_PATH env or /opt/rocm).

    Returns:
        GPU tool manager instance for the specified vendor

    Raises:
        ValueError: If vendor is unknown or unsupported
        ImportError: If vendor-specific manager module cannot be imported

    Example:
        >>> from madengine.utils.gpu_tool_factory import get_gpu_tool_manager
        >>> from madengine.utils.gpu_validator import GPUVendor
        >>>
        >>> # Auto-detect vendor
        >>> manager = get_gpu_tool_manager()
        >>>
        >>> # Explicit vendor
        >>> amd_manager = get_gpu_tool_manager(GPUVendor.AMD)
        >>> nvidia_manager = get_gpu_tool_manager(GPUVendor.NVIDIA)
    """
    # Auto-detect vendor if not specified
    if vendor is None:
        vendor = detect_gpu_vendor(rocm_path=rocm_path)
        logger.debug(f"Auto-detected GPU vendor: {vendor.value}")

    # Cache key: (vendor, rocm_path) for AMD so different paths get different managers
    resolved_rocm = get_rocm_path(rocm_path) if vendor == GPUVendor.AMD else ""
    cache_key = (vendor, resolved_rocm)

    if cache_key in _manager_instances:
        logger.debug(f"Returning cached {vendor.value} tool manager")
        return _manager_instances[cache_key]

    # Create new manager instance based on vendor
    if vendor == GPUVendor.AMD:
        try:
            from madengine.utils.rocm_tool_manager import ROCmToolManager

            manager = ROCmToolManager(rocm_path=rocm_path)
            logger.info(f"Created new ROCm tool manager")
        except ImportError as e:
            raise ImportError(f"Failed to import ROCm tool manager: {e}")

    elif vendor == GPUVendor.NVIDIA:
        try:
            from madengine.utils.nvidia_tool_manager import NvidiaToolManager

            manager = NvidiaToolManager()
            logger.info(f"Created new NVIDIA tool manager")
        except ImportError as e:
            raise ImportError(f"Failed to import NVIDIA tool manager: {e}")

    elif vendor == GPUVendor.UNKNOWN:
        raise ValueError(
            "Unable to detect GPU vendor. Ensure GPU drivers and tools are installed.\n"
            "For AMD: Install ROCm (https://github.com/ROCm/ROCm)\n"
            "For NVIDIA: Install CUDA toolkit"
        )

    else:
        raise ValueError(f"Unsupported GPU vendor: {vendor.value}")

    # Cache the manager instance
    _manager_instances[cache_key] = manager

    return manager


def clear_manager_cache() -> None:
    """Clear all cached manager instances.

    Useful for testing or when GPU configuration changes during runtime.
    This will force recreation of managers on next call to get_gpu_tool_manager().

    Also clears internal caches within each manager before removing them.
    """
    global _manager_instances

    # Clear caches within managers before removing them
    for manager in _manager_instances.values():
        manager.clear_cache()

    _manager_instances.clear()
    logger.debug("Cleared all GPU tool manager instances")


def get_cached_managers() -> Dict[tuple, BaseGPUToolManager]:
    """Get dictionary of currently cached manager instances.

    Primarily for debugging and testing purposes.
    Keys are (GPUVendor, rocm_path) for AMD, (GPUVendor, "") for NVIDIA.

    Returns:
        Dictionary mapping (vendor, rocm_path) to manager instances
    """
    return _manager_instances.copy()
