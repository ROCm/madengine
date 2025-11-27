"""Utility functions for tests.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import os
import sys
import json
import subprocess
import shutil
import re
import pytest
from unittest.mock import MagicMock

MODEL_DIR = "tests/fixtures/dummy"
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(1, BASE_DIR)

# Cache variables to avoid repeated system checks during collection
_gpu_vendor_cache = None
_gpu_nodeid_map_cache = None
_num_gpus_cache = None
_num_cpus_cache = None

# GPU detection cache to avoid multiple expensive calls
_has_gpu_cache = None


def has_gpu() -> bool:
    """Simple function to check if GPU is available for testing.

    This is the primary function for test skipping decisions.
    Uses caching to avoid repeated expensive detection calls.

    Returns:
        bool: True if GPU is available, False if CPU-only machine
    """
    global _has_gpu_cache

    if _has_gpu_cache is not None:
        return _has_gpu_cache

    try:
        # Ultra-simple file existence check (no subprocess calls)
        # This is safe for pytest collection and avoids hanging
        nvidia_exists = os.path.exists("/usr/bin/nvidia-smi")
        amd_rocm_exists = os.path.exists("/opt/rocm/bin/rocm-smi") or os.path.exists(
            "/usr/local/bin/rocm-smi"
        )

        _has_gpu_cache = nvidia_exists or amd_rocm_exists

    except Exception:
        # If file checks fail, assume no GPU (safe default for tests)
        _has_gpu_cache = False

    return _has_gpu_cache


def requires_gpu(reason: str = "test requires GPU functionality"):
    """Simple decorator to skip tests that require GPU.

    This is the only decorator needed for GPU-dependent tests.

    Args:
        reason: Custom reason for skipping

    Returns:
        pytest.mark.skipif decorator
    """
    return pytest.mark.skipif(not has_gpu(), reason=reason)


@pytest.fixture
def global_data():
    # Lazy import to avoid collection issues
    if "Console" not in globals():
        from madengine.core.console import Console
    return {"console": Console(live_output=True)}


@pytest.fixture()
def clean_test_temp_files(request):

    yield

    for filename in request.param:
        file_path = os.path.join(BASE_DIR, filename)
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)


def generate_additional_context_for_machine() -> dict:
    """Generate appropriate additional context based on detected machine capabilities.

    Returns:
        dict: Additional context with gpu_vendor and guest_os suitable for current machine
    """
    if has_gpu():
        # Simple vendor detection for GPU machines
        vendor = "NVIDIA" if os.path.exists("/usr/bin/nvidia-smi") else "AMD"
        return {"gpu_vendor": vendor, "guest_os": "UBUNTU"}
    else:
        # On CPU-only machines, use defaults suitable for build-only operations
        return {
            "gpu_vendor": "AMD",  # Default for build-only nodes
            "guest_os": "UBUNTU",  # Default OS
        }


def generate_additional_context_json() -> str:
    """Generate JSON string of additional context for current machine.

    Returns:
        str: JSON string representation of additional context
    """
    return json.dumps(generate_additional_context_for_machine())


def create_mock_args_with_auto_context(**kwargs) -> MagicMock:
    """Create mock args with automatically generated additional context.

    Args:
        **kwargs: Additional attributes to set on the mock args

    Returns:
        MagicMock: Mock args object with auto-generated additional context
    """
    mock_args = MagicMock()

    # Set auto-generated context
    mock_args.additional_context = generate_additional_context_json()
    mock_args.additional_context_file = None

    # Set any additional attributes
    for key, value in kwargs.items():
        setattr(mock_args, key, value)

    return mock_args


def is_nvidia() -> bool:
    """Check if the GPU is NVIDIA or not.

    Returns:
        bool: True if NVIDIA GPU is present, False otherwise.
    """
    global _gpu_vendor_cache
    
    if _gpu_vendor_cache is not None:
        return _gpu_vendor_cache == "NVIDIA"
    
    try:
        # Lazy import to avoid collection issues
        from madengine.core.context import Context
        context = Context()
        _gpu_vendor_cache = context.ctx["gpu_vendor"]
        return _gpu_vendor_cache == "NVIDIA"
    except Exception:
        # If context creation fails during collection, assume AMD
        _gpu_vendor_cache = "AMD"
        return False


def get_gpu_nodeid_map() -> dict:
    """Get the GPU node id map using amd-smi

    Returns:
        dict: GPU node id map mapping node_id strings to GPU indices.
    """
    global _gpu_nodeid_map_cache
    
    if _gpu_nodeid_map_cache is not None:
        return _gpu_nodeid_map_cache
    
    try:
        # Lazy import to avoid collection issues
        from madengine.core.console import Console
        gpu_map = {}
        console = Console(live_output=True)
        nvidia = is_nvidia()
        
        if nvidia:
            command = "nvidia-smi --list-gpus"
            output = console.sh(command)
            lines = output.split("\n")
            for line in lines:
                if line.strip():
                    gpu_id = int(line.split(":")[0].split()[1])
                    unique_id = line.split(":")[2].split(")")[0].strip()
                    gpu_map[unique_id] = gpu_id
        else:
            try:
                # Try the new amd-smi tool first (ROCm 6.4+)
                output = console.sh("amd-smi list --json")
                gpu_data = json.loads(output)
                for gpu_info in gpu_data:
                    node_id = str(gpu_info["node_id"])
                    gpu_id = gpu_info["gpu"]
                    gpu_map[node_id] = gpu_id
            except:
                # Fall back to older rocm-smi tools
                try:
                    rocm_version = console.sh("hipconfig --version")
                    rocm_version = float(".".join(rocm_version.split(".")[:2]))
                    command = (
                        "rocm-smi --showuniqueid" if rocm_version < 6.4 else "rocm-smi --showhw"
                    )
                    output = console.sh(command)
                    lines = output.split("\n")

                    for line in lines:
                        if rocm_version < 6.4:
                            if "Unique ID:" in line:
                                gpu_id = int(line.split(":")[0].split("[")[1].split("]")[0])
                                unique_id = line.split(":")[2].strip()
                                gpu_map[unique_id] = gpu_id
                        else:
                            if re.match(r"\d+\s+\d+", line):
                                gpu_id = int(line.split()[0])
                                node_id = line.split()[1]
                                gpu_map[node_id] = gpu_id
                except:
                    # If all else fails, return empty map
                    pass
        
        _gpu_nodeid_map_cache = gpu_map
        return gpu_map
    
    except Exception:
        # If detection fails during collection, return a default mapping
        _gpu_nodeid_map_cache = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7}
        return _gpu_nodeid_map_cache


def get_num_gpus() -> int:
    """Get the number of GPUs present.

    Returns:
        int: Number of GPUs present.
    """
    global _num_gpus_cache
    
    if _num_gpus_cache is not None:
        return _num_gpus_cache
    
    try:
        gpu_map = get_gpu_nodeid_map()
        _num_gpus_cache = len(gpu_map)
        return _num_gpus_cache
    except Exception:
        # Default to 8 GPUs if detection fails during collection
        _num_gpus_cache = 8
        return _num_gpus_cache


def get_num_cpus() -> int:
    """Get the number of CPUs present.

    Returns:
        int: Number of CPUs present.
    """
    global _num_cpus_cache
    
    if _num_cpus_cache is not None:
        return _num_cpus_cache
    
    try:
        # Lazy import to avoid collection issues
        from madengine.core.console import Console
        console = Console(live_output=True)
        _num_cpus_cache = int(console.sh("lscpu | grep \"^CPU(s):\" | awk '{print $2}'"))
        return _num_cpus_cache
    except Exception:
        # Default to 64 CPUs if detection fails during collection
        _num_cpus_cache = 64
        return _num_cpus_cache
