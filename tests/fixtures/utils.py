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
import re
import json


MODEL_DIR = "tests/fixtures/dummy"
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(1, BASE_DIR)

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
        amd_rocm_exists = os.path.exists("/opt/rocm/bin/rocm-smi") or os.path.exists("/usr/local/bin/rocm-smi")

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
    """Simple function to check if NVIDIA GPU tools are available.

    Returns:
        bool: True if NVIDIA GPU tools are detected
    """
    try:
        return os.path.exists("/usr/bin/nvidia-smi")
    except Exception:
        return False


def is_amd() -> bool:
    """Simple function to check if AMD GPU tools are available.

    Returns:
        bool: True if AMD GPU tools are detected
    """
    try:
        return os.path.exists("/opt/rocm/bin/rocm-smi") or os.path.exists("/usr/bin/rocm-smi")
    except Exception:
        return False


def get_gpu_nodeid_map() -> dict:
    """Get the GPU node id map using amd-smi.

    Returns:
        dict: GPU node id map.
    """
    # Lazy import to avoid collection issues
    if "Console" not in globals():
        from madengine.core.console import Console
    gpu_map = {}
    console = Console(live_output=True)
    if is_nvidia():
        command = "nvidia-smi --list-gpus"
        output = console.sh(command)
        lines = output.split("\n")
        for line in lines:
            gpu_id = int(line.split(":")[0].split()[1])
            unique_id = line.split(":")[2].split(")")[0].strip()
            gpu_map[unique_id] = gpu_id
        print(f"NVIDIA GPU data: {gpu_map}")
    else:
        # example output of hipconfig --version: 6.1.40092-038397aaa
        rocm_version = console.sh("hipconfig --version")
        rocm_version = float(".".join(rocm_version.split(".")[:2]))

        if rocm_version < 6.1:
            command = "rocm-smi --showuniqueid"
            output = console.sh(command)
            lines = output.split("\n")
            for line in lines:
                if "Unique ID:" in line:
                    gpu_id = int(line.split(":")[0].split("[")[1].split("]")[0])
                    unique_id = line.split(":")[2].strip()
                    gpu_map[unique_id] = gpu_id
        else:
            command = "amd-smi list --json"
            output = console.sh(command)
            if output:
                data = json.loads(output)
            else:
                raise ValueError("Failed to retrieve AMD GPU data")

            for item in data:
                node_id = item["node_id"]
                gpu_map[node_id] = item["gpu"]

        print(f"AMD GPU data: {gpu_map}")
    return gpu_map


def get_num_gpus() -> int:
    """Get the number of GPUs present.

    Returns:
        int: Number of GPUs present.
    """
    gpu_map = get_gpu_nodeid_map()
    return len(gpu_map)


def get_num_cpus() -> int:
    """Get the number of CPUs present.

    Returns:
        int: Number of CPUs present.
    """
    # Lazy import to avoid collection issues
    if "Console" not in globals():
        from madengine.core.console import Console
    console = Console(live_output=True)
    return int(console.sh("lscpu | grep \"^CPU(s):\" | awk '{print $2}'"))
