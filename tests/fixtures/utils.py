"""Utility functions for tests.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import csv
import os
import re
import shutil
import subprocess
import sys
import json
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
        from madengine.core.constants import get_rocm_path
        rocm_path = get_rocm_path()
        amd_rocm_exists = (
            os.path.exists(os.path.join(rocm_path, "bin", "rocm-smi"))
            or os.path.exists("/usr/local/bin/rocm-smi")
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
    """
    Fixture to clean up test temporary files and Docker containers.
    
    Cleans up both before (to ensure clean state) and after (to avoid conflicts).
    """
    import subprocess
    
    # Clean up Docker containers BEFORE test (ensure clean state)
    try:
        subprocess.run(
            "docker ps -a | grep 'container_ci-dummy' | awk '{print $1}' | xargs -r docker rm -f",
            shell=True,
            capture_output=True,
            timeout=30
        )
    except:
        pass  # Ignore cleanup errors before test

    yield

    # Clean up files after test
    for filename in request.param:
        file_path = os.path.join(BASE_DIR, filename)
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
    
    # Clean up Docker containers AFTER test (avoid conflicts with next test)
    try:
        subprocess.run(
            "docker ps -a | grep 'container_ci-dummy' | awk '{print $1}' | xargs -r docker rm -f",
            shell=True,
            capture_output=True,
            timeout=30
        )
    except:
        pass  # Ignore cleanup errors after test


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
                # Try the new amd-smi tool first (ROCm 6.4.1+, PR #54)
                output = console.sh("amd-smi list --json")
                gpu_data = json.loads(output)
                for gpu_info in gpu_data:
                    node_id = str(gpu_info["node_id"])
                    gpu_id = gpu_info["gpu"]
                    gpu_map[node_id] = gpu_id
            except:
                # Fall back to older rocm-smi tools
                try:
                    rocm_version_str = console.sh("hipconfig --version")
                    # Parse version as tuple for proper comparison (6.4.1 vs 6.4.0)
                    version_parts = rocm_version_str.split(".")
                    if len(version_parts) >= 3:
                        rocm_version = tuple(int(p.split('-')[0]) for p in version_parts[:3])
                    else:
                        # Fallback to float comparison for versions without patch
                        rocm_version = (int(version_parts[0]), int(version_parts[1]), 0)
                    
                    # Use appropriate rocm-smi command based on version (PR #54: threshold is 6.4.1)
                    command = (
                        "rocm-smi --showuniqueid" if rocm_version < (6, 4, 1) else "rocm-smi --showhw"
                    )
                    output = console.sh(command)
                    lines = output.split("\n")

                    for line in lines:
                        if rocm_version < (6, 4, 1):
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


# =============================================================================
# E2E test helpers (run command, perf CSV, log path, timeout from log)
# =============================================================================

# Default list of perf output files to clean before/after e2e tests.
DEFAULT_CLEAN_FILES = ["perf.csv", "perf.html"]


def build_run_command(tags, extra_args="", output_file=None, additional_context=None):
    """Build the base shell command for 'madengine run' e2e tests.

    Args:
        tags: Model tag(s) string, e.g. 'dummy' or 'dummy_ctxtest'.
        extra_args: Optional CLI args to append (e.g. '--timeout 120').
        output_file: Optional output CSV path (e.g. 'perf.csv' or 'perf_test.csv').
        additional_context: Optional dict or JSON-str for --additional-context.
            If dict, it is json.dumps'd and wrapped in single quotes.

    Returns:
        str: Full command to run in shell (cd BASE_DIR; MODEL_DIR=... python3 -m ...).
    """
    parts = [
        "cd " + BASE_DIR + ";",
        "MODEL_DIR=" + MODEL_DIR,
        "python3 -m madengine.cli.app run --live-output --tags " + tags,
    ]
    cmd = " ".join(parts)
    if output_file:
        cmd += " -o " + output_file
    if additional_context is not None:
        if isinstance(additional_context, dict):
            ctx_str = json.dumps(additional_context)
            cmd += ' --additional-context "' + ctx_str.replace('"', '\\"') + '"'
        else:
            ctx_str = additional_context
            cmd += " --additional-context '" + ctx_str.replace("'", "'\"'\"'") + "'"
    if extra_args:
        cmd += " " + extra_args.strip()
    return cmd


def assert_model_in_perf_csv(csv_path, model, status="SUCCESS", performance=None):
    """Assert that a model row exists in perf CSV with given status and optional performance.

    Fails with pytest.fail if no matching row or row does not match expectations.

    Args:
        csv_path: Path to the perf CSV file.
        model: Model name to find (e.g. 'dummy_ctxtest' or 'dummy').
        status: Expected status string (default 'SUCCESS').
        performance: Optional expected performance value (compared as string to CSV).
    """
    if not os.path.exists(csv_path):
        pytest.fail(f"Perf CSV not found: {csv_path}")
    with open(csv_path, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row.get("model") != model:
                continue
            if row.get("status") != status:
                pytest.fail(
                    f"model {model} in perf CSV did not run successfully (status={row.get('status')})."
                )
            if performance is not None and str(row.get("performance", "")) != str(performance):
                pytest.fail(
                    f"model {model} expected performance {performance}, got {row.get('performance')}."
                )
            return
    pytest.fail(f"model {model} not found in perf CSV.")


def get_run_live_log_path(log_base_name, suffix=".run.live.log"):
    """Return the path to a run live log file for the given base name and vendor.

    Args:
        log_base_name: Base name without extension, e.g. 'dummy_dummy' or 'dummy_timeout_dummy'.
        suffix: Log file suffix (default '.run.live.log').

    Returns:
        str: Absolute path under BASE_DIR to the log file.
    """
    vendor = "amd" if not is_nvidia() else "nvidia"
    return os.path.join(BASE_DIR, log_base_name + ".ubuntu." + vendor + suffix)


def get_timeout_seconds_from_log(log_path, timeout_regex=None):
    """Read the first 'Setting timeout to N seconds' line from a run log and return N.

    Args:
        log_path: Path to the run live log file.
        timeout_regex: Optional compiled regex; default matches '⏰ Setting timeout to ([0-9]*) seconds.'

    Returns:
        str or None: The timeout seconds string, or None if not found.
    """
    if timeout_regex is None:
        timeout_regex = re.compile(r"⏰ Setting timeout to ([0-9]*) seconds.")
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r") as f:
        for line in f:
            match = timeout_regex.search(line)
            if match:
                return match.group(1)
    return None
