"""Utility functions for tests.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import os
import sys
import subprocess
import shutil
import re
import pytest
import re

# project modules
from madengine.core.console import Console
from madengine.core.context import Context


MODEL_DIR = "tests/fixtures/dummy"
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(1, BASE_DIR)
print(f'BASE DIR:: {BASE_DIR}')


@pytest.fixture
def global_data():
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


def is_nvidia() -> bool:
    """Check if the GPU is NVIDIA or not.

    Returns:
        bool: True if NVIDIA GPU is present, False otherwise.
    """
    context = Context()
    if context.ctx["gpu_vendor"] == "NVIDIA":
        return True
    else:
        return False


def get_gpu_nodeid_map() -> dict:
    """Get the GPU node id map using amd-smi

    Returns:
        dict: GPU node id map.
    """
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
        rocm_version = console.sh("hipconfig --version")
        rocm_version = float(".".join(rocm_version.split(".")[:2]))
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
    console = Console(live_output=True)
    return int(console.sh("lscpu | grep \"^CPU(s):\" | awk '{print $2}'"))
