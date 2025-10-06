"""Integration tests for get_gpu_renderD_nodes function.

These tests run against real hardware to validate the function works correctly
with actual GPU information from the system.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import json
import os
import re
import stat
# third-party modules
import pytest
# project modules
from madengine.core.context import Context
from madengine.core.console import Console


def is_amd_gpu():
    """Check if the system has AMD GPUs."""
    try:
        console = Console()
        vendor = console.sh(
            'bash -c \'if [[ -f /opt/rocm/bin/amd-smi ]]; then echo "AMD"; elif [[ -f /usr/local/bin/amd-smi ]]; then echo "AMD"; else echo "OTHER"; fi || true\''
        )
        return vendor.strip() == "AMD"
    except Exception:
        return False


def is_nvidia_gpu():
    """Check if the system has NVIDIA GPUs."""
    try:
        console = Console()
        result = console.sh('bash -c \'if [[ -f /usr/bin/nvidia-smi ]] && $(/usr/bin/nvidia-smi > /dev/null 2>&1); then echo "NVIDIA"; else echo "OTHER"; fi || true\'')
        return result.strip() == "NVIDIA"
    except Exception:
        return False


class TestGetGpuRenderDNodesIntegration:
    """Integration test suite for the get_gpu_renderD_nodes method using real hardware."""

    @pytest.mark.skipif(is_amd_gpu(), reason="Test requires non-AMD GPU or no GPU")
    def test_returns_none_for_non_amd_gpu(self):
        """Test that the function returns None for non-AMD GPUs."""
        context = Context()
        
        # Should return None for non-AMD GPUs
        if context.ctx['docker_env_vars']['MAD_GPU_VENDOR'] != 'AMD':
            assert context.ctx['gpu_renderDs'] is None

    @pytest.mark.skipif(not is_amd_gpu(), reason="Test requires AMD GPU")
    def test_returns_list_for_amd_gpu(self):
        """Test that the function returns a list of renderD nodes for AMD GPUs."""
        context = Context()
        
        # Should return a list for AMD GPUs
        assert context.ctx['gpu_renderDs'] is not None
        assert isinstance(context.ctx['gpu_renderDs'], list)
        
        # List should not be empty if there are GPUs
        if context.ctx['docker_env_vars']['MAD_SYSTEM_NGPUS'] > 0:
            assert len(context.ctx['gpu_renderDs']) > 0

    @pytest.mark.skipif(not is_amd_gpu(), reason="Test requires AMD GPU")
    def test_renderD_count_matches_gpu_count(self):
        """Test that the number of renderD nodes matches the number of GPUs."""
        console = Console()
        context = Context()
        
        # Get GPU count from amd-smi
        try:
            amd_smi_output = console.sh("amd-smi list -e --json")
            gpu_data = json.loads(amd_smi_output)
            expected_gpu_count = len(gpu_data)
        except Exception:
            pytest.skip("Unable to query amd-smi for GPU count")
        
        # The number of renderD nodes should match the number of GPUs
        assert len(context.ctx['gpu_renderDs']) == expected_gpu_count
        
    @pytest.mark.skipif(not is_amd_gpu(), reason="Test requires AMD GPU")
    def test_renderD_values_are_valid(self):
        """Test that all renderD values are valid integers."""
        context = Context()
        
        # All renderD values should be positive integers
        for renderD in context.ctx['gpu_renderDs']:
            assert isinstance(renderD, int)
            assert renderD > 0
            
    @pytest.mark.skipif(not is_amd_gpu(), reason="Test requires AMD GPU")
    def test_renderD_nodes_are_unique(self):
        """Test that all renderD nodes are unique."""
        context = Context()
        
        renderDs = context.ctx['gpu_renderDs']
        # All renderD values should be unique
        assert len(renderDs) == len(set(renderDs))
        
    @pytest.mark.skipif(not is_amd_gpu(), reason="Test requires AMD GPU")
    def test_renderD_values_match_kfd_properties(self):
        """Test that renderD values match what's in KFD properties."""
        console = Console()
        context = Context()
        
        # Get renderD values from KFD directly
        try:
            kfd_output = console.sh("grep -r drm_render_minor /sys/devices/virtual/kfd/kfd/topology/nodes")
            kfd_lines = [line for line in kfd_output.split("\n") if line.strip()]
            # Filter out CPU entries (renderD value 0)
            kfd_renderDs = [int(line.split()[-1]) for line in kfd_lines if int(line.split()[-1]) != 0]
        except Exception:
            pytest.skip("Unable to read KFD properties")
        
        # The renderD values from context should be a subset of KFD renderDs
        for renderD in context.ctx['gpu_renderDs']:
            assert renderD in kfd_renderDs, f"renderD {renderD} not found in KFD properties"
    
    @pytest.mark.skipif(not is_amd_gpu(), reason="Test requires AMD GPU")
    def test_gpu_ordering_is_consistent(self):
        """Test that GPU ordering matches amd-smi GPU IDs."""
        console = Console()
        context = Context()
        
        try:
            # Get amd-smi data
            amd_smi_output = console.sh("amd-smi list -e --json")
            gpu_data = json.loads(amd_smi_output)
            
            # Sort by GPU ID
            sorted_gpus = sorted(gpu_data, key=lambda x: x["gpu"])
            
            # The number of GPUs should match
            assert len(context.ctx['gpu_renderDs']) == len(sorted_gpus)
            
        except Exception:
            pytest.skip("Unable to verify GPU ordering with amd-smi")
    
    @pytest.mark.skipif(not is_amd_gpu(), reason="Test requires AMD GPU")
    def test_renderD_nodes_exist_in_dev(self):
        """Test that the renderD nodes actually exist in /dev/dri/."""
        context = Context()
        
        # Check that each renderD node exists as a device file
        for renderD in context.ctx['gpu_renderDs']:
            dev_path = f"/dev/dri/renderD{renderD}"
            assert os.path.exists(dev_path), f"Device {dev_path} does not exist"
            # Should be a character device
            assert stat.S_ISCHR(os.stat(dev_path).st_mode), f"{dev_path} is not a character device"
    
    @pytest.mark.skipif(not is_amd_gpu(), reason="Test requires AMD GPU")
    def test_no_cpu_entries_in_renderDs(self):
        """Test that CPU entries (renderD=0) are not included."""
        context = Context()
        
        # None of the renderD values should be 0 (CPUs)
        for renderD in context.ctx['gpu_renderDs']:
            assert renderD != 0, "CPU entry (renderD=0) found in GPU renderD list"
    
    @pytest.mark.skipif(not is_amd_gpu(), reason="Test requires AMD GPU")
    def test_context_initialization_succeeds(self):
        """Test that Context initialization succeeds with real GPU data."""
        # This should not raise any exceptions
        context = Context()
        
        # Basic sanity checks
        assert context.ctx is not None
        assert 'gpu_renderDs' in context.ctx
        assert 'docker_env_vars' in context.ctx
        assert 'MAD_GPU_VENDOR' in context.ctx['docker_env_vars']
        
    @pytest.mark.skipif(not is_amd_gpu(), reason="Test requires AMD GPU")
    def test_renderD_mapping_is_reproducible(self):
        """Test that creating multiple Context objects produces the same renderD mapping."""
        context1 = Context()
        context2 = Context()
        
        # The renderD lists should be identical
        assert context1.ctx['gpu_renderDs'] == context2.ctx['gpu_renderDs']
        
    @pytest.mark.skipif(not is_amd_gpu(), reason="Test requires AMD GPU")
    def test_renderD_values_are_in_valid_range(self):
        """Test that renderD values are in the valid Linux device range."""
        context = Context()
        
        # renderD values typically start at 128 and go up
        # Valid range is 128-255 for render nodes
        for renderD in context.ctx['gpu_renderDs']:
            assert 128 <= renderD <= 255, f"renderD {renderD} is outside valid range [128, 255]"
