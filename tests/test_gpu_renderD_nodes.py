"""Unit tests for get_gpu_renderD_nodes function.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import json
from unittest.mock import MagicMock, patch, PropertyMock
# third-party modules
import pytest
# project modules
from madengine.core.context import Context


class TestGetGpuRenderDNodes:
    """Test suite for the get_gpu_renderD_nodes method."""

    @patch('madengine.core.context.Console')
    def test_returns_none_for_nvidia_gpu(self, mock_console_class):
        """Test that the function returns None for NVIDIA GPUs."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        # Mock the GPU vendor detection
        mock_console.sh.side_effect = [
            'None',  # ctx_test
            'HOST_UBUNTU',  # host_os
            '0',  # numa_balancing
            'NVIDIA',  # gpu_vendor
        ]
        
        with patch('madengine.core.context.validate_rocm_installation'):
            context = Context()
        
        # Should return None for NVIDIA
        assert context.ctx['gpu_renderDs'] is None

    @patch('madengine.core.context.Console')
    @patch('madengine.core.context.validate_rocm_installation')
    def test_rocm_6_1_2_or_later_uses_node_id(self, mock_validate, mock_console_class):
        """Test that ROCm >= 6.1.2 uses node_id method."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        # Mock amd-smi output for 2 GPUs
        amd_smi_output = json.dumps([
            {"gpu": 0, "node_id": 1},
            {"gpu": 1, "node_id": 2}
        ])
        
        kfd_drm_output = (
            "/sys/devices/virtual/kfd/kfd/topology/nodes/1/drm_render_minor 128\n"
            "/sys/devices/virtual/kfd/kfd/topology/nodes/2/drm_render_minor 129"
        )
        
        mock_console.sh.side_effect = [
            'None',  # ctx_test
            'HOST_UBUNTU',  # host_os
            '0',  # numa_balancing
            'AMD',  # gpu_vendor
            '8',  # system_ngpus
            'gfx90a',  # gpu_architecture
            'AMD Instinct MI210',  # gpu_product_name
            '6.2',  # hip_version
            '6.2.0',  # rocm_version for get_gpu_renderD_nodes
            kfd_drm_output,  # drm_render_minor grep
            amd_smi_output,  # amd-smi list -e --json
        ]
        
        context = Context()
        
        # Verify the result
        assert context.ctx['gpu_renderDs'] == [128, 129]

    @patch('madengine.core.context.Console')
    @patch('madengine.core.context.validate_rocm_installation')
    def test_rocm_before_6_1_2_uses_unique_id(self, mock_validate, mock_console_class):
        """Test that ROCm < 6.1.2 uses unique_id method."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        # Mock amd-smi output with hip_uuid
        amd_smi_output = json.dumps([
            {"gpu": 0, "hip_uuid": "GPU-12345678-1234"},
            {"gpu": 1, "hip_uuid": "GPU-87654321-5678"}
        ])
        
        kfd_drm_output = (
            "/sys/devices/virtual/kfd/kfd/topology/nodes/1/drm_render_minor 128\n"
            "/sys/devices/virtual/kfd/kfd/topology/nodes/2/drm_render_minor 129"
        )
        
        kfd_unique_output = (
            "/sys/devices/virtual/kfd/kfd/topology/nodes/1/unique_id 305419896\n"  # hex: 0x12345678
            "/sys/devices/virtual/kfd/kfd/topology/nodes/2/unique_id 2271560481"   # hex: 0x87654321
        )
        
        mock_console.sh.side_effect = [
            'None',  # ctx_test
            'HOST_UBUNTU',  # host_os
            '0',  # numa_balancing
            'AMD',  # gpu_vendor
            '8',  # system_ngpus
            'gfx90a',  # gpu_architecture
            'AMD Instinct MI210',  # gpu_product_name
            '5.7',  # hip_version
            '6.0.0',  # rocm_version for get_gpu_renderD_nodes
            kfd_drm_output,  # drm_render_minor grep
            amd_smi_output,  # amd-smi list -e --json
            kfd_unique_output,  # unique_id grep
        ]
        
        context = Context()
        
        # Verify the result
        assert context.ctx['gpu_renderDs'] == [128, 129]

    @patch('madengine.core.context.Console')
    @patch('madengine.core.context.validate_rocm_installation')
    def test_empty_rocm_version_raises_error(self, mock_validate, mock_console_class):
        """Test that empty ROCm version raises RuntimeError."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        mock_console.sh.side_effect = [
            'None',  # ctx_test
            'HOST_UBUNTU',  # host_os
            '0',  # numa_balancing
            'AMD',  # gpu_vendor
            '8',  # system_ngpus
            'gfx90a',  # gpu_architecture
            'AMD Instinct MI210',  # gpu_product_name
            '6.2',  # hip_version
            '',  # empty rocm_version - should raise error
        ]
        
        with pytest.raises(RuntimeError, match="Failed to retrieve ROCm version"):
            context = Context()

    @patch('madengine.core.context.Console')
    @patch('madengine.core.context.validate_rocm_installation')
    def test_invalid_rocm_version_format_raises_error(self, mock_validate, mock_console_class):
        """Test that invalid ROCm version format raises RuntimeError."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        mock_console.sh.side_effect = [
            'None',  # ctx_test
            'HOST_UBUNTU',  # host_os
            '0',  # numa_balancing
            'AMD',  # gpu_vendor
            '8',  # system_ngpus
            'gfx90a',  # gpu_architecture
            'AMD Instinct MI210',  # gpu_product_name
            '6.2',  # hip_version
            'invalid.version.format.extra',  # invalid rocm_version
        ]
        
        with pytest.raises(RuntimeError, match="Failed to parse ROCm version"):
            context = Context()

    @patch('madengine.core.context.Console')
    @patch('madengine.core.context.validate_rocm_installation')
    def test_empty_kfd_properties_raises_error(self, mock_validate, mock_console_class):
        """Test that empty KFD properties raises RuntimeError."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        mock_console.sh.side_effect = [
            'None',  # ctx_test
            'HOST_UBUNTU',  # host_os
            '0',  # numa_balancing
            'AMD',  # gpu_vendor
            '8',  # system_ngpus
            'gfx90a',  # gpu_architecture
            'AMD Instinct MI210',  # gpu_product_name
            '6.2',  # hip_version
            '6.2.0',  # rocm_version
            '',  # empty kfd_properties - should raise error
        ]
        
        with pytest.raises(RuntimeError, match="Failed to retrieve KFD properties"):
            context = Context()

    @patch('madengine.core.context.Console')
    @patch('madengine.core.context.validate_rocm_installation')
    def test_filters_cpu_entries_with_zero_renderD(self, mock_validate, mock_console_class):
        """Test that CPU entries (renderD=0) are filtered out."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        # Mock amd-smi output for 2 GPUs
        amd_smi_output = json.dumps([
            {"gpu": 0, "node_id": 2},
            {"gpu": 1, "node_id": 3}
        ])
        
        # KFD output includes CPU nodes with renderD=0
        kfd_drm_output = (
            "/sys/devices/virtual/kfd/kfd/topology/nodes/0/drm_render_minor 0\n"  # CPU
            "/sys/devices/virtual/kfd/kfd/topology/nodes/1/drm_render_minor 0\n"  # CPU
            "/sys/devices/virtual/kfd/kfd/topology/nodes/2/drm_render_minor 128\n"  # GPU
            "/sys/devices/virtual/kfd/kfd/topology/nodes/3/drm_render_minor 129"   # GPU
        )
        
        mock_console.sh.side_effect = [
            'None',  # ctx_test
            'HOST_UBUNTU',  # host_os
            '0',  # numa_balancing
            'AMD',  # gpu_vendor
            '8',  # system_ngpus
            'gfx90a',  # gpu_architecture
            'AMD Instinct MI210',  # gpu_product_name
            '6.2',  # hip_version
            '6.2.0',  # rocm_version
            kfd_drm_output,  # drm_render_minor grep
            amd_smi_output,  # amd-smi list -e --json
        ]
        
        context = Context()
        
        # Verify that only GPU entries are included
        assert context.ctx['gpu_renderDs'] == [128, 129]

    @patch('madengine.core.context.Console')
    @patch('madengine.core.context.validate_rocm_installation')
    def test_empty_amd_smi_output_raises_error(self, mock_validate, mock_console_class):
        """Test that empty amd-smi output raises ValueError."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        kfd_drm_output = (
            "/sys/devices/virtual/kfd/kfd/topology/nodes/1/drm_render_minor 128\n"
            "/sys/devices/virtual/kfd/kfd/topology/nodes/2/drm_render_minor 129"
        )
        
        mock_console.sh.side_effect = [
            'None',  # ctx_test
            'HOST_UBUNTU',  # host_os
            '0',  # numa_balancing
            'AMD',  # gpu_vendor
            '8',  # system_ngpus
            'gfx90a',  # gpu_architecture
            'AMD Instinct MI210',  # gpu_product_name
            '6.2',  # hip_version
            '6.2.0',  # rocm_version
            kfd_drm_output,  # drm_render_minor grep
            '',  # empty amd-smi output
        ]
        
        with pytest.raises(RuntimeError, match="Failed to retrieve AMD GPU data"):
            context = Context()

    @patch('madengine.core.context.Console')
    @patch('madengine.core.context.validate_rocm_installation')
    def test_invalid_json_from_amd_smi_raises_error(self, mock_validate, mock_console_class):
        """Test that invalid JSON from amd-smi raises ValueError."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        kfd_drm_output = (
            "/sys/devices/virtual/kfd/kfd/topology/nodes/1/drm_render_minor 128\n"
            "/sys/devices/virtual/kfd/kfd/topology/nodes/2/drm_render_minor 129"
        )
        
        mock_console.sh.side_effect = [
            'None',  # ctx_test
            'HOST_UBUNTU',  # host_os
            '0',  # numa_balancing
            'AMD',  # gpu_vendor
            '8',  # system_ngpus
            'gfx90a',  # gpu_architecture
            'AMD Instinct MI210',  # gpu_product_name
            '6.2',  # hip_version
            '6.2.0',  # rocm_version
            kfd_drm_output,  # drm_render_minor grep
            'invalid json output',  # invalid JSON
        ]
        
        with pytest.raises(RuntimeError, match="Failed to parse amd-smi JSON output"):
            context = Context()

    @patch('madengine.core.context.Console')
    @patch('madengine.core.context.validate_rocm_installation')
    def test_mismatch_between_nodeids_and_renderDs_raises_error(self, mock_validate, mock_console_class):
        """Test that mismatch between node IDs and renderDs raises RuntimeError."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        # amd-smi has 2 GPUs
        amd_smi_output = json.dumps([
            {"gpu": 0, "node_id": 1},
            {"gpu": 1, "node_id": 2}
        ])
        
        # But KFD only has 1 GPU (mismatch)
        kfd_drm_output = (
            "/sys/devices/virtual/kfd/kfd/topology/nodes/1/drm_render_minor 128"
        )
        
        mock_console.sh.side_effect = [
            'None',  # ctx_test
            'HOST_UBUNTU',  # host_os
            '0',  # numa_balancing
            'AMD',  # gpu_vendor
            '8',  # system_ngpus
            'gfx90a',  # gpu_architecture
            'AMD Instinct MI210',  # gpu_product_name
            '6.2',  # hip_version
            '6.2.0',  # rocm_version
            kfd_drm_output,  # drm_render_minor grep (only 1 GPU)
            amd_smi_output,  # amd-smi list -e --json (2 GPUs)
        ]
        
        with pytest.raises(RuntimeError, match="Mismatch between node IDs count"):
            context = Context()

    @patch('madengine.core.context.Console')
    @patch('madengine.core.context.validate_rocm_installation')
    def test_missing_node_id_field_raises_error(self, mock_validate, mock_console_class):
        """Test that missing node_id field in amd-smi output raises KeyError."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        # amd-smi output missing node_id field
        amd_smi_output = json.dumps([
            {"gpu": 0},  # Missing node_id
            {"gpu": 1, "node_id": 2}
        ])
        
        kfd_drm_output = (
            "/sys/devices/virtual/kfd/kfd/topology/nodes/1/drm_render_minor 128\n"
            "/sys/devices/virtual/kfd/kfd/topology/nodes/2/drm_render_minor 129"
        )
        
        mock_console.sh.side_effect = [
            'None',  # ctx_test
            'HOST_UBUNTU',  # host_os
            '0',  # numa_balancing
            'AMD',  # gpu_vendor
            '8',  # system_ngpus
            'gfx90a',  # gpu_architecture
            'AMD Instinct MI210',  # gpu_product_name
            '6.2',  # hip_version
            '6.2.0',  # rocm_version
            kfd_drm_output,  # drm_render_minor grep
            amd_smi_output,  # amd-smi with missing node_id
        ]
        
        with pytest.raises(RuntimeError, match="Failed to parse node_id from amd-smi data"):
            context = Context()

    @patch('madengine.core.context.Console')
    @patch('madengine.core.context.validate_rocm_installation')
    def test_missing_hip_uuid_field_raises_error_old_rocm(self, mock_validate, mock_console_class):
        """Test that missing hip_uuid field raises KeyError for old ROCm."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        # amd-smi output missing hip_uuid field
        amd_smi_output = json.dumps([
            {"gpu": 0},  # Missing hip_uuid
            {"gpu": 1, "hip_uuid": "GPU-87654321-5678"}
        ])
        
        kfd_drm_output = (
            "/sys/devices/virtual/kfd/kfd/topology/nodes/1/drm_render_minor 128\n"
            "/sys/devices/virtual/kfd/kfd/topology/nodes/2/drm_render_minor 129"
        )
        
        kfd_unique_output = (
            "/sys/devices/virtual/kfd/kfd/topology/nodes/1/unique_id 305419896\n"
            "/sys/devices/virtual/kfd/kfd/topology/nodes/2/unique_id 2271560481"
        )
        
        mock_console.sh.side_effect = [
            'None',  # ctx_test
            'HOST_UBUNTU',  # host_os
            '0',  # numa_balancing
            'AMD',  # gpu_vendor
            '8',  # system_ngpus
            'gfx90a',  # gpu_architecture
            'AMD Instinct MI210',  # gpu_product_name
            '5.7',  # hip_version
            '6.0.0',  # rocm_version (< 6.1.2)
            kfd_drm_output,  # drm_render_minor grep
            amd_smi_output,  # amd-smi with missing hip_uuid
            kfd_unique_output,  # unique_id grep
        ]
        
        with pytest.raises(RuntimeError, match="Failed to parse GPU data from amd-smi"):
            context = Context()

    @patch('madengine.core.context.Console')
    @patch('madengine.core.context.validate_rocm_installation')
    def test_multi_gpu_system_correct_ordering(self, mock_validate, mock_console_class):
        """Test that multi-GPU system returns correct ordering."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        # Mock amd-smi output for 4 GPUs (not in order)
        amd_smi_output = json.dumps([
            {"gpu": 2, "node_id": 4},
            {"gpu": 0, "node_id": 2},
            {"gpu": 3, "node_id": 5},
            {"gpu": 1, "node_id": 3}
        ])
        
        kfd_drm_output = (
            "/sys/devices/virtual/kfd/kfd/topology/nodes/2/drm_render_minor 128\n"
            "/sys/devices/virtual/kfd/kfd/topology/nodes/3/drm_render_minor 129\n"
            "/sys/devices/virtual/kfd/kfd/topology/nodes/4/drm_render_minor 130\n"
            "/sys/devices/virtual/kfd/kfd/topology/nodes/5/drm_render_minor 131"
        )
        
        mock_console.sh.side_effect = [
            'None',  # ctx_test
            'HOST_UBUNTU',  # host_os
            '0',  # numa_balancing
            'AMD',  # gpu_vendor
            '8',  # system_ngpus
            'gfx90a',  # gpu_architecture
            'AMD Instinct MI210',  # gpu_product_name
            '6.2',  # hip_version
            '6.2.0',  # rocm_version
            kfd_drm_output,  # drm_render_minor grep
            amd_smi_output,  # amd-smi list -e --json
        ]
        
        context = Context()
        
        # Should be sorted by GPU ID: 0->128, 1->129, 2->130, 3->131
        assert context.ctx['gpu_renderDs'] == [128, 129, 130, 131]

    @patch('madengine.core.context.Console')
    @patch('madengine.core.context.validate_rocm_installation')
    def test_handles_empty_lines_in_kfd_output(self, mock_validate, mock_console_class):
        """Test that empty lines in KFD output are handled correctly."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        amd_smi_output = json.dumps([
            {"gpu": 0, "node_id": 1},
            {"gpu": 1, "node_id": 2}
        ])
        
        # KFD output with empty lines
        kfd_drm_output = (
            "/sys/devices/virtual/kfd/kfd/topology/nodes/1/drm_render_minor 128\n"
            "\n"  # Empty line
            "/sys/devices/virtual/kfd/kfd/topology/nodes/2/drm_render_minor 129\n"
            "\n"  # Another empty line
        )
        
        mock_console.sh.side_effect = [
            'None',  # ctx_test
            'HOST_UBUNTU',  # host_os
            '0',  # numa_balancing
            'AMD',  # gpu_vendor
            '8',  # system_ngpus
            'gfx90a',  # gpu_architecture
            'AMD Instinct MI210',  # gpu_product_name
            '6.2',  # hip_version
            '6.2.0',  # rocm_version
            kfd_drm_output,  # drm_render_minor grep with empty lines
            amd_smi_output,  # amd-smi list -e --json
        ]
        
        context = Context()
        
        # Should handle empty lines correctly
        assert context.ctx['gpu_renderDs'] == [128, 129]

    @patch('madengine.core.context.Console')
    @patch('madengine.core.context.validate_rocm_installation')
    def test_no_valid_gpu_entries_raises_error(self, mock_validate, mock_console_class):
        """Test that no valid GPU entries (all CPUs) raises RuntimeError."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        # KFD output with only CPU entries
        kfd_drm_output = (
            "/sys/devices/virtual/kfd/kfd/topology/nodes/0/drm_render_minor 0\n"
            "/sys/devices/virtual/kfd/kfd/topology/nodes/1/drm_render_minor 0"
        )
        
        mock_console.sh.side_effect = [
            'None',  # ctx_test
            'HOST_UBUNTU',  # host_os
            '0',  # numa_balancing
            'AMD',  # gpu_vendor
            '8',  # system_ngpus
            'gfx90a',  # gpu_architecture
            'AMD Instinct MI210',  # gpu_product_name
            '6.2',  # hip_version
            '6.2.0',  # rocm_version
            kfd_drm_output,  # drm_render_minor grep (only CPUs)
        ]
        
        with pytest.raises(RuntimeError, match="No valid GPU renderD entries found"):
            context = Context()
