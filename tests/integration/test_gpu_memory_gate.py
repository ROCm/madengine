"""Test GPU memory usage queries and the GPU-memory-free run gate.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
from unittest.mock import Mock, patch

import pytest

from madengine.core.errors import ExecutionError
from madengine.utils.gpu_memory_gate import check_gpu_memory_free
from madengine.utils.nvidia_tool_manager import NvidiaToolManager
from madengine.utils.rocm_tool_manager import ROCmToolManager

AMD_SMI_MEM_USAGE_JSON = json.dumps(
    {
        "gpu_data": [
            {
                "gpu": 0,
                "mem_usage": {
                    "total_vram": {"value": 294896, "unit": "MB"},
                    "used_vram": {"value": 283, "unit": "MB"},
                    "free_vram": {"value": 294613, "unit": "MB"},
                },
            },
            {
                "gpu": 1,
                "mem_usage": {
                    "total_vram": {"value": 294896, "unit": "MB"},
                    "used_vram": {"value": 50000, "unit": "MB"},
                    "free_vram": {"value": 244896, "unit": "MB"},
                },
            },
        ]
    }
)

ROCM_SMI_MEM_INFO_JSON = json.dumps(
    {
        "card0": {
            "VRAM Total Memory (B)": "309220868096",
            "VRAM Total Used Memory (B)": "297766912",
        },
        "card1": {
            "VRAM Total Memory (B)": "309220868096",
            "VRAM Total Used Memory (B)": "52428800000",
        },
    }
)


class TestROCmToolManagerMemoryUsage:
    """Test ROCmToolManager.get_gpu_memory_usage_mb."""

    def test_uses_amd_smi_when_available(self):
        manager = ROCmToolManager()

        with patch.object(
            manager, "is_tool_available", return_value=True
        ), patch.object(
            manager, "execute_command", return_value=AMD_SMI_MEM_USAGE_JSON
        ):
            usage = manager.get_gpu_memory_usage_mb()

        assert usage == [
            {"gpu": 0, "used_mb": 283, "total_mb": 294896},
            {"gpu": 1, "used_mb": 50000, "total_mb": 294896},
        ]

    def test_falls_back_to_rocm_smi(self):
        manager = ROCmToolManager()

        def fake_execute_command(command):
            if "amd-smi" in command:
                raise RuntimeError("amd-smi not available")
            return ROCM_SMI_MEM_INFO_JSON

        with patch.object(
            manager, "is_tool_available", return_value=False
        ), patch.object(manager, "execute_command", side_effect=fake_execute_command):
            usage = manager.get_gpu_memory_usage_mb()

        assert usage == [
            {
                "gpu": 0,
                "used_mb": 297766912 // (1024 * 1024),
                "total_mb": 309220868096 // (1024 * 1024),
            },
            {
                "gpu": 1,
                "used_mb": 52428800000 // (1024 * 1024),
                "total_mb": 309220868096 // (1024 * 1024),
            },
        ]

    def test_raises_when_both_tools_fail(self):
        manager = ROCmToolManager()

        with patch.object(
            manager, "is_tool_available", return_value=False
        ), patch.object(
            manager, "execute_command", side_effect=RuntimeError("no tools")
        ), pytest.raises(
            RuntimeError, match="Unable to determine GPU memory usage"
        ):
            manager.get_gpu_memory_usage_mb()


class TestNvidiaToolManagerMemoryUsage:
    """Test NvidiaToolManager.get_gpu_memory_usage_mb."""

    def test_not_implemented(self):
        manager = NvidiaToolManager()

        with pytest.raises(NotImplementedError):
            manager.get_gpu_memory_usage_mb()


class TestCheckGpuMemoryFree:
    """Test the check_gpu_memory_free gating function."""

    def test_passes_when_all_under_threshold(self):
        manager = Mock()
        manager.get_gpu_memory_usage_mb.return_value = [
            {"gpu": 0, "used_mb": 283, "total_mb": 294896},
            {"gpu": 1, "used_mb": 100, "total_mb": 294896},
        ]

        check_gpu_memory_free(manager, timeout_s=30)

    def test_fails_fast_when_busy(self):
        manager = Mock()
        manager.get_gpu_memory_usage_mb.return_value = [
            {"gpu": 0, "used_mb": 50000, "total_mb": 294896},
        ]

        with pytest.raises(ExecutionError, match="Refusing to run"):
            check_gpu_memory_free(manager, timeout_s=0)

        manager.get_gpu_memory_usage_mb.assert_called_once()

    def test_wait_succeeds_once_memory_frees(self):
        manager = Mock()
        manager.get_gpu_memory_usage_mb.side_effect = [
            [{"gpu": 0, "used_mb": 50000, "total_mb": 294896}],
            [{"gpu": 0, "used_mb": 50000, "total_mb": 294896}],
            [{"gpu": 0, "used_mb": 100, "total_mb": 294896}],
        ]

        with patch("madengine.utils.gpu_memory_gate.time.sleep") as mock_sleep, patch(
            "madengine.utils.gpu_memory_gate.DEFAULT_POLL_INTERVAL_S", 1
        ):
            check_gpu_memory_free(manager, timeout_s=10)

        assert mock_sleep.call_count == 2
        assert manager.get_gpu_memory_usage_mb.call_count == 3

    def test_wait_raises_after_timeout(self):
        manager = Mock()
        manager.get_gpu_memory_usage_mb.return_value = [
            {"gpu": 0, "used_mb": 50000, "total_mb": 294896},
        ]

        with patch("madengine.utils.gpu_memory_gate.time.sleep"), patch(
            "madengine.utils.gpu_memory_gate.DEFAULT_POLL_INTERVAL_S", 5
        ):
            with pytest.raises(ExecutionError, match="Refusing to run"):
                check_gpu_memory_free(manager, timeout_s=10)
