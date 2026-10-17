"""GPU memory queries and the pre-run memory gate.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from madengine.core.errors import ExecutionError
from madengine.utils.gpu_memory_gate import (
    DEFAULT_THRESHOLD_MB,
    check_gpu_memory_free,
)
from madengine.utils.nvidia_tool_manager import NvidiaToolManager
from madengine.utils.rocm_tool_manager import ROCmToolManager

_MIB = 1024 * 1024

AMD_SMI_GPUS = [
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

AMD_SMI_LIST_JSON = json.dumps(AMD_SMI_GPUS)
AMD_SMI_WRAPPED_JSON = json.dumps({"gpu_data": AMD_SMI_GPUS})

ROCM_SMI_MEM_INFO_JSON = json.dumps(
    {
        "system": {"driver": "rocm"},
        "card0": {
            "VRAM Total Memory (B)": "309220868096",
            "VRAM Total Used Memory (B)": "297766912",
        },
        "card1": {
            "VRAM": {
                "VRAM Total Memory (B)": "309220868096",
                "VRAM Total Used Memory (B)": "52428800000",
            }
        },
    }
)

EXPECTED_AMD = [
    {"gpu": 0, "used_mb": 283, "total_mb": 294896},
    {"gpu": 1, "used_mb": 50000, "total_mb": 294896},
]

EXPECTED_ROCM = [
    {
        "gpu": 0,
        "used_mb": 297766912 // _MIB,
        "total_mb": 309220868096 // _MIB,
    },
    {
        "gpu": 1,
        "used_mb": 52428800000 // _MIB,
        "total_mb": 309220868096 // _MIB,
    },
]


def _rocm_manager():
    return ROCmToolManager()


@pytest.mark.unit
class TestROCmToolManagerMemoryUsage:
    """Test ROCmToolManager.get_gpu_memory_usage_mb."""

    def test_parses_amd_smi_list(self):
        """amd-smi metric JSON is a top-level list, same as amd-smi list."""
        manager = _rocm_manager()
        with patch.object(
            manager, "is_tool_available", return_value=True
        ), patch.object(
            manager, "execute_command", return_value=AMD_SMI_LIST_JSON
        ) as execute:
            usage = manager.get_gpu_memory_usage_mb()

        assert usage == EXPECTED_AMD
        assert execute.call_count == 1
        assert "amd-smi" in execute.call_args.args[0]
        assert "rocm-smi" not in execute.call_args.args[0]

    def test_parses_amd_smi_gpu_data_wrapper(self):
        """Older or alternate amd-smi builds wrap the list in gpu_data."""
        manager = _rocm_manager()
        with patch.object(
            manager, "is_tool_available", return_value=True
        ), patch.object(
            manager, "execute_command", return_value=AMD_SMI_WRAPPED_JSON
        ) as execute:
            usage = manager.get_gpu_memory_usage_mb()

        assert usage == EXPECTED_AMD
        assert execute.call_count == 1

    def test_falls_back_when_amd_smi_command_fails(self):
        manager = _rocm_manager()

        def fake_execute(command):
            if "amd-smi" in command:
                raise RuntimeError("amd-smi not available")
            return ROCM_SMI_MEM_INFO_JSON

        with patch.object(
            manager, "is_tool_available", return_value=True
        ), patch.object(manager, "execute_command", side_effect=fake_execute):
            usage = manager.get_gpu_memory_usage_mb()

        assert usage == EXPECTED_ROCM

    def test_falls_back_when_amd_smi_returns_no_gpus(self):
        manager = _rocm_manager()

        def fake_execute(command):
            if "amd-smi" in command:
                return "[]"
            return ROCM_SMI_MEM_INFO_JSON

        with patch.object(
            manager, "is_tool_available", return_value=True
        ), patch.object(manager, "execute_command", side_effect=fake_execute):
            usage = manager.get_gpu_memory_usage_mb()

        assert usage == EXPECTED_ROCM

    def test_uses_rocm_smi_when_amd_smi_is_absent(self):
        manager = _rocm_manager()
        with patch.object(
            manager, "is_tool_available", return_value=False
        ), patch.object(
            manager, "execute_command", return_value=ROCM_SMI_MEM_INFO_JSON
        ) as execute:
            usage = manager.get_gpu_memory_usage_mb()

        assert usage == EXPECTED_ROCM
        assert "rocm-smi" in execute.call_args.args[0]

    def test_raises_when_both_tools_fail(self):
        manager = _rocm_manager()
        with patch.object(
            manager, "is_tool_available", return_value=False
        ), patch.object(
            manager, "execute_command", side_effect=RuntimeError("no tools")
        ), pytest.raises(
            RuntimeError, match="Unable to determine GPU memory usage"
        ):
            manager.get_gpu_memory_usage_mb()

    def test_raises_when_both_tools_report_no_gpus(self):
        manager = _rocm_manager()

        def fake_execute(command):
            if "amd-smi" in command:
                return "[]"
            return "{}"

        with patch.object(
            manager, "is_tool_available", return_value=True
        ), patch.object(
            manager, "execute_command", side_effect=fake_execute
        ), pytest.raises(
            RuntimeError, match="Unable to determine GPU memory usage"
        ):
            manager.get_gpu_memory_usage_mb()


@pytest.mark.unit
class TestNvidiaToolManagerMemoryUsage:
    """Test NvidiaToolManager.get_gpu_memory_usage_mb."""

    def test_parses_query_gpu_csv(self):
        manager = NvidiaToolManager()
        output = "0, 283, 81920\n1, 50000, 81920\n\n"
        with patch.object(manager, "execute_nvidia_smi", return_value=output) as run:
            usage = manager.get_gpu_memory_usage_mb()

        assert usage == [
            {"gpu": 0, "used_mb": 283, "total_mb": 81920},
            {"gpu": 1, "used_mb": 50000, "total_mb": 81920},
        ]
        query = run.call_args.args[0]
        assert "memory.used" in query
        assert "nounits" in query

    def test_raises_on_unparseable_line(self):
        manager = NvidiaToolManager()
        with patch.object(
            manager, "execute_nvidia_smi", return_value="0, N/A, 81920\n"
        ), pytest.raises(RuntimeError, match="Unable to parse"):
            manager.get_gpu_memory_usage_mb()

    def test_raises_when_no_gpus(self):
        manager = NvidiaToolManager()
        with patch.object(
            manager, "execute_nvidia_smi", return_value="\n"
        ), pytest.raises(RuntimeError, match="no GPUs"):
            manager.get_gpu_memory_usage_mb()


def _manager(usage):
    manager = Mock()
    manager.get_gpu_memory_usage_mb.return_value = usage
    return manager


@pytest.mark.unit
class TestCheckGpuMemoryFree:
    """Test the check_gpu_memory_free gating function."""

    def test_passes_when_all_under_threshold(self):
        manager = _manager(
            [
                {"gpu": 0, "used_mb": 283, "total_mb": 294896},
                {"gpu": 1, "used_mb": 100, "total_mb": 294896},
            ]
        )
        check_gpu_memory_free(manager, timeout_s=30)
        manager.get_gpu_memory_usage_mb.assert_called_once()

    def test_fails_fast_when_busy(self):
        manager = _manager([{"gpu": 0, "used_mb": 50000, "total_mb": 294896}])

        with pytest.raises(ExecutionError, match="Refusing to run") as exc_info:
            check_gpu_memory_free(manager, timeout_s=0)

        message = str(exc_info.value)
        assert "50000 MB used" in message
        assert f"{DEFAULT_THRESHOLD_MB} MB" in message
        suggestions = " ".join(exc_info.value.suggestions)
        assert "--keep-alive" in suggestions
        assert "positive number of seconds" in suggestions
        assert "Raise --wait-for-gpu-memory" not in suggestions
        manager.get_gpu_memory_usage_mb.assert_called_once()

    def test_wait_succeeds_once_memory_frees(self):
        manager = Mock()
        manager.get_gpu_memory_usage_mb.side_effect = [
            [{"gpu": 0, "used_mb": 50000, "total_mb": 294896}],
            [{"gpu": 0, "used_mb": 50000, "total_mb": 294896}],
            [{"gpu": 0, "used_mb": 100, "total_mb": 294896}],
        ]

        with patch("madengine.utils.gpu_memory_gate.time.sleep") as mock_sleep:
            check_gpu_memory_free(manager, timeout_s=10)

        assert mock_sleep.call_count == 2
        assert manager.get_gpu_memory_usage_mb.call_count == 3

    def test_wait_logs_busy_gpus(self, caplog):
        manager = Mock()
        manager.get_gpu_memory_usage_mb.side_effect = [
            [{"gpu": 0, "used_mb": 50000, "total_mb": 294896}],
            [{"gpu": 0, "used_mb": 100, "total_mb": 294896}],
        ]
        with caplog.at_level(logging.INFO), patch(
            "madengine.utils.gpu_memory_gate.time.sleep"
        ):
            check_gpu_memory_free(manager, timeout_s=10)

        assert "50000 MB used" in caplog.text
        assert "still above" in caplog.text

    def test_wait_raises_after_wall_clock_timeout(self):
        manager = _manager([{"gpu": 0, "used_mb": 50000, "total_mb": 294896}])
        clock = {"now": 100.0}

        def monotonic():
            return clock["now"]

        def sleep(seconds):
            clock["now"] += seconds

        with patch(
            "madengine.utils.gpu_memory_gate.time.monotonic", side_effect=monotonic
        ), patch(
            "madengine.utils.gpu_memory_gate.time.sleep", side_effect=sleep
        ) as mock_sleep, patch(
            "madengine.utils.gpu_memory_gate.DEFAULT_POLL_INTERVAL_S", 1
        ):
            with pytest.raises(ExecutionError, match="Refusing to run") as exc_info:
                check_gpu_memory_free(manager, timeout_s=2)

        assert mock_sleep.call_count == 2
        assert manager.get_gpu_memory_usage_mb.call_count == 3
        suggestions = " ".join(exc_info.value.suggestions)
        assert "Raise --wait-for-gpu-memory above 2" in suggestions
        assert "--keep-alive" in suggestions
        assert "positive number of seconds" not in suggestions
        assert "50000 MB used" in str(exc_info.value)

    def test_query_failure_is_execution_error(self):
        manager = Mock()
        manager.get_gpu_memory_usage_mb.side_effect = RuntimeError("smi down")

        with pytest.raises(
            ExecutionError, match="unable to read GPU memory"
        ) as exc_info:
            check_gpu_memory_free(manager, timeout_s=30)

        assert exc_info.value.__cause__ is manager.get_gpu_memory_usage_mb.side_effect

    def test_empty_query_is_execution_error(self):
        manager = _manager([])
        with pytest.raises(ExecutionError, match="no devices"):
            check_gpu_memory_free(manager, timeout_s=30)

    def test_rejects_negative_timeout(self):
        manager = _manager([{"gpu": 0, "used_mb": 0, "total_mb": 1}])
        with pytest.raises(ExecutionError, match="non-negative"):
            check_gpu_memory_free(manager, timeout_s=-1)
        manager.get_gpu_memory_usage_mb.assert_not_called()


@pytest.mark.unit
def test_cli_reference_documents_gpu_memory_gate():
    text = (
        Path(__file__).resolve().parents[2] / "docs" / "cli-reference.md"
    ).read_text()
    assert "--wait-for-gpu-memory" in text
    assert f"{DEFAULT_THRESHOLD_MB} MB" in text
    assert "SLURM" in text
