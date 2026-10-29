"""Tests for GPU counting from amd-smi / rocm-smi CSV output (issue #123).

These run the real shell pipeline in ``ROCmToolManager.get_gpu_count`` against
fake SMI executables, so the parsing is exercised end to end.
"""

import stat

import pytest

from madengine.utils.rocm_tool_manager import ROCmToolManager

GROUP_WARNING = (
    "WARNING: User is missing the following required groups: render, video. "
    "Please add user to these groups."
)


def _amd_smi_csv(n_gpus: int, warnings: int = 0, eol: str = "\r\n") -> str:
    """Mimic `amd-smi list --csv`: optional warnings, header, rows, blank line."""
    lines = [GROUP_WARNING] * warnings
    lines.append("gpu,gpu_bdf,gpu_uuid,kfd_id,node_id,partition_id")
    lines += [f"{i},0000:{i:02x}:00.0,uuid-{i},{i + 1000},{i},0" for i in range(n_gpus)]
    return eol.join(lines) + eol + eol


def _rocm_smi_csv(n_gpus: int, warnings: int = 0) -> str:
    """Mimic `rocm-smi --showid --csv`: optional warnings, header, cardN rows."""
    lines = [GROUP_WARNING] * warnings
    lines.append("device,Device Name,Device ID,Device Rev,Subsystem ID,GUID")
    lines += [
        f"card{i},AMD Instinct MI300X,0x74a1,0x00,0x74a1,{i + 1000}"
        for i in range(n_gpus)
    ]
    return "\n".join(lines) + "\n\n"


def _install_fake_tool(tmp_path, name: str, stdout: str) -> None:
    """Install an executable `bin/<name>` that prints `stdout` verbatim."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    data = tmp_path / f"{name}.out"
    data.write_bytes(stdout.encode())
    script = bin_dir / name
    script.write_text(f'#!/bin/bash\ncat "{data}"\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


def _manager(tmp_path, monkeypatch, preferred: str) -> ROCmToolManager:
    manager = ROCmToolManager(rocm_path=str(tmp_path))
    monkeypatch.setattr(manager, "get_preferred_smi_tool", lambda: preferred)
    return manager


class TestAmdSmiGpuCount:
    """amd-smi is preferred for ROCm >= 6.4.1."""

    @pytest.mark.parametrize("warnings", [0, 2, 8])
    @pytest.mark.parametrize("eol", ["\r\n", "\n"])
    def test_counts_only_gpu_rows(self, tmp_path, monkeypatch, warnings, eol):
        _install_fake_tool(tmp_path, "amd-smi", _amd_smi_csv(8, warnings, eol))

        assert _manager(tmp_path, monkeypatch, "amd-smi").get_gpu_count() == 8

    def test_group_warnings_do_not_change_count(self, tmp_path, monkeypatch):
        """A user outside render/video gets the same count as one inside."""
        _install_fake_tool(tmp_path, "amd-smi", _amd_smi_csv(8, warnings=0))
        in_group = _manager(tmp_path, monkeypatch, "amd-smi").get_gpu_count()

        _install_fake_tool(tmp_path, "amd-smi", _amd_smi_csv(8, warnings=8))
        out_of_group = _manager(tmp_path, monkeypatch, "amd-smi").get_gpu_count()

        assert in_group == out_of_group == 8


class TestRocmSmiGpuCount:
    """rocm-smi is used for ROCm < 6.4.1."""

    @pytest.mark.parametrize("warnings", [0, 1, 8])
    def test_counts_only_card_rows(self, tmp_path, monkeypatch, warnings):
        _install_fake_tool(tmp_path, "rocm-smi", _rocm_smi_csv(8, warnings))

        assert _manager(tmp_path, monkeypatch, "rocm-smi").get_gpu_count() == 8
