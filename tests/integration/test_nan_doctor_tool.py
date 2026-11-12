"""Integration tests for the nan_doctor tool: tools.json wiring and its scripts.

nan_doctor arms nan-doctor's NaN capture in an unmodified PyTorch workload: the
pre-script installs the vendored capture runtime into site-packages with a .pth
hook, the post-script removes it and summarizes the capsules. The scripts run
here against a throwaway venv with a stub ``torch`` package, so no GPU or real
torch is needed and the test interpreter's site-packages is never touched.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import hashlib
import json
import os
import subprocess
import venv
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from madengine.deployment.common import tool_output_dirs
from madengine.execution.container_runner import ContainerRunner
from madengine.utils.path_utils import get_madengine_root

COMMON = get_madengine_root() / "scripts" / "common"
RUNTIME = COMMON / "tools" / "nan_doctor"
START = COMMON / "pre_scripts" / "nan_doctor_start.sh"
COLLECT = COMMON / "post_scripts" / "nan_doctor_collect.sh"


@pytest.fixture(scope="module")
def tool() -> dict:
    with open(COMMON / "tools.json", encoding="utf-8") as f:
        return json.load(f)["tools"]["nan_doctor"]


class TestNanDoctorToolConfig:
    def test_does_not_wrap_the_model_command(self, tool):
        """Capture rides in on a .pth hook in every rank, not on a wrapper."""
        assert tool["cmd"] == ""

    def test_env_arms_the_auto_capture_under_the_workspace(self, tool):
        env = tool["env_vars"]
        assert env["NAN_CAPTURE_AUTO"] == "optimizer"
        assert env["NAN_CAPTURE_DIR"] == "/myworkspace/nan_doctor_output/capsules"

    def test_declares_its_output_dir(self, tool):
        assert tool["output_dir"] == "nan_doctor_output"
        assert tool_output_dirs([{"name": "nan_doctor"}]) == ["nan_doctor_output"]

    def test_referenced_scripts_exist(self, tool):
        root = get_madengine_root()
        for entry in tool["pre_scripts"] + tool["post_scripts"]:
            assert (root / entry["path"]).is_file(), entry["path"]
        assert (RUNTIME / "install_autocapture.py").is_file()

    def test_apply_tools_prepends_start_and_appends_collect(self):
        ctx = MagicMock()
        ctx.ctx = {"tools": [{"name": "nan_doctor"}]}
        runner = ContainerRunner(context=ctx, console=MagicMock())
        scripts = {
            "pre_scripts": [{"path": "existing_pre.sh"}],
            "encapsulate_script": "bash run.sh",
            "post_scripts": [{"path": "existing_post.sh"}],
        }
        env: dict = {}
        runner.apply_tools(scripts, env, str(COMMON / "tools.json"))
        assert Path(scripts["pre_scripts"][0]["path"]).name == "nan_doctor_start.sh"
        assert Path(scripts["post_scripts"][-1]["path"]).name == "nan_doctor_collect.sh"
        assert scripts["encapsulate_script"].strip() == "bash run.sh"
        assert env["NAN_CAPTURE_AUTO"] == "optimizer"


class TestVendoredRuntime:
    def test_runtime_files_match_their_version_record(self):
        """The copy is synced from nan-doctor; a hand edit shows up here."""
        lines = (RUNTIME / "VERSION").read_text().splitlines()
        assert lines[0].startswith("nan-doctor ")
        recorded = dict(reversed(l.split("  ", 1)) for l in lines[2:] if l.strip())
        assert set(recorded) == {
            "op_nan_capture.py",
            "nan_doctor_autocapture.py",
            "install_autocapture.py",
        }
        for name, digest in recorded.items():
            assert (
                hashlib.sha256((RUNTIME / name).read_bytes()).hexdigest() == digest
            ), name


class TestToolOutputDirs:
    def test_tools_without_output_dir_add_nothing(self):
        assert tool_output_dirs([{"name": "rocprofv3"}, {"name": "tracelens"}]) == []

    def test_unknown_and_string_tools(self):
        assert tool_output_dirs([{"name": "no_such_tool"}, "nan_doctor"]) == [
            "nan_doctor_output"
        ]

    def test_rejects_a_path_as_output_dir(self, tmp_path):
        reg = tmp_path / "tools.json"
        reg.write_text(json.dumps({"tools": {"bad": {"output_dir": "../../etc"}}}))
        with pytest.raises(Exception, match="plain directory"):
            tool_output_dirs([{"name": "bad"}], str(reg))


# ---------------------------------------------------------------------------
# The scripts, run for real in a throwaway venv


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    """A venv whose python3 has a stub torch, plus a fake model workspace."""
    root = tmp_path_factory.mktemp("nan_doctor_tool")
    env_dir = root / "venv"
    venv.EnvBuilder(with_pip=False).create(env_dir)
    py = env_dir / "bin" / "python3"
    site = subprocess.run(
        [str(py), "-c", "import site; print(site.getsitepackages()[0])"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    (Path(site) / "torch").mkdir()
    (Path(site) / "torch" / "__init__.py").write_text("")
    work = root / "work"
    (work / "scripts" / "common" / "tools").mkdir(parents=True)
    os.symlink(RUNTIME, work / "scripts" / "common" / "tools" / "nan_doctor")
    (work / "run_directory").mkdir()
    return {"bin": env_dir / "bin", "site": Path(site), "work": work}


def _run(script, sandbox, **env):
    e = {k: v for k, v in os.environ.items() if not k.startswith("NAN_CAPTURE")}
    e["PATH"] = f"{sandbox['bin']}{os.pathsep}{e.get('PATH', '')}"
    e.update(env)
    return subprocess.run(
        ["bash", str(script)],
        cwd=sandbox["work"] / "run_directory",
        capture_output=True,
        text=True,
        env=e,
        timeout=120,
    )


class TestNanDoctorScripts:
    def test_start_refuses_without_capture_dir(self, sandbox):
        r = _run(START, sandbox, NAN_CAPTURE_AUTO="optimizer")
        assert r.returncode != 0 and "NAN_CAPTURE_DIR" in r.stderr

    def test_start_refuses_without_torch(self, sandbox, tmp_path):
        r = _run(
            START,
            sandbox,
            NAN_CAPTURE_DIR=str(tmp_path / "c"),
            NAN_CAPTURE_AUTO="optimizer",
            PATH="/usr/bin:/bin",
        )
        if "torch is not importable" not in r.stderr:
            pytest.skip("system python3 has torch")
        assert r.returncode != 0

    def test_start_installs_then_collect_removes_and_summarizes(
        self, sandbox, tmp_path
    ):
        caps = tmp_path / "out" / "capsules"
        env = {"NAN_CAPTURE_DIR": str(caps), "NAN_CAPTURE_AUTO": "optimizer"}
        r = _run(START, sandbox, **env)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "capture armed" in r.stdout
        assert caps.is_dir()
        assert (sandbox["site"] / "nan_doctor_autocapture.pth").exists()

        (caps / "rank0_abc.pt").write_bytes(b"")  # what a NaN run leaves
        r = _run(COLLECT, sandbox, **env)
        assert r.returncode == 0, r.stdout + r.stderr
        assert not (sandbox["site"] / "nan_doctor_autocapture.pth").exists()
        assert not (sandbox["site"] / "op_nan_capture.py").exists()
        summary = (tmp_path / "out" / "summary.txt").read_text()
        assert "capsules=1" in summary and "manifest=no" in summary
        assert "runtime=nan-doctor " in summary

    def test_collect_never_fails(self, sandbox, tmp_path):
        r = _run(
            COLLECT, sandbox, NAN_CAPTURE_DIR=str(tmp_path / "missing" / "capsules")
        )
        assert r.returncode == 0
        assert "capsules=0" in (tmp_path / "missing" / "summary.txt").read_text()
