"""Behavioral tests for the profiler wrapper scripts, run against stub profilers.

The wrappers are prepended to a model's command and run from its run_directory;
the workload itself may change directory before it starts GPU work.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import sqlite3
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

from madengine.utils.path_utils import get_madengine_root

TOOLS = get_madengine_root() / "scripts" / "common" / "tools"


def _stub(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text("#!/usr/bin/env bash\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _run(script: str, args, cwd: Path, bin_dir: Path, **env):
    full_env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", **env}
    return subprocess.run(["bash", str(TOOLS / script), *args], cwd=cwd, env=full_env,
                          capture_output=True, text=True)


@pytest.fixture
def bin_dir(tmp_path):
    d = tmp_path / "bin"
    d.mkdir()
    return d


class TestRocprofWrapperOutputDir:
    @pytest.fixture(autouse=True)
    def _rocprofv3(self, bin_dir):
        # Record the arguments rocprofv3 receives, one per line.
        _stub(bin_dir, "rocprofv3", 'printf "%s\\n" "$@" > "$ARGS_OUT"\n')

    def _args(self, tmp_path, bin_dir, *opts):
        run_dir = tmp_path / "run_directory"
        run_dir.mkdir(exist_ok=True)
        out = tmp_path / "args.txt"
        res = _run("rocprof_wrapper.sh", [*opts, "--", "bash", "-c", "cd /tmp && true"],
                   run_dir, bin_dir, ARGS_OUT=str(out))
        assert res.returncode == 0, res.stderr
        return run_dir, out.read_text().splitlines()

    def test_relative_dir_made_absolute(self, tmp_path, bin_dir):
        run_dir, args = self._args(tmp_path, bin_dir, "--kernel-trace", "-d", "./rocprof_output")
        assert args[args.index("-d") + 1] == f"{run_dir}/rocprof_output"

    def test_long_option_with_equals(self, tmp_path, bin_dir):
        run_dir, args = self._args(tmp_path, bin_dir, "--output-directory=out")
        assert f"--output-directory={run_dir}/out" in args

    def test_absolute_dir_unchanged(self, tmp_path, bin_dir):
        _, args = self._args(tmp_path, bin_dir, "-d", "/abs/out")
        assert args[args.index("-d") + 1] == "/abs/out"

    def test_application_args_untouched(self, tmp_path, bin_dir):
        _, args = self._args(tmp_path, bin_dir, "-d", "out")
        assert args[args.index("--") + 1:] == ["bash", "-c", "cd /tmp && true"]


def _write_rtl_db(path: Path, op_types) -> None:
    con = sqlite3.connect(path)
    con.execute("create table rocpd_string (id integer primary key, string text)")
    con.execute("create table rocpd_op (id integer primary key, gpuId int, opType_id int)")
    for i, t in enumerate(sorted(set(op_types)), start=1):
        con.execute("insert into rocpd_string values (?, ?)", (i, t))
    ids = {t: i for i, t in enumerate(sorted(set(op_types)), start=1)}
    con.executemany("insert into rocpd_op (gpuId, opType_id) values (0, ?)",
                    [(ids[t],) for t in op_types])
    con.commit()
    con.close()


class TestRtlWrapperEmptyTrace:
    @pytest.fixture(autouse=True)
    def _rtl(self, bin_dir, tmp_path):
        # Stub `rtl trace -o DB <cmd>`: run the workload, then copy a canned DB to -o.
        _stub(bin_dir, "rtl", """
            shift  # "trace"
            while [ "$1" != "-o" ]; do shift; done
            db="$2"; shift 2
            "$@" || exit $?
            [ -n "$CANNED_DB" ] && cp "$CANNED_DB" "$db"
            exit 0
        """)

    def _run_rtl(self, tmp_path, bin_dir, op_types=None, cmd=("true",), **env):
        canned = ""
        if op_types is not None:
            canned = str(tmp_path / "canned.db")
            _write_rtl_db(Path(canned), op_types)
        run_dir = tmp_path / "run_directory"
        run_dir.mkdir(exist_ok=True)
        return _run("rtl_trace_wrapper.sh", list(cmd), run_dir, bin_dir, CANNED_DB=canned, **env), run_dir

    def test_kernel_ops_pass(self, tmp_path, bin_dir):
        res, run_dir = self._run_rtl(tmp_path, bin_dir, ["KernelExecution", "UserMarker"])
        assert res.returncode == 0, res.stderr
        assert (run_dir / "rocm_trace_lite_output" / "trace.db").exists()

    def test_markers_only_fails(self, tmp_path, bin_dir):
        res, _ = self._run_rtl(tmp_path, bin_dir, ["UserMarker", "UserMarker"])
        assert res.returncode == 3
        assert "captured no GPU operations" in res.stderr

    def test_missing_trace_fails(self, tmp_path, bin_dir):
        res, _ = self._run_rtl(tmp_path, bin_dir, None)
        assert res.returncode == 3

    def test_allow_empty(self, tmp_path, bin_dir):
        res, _ = self._run_rtl(tmp_path, bin_dir, ["UserMarker"], RTL_WRAPPER_ALLOW_EMPTY="1")
        assert res.returncode == 0
        assert "captured no GPU operations" in res.stderr

    def test_workload_failure_propagates(self, tmp_path, bin_dir):
        res, _ = self._run_rtl(tmp_path, bin_dir, ["KernelExecution"], cmd=("bash", "-c", "exit 7"))
        assert res.returncode == 7

    def test_hsa_tools_lib_reaches_workers(self, tmp_path, bin_dir):
        seen = tmp_path / "seen"
        res, _ = self._run_rtl(
            tmp_path, bin_dir, ["KernelExecution"],
            cmd=("bash", "-c", 'printf "%s\\n%s\\n" "$HSA_TOOLS_LIB" "$HSA_TOOLS_DISABLE_REGISTER" > "$SEEN"'),
            HSA_TOOLS_LIB="/opt/librtl.so", SEEN=str(seen),
        )
        assert res.returncode == 0, res.stderr
        lib, disabled = seen.read_text().splitlines()
        assert lib == "/opt/librtl.so"
        assert disabled == "1"

    def test_keep_register_does_not_disable_it(self, tmp_path, bin_dir):
        seen = tmp_path / "seen"
        res, _ = self._run_rtl(
            tmp_path, bin_dir, ["KernelExecution"],
            cmd=("bash", "-c", 'printf "%s\\n" "${HSA_TOOLS_DISABLE_REGISTER-unset}" > "$SEEN"'),
            HSA_TOOLS_LIB="/opt/librtl.so", RTL_WRAPPER_KEEP_REGISTER="1", SEEN=str(seen),
        )
        assert res.returncode == 0, res.stderr
        assert seen.read_text().strip() == "unset"

    def test_per_process_db_counts_as_gpu_ops(self, tmp_path, bin_dir):
        # Workers write trace_<pid>.db. The merged trace.db may be absent.
        writer = tmp_path / "write_trace.py"
        writer.write_text(
            "import os, sqlite3\n"
            "from pathlib import Path\n"
            "path = Path(os.environ['RTL_OUTPUT'].replace('%p', '42'))\n"
            "path.parent.mkdir(parents=True, exist_ok=True)\n"
            "con = sqlite3.connect(path)\n"
            "con.execute('create table rocpd_string (id integer primary key, string text)')\n"
            "con.execute('create table rocpd_op (id integer primary key, gpuId int, opType_id int)')\n"
            "con.execute(\"insert into rocpd_string values (1, 'KernelExecution')\")\n"
            "con.execute('insert into rocpd_op (gpuId, opType_id) values (0, 1)')\n"
            "con.commit()\n"
        )
        res, run_dir = self._run_rtl(tmp_path, bin_dir, None, cmd=("python3", str(writer)))
        assert res.returncode == 0, res.stderr
        assert (run_dir / "rocm_trace_lite_output" / "trace_42.db").exists()
