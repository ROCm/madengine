"""Whole-pipeline TraceLens tests that run wherever CI runs.

The real TraceLens pins ``protobuf`` and ``xprof`` and only produces reports from
recorded GPU traces, so CI can neither install it nor feed it real input. These
tests put the dummy TraceLens in ``tests/fixtures/dummy_tracelens`` on
``PYTHONPATH`` in its place and then drive the integration for real: fabricated
trace artifacts, the packaged analyzer, the in-container ``tracelens``
post-script, and ``madengine report tracelens``. No GPU, Docker, or network.

This covers the half of the integration madengine owns — which report generator
each trace kind is routed to, the flags it receives, where reports land, what the
summary records, and the guarantee that a failed analysis never fails a model
run. It cannot confirm that upstream TraceLens still accepts those flags.

Substituting TraceLens is not just a convenience: the real one reports only on
kernels it can link back to the runtime calls that launched them, so no
fabricated trace will produce a report, however well shaped. The trace artifacts
below are therefore structural stand-ins, and the GPU-gated tests in
test_tracelens_workflows.py are what exercise real analysis.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import csv
import gzip
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# third-party modules
import pytest

# project modules
from madengine.utils.path_utils import get_madengine_root

# The in-container half of the integration is shell, and the venv layout the
# analyzer looks into is POSIX-only.
pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="drives Linux container scripts and a POSIX venv layout"
)

DUMMY_TRACELENS = Path(__file__).resolve().parents[1] / "fixtures" / "dummy_tracelens"
COMMON_SCRIPTS = get_madengine_root() / "scripts" / "common"
ANALYZER = COMMON_SCRIPTS / "tools" / "tracelens_analyze.py"
POST_SCRIPT = COMMON_SCRIPTS / "post_scripts" / "tracelens.sh"

PYTORCH_REPORT = "TraceLens_generate_perf_report_pytorch"
ROCPROF_REPORT = "TraceLens_generate_perf_report_rocprof"
PFTRACE_REPORTS = (
    "TraceLens_generate_perf_report_pftrace_hip_activity",
    "TraceLens_generate_perf_report_pftrace_hip_api",
    "TraceLens_generate_perf_report_pftrace_memory_copy",
)
COLLECTIVE_REPORT = "TraceLens_generate_multi_rank_collective_report_pytorch"


def _entry_points() -> dict:
    """Return the analyzer's entry point -> module map.

    Read from the analyzer itself so the dummy console scripts are always named
    after what madengine actually looks for.
    """
    spec = importlib.util.spec_from_file_location("tracelens_analyze", ANALYZER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._ENTRY_POINTS


ENTRY_POINTS = _entry_points()


class DummyTraceLens:
    """The dummy package, installed the way the pre-script installs the real one.

    ``pre_scripts/trace.sh tracelens`` builds an isolated venv whose ``bin/``
    holds a python and TraceLens' report console scripts, and the analyzer
    prefers those console scripts over importing the modules. Reproducing that
    layout means these tests exercise the same lookup a real run does.
    """

    def __init__(self, root: Path, console_scripts: bool = True) -> None:
        self.root = root
        self.log = root / "invocations.jsonl"
        bin_dir = root / "bin"
        bin_dir.mkdir(parents=True)

        self.python = bin_dir / "python3"
        self._write_executable(
            self.python, f'#!/bin/sh\nexec "{sys.executable}" "$@"\n'
        )
        if console_scripts:
            for name, module in ENTRY_POINTS.items():
                self._write_executable(
                    bin_dir / name,
                    f"#!{sys.executable}\n"
                    "import sys\n"
                    f"from {module} import main\n"
                    "sys.exit(main())\n",
                )
        self._vars = {
            "DUMMY_TRACELENS_LOG": str(self.log),
            "TRACELENS_VENV": str(root),
        }

    @staticmethod
    def _write_executable(path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)

    def fail(self, *entry_points: str) -> None:
        """Make the named report generators fail. ``"all"`` fails every one."""
        self._vars["DUMMY_TRACELENS_FAIL"] = ",".join(entry_points)

    def environ(self, **overrides: str) -> dict:
        """Return an environment in which the dummy shadows any real TraceLens."""
        existing = os.environ.get("PYTHONPATH", "")
        python_path = os.pathsep.join(p for p in (str(DUMMY_TRACELENS), existing) if p)
        env = dict(os.environ, PYTHONPATH=python_path, **self._vars)
        env.update(overrides)
        return env

    def invocations(self, entry_point: str = "") -> list:
        """Return the report generator calls recorded so far."""
        if not self.log.is_file():
            return []
        records = [
            json.loads(line)
            for line in self.log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if entry_point:
            records = [r for r in records if r["entry_point"] == entry_point]
        return records

    def reports_run(self) -> list:
        """Return the entry point names that ran, in order."""
        return [record["entry_point"] for record in self.invocations()]


@pytest.fixture
def dummy_tracelens(tmp_path):
    return DummyTraceLens(tmp_path / "tracelens-venv")


def write_chrome_trace(path: Path) -> Path:
    """Write a minimal Chrome Trace Event document, as Kineto does.

    Gzips the payload when ``path`` ends in ``.gz``, which is how
    ``tensorboard_trace_handler`` writes traces.
    """
    payload = {
        "schemaVersion": 1,
        "distributedInfo": {"backend": "nccl", "rank": 0, "world_size": 1},
        "traceEvents": [
            {
                "ph": "X",
                "cat": "kernel",
                "name": "void gemm_kernel<float>(float*, float*)",
                "pid": 1,
                "tid": 7,
                "ts": 100,
                "dur": 42,
                "args": {"stream": 7, "grid": [8, 1, 1], "block": [256, 1, 1]},
            },
            {
                "ph": "X",
                "cat": "gpu_memcpy",
                "name": "Memcpy DtoH",
                "pid": 1,
                "tid": 7,
                "ts": 200,
                "dur": 8,
                "args": {"bytes": 4096},
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_rocprof_json(path: Path) -> Path:
    """Write a rocprofv3 JSON result document, as ``rocprofv3_lightweight`` does.

    Shaped like the real output down to the ``buffer_records`` TraceLens reads
    kernel dispatches from, but structural only: see the module docstring.
    """
    payload = {
        "rocprofiler-sdk-tool": [
            {
                "metadata": {"pid": 1234},
                "agents": [{"id": {"handle": 0}, "type": 2, "name": "gfx942"}],
                "kernel_symbols": [
                    {"id": 1, "formatted_kernel_name": "gemm_kernel(float*, float*)"}
                ],
                "buffer_records": {
                    "kernel_dispatch": [
                        {
                            "correlation_id": {"internal": 1},
                            "start_timestamp": 1000,
                            "end_timestamp": 43000,
                            "dispatch_info": {
                                "kernel_id": 1,
                                "agent_id": {"handle": 0},
                            },
                        }
                    ],
                    "memory_copy": [],
                },
            }
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_pftrace(path: Path) -> Path:
    """Write a placeholder Perfetto trace, as ``rocprofv3_perfetto`` does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x0a\x0fdummy-perfetto-trace")
    return path


def write_rocprof_db(path: Path) -> Path:
    """Write rocprofv3's default SQLite output, which TraceLens cannot read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"SQLite format 3\x00")
    return path


@pytest.fixture
def profiled_run(tmp_path):
    """A directory shaped like the one a profiled madengine run leaves behind.

    Holds one artifact of every kind madengine's profiling tools can produce,
    including the SQLite output TraceLens cannot read.
    """
    work = tmp_path / "workdir"
    write_chrome_trace(work / "torch_profiler_output" / "libkineto_trace_1234.json")
    write_rocprof_json(work / "rocprof_output" / "1234_results.json")
    write_pftrace(work / "rocprof_output" / "model.pftrace")
    write_rocprof_db(work / "rocprof_output" / "1234_results.db")
    return work


def run_analyzer(
    work: Path, dummy: DummyTraceLens, *extra: str, output_dir: str = "tracelens_output"
) -> subprocess.CompletedProcess:
    """Run the packaged analyzer from ``work``, as the post-script does."""
    return subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--root",
            ".",
            "--output-dir",
            output_dir,
            "--python",
            str(dummy.python),
            "--json-summary",
            f"{output_dir}/tracelens_summary.json",
            *extra,
        ],
        cwd=work,
        env=dummy.environ(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )


def summary_rows(work: Path, output_dir: str = "tracelens_output") -> list:
    """Read the analyzer's summary CSV into a list of dict rows."""
    path = work / output_dir / "tracelens_summary.csv"
    assert path.is_file(), f"analyzer wrote no summary at {path}"
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def only(records: list, entry_point: str) -> dict:
    """Return the single recorded invocation of ``entry_point``."""
    matches = [r for r in records if r["entry_point"] == entry_point]
    assert len(matches) == 1, f"expected one {entry_point} call, got {len(matches)}"
    return matches[0]


class TestAnalyzerWithDummyTraceLens:
    """Every trace kind a run can produce, analyzed by the real analyzer."""

    def test_each_trace_kind_is_routed_to_its_report_generator(
        self, profiled_run, dummy_tracelens
    ):
        """A Kineto trace, a rocprofv3 JSON, and a pftrace pick different reports."""
        result = run_analyzer(profiled_run, dummy_tracelens)

        assert result.returncode == 0, result.stdout
        assert sorted(dummy_tracelens.reports_run()) == sorted(
            [PYTORCH_REPORT, ROCPROF_REPORT, *PFTRACE_REPORTS]
        )
        records = dummy_tracelens.invocations()
        assert only(records, PYTORCH_REPORT)["args"]["profile_json_path"].endswith(
            "libkineto_trace_1234.json"
        )
        assert only(records, ROCPROF_REPORT)["args"]["profile_json_path"].endswith(
            "1234_results.json"
        )
        for report in PFTRACE_REPORTS:
            assert only(records, report)["args"]["trace_path"].endswith(".pftrace")

    def test_reports_are_written_into_the_output_directory(
        self, profiled_run, dummy_tracelens
    ):
        """Each report generator's workbook and per-sheet CSVs land together."""
        run_analyzer(profiled_run, dummy_tracelens)

        output_dir = profiled_run / "tracelens_output"
        written = sorted(p.name for p in output_dir.iterdir())
        assert "tracelens_summary.csv" in written
        assert "tracelens_summary.json" in written
        # One workbook plus one CSV directory per PyTorch trace analyzed.
        pytorch_stem = "torch_profiler_output_libkineto_trace_1234"
        assert (output_dir / f"{pytorch_stem}.xlsx").is_file(), written
        assert (output_dir / f"{pytorch_stem}_csv" / "kernel_summary.csv").is_file()
        # The pftrace activity report is markdown rather than a workbook.
        assert list(output_dir.glob("*_activity.md")), written

    def test_summary_records_each_report_and_the_unreadable_db(
        self, profiled_run, dummy_tracelens
    ):
        """The summary is the run's record: five reports, one skipped artifact."""
        run_analyzer(profiled_run, dummy_tracelens)

        rows = summary_rows(profiled_run)
        assert [r["status"] for r in rows].count("SUCCESS") == 5
        skipped = [r for r in rows if r["status"] == "SKIPPED"]
        assert len(skipped) == 1
        assert skipped[0]["trace_file"].endswith("1234_results.db")
        # The skip has to say which preset the user should have run instead.
        assert "rocprofv3_lightweight" in skipped[0]["detail"]

    def test_trace_with_undecodable_bytes_is_still_analyzed(
        self, tmp_path, dummy_tracelens
    ):
        """rocprofv3 traces are not always valid UTF-8, and TraceLens demands it.

        rocprofv3 copies HIP API ``const char *`` arguments into its JSON as they
        are, so an argument that does not point at a string leaves raw bytes in an
        otherwise perfectly good 300MB trace. TraceLens loads traces with orjson,
        which rejects the whole document over those few bytes.
        """
        work = tmp_path / "raw-bytes"
        trace = write_rocprof_json(work / "rocprof_output" / "510_results.json")
        original = trace.read_bytes().replace(
            b'"gemm_kernel(float*, float*)"', b'"\x90{-\xad\xcb\x7f"'
        )
        trace.write_bytes(original)

        result = run_analyzer(work, dummy_tracelens)

        assert result.returncode == 0, result.stdout
        rows = summary_rows(work)
        assert [r["status"] for r in rows] == ["SUCCESS"], rows
        # Analyzed through a copy, so the trace the run collected is untouched.
        assert trace.read_bytes() == original
        analyzed = only(dummy_tracelens.invocations(), ROCPROF_REPORT)["args"][
            "profile_json_path"
        ]
        assert analyzed != str(trace)
        assert "sanitized copy" in result.stdout

    def test_a_valid_trace_is_analyzed_where_it_lies(self, profiled_run, dummy_tracelens):
        """Traces TraceLens can already read are not copied; they can be huge."""
        result = run_analyzer(profiled_run, dummy_tracelens)

        assert result.returncode == 0, result.stdout
        analyzed = only(dummy_tracelens.invocations(), ROCPROF_REPORT)["args"][
            "profile_json_path"
        ]
        assert analyzed == str(profiled_run / "rocprof_output" / "1234_results.json")
        assert "sanitized copy" not in result.stdout

    def test_gzipped_kineto_trace_is_analyzed(self, tmp_path, dummy_tracelens):
        """tensorboard_trace_handler's gzipped traces are picked up too."""
        work = tmp_path / "gz"
        write_chrome_trace(work / "traces" / "worker0.pt.trace.json.gz")

        result = run_analyzer(work, dummy_tracelens)

        assert result.returncode == 0, result.stdout
        record = only(dummy_tracelens.invocations(), PYTORCH_REPORT)
        assert record["args"]["profile_json_path"].endswith(".pt.trace.json.gz")

    def test_per_rank_traces_add_a_collective_report(self, tmp_path, dummy_tracelens):
        """Several ranks' traces also produce the multi-rank collective report."""
        work = tmp_path / "distributed"
        for rank in (0, 1):
            write_chrome_trace(
                work / "torch_profiler_output" / f"libkineto_trace_rank{rank}_99.json"
            )

        result = run_analyzer(work, dummy_tracelens)

        assert result.returncode == 0, result.stdout
        assert dummy_tracelens.reports_run().count(PYTORCH_REPORT) == 2
        collective = only(dummy_tracelens.invocations(), COLLECTIVE_REPORT)
        # The dummy fails unless the glob really matched both per-rank traces.
        assert collective["args"]["world_size"] == "2"
        assert collective["args"]["use_multiprocessing"] is True

    def test_single_rank_run_skips_the_collective_report(
        self, profiled_run, dummy_tracelens
    ):
        """One rank cannot have collectives to compare, so that report is skipped."""
        run_analyzer(profiled_run, dummy_tracelens)

        assert COLLECTIVE_REPORT not in dummy_tracelens.reports_run()

    def test_gpu_arch_enables_roofline_classification(
        self, profiled_run, dummy_tracelens
    ):
        """--gpu-arch reaches the PyTorch report as TraceLens' arch flag."""
        run_analyzer(profiled_run, dummy_tracelens, "--gpu-arch", "MI300X")

        record = only(dummy_tracelens.invocations(), PYTORCH_REPORT)
        assert record["args"]["gpu_arch_platform"] == "MI300X"

    def test_mode_restricts_analysis_to_one_trace_kind(
        self, profiled_run, dummy_tracelens
    ):
        """--mode rocprof leaves the Kineto trace and the pftrace alone."""
        run_analyzer(profiled_run, dummy_tracelens, "--mode", "rocprof")

        assert dummy_tracelens.reports_run() == [ROCPROF_REPORT]

    def test_extra_flags_are_forwarded_to_every_report(
        self, profiled_run, dummy_tracelens
    ):
        """Flags after ``--`` reach TraceLens untouched, for options we do not wrap."""
        run_analyzer(profiled_run, dummy_tracelens, "--", "--top_k_kernels", "5")

        for record in dummy_tracelens.invocations():
            assert record["extra"] == ["--top_k_kernels", "5"], record["entry_point"]

    def test_console_script_from_the_venv_is_preferred(
        self, profiled_run, dummy_tracelens
    ):
        """With TraceLens' console scripts installed, they are what we run."""
        run_analyzer(profiled_run, dummy_tracelens)

        record = only(dummy_tracelens.invocations(), PYTORCH_REPORT)
        assert record["argv0"] == str(dummy_tracelens.root / "bin" / PYTORCH_REPORT)

    def test_module_fallback_still_reports_failures(self, tmp_path, profiled_run):
        """Without console scripts we import the module, and still see its exit code.

        The fallback runs ``main()`` in a fresh interpreter; if its return value
        were dropped, every failed report would be recorded as a success.
        """
        dummy = DummyTraceLens(tmp_path / "no-console-scripts", console_scripts=False)
        dummy.fail(PYTORCH_REPORT)

        result = run_analyzer(profiled_run, dummy)

        assert result.returncode != 0, result.stdout
        assert only(dummy.invocations(), PYTORCH_REPORT)["argv0"] == "-c"
        pytorch_rows = [r for r in summary_rows(profiled_run) if r["kind"] == "pytorch"]
        assert [r["status"] for r in pytorch_rows] == ["FAILURE"]
        assert "forced failure" in pytorch_rows[0]["detail"]

    def test_one_failed_report_does_not_hide_the_others(
        self, profiled_run, dummy_tracelens
    ):
        """A failing report is recorded as such; the rest still run and succeed."""
        dummy_tracelens.fail(ROCPROF_REPORT)

        result = run_analyzer(profiled_run, dummy_tracelens)

        assert result.returncode != 0, result.stdout
        statuses = {
            r["tracelens_tool"]: r["status"] for r in summary_rows(profiled_run)
        }
        assert statuses[ROCPROF_REPORT] == "FAILURE"
        assert statuses[PYTORCH_REPORT] == "SUCCESS"
        assert all(statuses[report] == "SUCCESS" for report in PFTRACE_REPORTS)

    def test_run_without_traces_says_which_tool_to_stack(
        self, tmp_path, dummy_tracelens
    ):
        """No traces is a no-op with guidance, not a failure."""
        work = tmp_path / "empty"
        work.mkdir()

        result = run_analyzer(work, dummy_tracelens)

        assert result.returncode == 0, result.stdout
        assert "No supported trace artifacts found" in result.stdout
        assert "torch_profiler_dynolog" in result.stdout
        assert dummy_tracelens.reports_run() == []
        assert summary_rows(work) == []


class TestInContainerPostScript:
    """The `tracelens` tool's post-script, run the way a container runs it."""

    @staticmethod
    def stage_scripts(work: Path) -> None:
        """Copy the analyzer to where ContainerRunner puts it during a run."""
        tools_dir = work / "scripts" / "common" / "tools"
        tools_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ANALYZER, tools_dir / ANALYZER.name)

    @classmethod
    def run_post_script(
        cls, work: Path, dummy: DummyTraceLens, **env: str
    ) -> subprocess.CompletedProcess:
        cls.stage_scripts(work)
        return subprocess.run(
            ["bash", str(POST_SCRIPT)],
            cwd=work,
            env=dummy.environ(**env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )

    def test_post_script_analyzes_the_traces_the_run_produced(
        self, profiled_run, dummy_tracelens
    ):
        """The tool needs no configuration: it finds the traces and reports on them."""
        result = self.run_post_script(profiled_run, dummy_tracelens)

        assert result.returncode == 0, result.stdout
        assert sorted(dummy_tracelens.reports_run()) == sorted(
            [PYTORCH_REPORT, ROCPROF_REPORT, *PFTRACE_REPORTS]
        )
        # tracelens_output/ is the directory the post-script hands to collection.
        assert (profiled_run / "tracelens_output" / "tracelens_summary.csv").is_file()
        assert (profiled_run / "tracelens_output" / "tracelens_summary.json").is_file()

    def test_failed_analysis_does_not_fail_the_model_run(
        self, profiled_run, dummy_tracelens
    ):
        """Reporting is not the workload: TraceLens failing must not fail the run."""
        dummy_tracelens.fail("all")

        result = self.run_post_script(profiled_run, dummy_tracelens)

        assert result.returncode == 0, result.stdout
        assert "WARNING: TraceLens analysis reported failures" in result.stdout
        statuses = {r["status"] for r in summary_rows(profiled_run)}
        assert "FAILURE" in statuses

    def test_missing_venv_fails_loudly(self, profiled_run, dummy_tracelens, tmp_path):
        """A missing venv means the pre-script never ran, which is worth failing on."""
        result = self.run_post_script(
            profiled_run, dummy_tracelens, TRACELENS_VENV=str(tmp_path / "absent")
        )

        assert result.returncode != 0
        assert "pre-script must run first" in result.stdout

    def test_mode_and_output_dir_are_configurable_by_env_var(
        self, profiled_run, dummy_tracelens
    ):
        """The tool's env vars are how a user narrows analysis in a models.json."""
        result = self.run_post_script(
            profiled_run,
            dummy_tracelens,
            TRACELENS_MODE="pytorch",
            TRACELENS_OUTPUT_DIR="custom_tracelens",
        )

        assert result.returncode == 0, result.stdout
        assert dummy_tracelens.reports_run() == [PYTORCH_REPORT]
        rows = summary_rows(profiled_run, output_dir="custom_tracelens")
        assert [r["kind"] for r in rows] == ["pytorch", "unsupported"]

    def test_gpu_arch_and_world_size_reach_tracelens(self, tmp_path, dummy_tracelens):
        """Distributed runs pass rank count and arch through the tool's env vars."""
        work = tmp_path / "distributed"
        for rank in (0, 1):
            write_chrome_trace(
                work / "torch_profiler_output" / f"libkineto_trace_rank{rank}_99.json"
            )

        result = self.run_post_script(
            work,
            dummy_tracelens,
            TRACELENS_GPU_ARCH="MI300X",
            TRACELENS_WORLD_SIZE="2",
        )

        assert result.returncode == 0, result.stdout
        records = dummy_tracelens.invocations()
        assert only(records, COLLECTIVE_REPORT)["args"]["world_size"] == "2"
        assert all(
            r["args"]["gpu_arch_platform"] == "MI300X"
            for r in records
            if r["entry_point"] == PYTORCH_REPORT
        )


class TestHostSideReportCommand:
    """`madengine report tracelens` over artifacts already collected to the host."""

    @staticmethod
    def run_cli(dummy: DummyTraceLens, *args: str) -> subprocess.CompletedProcess:
        """Run the CLI in a wide, colourless console so assertions survive wrapping."""
        env = dummy.environ(
            COLUMNS="300", NO_COLOR="1", TERM="dumb", PYTHONIOENCODING="utf-8"
        )
        return subprocess.run(
            [sys.executable, "-m", "madengine.cli.app", "report", *args],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )

    def test_report_tracelens_analyzes_collected_artifacts(
        self, profiled_run, dummy_tracelens
    ):
        """The host path produces the same reports without touching the container."""
        output_dir = profiled_run / "host_reports"

        result = self.run_cli(
            dummy_tracelens,
            "tracelens",
            "--root",
            str(profiled_run),
            "--output-dir",
            str(output_dir),
        )

        assert result.returncode == 0, result.stdout
        assert sorted(dummy_tracelens.reports_run()) == sorted(
            [PYTORCH_REPORT, ROCPROF_REPORT, *PFTRACE_REPORTS]
        )
        assert (output_dir / "tracelens_summary.csv").is_file()
        # The rendered table is how a user sees what was analyzed.
        assert "SUCCESS" in result.stdout
        assert "SKIPPED" in result.stdout

    def test_report_tracelens_compare_diffs_two_reports(
        self, profiled_run, dummy_tracelens
    ):
        """tracelens-compare turns two runs' reports into one comparison workbook."""
        output_dir = profiled_run / "host_reports"
        self.run_cli(
            dummy_tracelens,
            "tracelens",
            "--root",
            str(profiled_run),
            "--output-dir",
            str(output_dir),
        )
        reports = sorted(output_dir.glob("*.xlsx"))
        assert len(reports) >= 2, [p.name for p in output_dir.iterdir()]
        comparison = profiled_run / "comparison.xlsx"

        result = self.run_cli(
            dummy_tracelens,
            "tracelens-compare",
            str(reports[0]),
            str(reports[1]),
            "--output",
            str(comparison),
            "--names",
            "baseline",
            "--names",
            "candidate",
        )

        assert result.returncode == 0, result.stdout
        assert comparison.is_file()
        record = only(
            dummy_tracelens.invocations(), "TraceLens_compare_perf_reports_pytorch"
        )
        assert record["args"]["names"] == ["baseline", "candidate"]
