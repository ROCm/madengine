"""End-to-end tests for the TraceLens integration.

Two independent execution paths are covered:

* the host-side reporting path (``madengine report tracelens``), which needs
  neither Docker nor a GPU and therefore runs everywhere
* the in-container tool path (``tracelens`` / ``torch_profiler_dynolog``
  stacked onto a run), which needs Docker and a GPU

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import csv
import gzip
import json
import os
import subprocess
import sys

# third-party modules
import pytest

# project modules
from tests.fixtures.utils import (
    BASE_DIR,
    DEFAULT_CLEAN_FILES,
    build_run_command,
    clean_test_temp_files,
    global_data,
    is_nvidia,
    requires_gpu,
)
from madengine.reporting.tracelens_report import (
    check_tracelens_available,
    resolve_python,
)


def tracelens_installed() -> bool:
    """Return True when the interpreter running the tests can import TraceLens."""
    try:
        return check_tracelens_available(resolve_python())
    except Exception:
        return False


def run_report_cli(*args: str) -> subprocess.CompletedProcess:
    """Run ``madengine report ...`` in a wide, colourless console.

    Rich wraps and colourises output based on the terminal, which would make
    substring assertions brittle. A wide COLUMNS plus NO_COLOR keeps the text
    intact, and callers still normalise whitespace via :func:`flatten`.
    PYTHONIOENCODING is needed because the CLI prints emoji, which a piped
    stdout on a non-UTF-8 Windows console cannot encode.
    """
    env = dict(
        os.environ,
        COLUMNS="300",
        NO_COLOR="1",
        TERM="dumb",
        PYTHONIOENCODING="utf-8",
    )
    return subprocess.run(
        [sys.executable, "-m", "madengine.cli.app", "report", *args],
        cwd=BASE_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )


def flatten(text: str) -> str:
    """Collapse all whitespace so assertions survive console line wrapping."""
    return " ".join(text.split())


def run_context(tools: list) -> dict:
    """Return an additional-context dict selecting ``tools`` on an AMD host."""
    return {"gpu_vendor": "AMD", "guest_os": "UBUNTU", "tools": tools}


def write_pytorch_trace(path) -> None:
    """Write a minimal Chrome Trace Event document, as Kineto would."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schemaVersion": 1,
        "traceEvents": [
            {
                "ph": "X",
                "cat": "kernel",
                "name": "void gemm_kernel<float>(float*)",
                "pid": 1,
                "tid": 7,
                "ts": 100,
                "dur": 42,
                "args": {"stream": 7, "grid": [8, 1, 1], "block": [256, 1, 1]},
            }
        ],
    }
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


def summary_rows(summary_csv: str) -> list:
    """Read the analyzer summary CSV into a list of dict rows."""
    with open(summary_csv, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestTraceLensHostSideReporting:
    """`madengine report tracelens` against traces already on the filesystem."""

    def test_discover_only_classifies_every_trace_kind(self, tmp_path):
        """Discovery reports each trace kind and explains unreadable formats."""
        write_pytorch_trace(tmp_path / "torch_profiler_output" / "libkineto_trace.json")
        rocprof_dir = tmp_path / "rocprof_output"
        rocprof_dir.mkdir()
        (rocprof_dir / "run_results.json").write_text(
            json.dumps({"rocprofiler-sdk-tool": []}), encoding="utf-8"
        )
        (rocprof_dir / "run.pftrace").write_bytes(b"\x0a\x00perfetto-ish")
        # rocprofv3's default SQLite output: found, but unusable by TraceLens.
        (rocprof_dir / "run_results.db").write_bytes(b"SQLite format 3\x00")

        result = run_report_cli(
            "tracelens",
            "--root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "tracelens_output"),
            "--discover-only",
        )

        assert result.returncode == 0, result.stdout
        output = flatten(result.stdout)
        assert "pytorch: 1 trace(s)" in output
        assert "rocprof_json: 1 trace(s)" in output
        assert "pftrace: 1 trace(s)" in output
        # The .db is surfaced with a pointer at the presets that do work.
        assert "unsupported" in output
        assert "rocprofv3_lightweight" in output

    def test_discover_only_reports_empty_root(self, tmp_path):
        """An empty search root is a warning, not a failure."""
        result = run_report_cli(
            "tracelens",
            "--root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "tracelens_output"),
            "--discover-only",
        )

        assert result.returncode == 0, result.stdout
        assert "No trace artifacts found" in flatten(result.stdout)

    def test_missing_root_fails_with_clear_error(self, tmp_path):
        """A nonexistent search root fails before any analysis is attempted."""
        result = run_report_cli(
            "tracelens", "--root", str(tmp_path / "nope"), "--discover-only"
        )

        assert result.returncode != 0
        assert "directory not found" in flatten(result.stdout)

    def test_invalid_mode_is_rejected(self, tmp_path):
        """--mode is validated against the supported trace kinds."""
        result = run_report_cli("tracelens", "--root", str(tmp_path), "--mode", "bogus")

        assert result.returncode != 0
        output = flatten(result.stdout)
        assert "invalid --mode" in output
        assert "collective" in output

    @pytest.mark.skipif(
        tracelens_installed(), reason="test covers the TraceLens-missing path"
    )
    def test_analysis_without_tracelens_explains_how_to_install(self, tmp_path):
        """Without TraceLens the command fails with install guidance, not a crash."""
        write_pytorch_trace(tmp_path / "torch_profiler_output" / "libkineto_trace.json")

        result = run_report_cli(
            "tracelens",
            "--root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "tracelens_output"),
        )

        assert result.returncode != 0
        output = flatten(result.stdout)
        assert "TraceLens is not importable" in output
        assert "madengine[tracelens]" in output

    def test_compare_rejects_missing_reports(self, tmp_path):
        """tracelens-compare validates its inputs before invoking TraceLens."""
        existing = tmp_path / "baseline.xlsx"
        existing.write_bytes(b"not really a workbook")

        result = run_report_cli(
            "tracelens-compare",
            str(existing),
            str(tmp_path / "missing.xlsx"),
            "--output",
            str(tmp_path / "diff.xlsx"),
        )

        assert result.returncode != 0
        output = flatten(result.stdout)
        assert "not found" in output
        assert "missing.xlsx" in output

    @pytest.mark.skipif(
        tracelens_installed(), reason="test covers the TraceLens-missing path"
    )
    def test_compare_without_tracelens_explains_how_to_install(self, tmp_path):
        """tracelens-compare surfaces the same install guidance as analysis."""
        first = tmp_path / "a.xlsx"
        second = tmp_path / "b.xlsx"
        for report in (first, second):
            report.write_bytes(b"not really a workbook")

        result = run_report_cli(
            "tracelens-compare",
            str(first),
            str(second),
            "--output",
            str(tmp_path / "diff.xlsx"),
        )

        assert result.returncode != 0
        assert "madengine[tracelens]" in flatten(result.stdout)

    @pytest.mark.skipif(
        not tracelens_installed(), reason="requires pip install 'madengine[tracelens]'"
    )
    def test_analysis_generates_reports_for_a_pytorch_trace(self, tmp_path):
        """With TraceLens installed, a Kineto trace yields a summarised report."""
        write_pytorch_trace(tmp_path / "torch_profiler_output" / "libkineto_trace.json")
        output_dir = tmp_path / "tracelens_output"

        result = run_report_cli(
            "tracelens",
            "--root",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
        )

        summary_csv = output_dir / "tracelens_summary.csv"
        assert summary_csv.is_file(), result.stdout
        rows = summary_rows(str(summary_csv))
        assert [r for r in rows if r["kind"] == "pytorch"], rows
        assert (output_dir / "tracelens_summary.json").is_file()


@requires_gpu("in-container TraceLens tools require GPU hardware")
@pytest.mark.skipif(is_nvidia(), reason="TraceLens targets AMD GPU traces")
@pytest.mark.slow
class TestTraceLensContainerTools:
    """The `tracelens` and `torch_profiler_dynolog` tools stacked onto a run."""

    @pytest.mark.parametrize(
        "clean_test_temp_files",
        [DEFAULT_CLEAN_FILES + ["rocprof_output", "tracelens_output"]],
        indirect=True,
    )
    def test_tracelens_analyzes_rocprofv3_json_traces(
        self, global_data, clean_test_temp_files
    ):
        """rocprofv3 JSON output is analyzed in-container and collected to cwd."""
        global_data["console"].sh(
            build_run_command(
                "dummy_prof",
                additional_context=run_context(
                    [{"name": "rocprofv3_lightweight"}, {"name": "tracelens"}]
                ),
            ),
            canFail=True,
        )

        summary_csv = os.path.join(
            BASE_DIR, "tracelens_output", "tracelens_summary.csv"
        )
        if not os.path.isfile(summary_csv):
            pytest.fail(
                "tracelens_output/tracelens_summary.csv not collected when stacking "
                "tracelens onto rocprofv3_lightweight."
            )
        rows = summary_rows(summary_csv)
        if not [r for r in rows if r["kind"] == "rocprof_json"]:
            pytest.fail(f"no rocprofv3 JSON trace was analyzed; summary rows: {rows}")

    @pytest.mark.parametrize(
        "clean_test_temp_files",
        [DEFAULT_CLEAN_FILES + ["rocprof_output", "tracelens_output"]],
        indirect=True,
    )
    def test_tracelens_explains_unreadable_rocprofv3_db_output(
        self, global_data, clean_test_temp_files
    ):
        """The default .db output is reported as skipped with actionable guidance."""
        global_data["console"].sh(
            build_run_command(
                "dummy_prof",
                additional_context=run_context(
                    [{"name": "rocprofv3"}, {"name": "tracelens"}]
                ),
            ),
            canFail=True,
        )

        summary_csv = os.path.join(
            BASE_DIR, "tracelens_output", "tracelens_summary.csv"
        )
        if not os.path.isfile(summary_csv):
            pytest.fail(
                "tracelens_output/tracelens_summary.csv not collected when stacking "
                "tracelens onto rocprofv3."
            )
        skipped = [r for r in summary_rows(summary_csv) if r["status"] == "SKIPPED"]
        if not skipped:
            pytest.fail(
                "rocprofv3 .db output should be reported as SKIPPED with guidance."
            )
        if not any("rocprofv3_lightweight" in r["detail"] for r in skipped):
            pytest.fail(f"skip reason does not point at a usable preset: {skipped}")

    @pytest.mark.parametrize(
        "clean_test_temp_files",
        [DEFAULT_CLEAN_FILES + ["torch_profiler_output"]],
        indirect=True,
    )
    def test_dynolog_collects_a_kineto_trace_from_pytorch(
        self, global_data, clean_test_temp_files
    ):
        """torch_profiler_dynolog captures an on-demand trace from a PyTorch run.

        The warmup is shortened from the 60s default because the fixture workload
        is far shorter lived than a real training job.
        """
        global_data["console"].sh(
            build_run_command(
                "dummy_torchrun",
                additional_context=run_context(
                    [
                        {
                            "name": "torch_profiler_dynolog",
                            "env_vars": {
                                "TORCH_PROFILE_WARMUP_S": "20",
                                "TORCH_PROFILE_RETRY_INTERVAL_S": "5",
                                "TORCH_PROFILE_MAX_ATTEMPTS": "10",
                            },
                        }
                    ]
                ),
            ),
            canFail=True,
        )

        output_dir = os.path.join(BASE_DIR, "torch_profiler_output")
        if not os.path.isdir(output_dir):
            pytest.fail(
                "torch_profiler_output/ not collected with the "
                "torch_profiler_dynolog tool."
            )
        collected = os.listdir(output_dir)
        traces = [f for f in collected if f.endswith((".json", ".json.gz"))]
        if not traces:
            pytest.fail(
                "no Kineto trace captured; dynolog never matched a PyTorch process "
                f"(torch_profiler_output/ contains {collected})."
            )
