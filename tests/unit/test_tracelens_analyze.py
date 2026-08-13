"""Unit tests for the TraceLens trace analyzer script.

The analyzer is shipped as a standalone stdlib-only script under
``scripts/common/tools/`` so it can run both inside a workload container and on
the host, so it is loaded here by path rather than imported as a module.
"""

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from madengine.utils.path_utils import get_madengine_root

CHROME_TRACE = b'{"traceEvents": [], "schemaVersion": 1}'


def _load_analyzer():
    script = (
        get_madengine_root() / "scripts" / "common" / "tools" / "tracelens_analyze.py"
    )
    assert script.is_file(), f"analyzer script missing at {script}"
    spec = importlib.util.spec_from_file_location("tracelens_analyze", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def analyzer():
    return _load_analyzer()


def _write(root: Path, rel: str, content: bytes = b"{}") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


@pytest.fixture
def trace_tree(tmp_path):
    """A directory shaped like the working directory after a profiled run."""
    _write(tmp_path, "torch_profiler_output/libkineto_trace_rank0_1.json", CHROME_TRACE)
    _write(tmp_path, "torch_profiler_output/libkineto_trace_rank1_2.json", CHROME_TRACE)
    _write(tmp_path, "traces/model_rank2.pt.trace.json", CHROME_TRACE)
    _write(tmp_path, "rocprof_output/9999_results.json", b"{}")
    _write(tmp_path, "rocprof_output/model_trace.pftrace", b"\x00\x01")
    _write(tmp_path, "rocprof_output/1e4d92661463/1234_results.db", b"sqlite")
    _write(tmp_path, "rpd_output/trace.rpd", b"sqlite")
    _write(tmp_path, "rpd_output/trace.json", CHROME_TRACE)
    _write(tmp_path, "perf.csv", b"model,performance\n")
    return tmp_path


class TestDiscovery:
    """Trace classification must route each artifact to the right TraceLens reader."""

    def test_classifies_each_trace_kind(self, analyzer, trace_tree):
        traces, _ = analyzer.discover_traces(str(trace_tree))
        names = {
            kind: sorted(Path(p).name for p in paths) for kind, paths in traces.items()
        }

        assert names[analyzer.KIND_PYTORCH] == [
            "libkineto_trace_rank0_1.json",
            "libkineto_trace_rank1_2.json",
            "model_rank2.pt.trace.json",
            "trace.json",
        ]
        assert names[analyzer.KIND_ROCPROF_JSON] == ["9999_results.json"]
        assert names[analyzer.KIND_PFTRACE] == ["model_trace.pftrace"]

    def test_reports_unreadable_formats_with_guidance(self, analyzer, trace_tree):
        _, unsupported = analyzer.discover_traces(str(trace_tree))
        by_name = {Path(p).name: reason for p, reason in unsupported}

        assert "1234_results.db" in by_name
        assert "--output-format" in by_name["1234_results.db"]
        assert "trace.rpd" in by_name

    def test_sniffs_ambiguous_trace_json(self, analyzer, tmp_path):
        """trace.json is written by both rpd (Chrome trace) and rocm-trace-lite."""
        _write(tmp_path, "rpd_output/trace.json", CHROME_TRACE)
        _write(tmp_path, "other_output/trace.json", b'{"not": "a trace"}')

        traces, _ = analyzer.discover_traces(str(tmp_path))
        claimed = [Path(p).parent.name for p in traces.get(analyzer.KIND_PYTORCH, [])]
        assert claimed == ["rpd_output"]

    def test_excludes_report_output_directory(self, analyzer, tmp_path):
        """Re-running analysis must not treat previous reports as new inputs."""
        _write(tmp_path, "torch_profiler_output/libkineto_trace_1.json", CHROME_TRACE)
        _write(tmp_path, "tracelens_output/stale_results.json", b"{}")

        traces, unsupported = analyzer.discover_traces(
            str(tmp_path), exclude_dirs=[str(tmp_path / "tracelens_output")]
        )
        assert analyzer.KIND_ROCPROF_JSON not in traces
        assert unsupported == []

    def test_empty_tree_discovers_nothing(self, analyzer, tmp_path):
        traces, unsupported = analyzer.discover_traces(str(tmp_path))
        assert traces == {}
        assert unsupported == []


class TestCommandConstruction:
    """TraceLens must be invoked with the flags each report generator expects."""

    def test_falls_back_to_module_when_console_script_absent(self, analyzer):
        command = analyzer._build_command(
            "/nonexistent/bin/python3",
            "TraceLens_generate_perf_report_pytorch",
            ["--profile_json_path", "trace.json"],
        )
        assert command[0] == "/nonexistent/bin/python3"
        assert command[1] == "-c"
        assert "TraceLens.Reporting.generate_perf_report_pytorch" in command[2]
        assert command[-2:] == ["--profile_json_path", "trace.json"]

    def test_prefers_console_script_in_interpreter_bindir(self, analyzer, tmp_path):
        bindir = tmp_path / "bin"
        bindir.mkdir()
        python = bindir / "python3"
        python.write_text("")
        script = bindir / "TraceLens_generate_perf_report_rocprof"
        script.write_text("")
        script.chmod(0o755)

        command = analyzer._build_command(
            str(python), "TraceLens_generate_perf_report_rocprof", ["--x"]
        )
        assert command == [str(script), "--x"]

    def test_every_entry_point_has_a_module_fallback(self, analyzer):
        for name, module in analyzer._ENTRY_POINTS.items():
            assert name.startswith("TraceLens_")
            assert module.startswith("TraceLens.Reporting.")

    def test_pytorch_args_request_shapes_and_roofline(self, analyzer):
        args = analyzer._pytorch_args("t.json", "/out/t", "MI300X", [])
        assert args[:2] == ["--profile_json_path", "t.json"]
        assert "--output_csvs_dir" in args
        assert args[args.index("--gpu_arch_platform") + 1] == "MI300X"

    def test_pytorch_args_omit_roofline_without_arch(self, analyzer):
        args = analyzer._pytorch_args("t.json", "/out/t", None, [])
        assert "--gpu_arch_platform" not in args

    def test_pftrace_produces_three_complementary_reports(self, analyzer):
        jobs = analyzer._pftrace_jobs("t.pftrace", "/out/t", [])
        assert [tool for tool, _ in jobs] == [
            "TraceLens_generate_perf_report_pftrace_hip_activity",
            "TraceLens_generate_perf_report_pftrace_hip_api",
            "TraceLens_generate_perf_report_pftrace_memory_copy",
        ]
        for _, args in jobs:
            assert args[:2] == ["--trace_path", "t.pftrace"]

    def test_collective_args_carry_rank_regex_and_world_size(self, analyzer):
        traces = [
            "/root/torch_profiler_output/libkineto_trace_rank0_1.json",
            "/root/torch_profiler_output/libkineto_trace_rank1_2.json",
        ]
        args = analyzer._collective_args(traces, "/out/coll", 8, [])
        assert args[args.index("--world_size") + 1] == "8"
        assert "rank" in args[args.index("--rank_regex") + 1]
        # Scoped to the traces' own directory: a wider glob sweeps up unrelated
        # JSON, and rocprofv3 results are hundreds of megabytes each.
        trace_glob = args[args.index("--trace_glob") + 1]
        assert "torch_profiler_output" in trace_glob
        assert trace_glob.endswith(os.path.join("**", "*.json*"))

    def test_extra_args_are_forwarded(self, analyzer):
        args = analyzer._pytorch_args("t.json", "/out/t", None, ["--detect_recompute"])
        assert args[-1] == "--detect_recompute"


class TestReportStem:
    """Report names must be unique per trace and safe as filenames."""

    def test_strips_known_trace_suffixes(self, analyzer, tmp_path):
        stem = analyzer._report_stem(
            str(tmp_path / "traces" / "model_rank0.pt.trace.json"), str(tmp_path)
        )
        assert stem == "traces_model_rank0"

    def test_distinguishes_same_name_in_different_directories(self, analyzer, tmp_path):
        a = analyzer._report_stem(str(tmp_path / "node_0" / "trace.json"), str(tmp_path))
        b = analyzer._report_stem(str(tmp_path / "node_1" / "trace.json"), str(tmp_path))
        assert a != b


class TestAnalyze:
    """The analyze() driver must schedule one job per trace and record outcomes."""

    def test_schedules_a_job_for_every_trace_and_writes_summary(
        self, analyzer, trace_tree, monkeypatch
    ):
        calls = []

        def fake_run(command, cwd=None):
            calls.append(list(command))
            return 0, ""

        monkeypatch.setattr(analyzer, "_run", fake_run)
        out = trace_tree / "tracelens_output"
        summary = analyzer.analyze(
            root=str(trace_tree), output_dir=str(out), python=sys.executable
        )

        # 4 pytorch + 1 rocprof + 3 pftrace + 1 collective
        assert len(calls) == 9
        assert summary["succeeded"] == 9
        assert summary["failed"] == 0
        # The unreadable .db and .rpd artifacts are surfaced as skipped.
        assert summary["skipped"] == 2

        with open(summary["summary_csv"], newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 11
        assert set(rows[0]) == set(analyzer.SUMMARY_CSV_FIELDS)

    def test_records_failure_detail(self, analyzer, tmp_path, monkeypatch):
        _write(tmp_path, "rocprof_output/1_results.json", b"{}")
        monkeypatch.setattr(
            analyzer, "_run", lambda command, cwd=None: (1, "boom\nNot a valid file")
        )

        summary = analyzer.analyze(
            root=str(tmp_path), output_dir=str(tmp_path / "out"), python=sys.executable
        )
        assert summary["failed"] == 1
        assert summary["results"][0]["detail"] == "Not a valid file"

    def test_mode_restricts_to_one_trace_kind(self, analyzer, trace_tree, monkeypatch):
        calls = []
        monkeypatch.setattr(
            analyzer,
            "_run",
            lambda command, cwd=None: (calls.append(list(command)), (0, ""))[1],
        )

        analyzer.analyze(
            root=str(trace_tree),
            output_dir=str(trace_tree / "out"),
            mode="rocprof",
            python=sys.executable,
        )
        assert len(calls) == 1
        assert any("generate_perf_report_rocprof" in part for part in calls[0])

    def test_collective_mode_emits_only_the_multi_rank_report(
        self, analyzer, trace_tree, monkeypatch
    ):
        calls = []
        monkeypatch.setattr(
            analyzer,
            "_run",
            lambda command, cwd=None: (calls.append(list(command)), (0, ""))[1],
        )

        analyzer.analyze(
            root=str(trace_tree),
            output_dir=str(trace_tree / "out"),
            mode="collective",
            python=sys.executable,
        )
        assert len(calls) == 1
        assert any("multi_rank_collective_report" in part for part in calls[0])

    def test_no_collective_report_for_a_single_rank(self, analyzer, tmp_path, monkeypatch):
        _write(tmp_path, "torch_profiler_output/libkineto_trace_1.json", CHROME_TRACE)
        calls = []
        monkeypatch.setattr(
            analyzer,
            "_run",
            lambda command, cwd=None: (calls.append(list(command)), (0, ""))[1],
        )

        analyzer.analyze(
            root=str(tmp_path), output_dir=str(tmp_path / "out"), python=sys.executable
        )
        assert len(calls) == 1
        assert not any("multi_rank" in part for call in calls for part in call)

    def test_collective_report_is_skipped_when_ranks_cannot_be_identified(
        self, analyzer, tmp_path, monkeypatch
    ):
        """dynolog names traces after the pid, and TraceLens needs the rank.

        Attempting the report anyway fails on every multi-process run profiled
        through dynolog, which reads as a broken tool rather than a limitation.
        """
        for pid in (724, 892):
            _write(
                tmp_path, f"torch_profiler_output/libkineto_trace_{pid}.json", CHROME_TRACE
            )
        calls = []
        monkeypatch.setattr(
            analyzer,
            "_run",
            lambda command, cwd=None: (calls.append(list(command)), (0, ""))[1],
        )

        summary = analyzer.analyze(
            root=str(tmp_path), output_dir=str(tmp_path / "out"), python=sys.executable
        )

        assert not any("multi_rank" in part for call in calls for part in call)
        skipped = [r for r in summary["results"] if r["status"] == "SKIPPED"]
        assert len(skipped) == 1
        assert "rank" in skipped[0]["detail"]
        assert "dynolog" in skipped[0]["detail"]

    def test_max_traces_caps_work_per_kind(self, analyzer, trace_tree, monkeypatch):
        calls = []
        monkeypatch.setattr(
            analyzer,
            "_run",
            lambda command, cwd=None: (calls.append(list(command)), (0, ""))[1],
        )

        analyzer.analyze(
            root=str(trace_tree),
            output_dir=str(trace_tree / "out"),
            mode="pytorch",
            max_traces=2,
            python=sys.executable,
        )
        assert len(calls) == 2

    def test_empty_tree_reports_nothing_without_error(self, analyzer, tmp_path):
        summary = analyzer.analyze(
            root=str(tmp_path), output_dir=str(tmp_path / "out"), python=sys.executable
        )
        assert summary["results"] == []
        assert summary["succeeded"] == 0


class TestCli:
    """The script's CLI is the contract used by tracelens.sh and the host wrapper."""

    def test_discover_only_writes_json_summary_and_succeeds(
        self, analyzer, trace_tree, capsys
    ):
        summary_path = trace_tree / "discovery.json"
        code = analyzer.main(
            ["--root", str(trace_tree), "--discover-only", "--json-summary", str(summary_path)]
        )
        capsys.readouterr()

        assert code == 0
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["discovered"]["pytorch"] == 4
        assert len(summary["unsupported"]) == 2

    def test_exit_code_reflects_failures(self, analyzer, tmp_path, monkeypatch, capsys):
        _write(tmp_path, "rocprof_output/1_results.json", b"{}")
        monkeypatch.setattr(analyzer, "_run", lambda command, cwd=None: (1, "failed"))

        code = analyzer.main(["--root", str(tmp_path), "--output-dir", str(tmp_path / "out")])
        capsys.readouterr()
        assert code == 1

    def test_rejects_unknown_mode(self, analyzer):
        with pytest.raises(SystemExit):
            analyzer.main(["--mode", "nonsense"])
