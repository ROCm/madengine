"""Unit tests for host-side TraceLens report generation and its CLI commands."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from madengine.reporting import tracelens_report as tlr


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def report_app():
    from madengine.cli.commands.report import report_app

    return report_app


class TestAnalyzerDiscovery:
    """The host path drives the same packaged analyzer as the in-container tool."""

    def test_finds_the_packaged_analyzer_script(self):
        script = tlr.find_analyzer_script()
        assert script.is_file()
        assert script.name == "tracelens_analyze.py"

    def test_raises_when_the_script_is_missing(self, tmp_path):
        with patch.object(tlr, "get_madengine_root", return_value=tmp_path):
            with pytest.raises(FileNotFoundError, match="Trace analyzer not found"):
                tlr.find_analyzer_script()


class TestResolvePython:
    """TraceLens runs out-of-process so its pinned deps stay isolated."""

    def test_explicit_python_wins(self):
        assert tlr.resolve_python("/custom/python3") == "/custom/python3"

    def test_falls_back_to_tracelens_venv(self, tmp_path, monkeypatch):
        bindir = tmp_path / "bin"
        bindir.mkdir()
        (bindir / "python3").write_text("")
        monkeypatch.setenv("TRACELENS_VENV", str(tmp_path))

        assert tlr.resolve_python() == str(bindir / "python3")

    def test_ignores_a_venv_without_an_interpreter(self, tmp_path, monkeypatch):
        import sys

        monkeypatch.setenv("TRACELENS_VENV", str(tmp_path))
        assert tlr.resolve_python() == sys.executable


class TestGenerateReports:
    """generate_tracelens_reports must fail fast and surface the analyzer summary."""

    def test_requires_tracelens_to_be_installed(self, tmp_path):
        with patch.object(tlr, "check_tracelens_available", return_value=False):
            with pytest.raises(tlr.TraceLensNotInstalledError, match="madengine\\[tracelens\\]"):
                tlr.generate_tracelens_reports(root=str(tmp_path))

    def test_passes_options_through_to_the_analyzer(self, tmp_path):
        captured = {}

        def fake_run(command):
            captured["command"] = list(command)
            summary_path = Path(command[command.index("--json-summary") + 1])
            summary_path.write_text(json.dumps({"succeeded": 2}), encoding="utf-8")
            return MagicMock(returncode=0)

        with patch.object(tlr, "check_tracelens_available", return_value=True), patch.object(
            tlr.subprocess, "run", side_effect=fake_run
        ):
            summary = tlr.generate_tracelens_reports(
                root=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                mode="pytorch",
                python="/venv/bin/python3",
                gpu_arch="MI300X",
                world_size=8,
                max_traces=4,
            )

        command = captured["command"]
        assert command[command.index("--mode") + 1] == "pytorch"
        assert command[command.index("--python") + 1] == "/venv/bin/python3"
        assert command[command.index("--gpu-arch") + 1] == "MI300X"
        assert command[command.index("--world-size") + 1] == "8"
        assert command[command.index("--max-traces") + 1] == "4"
        assert summary["succeeded"] == 2
        assert summary["exit_code"] == 0

    def test_omits_unset_options(self, tmp_path):
        captured = {}

        def fake_run(command):
            captured["command"] = list(command)
            return MagicMock(returncode=0)

        with patch.object(tlr, "check_tracelens_available", return_value=True), patch.object(
            tlr.subprocess, "run", side_effect=fake_run
        ):
            tlr.generate_tracelens_reports(
                root=str(tmp_path), output_dir=str(tmp_path / "out")
            )

        assert "--gpu-arch" not in captured["command"]
        assert "--world-size" not in captured["command"]

    def test_creates_the_output_directory(self, tmp_path):
        out = tmp_path / "nested" / "out"
        with patch.object(tlr, "check_tracelens_available", return_value=True), patch.object(
            tlr.subprocess, "run", return_value=MagicMock(returncode=0)
        ):
            tlr.generate_tracelens_reports(root=str(tmp_path), output_dir=str(out))
        assert out.is_dir()

    def test_discovery_does_not_require_tracelens(self, tmp_path):
        """Users must be able to see what traces exist before installing TraceLens."""
        with patch.object(tlr, "check_tracelens_available", return_value=False), patch.object(
            tlr.subprocess, "run", return_value=MagicMock(returncode=0)
        ) as run:
            tlr.discover_traces(root=str(tmp_path))
        assert "--discover-only" in run.call_args[0][0]


class TestCompareReports:
    def test_rejects_a_single_report(self):
        with pytest.raises(ValueError, match="at least two"):
            tlr.compare_tracelens_reports(["only.xlsx"])

    def test_rejects_mismatched_names(self):
        with pytest.raises(ValueError, match="counts must match"):
            tlr.compare_tracelens_reports(["a.xlsx", "b.xlsx"], names=["only-one"])

    def test_requires_tracelens_to_be_installed(self):
        with patch.object(tlr, "check_tracelens_available", return_value=False):
            with pytest.raises(tlr.TraceLensNotInstalledError):
                tlr.compare_tracelens_reports(["a.xlsx", "b.xlsx"])

    def test_forwards_reports_and_names(self):
        captured = {}

        def fake_run(command):
            captured["command"] = list(command)
            return MagicMock(returncode=0)

        with patch.object(tlr, "check_tracelens_available", return_value=True), patch.object(
            tlr.subprocess, "run", side_effect=fake_run
        ):
            tlr.compare_tracelens_reports(
                ["base.xlsx", "cand.xlsx"], output="diff.xlsx", names=["base", "cand"]
            )

        command = captured["command"]
        assert command[command.index("--compare") + 1 : command.index("--compare") + 3] == [
            "base.xlsx",
            "cand.xlsx",
        ]
        assert command[command.index("--compare-output") + 1] == "diff.xlsx"
        assert command[command.index("--compare-names") + 1 :] == ["base", "cand"]


class TestReportTraceLensCli:
    def test_rejects_an_invalid_mode(self, runner, report_app, tmp_path):
        result = runner.invoke(
            report_app, ["tracelens", "--root", str(tmp_path), "--mode", "nonsense"]
        )
        assert result.exit_code != 0
        assert "invalid --mode" in result.output

    def test_rejects_a_missing_root(self, runner, report_app):
        result = runner.invoke(report_app, ["tracelens", "--root", "does/not/exist"])
        assert result.exit_code != 0
        assert "directory not found" in result.output

    def test_reports_install_guidance_when_tracelens_is_absent(
        self, runner, report_app, tmp_path
    ):
        with patch(
            "madengine.cli.commands.report.generate_tracelens_reports",
            side_effect=tlr.TraceLensNotInstalledError(tlr.INSTALL_HINT),
        ):
            result = runner.invoke(report_app, ["tracelens", "--root", str(tmp_path)])
        assert result.exit_code != 0
        assert "madengine[tracelens]" in result.output

    def test_succeeds_and_lists_generated_reports(self, runner, report_app, tmp_path):
        summary = {
            "succeeded": 1,
            "failed": 0,
            "skipped": 0,
            "results": [
                {
                    "trace_file": "rocprof_output/1_results.json",
                    "kind": "rocprof_json",
                    "tracelens_tool": "TraceLens_generate_perf_report_rocprof",
                    "status": "SUCCESS",
                    "detail": "",
                }
            ],
        }
        with patch(
            "madengine.cli.commands.report.generate_tracelens_reports",
            return_value=summary,
        ):
            result = runner.invoke(report_app, ["tracelens", "--root", str(tmp_path)])
        assert result.exit_code == 0
        assert "1 report(s) generated" in result.output

    def test_fails_when_a_report_fails(self, runner, report_app, tmp_path):
        summary = {
            "succeeded": 0,
            "failed": 1,
            "skipped": 0,
            "results": [
                {
                    "trace_file": "t.json",
                    "kind": "pytorch",
                    "tracelens_tool": "TraceLens_generate_perf_report_pytorch",
                    "status": "FAILURE",
                    "detail": "Not a valid trace",
                }
            ],
        }
        with patch(
            "madengine.cli.commands.report.generate_tracelens_reports",
            return_value=summary,
        ):
            result = runner.invoke(report_app, ["tracelens", "--root", str(tmp_path)])
        assert result.exit_code != 0
        assert "1 failed" in result.output

    def test_guides_the_user_when_no_traces_exist(self, runner, report_app, tmp_path):
        with patch(
            "madengine.cli.commands.report.generate_tracelens_reports",
            return_value={"succeeded": 0, "failed": 0, "skipped": 0, "results": []},
        ):
            result = runner.invoke(report_app, ["tracelens", "--root", str(tmp_path)])
        assert result.exit_code == 0
        assert "torch_profiler_dynolog" in result.output

    def test_discover_only_lists_kinds(self, runner, report_app, tmp_path):
        with patch(
            "madengine.cli.commands.report.discover_traces",
            return_value={"discovered": {"pytorch": 3}, "unsupported": []},
        ):
            result = runner.invoke(
                report_app, ["tracelens", "--root", str(tmp_path), "--discover-only"]
            )
        assert result.exit_code == 0
        assert "pytorch" in result.output


class TestReportTraceLensCompareCli:
    def test_rejects_missing_report_files(self, runner, report_app):
        result = runner.invoke(report_app, ["tracelens-compare", "a.xlsx", "b.xlsx"])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_succeeds_on_a_clean_comparison(self, runner, report_app, tmp_path):
        a = tmp_path / "a.xlsx"
        b = tmp_path / "b.xlsx"
        a.write_text("")
        b.write_text("")
        out = str(tmp_path / "diff.xlsx")

        with patch(
            "madengine.cli.commands.report.compare_tracelens_reports",
            return_value={"status": "SUCCESS"},
        ):
            result = runner.invoke(
                report_app, ["tracelens-compare", str(a), str(b), "-o", out]
            )
        assert result.exit_code == 0
        assert "Comparison written to" in result.output
