"""Tests for the setup-page command and the workload entrypoint.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from madengine.cli import app
from madengine.setup_page.generator import render_setup_page

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_WORKLOAD = (
    REPO_ROOT
    / "src"
    / "madengine"
    / "scripts"
    / "common"
    / "entrypoints"
    / "run_workload.sh"
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_models(directory: Path) -> None:
    (directory / "models.json").write_text(
        json.dumps(
            [
                {
                    "name": "demo_model",
                    "tags": ["demo"],
                    "dockerfile": "docker/demo",
                    "scripts": "scripts/demo",
                    "n_gpus": "1",
                }
            ]
        ),
        encoding="utf-8",
    )


class TestSetupPageCommand:
    def test_writes_html_with_model_payload(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_models(tmp_path)
        monkeypatch.chdir(tmp_path)
        output = tmp_path / "picker.html"

        result = runner.invoke(
            app,
            ["setup-page", "--output", str(output), "--title", "Picker"],
        )

        assert result.exit_code == 0, result.output
        html = output.read_text(encoding="utf-8")
        assert "<title>Picker</title>" in html
        payload = html.split('id="madengine-data"', 1)[1]
        payload = payload.split(">", 1)[1].split("</script>", 1)[0]
        data = json.loads(payload)
        assert data["title"] == "Picker"
        assert data["models"][0]["name"] == "demo_model"

    def test_title_is_html_escaped(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_models(tmp_path)
        monkeypatch.chdir(tmp_path)
        output = tmp_path / "picker.html"
        title = "<script>alert(1)</script>"

        result = runner.invoke(
            app,
            ["setup-page", "--output", str(output), "--title", title],
        )

        assert result.exit_code == 0, result.output
        html = output.read_text(encoding="utf-8")
        assert "<title><script>alert(1)</script></title>" not in html
        assert "<title>&lt;script&gt;alert(1)&lt;/script&gt;</title>" in html
        payload = html.split('id="madengine-data"', 1)[1]
        payload = payload.split(">", 1)[1].split("</script>", 1)[0]
        assert "<script>alert(1)</script>" not in payload
        assert json.loads(payload)["title"] == title

    def test_missing_models_json_is_a_failure(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app, ["setup-page", "--output", str(tmp_path / "x.html")]
        )
        assert result.exit_code != 0
        assert not (tmp_path / "x.html").exists()


class TestRenderSetupPage:
    def test_json_payload_round_trips(self) -> None:
        html = render_setup_page(
            models=[{"name": "m", "tags": ["a<b"]}],
            title="T",
        )
        payload = html.split('id="madengine-data"', 1)[1]
        payload = payload.split(">", 1)[1].split("</script>", 1)[0]
        data = json.loads(payload)
        assert data["models"][0]["tags"] == ["a<b"]
        assert "<" not in payload


class TestRunWorkload:
    def _run(self, *args: str, cwd: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(RUN_WORKLOAD), *args],
            cwd=cwd,
            capture_output=True,
            text=True,
        )

    def test_dispatches_base_command(self, tmp_path: Path) -> None:
        result = self._run("--workload", "train", "--", "echo", "base", cwd=tmp_path)
        assert result.returncode == 0, result.stderr
        assert "madengine workload: train" in result.stdout
        assert result.stdout.strip().endswith("base")

    def test_prefers_model_script(self, tmp_path: Path) -> None:
        script = tmp_path / "train.sh"
        script.write_text("#!/bin/bash\necho from-script\n", encoding="utf-8")
        result = self._run("--workload", "train", "--", "echo", "base", cwd=tmp_path)
        assert result.returncode == 0, result.stderr
        assert "from-script" in result.stdout
        assert "base" not in result.stdout.splitlines()[-1]

    def test_workload_requires_a_name(self, tmp_path: Path) -> None:
        missing = self._run("--workload", cwd=tmp_path)
        assert missing.returncode == 1
        assert "--workload requires a name" in missing.stderr

        separator = self._run("--workload", "--", "echo", "base", cwd=tmp_path)
        assert separator.returncode == 1
        assert "--workload requires a name" in separator.stderr
        assert "base" not in separator.stdout
