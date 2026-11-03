#!/usr/bin/env python3
"""BuildOrchestrator MAD_SYSTEM_GPU_ARCHITECTURE messaging."""

import json
import subprocess
from unittest.mock import MagicMock

from madengine.orchestration.build_orchestrator import (
    BuildOrchestrator,
    _dockerhub_field_reader,
)


def test_warn_skipped_when_dockerfile_has_default(tmp_path):
    df = tmp_path / "d.Dockerfile"
    df.write_text("ARG MAD_SYSTEM_GPU_ARCHITECTURE=gfx942\n")
    orch = object.__new__(BuildOrchestrator)
    orch.rich_console = MagicMock()
    builder = MagicMock()
    builder._get_dockerfiles_for_model = lambda m: [str(df)]
    builder._get_effective_gpu_architecture = lambda m, d: None

    BuildOrchestrator._warn_if_mad_arch_unresolved_for_dockerfiles(
        orch, [{"name": "model_a"}], builder
    )
    orch.rich_console.print.assert_not_called()


def test_warn_when_bare_arg_and_unresolved(tmp_path):
    df = tmp_path / "d.Dockerfile"
    df.write_text("ARG MAD_SYSTEM_GPU_ARCHITECTURE\n")
    orch = object.__new__(BuildOrchestrator)
    orch.rich_console = MagicMock()
    builder = MagicMock()
    builder._get_dockerfiles_for_model = lambda m: [str(df)]
    builder._get_effective_gpu_architecture = lambda m, d: None

    BuildOrchestrator._warn_if_mad_arch_unresolved_for_dockerfiles(
        orch, [{"name": "model_a"}], builder
    )
    orch.rich_console.print.assert_called()


def test_compute_node_reader_skips_templates_and_keeps_real_secrets(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    cred = tmp_path / "credential.json"
    cred.write_text(
        json.dumps(
            {
                "dockerhub": {
                    "username": "your-docker-username",
                    "password": "your-docker-token",
                }
            }
        )
    )
    username = subprocess.check_output(
        ["python3", "-c", _dockerhub_field_reader("username")], text=True
    ).strip()
    assert username == ""

    cred.write_text(
        json.dumps({"dockerhub": {"username": "your-company", "password": "s3cret"}})
    )
    username = subprocess.check_output(
        ["python3", "-c", _dockerhub_field_reader("username")], text=True
    ).strip()
    password = subprocess.check_output(
        ["python3", "-c", _dockerhub_field_reader("password")], text=True
    ).strip()
    assert username == "your-company"
    assert password == "s3cret"
