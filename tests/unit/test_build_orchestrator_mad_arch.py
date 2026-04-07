#!/usr/bin/env python3
"""BuildOrchestrator MAD_SYSTEM_GPU_ARCHITECTURE messaging."""

from unittest.mock import MagicMock

from madengine.orchestration.build_orchestrator import BuildOrchestrator


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
