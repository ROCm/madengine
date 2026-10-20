"""Integration tests for the TraceLens and dynolog tools: tools.json wiring.

Verifies the shipped tools.json entries and that ContainerRunner.apply_tools
stacks them correctly, including the profiler-then-analysis ordering that makes
the tracelens tool useful.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from madengine.execution.container_runner import ContainerRunner
from madengine.utils.path_utils import get_madengine_root


def _tools_json() -> Path:
    return get_madengine_root() / "scripts" / "common" / "tools.json"


@pytest.fixture(scope="module")
def tools() -> dict:
    with open(_tools_json(), encoding="utf-8") as f:
        return json.load(f)["tools"]


def _apply(tool_names) -> tuple:
    """Run apply_tools for the given tool names, returning (scripts, env)."""
    ctx = MagicMock()
    ctx.ctx = {"tools": [{"name": name} for name in tool_names]}
    runner = ContainerRunner(context=ctx, console=MagicMock())
    scripts = {
        "pre_scripts": [],
        "encapsulate_script": "bash model_run.sh",
        "post_scripts": [],
    }
    env: dict = {}
    runner.apply_tools(scripts, env, str(_tools_json()))
    return scripts, env


class TestDynologToolConfig:
    """torch_profiler_dynolog drives torch.profiler on unmodified workloads."""

    def test_config_sets_kineto_daemon_env(self, tools):
        cfg = tools["torch_profiler_dynolog"]
        assert cfg["env_vars"]["KINETO_USE_DAEMON"] == "1"
        assert "KINETO_DAEMON_INIT_DELAY_S" in cfg["env_vars"]
        assert cfg["env_vars"]["TORCH_PROFILE_OUTPUT_DIR"] == "torch_profiler_output"

    def test_requests_the_metadata_tracelens_needs(self, tools):
        """Roofline and per-op analysis need input shapes; nn.Module view needs modules."""
        env = tools["torch_profiler_dynolog"]["env_vars"]
        assert env["TORCH_PROFILE_RECORD_SHAPES"] == "1"
        assert env["TORCH_PROFILE_WITH_STACKS"] == "1"
        assert env["TORCH_PROFILE_WITH_MODULES"] == "1"

    def test_raises_upstream_process_limit_for_multi_gpu(self, tools):
        """dyno gputrace defaults to 3 processes, which silently drops most ranks."""
        assert int(tools["torch_profiler_dynolog"]["env_vars"]["TORCH_PROFILE_PROCESS_LIMIT"]) >= 8

    def test_does_not_wrap_the_model_command(self, tools):
        """Tracing is triggered out-of-band, so the workload command is untouched."""
        assert tools["torch_profiler_dynolog"]["cmd"] == ""

    def test_installs_then_starts_then_stops_then_collects(self, tools):
        cfg = tools["torch_profiler_dynolog"]
        assert [Path(s["path"]).name for s in cfg["pre_scripts"]] == [
            "trace.sh",
            "dynolog_start.sh",
        ]
        assert cfg["pre_scripts"][0]["args"] == "dynolog"
        assert [Path(s["path"]).name for s in cfg["post_scripts"]] == [
            "dynolog_stop.sh",
            "trace.sh",
        ]
        assert cfg["post_scripts"][1]["args"] == "torch_profiler"

    def test_referenced_scripts_exist(self, tools):
        root = get_madengine_root()
        cfg = tools["torch_profiler_dynolog"]
        for script in cfg["pre_scripts"] + cfg["post_scripts"]:
            assert (root / script["path"]).is_file(), script["path"]
        # dynolog_start.sh launches the trigger, which tools.json does not name.
        assert (root / "scripts/common/tools/dynolog_trigger.sh").is_file()

    def test_apply_tools_wires_env_and_scripts(self):
        scripts, env = _apply(["torch_profiler_dynolog"])
        assert env["KINETO_USE_DAEMON"] == "1"
        # An empty cmd must not disturb the model invocation.
        assert scripts["encapsulate_script"].strip() == "bash model_run.sh"
        assert any(
            Path(s["path"]).name == "dynolog_start.sh" for s in scripts["pre_scripts"]
        )
        assert any(
            Path(s["path"]).name == "dynolog_stop.sh" for s in scripts["post_scripts"]
        )

    def test_not_in_rocprof_family(self):
        """dynolog does not need rocprofv3, so multi-node runs must not drop it."""
        from madengine.deployment.common import tools_include_rocprof_family

        assert not tools_include_rocprof_family([{"name": "torch_profiler_dynolog"}])


class TestTraceLensToolConfig:
    """The tracelens tools run TraceLens over whatever traces the run produced."""

    ALL_VARIANTS = (
        "tracelens",
        "tracelens_pytorch",
        "tracelens_rocprof",
        "tracelens_pftrace",
        "tracelens_collective",
    )

    def test_all_variants_are_defined(self, tools):
        for name in self.ALL_VARIANTS:
            assert name in tools, name

    def test_variants_differ_only_by_mode(self, tools):
        modes = {
            name: tools[name]["env_vars"]["TRACELENS_MODE"] for name in self.ALL_VARIANTS
        }
        assert modes == {
            "tracelens": "auto",
            "tracelens_pytorch": "pytorch",
            "tracelens_rocprof": "rocprof",
            "tracelens_pftrace": "pftrace",
            "tracelens_collective": "collective",
        }

    def test_installs_into_an_isolated_venv(self, tools):
        """TraceLens pins protobuf/xprof; it must not touch the workload's env."""
        venv = tools["tracelens"]["env_vars"]["TRACELENS_VENV"]
        assert venv.startswith("/")
        assert "site-packages" not in venv

    def test_analysis_runs_after_the_model_not_around_it(self, tools):
        cfg = tools["tracelens"]
        assert cfg["cmd"] == ""
        assert [Path(s["path"]).name for s in cfg["post_scripts"]] == [
            "tracelens.sh",
            "trace.sh",
        ]
        assert cfg["post_scripts"][1]["args"] == "tracelens"

    def test_referenced_scripts_exist(self, tools):
        root = get_madengine_root()
        for name in self.ALL_VARIANTS:
            cfg = tools[name]
            for script in cfg["pre_scripts"] + cfg["post_scripts"]:
                assert (root / script["path"]).is_file(), script["path"]
        assert (root / "scripts/common/tools/tracelens_analyze.py").is_file()

    def test_stacks_after_a_profiler(self):
        """Profiler setup must precede analysis, and analysis must run last."""
        scripts, env = _apply(["rocprofv3_perfetto", "tracelens"])

        # The profiler still wraps the model command.
        assert "rocprof_wrapper.sh" in scripts["encapsulate_script"]
        assert "bash model_run.sh" in scripts["encapsulate_script"]

        post = [Path(s["path"]).name for s in scripts["post_scripts"]]
        # rocprofv3_perfetto collects its trace before TraceLens reads it.
        assert post.index("trace.sh") < post.index("tracelens.sh")
        assert env["TRACELENS_MODE"] == "auto"

    def test_stacks_with_dynolog_for_the_full_pytorch_path(self):
        scripts, env = _apply(["torch_profiler_dynolog", "tracelens"])

        assert env["KINETO_USE_DAEMON"] == "1"
        assert env["TRACELENS_MODE"] == "auto"
        pre = [Path(s["path"]).name for s in scripts["pre_scripts"]]
        assert "dynolog_start.sh" in pre
        post = [Path(s["path"]).name for s in scripts["post_scripts"]]
        # Kineto traces are collected before TraceLens analyses them.
        assert post.index("dynolog_stop.sh") < post.index("tracelens.sh")

    def test_env_vars_can_be_overridden_per_run(self):
        ctx = MagicMock()
        ctx.ctx = {
            "tools": [
                {"name": "tracelens", "env_vars": {"TRACELENS_GPU_ARCH": "MI300X"}}
            ]
        }
        runner = ContainerRunner(context=ctx, console=MagicMock())
        scripts = {
            "pre_scripts": [],
            "encapsulate_script": "bash model_run.sh",
            "post_scripts": [],
        }
        env: dict = {}
        runner.apply_tools(scripts, env, str(_tools_json()))
        assert env["TRACELENS_GPU_ARCH"] == "MI300X"
        assert env["TRACELENS_MODE"] == "auto"

    def test_not_in_rocprof_family(self):
        from madengine.deployment.common import tools_include_rocprof_family

        assert not tools_include_rocprof_family([{"name": name} for name in self.ALL_VARIANTS])
