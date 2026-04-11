"""Unit tests for rocm_trace_lite: tools.json entry and apply_tools wiring (no Docker)."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from madengine.execution.container_runner import ContainerRunner
from madengine.utils.path_utils import get_madengine_root


def _tools_json() -> Path:
    return get_madengine_root() / "scripts" / "common" / "tools.json"


def test_rocm_trace_lite_config_and_apply_tools():
    """Shipped tools.json lists rocm_trace_lite; wrapper exists; apply_tools prepends it."""
    with open(_tools_json(), encoding="utf-8") as f:
        cfg = json.load(f)["tools"]["rocm_trace_lite"]

    assert "rtl_trace_wrapper.sh" in cfg["cmd"]
    assert cfg["pre_scripts"][0]["args"] == "rocm_trace_lite"
    assert cfg["post_scripts"][0]["args"] == "rocm_trace_lite"

    wrapper = get_madengine_root() / "scripts" / "common" / "tools" / "rtl_trace_wrapper.sh"
    assert wrapper.is_file()

    ctx = MagicMock()
    ctx.ctx = {"tools": [{"name": "rocm_trace_lite"}]}
    runner = ContainerRunner(context=ctx, console=MagicMock())
    pre_encap_post = {
        "pre_scripts": [],
        "encapsulate_script": "bash model_run.sh",
        "post_scripts": [],
    }
    runner.apply_tools(pre_encap_post, {}, str(_tools_json()))

    enc = pre_encap_post["encapsulate_script"]
    assert "rtl_trace_wrapper.sh" in enc
    assert "bash model_run.sh" in enc
    assert any(
        s.get("args") == "rocm_trace_lite" for s in pre_encap_post["pre_scripts"]
    )
    assert any(
        s.get("args") == "rocm_trace_lite" for s in pre_encap_post["post_scripts"]
    )
