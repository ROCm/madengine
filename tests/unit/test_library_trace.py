"""Parser coverage for library-trace log lines.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import importlib.util

from madengine.utils.path_utils import get_madengine_root


def _load():
    path = (
        get_madengine_root()
        / "scripts"
        / "common"
        / "tools"
        / "get_library_trace.py"
    )
    spec = importlib.util.spec_from_file_location("get_library_trace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_miopen_parser_accepts_hip_and_plain_prefixes():
    """TheRock logs 'MIOpen:'; older ROCm logs 'MIOpen(HIP):'. Both are commands."""
    module = _load()
    module.filtered_configs["miopen"] = {}
    lines = [
        "MIOpen(HIP): Command [LogCmdConvolution] ./bin/MIOpenDriver conv -n 1 -c 1",
        "MIOpen: Command [LogCmdFindConvolution] ./bin/MIOpenDriver conv -n 2 -c 3",
        "rocblas: nothing to see here",
    ]

    assert module.process_miopen_trace(lines) is True
    stored = " ".join(module.filtered_configs["miopen"])
    assert "./bin/MIOpenDriver conv -n 1 -c 1" in stored
    assert "./bin/MIOpenDriver conv -n 2 -c 3" in stored


def test_miopen_parser_ignores_lines_without_a_driver_command():
    module = _load()
    module.filtered_configs["miopen"] = {}

    assert module.process_miopen_trace(["MIOpen: find algorithm done"]) is False
    assert module.filtered_configs["miopen"] == {}
