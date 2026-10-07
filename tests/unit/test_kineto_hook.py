"""The dynolog site hook imports torch.profiler only while the daemon is on.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from madengine.utils.path_utils import get_madengine_root

HOOK = (
    get_madengine_root() / "scripts" / "common" / "tools" / "madengine_kineto_hook.py"
)


def _run(tmp_path: Path, daemon: str) -> str:
    fake = tmp_path / "fake"
    profiler = fake / "torch" / "profiler"
    profiler.mkdir(parents=True)
    (fake / "torch" / "__init__.py").write_text("", encoding="utf-8")
    (profiler / "__init__.py").write_text("IMPORTED = True\n", encoding="utf-8")
    script = textwrap.dedent(
        """
        import os
        import sys
        sys.path.insert(0, os.environ["FAKE_TORCH"])
        sys.path.insert(0, os.path.dirname(os.environ["HOOK"]))
        import madengine_kineto_hook
        print("imported", "torch.profiler" in sys.modules)
        """
    )
    env = dict(os.environ)
    env["FAKE_TORCH"] = str(fake)
    env["HOOK"] = str(HOOK)
    if daemon:
        env["KINETO_USE_DAEMON"] = daemon
    else:
        env.pop("KINETO_USE_DAEMON", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_hook_imports_profiler_when_the_daemon_is_enabled(tmp_path):
    assert "imported True" in _run(tmp_path, "1")


def test_hook_leaves_profiler_alone_without_the_daemon(tmp_path):
    assert "imported False" in _run(tmp_path, "")
