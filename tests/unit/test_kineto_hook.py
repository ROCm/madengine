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


def test_hook_does_not_register_the_torchrun_launcher(tmp_path):
    """The launcher imports torch. It must not also register with dynolog."""
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
        sys.argv = ["/usr/local/bin/torchrun", "--standalone", "--nproc_per_node=1"]
        import madengine_kineto_hook
        print("imported", "torch.profiler" in sys.modules)
        print("daemon", os.environ.get("KINETO_USE_DAEMON", "<unset>"))
        print("saved", os.environ.get("MADENGINE_KINETO_USE_DAEMON"))
        """
    )
    env = dict(os.environ)
    env["FAKE_TORCH"] = str(fake)
    env["HOOK"] = str(HOOK)
    env["KINETO_USE_DAEMON"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "imported False" in result.stdout
    assert "daemon <unset>" in result.stdout
    assert "saved 1" in result.stdout


def test_hook_does_not_register_startup_helpers(tmp_path):
    """rocenv, pip, and rocm-sdk import torch and exit before the worker.

    A dead registrant left in the dynolog job makes the worker SIGSEGV while
    flushing the trace that was requested for it.
    """
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
        sys.argv = ["python3", "rocenv_tool.py", "--lite"]
        import madengine_kineto_hook
        print("imported", "torch.profiler" in sys.modules)
        print("daemon", os.environ.get("KINETO_USE_DAEMON", "<unset>"))
        """
    )
    env = dict(os.environ)
    env["FAKE_TORCH"] = str(fake)
    env["HOOK"] = str(HOOK)
    env["KINETO_USE_DAEMON"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "imported False" in result.stdout
    assert "daemon <unset>" in result.stdout


def test_hook_restores_the_daemon_for_the_worker(tmp_path):
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
        sys.argv = ["run_torchrun.py"]
        os.environ.pop("KINETO_USE_DAEMON", None)
        os.environ["MADENGINE_KINETO_USE_DAEMON"] = "1"
        import madengine_kineto_hook
        print("imported", "torch.profiler" in sys.modules)
        print("daemon", os.environ.get("KINETO_USE_DAEMON"))
        """
    )
    env = dict(os.environ)
    env["FAKE_TORCH"] = str(fake)
    env["HOOK"] = str(HOOK)
    env["KINETO_USE_DAEMON"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "imported True" in result.stdout
    assert "daemon 1" in result.stdout
