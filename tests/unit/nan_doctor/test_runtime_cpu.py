"""The nan-doctor CPU test scripts (cpu/cpu_*.py), run under pytest.

The scripts check many cases each and print `PASS|FAIL  <name>` lines; this wraps each one so a
failure names the failing checks. Host-side scripts need only the standard library; the capture /
replay scripts need torch (CPU is enough, no GPU) and some spawn gloo workers or call `torchrun`
(it must be on PATH).
"""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

CPU = Path(__file__).resolve().parent / "cpu"
SRC = Path(__file__).resolve().parents[3] / "src"
NO_TORCH = {"cpu_host_tools.py", "cpu_intake.py", "cpu_diff_unit.py"}
SCRIPTS = sorted(p.name for p in CPU.glob("cpu_*.py"))

# Checks that fail identically before the code moved into madengine, on torch 2.15 CPU builds.
# Listed so the suite is green and the failure stays visible; remove an entry when fixed.
KNOWN_FAILURES = {}


@pytest.mark.parametrize("script", SCRIPTS)
def test_cpu_script(script):
    if script not in NO_TORCH and importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]),
    )
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    r = subprocess.run(
        [sys.executable, str(CPU / script)],
        capture_output=True,
        text=True,
        timeout=1500,
        env=env,
    )
    out = r.stdout + r.stderr
    passed = re.findall(r"^PASS\s+(.*)$", out, re.M)
    failed = re.findall(r"^FAIL\s+(.*)$", out, re.M)
    known = KNOWN_FAILURES.get(script, [])
    unexpected = [f for f in failed if not any(f.startswith(k) for k in known)]
    assert not unexpected, "failed checks in %s:\n  %s\n\n%s" % (
        script,
        "\n  ".join(unexpected),
        out[-3000:],
    )
    if not failed:
        assert r.returncode == 0, out[-3000:]
    assert passed, "no checks ran in %s:\n%s" % (script, out[-1500:])
