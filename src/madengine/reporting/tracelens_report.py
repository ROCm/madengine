"""Host-side TraceLens report generation for collected madengine trace artifacts.

This module drives ``scripts/common/tools/tracelens_analyze.py``, the same
analyzer the in-container ``tracelens`` tool runs, against artifacts that a run
already copied back to the host working directory (``rocprof_output/``,
``torch_profiler_output/``, ``slurm_results/``, ``k8s_results/``, ...).

Running TraceLens on the host keeps its pinned ``protobuf`` and ``xprof``
dependencies out of the workload container. Install support with::

    pip install 'madengine[tracelens]'

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from madengine.utils.path_utils import get_madengine_root

logger = logging.getLogger(__name__)

ANALYZER_RELATIVE_PATH = Path("scripts") / "common" / "tools" / "tracelens_analyze.py"

INSTALL_HINT = (
    "TraceLens is not importable by the selected interpreter. Install it with "
    "\"pip install 'madengine[tracelens]'\", or point --python at an "
    "environment that has it (or set TRACELENS_VENV)."
)


class TraceLensNotInstalledError(RuntimeError):
    """Raised when the selected interpreter cannot import TraceLens."""


def find_analyzer_script() -> Path:
    """Return the path to the packaged trace analyzer script.

    Raises:
        FileNotFoundError: If the script is missing from the installation.
    """
    script = get_madengine_root() / ANALYZER_RELATIVE_PATH
    if not script.is_file():
        raise FileNotFoundError(
            f"Trace analyzer not found at {script}. The madengine installation "
            "appears to be missing its bundled scripts."
        )
    return script


def resolve_python(python: Optional[str] = None) -> str:
    """Return the interpreter used to run TraceLens itself.

    Prefers an explicit ``python``, then ``$TRACELENS_VENV``, then the
    interpreter running madengine.
    """
    if python:
        return python
    venv = os.environ.get("TRACELENS_VENV", "").strip()
    if venv:
        for candidate in (
            Path(venv) / "bin" / "python3",
            Path(venv) / "Scripts" / "python.exe",
        ):
            if candidate.is_file():
                return str(candidate)
    return sys.executable


def check_tracelens_available(python: str) -> bool:
    """Return True if ``python`` can import TraceLens."""
    try:
        completed = subprocess.run(
            [python, "-c", "import TraceLens"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return completed.returncode == 0


def _run_analyzer(args: Sequence[str], summary_path: Path) -> Dict[str, object]:
    """Run the analyzer script and return the summary it wrote."""
    script = find_analyzer_script()
    command = [sys.executable, str(script), *args]
    logger.info("Running trace analyzer: %s", " ".join(command))
    completed = subprocess.run(command)

    summary: Dict[str, object] = {}
    if summary_path.is_file():
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
    summary["exit_code"] = completed.returncode
    return summary


def generate_tracelens_reports(
    root: str = ".",
    output_dir: str = "tracelens_output",
    mode: str = "auto",
    python: Optional[str] = None,
    gpu_arch: Optional[str] = None,
    world_size: int = 0,
    max_traces: int = 0,
    extra_args: Sequence[str] = (),
) -> Dict[str, object]:
    """Generate TraceLens reports for every supported trace found under ``root``.

    Args:
        root: Directory to search recursively for trace artifacts.
        output_dir: Directory that receives the generated reports.
        mode: ``auto``, or one of ``pytorch``, ``rocprof``, ``pftrace``,
            ``collective`` to restrict the run to a single trace kind.
        python: Interpreter that has TraceLens installed.
        gpu_arch: Bundled TraceLens GPU arch name (e.g. ``MI300X``) that enables
            roofline bound classification on PyTorch reports.
        world_size: Rank count for the multi-rank collective report. Inferred
            from the number of discovered PyTorch traces when 0.
        max_traces: Cap on traces analyzed per kind. 0 means no cap.
        extra_args: Extra flags forwarded verbatim to every TraceLens command.

    Returns:
        The analyzer summary, including ``results``, ``discovered``, counters,
        and ``exit_code``.

    Raises:
        TraceLensNotInstalledError: If the interpreter cannot import TraceLens.
    """
    interpreter = resolve_python(python)
    if not check_tracelens_available(interpreter):
        raise TraceLensNotInstalledError(INSTALL_HINT)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "tracelens_summary.json"

    args: List[str] = [
        "--root",
        root,
        "--output-dir",
        str(output_path),
        "--mode",
        mode,
        "--python",
        interpreter,
        "--json-summary",
        str(summary_path),
    ]
    if gpu_arch:
        args += ["--gpu-arch", gpu_arch]
    if world_size:
        args += ["--world-size", str(world_size)]
    if max_traces:
        args += ["--max-traces", str(max_traces)]
    args += list(extra_args)

    return _run_analyzer(args, summary_path)


def discover_traces(root: str = ".", output_dir: str = "tracelens_output") -> Dict[str, object]:
    """List the trace artifacts under ``root`` without running TraceLens.

    Unlike :func:`generate_tracelens_reports`, this does not require TraceLens to
    be installed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        summary_path = Path(tmp) / "discovery.json"
        return _run_analyzer(
            [
                "--root",
                root,
                "--output-dir",
                output_dir,
                "--discover-only",
                "--json-summary",
                str(summary_path),
            ],
            summary_path,
        )


def compare_tracelens_reports(
    reports: Sequence[str],
    output: str = "tracelens_comparison.xlsx",
    names: Sequence[str] = (),
    python: Optional[str] = None,
) -> Dict[str, object]:
    """Diff two or more TraceLens reports into a single comparison workbook.

    Args:
        reports: TraceLens ``.xlsx`` reports or per-sheet CSV directories. The
            first is treated as the baseline.
        output: Output workbook path.
        names: Display tags, one per report.
        python: Interpreter that has TraceLens installed.

    Raises:
        TraceLensNotInstalledError: If the interpreter cannot import TraceLens.
        ValueError: If fewer than two reports were given.
    """
    if len(reports) < 2:
        raise ValueError("Comparing reports needs at least two inputs.")
    if names and len(names) != len(reports):
        raise ValueError(
            f"Got {len(names)} names for {len(reports)} reports; counts must match."
        )

    interpreter = resolve_python(python)
    if not check_tracelens_available(interpreter):
        raise TraceLensNotInstalledError(INSTALL_HINT)

    with tempfile.TemporaryDirectory() as tmp:
        summary_path = Path(tmp) / "comparison.json"
        args: List[str] = [
            "--python",
            interpreter,
            "--compare",
            *reports,
            "--compare-output",
            output,
            "--json-summary",
            str(summary_path),
        ]
        if names:
            args += ["--compare-names", *names]
        return _run_analyzer(args, summary_path)
