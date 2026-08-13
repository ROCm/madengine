#!/usr/bin/env python3
"""Discover GPU trace artifacts and generate TraceLens reports for them.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

This script is the single implementation shared by both TraceLens execution
paths in madengine:

* in-container, as the ``tracelens`` tool post-script (``post_scripts/tracelens.sh``)
* on the host, via ``madengine report tracelens``

It therefore uses only the Python standard library and never imports madengine.
TraceLens itself is invoked out-of-process through ``--python``, which lets the
caller point at an isolated virtualenv so that TraceLens' pinned ``protobuf``
and ``xprof`` cannot disturb the workload's own Python environment.
"""

import argparse
import codecs
import csv
import glob
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

# Trace kinds, in discovery precedence order. The first pattern set that claims a
# file wins, so PyTorch traces are matched before the broader JSON patterns.
KIND_PYTORCH = "pytorch"
KIND_ROCPROF_JSON = "rocprof_json"
KIND_PFTRACE = "pftrace"
KIND_UNSUPPORTED = "unsupported"

# Glob patterns are matched against paths relative to the discovery root.
_PYTORCH_PATTERNS = (
    "**/*.pt.trace.json",
    "**/*.pt.trace.json.gz",
    "**/libkineto_trace*.json",
    "**/libkineto_trace*.json.gz",
    "**/torch_profiler_output/*.json",
    "**/torch_profiler_output/*.json.gz",
)
_ROCPROF_JSON_PATTERNS = ("**/*_results.json",)
_PFTRACE_PATTERNS = ("**/*.pftrace",)
# Formats madengine can produce that TraceLens cannot read. Reported with
# actionable guidance rather than silently ignored.
_UNSUPPORTED_PATTERNS = {
    "**/*_results.db": (
        "rocprofv3 SQLite output is not readable by TraceLens. Re-run with a "
        "preset that sets an explicit --output-format, e.g. "
        "rocprofv3_lightweight (JSON) or rocprofv3_perfetto (pftrace)."
    ),
    "**/*.rpd": (
        "RPD databases are not readable by TraceLens. The rpd post-script also "
        "writes a converted trace.json alongside it; point TraceLens at that."
    ),
    "**/*.pb": (
        "JAX XPlane protobuf traces need TraceLens_generate_perf_report_jax, "
        "which madengine does not drive yet."
    ),
}

# Ambiguous names written by more than one madengine tool (rpd writes a Chrome
# trace here, rocm-trace-lite writes its own). Sniffed rather than assumed.
_AMBIGUOUS_NAMES = ("trace.json", "trace.json.gz")

# Directories never worth walking: our own output, virtualenvs, VCS metadata.
_SKIP_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        "site-packages",
        "venv",
        ".venv",
    }
)

# TraceLens console script -> module providing main(). The console script is
# preferred when present; the module is the fallback so the integration keeps
# working if entry points were not installed onto PATH.
_ENTRY_POINTS = {
    "TraceLens_generate_perf_report_pytorch": "TraceLens.Reporting.generate_perf_report_pytorch",
    "TraceLens_generate_perf_report_rocprof": "TraceLens.Reporting.generate_perf_report_rocprof",
    "TraceLens_generate_perf_report_pftrace_hip_activity": "TraceLens.Reporting.generate_perf_report_pftrace_hip_activity",
    "TraceLens_generate_perf_report_pftrace_hip_api": "TraceLens.Reporting.generate_perf_report_pftrace_hip_api",
    "TraceLens_generate_perf_report_pftrace_memory_copy": "TraceLens.Reporting.generate_perf_report_pftrace_memory_copy",
    "TraceLens_generate_multi_rank_collective_report_pytorch": "TraceLens.Reporting.generate_multi_rank_collective_report_pytorch",
    "TraceLens_compare_perf_reports_pytorch": "TraceLens.Reporting.compare_perf_reports_pytorch",
}

# Read size used when checking a trace for undecodable bytes and rewriting it.
_SANITIZE_CHUNK_BYTES = 1 << 20

# What TraceLens says when a trace holds no GPU activity for it to report on.
_NO_GPU_EVENTS_ERROR = "No GPU events found in the trace"
_NO_GPU_EVENTS_REASON = (
    "the trace holds no GPU activity, so there is nothing to report. dynolog "
    "configures every process that registered with it, which for a torchrun job "
    "includes the launcher: it only supervises its children and runs no kernels."
)

SUMMARY_CSV_FIELDS = (
    "trace_file",
    "kind",
    "tracelens_tool",
    "status",
    "output",
    "detail",
)


def _read_head(path: str, size: int = 4096) -> str:
    """Return the first ``size`` bytes of a plain or gzipped file as text."""
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rb") as handle:  # type: ignore[operator]
            return handle.read(size).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _is_chrome_trace(path: str) -> bool:
    """Return True if the file looks like a Chrome Trace Event JSON document."""
    return '"traceEvents"' in _read_head(path)


def _iter_files(root: str) -> List[str]:
    """Return every file under ``root``, skipping uninteresting directories."""
    found: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            found.append(os.path.join(dirpath, name))
    return found


def _match(root: str, patterns: Sequence[str]) -> List[str]:
    matches: List[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(os.path.join(root, pattern), recursive=True))
    return matches


def discover_traces(
    root: str, exclude_dirs: Sequence[str] = ()
) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
    """Classify trace artifacts under ``root`` by the TraceLens reader they need.

    Args:
        root: Directory to search recursively.
        exclude_dirs: Absolute or relative directories to omit from results,
            typically the report output directory.

    Returns:
        A ``(traces, unsupported)`` pair. ``traces`` maps a trace kind to sorted
        file paths. ``unsupported`` is a list of ``(path, reason)`` for artifacts
        that were found but cannot be analyzed.
    """
    excluded = [os.path.abspath(d) for d in exclude_dirs]

    def is_excluded(path: str) -> bool:
        absolute = os.path.abspath(path)
        return any(
            absolute == prefix or absolute.startswith(prefix + os.sep)
            for prefix in excluded
        )

    claimed = set()
    traces: Dict[str, List[str]] = {}
    for kind, patterns in (
        (KIND_PYTORCH, _PYTORCH_PATTERNS),
        (KIND_ROCPROF_JSON, _ROCPROF_JSON_PATTERNS),
        (KIND_PFTRACE, _PFTRACE_PATTERNS),
    ):
        for path in _match(root, patterns):
            if not os.path.isfile(path) or is_excluded(path) or path in claimed:
                continue
            claimed.add(path)
            traces.setdefault(kind, []).append(path)

    # Sniff ambiguously named files that no pattern claimed.
    for path in _iter_files(root):
        if path in claimed or is_excluded(path):
            continue
        if os.path.basename(path) in _AMBIGUOUS_NAMES and _is_chrome_trace(path):
            claimed.add(path)
            traces.setdefault(KIND_PYTORCH, []).append(path)

    unsupported: List[Tuple[str, str]] = []
    for pattern, reason in _UNSUPPORTED_PATTERNS.items():
        for path in _match(root, [pattern]):
            if os.path.isfile(path) and not is_excluded(path) and path not in claimed:
                unsupported.append((path, reason))

    for kind in traces:
        traces[kind] = sorted(set(traces[kind]))
    return traces, sorted(set(unsupported))


def _resolve_python(python: Optional[str]) -> str:
    """Return the interpreter used to run TraceLens."""
    if python:
        return python
    venv = os.environ.get("TRACELENS_VENV", "").strip()
    if venv:
        candidate = os.path.join(venv, "bin", "python3")
        if os.path.isfile(candidate):
            return candidate
    return sys.executable or "python3"


def _build_command(python: str, script_name: str, args: Sequence[str]) -> List[str]:
    """Return argv invoking a TraceLens entry point with ``args``.

    Prefers the console script installed alongside ``python`` (clearer logs,
    honours the package's own entry-point wiring) and falls back to importing the
    module's ``main`` with that same interpreter. The fallback exits with
    ``main()``'s return value, the same way the console scripts pip generates do,
    so a report failure is not silently swallowed.

    Both forms stay inside the environment the caller asked for. Searching PATH
    instead would defeat the isolation ``--python`` exists to provide: TraceLens
    pins protobuf and xprof, and is installed in a venv of its own.
    """
    bindir = os.path.dirname(os.path.abspath(python))
    candidate = os.path.join(bindir, script_name)
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return [candidate, *args]
    module = _ENTRY_POINTS[script_name]
    return [
        python,
        "-c",
        f"import sys; from {module} import main; sys.exit(main())",
        *args,
    ]


def _run(command: Sequence[str], cwd: Optional[str] = None) -> Tuple[int, str]:
    """Run ``command``, streaming nothing, returning ``(returncode, output)``."""
    printable = " ".join(command)
    print(f"  $ {printable}", flush=True)
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
    except OSError as exc:
        return 1, str(exc)
    output = completed.stdout or ""
    if output:
        for line in output.splitlines():
            print(f"    {line}", flush=True)
    return completed.returncode, output


def _has_invalid_utf8(path: str) -> bool:
    """Return True when ``path`` is not decodable as UTF-8."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(_SANITIZE_CHUNK_BYTES)
                if not chunk:
                    decoder.decode(b"", final=True)
                    return False
                decoder.decode(chunk)
    except UnicodeDecodeError:
        return True
    except OSError:
        return False


def _write_sanitized_copy(path: str, destination: str) -> None:
    """Copy ``path`` to ``destination`` with undecodable bytes replaced."""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    with open(path, "rb") as source:
        with open(destination, "w", encoding="utf-8") as target:
            while True:
                chunk = source.read(_SANITIZE_CHUNK_BYTES)
                if not chunk:
                    target.write(decoder.decode(b"", final=True))
                    return
                target.write(decoder.decode(chunk))


def _sanitized_trace(trace: str, kind: str, workspace: List[Optional[str]]) -> str:
    """Return a trace path TraceLens can load, sanitizing bytes if it must.

    rocprofv3 copies HIP API ``const char *`` arguments into its JSON verbatim, so
    arguments that do not point at a string (``fname``, ``kname``) leave raw bytes
    in the trace. TraceLens loads traces with orjson, which rejects the whole file
    when any byte is not valid UTF-8, so a single stray pointer costs the entire
    report. Analyzing a sanitized copy keeps the original trace untouched.

    Args:
        trace: Path to the discovered trace.
        kind: Discovered trace kind.
        workspace: Single-element list caching the scratch directory, so it is
            created only once and only when a trace actually needs sanitizing.

    Returns:
        The path to analyze: ``trace`` itself, or a sanitized copy of it.
    """
    if kind not in (KIND_PYTORCH, KIND_ROCPROF_JSON) or not trace.endswith(".json"):
        return trace
    if not _has_invalid_utf8(trace):
        return trace

    if workspace[0] is None:
        workspace[0] = tempfile.mkdtemp(prefix="madengine-tracelens-")
    destination = os.path.join(workspace[0], os.path.basename(trace))
    print(
        f"[tracelens] {trace} is not valid UTF-8 (rocprofv3 writes raw pointer "
        "bytes for some HIP API string arguments); analyzing a sanitized copy",
        flush=True,
    )
    try:
        _write_sanitized_copy(trace, destination)
    except OSError as exc:
        print(f"[tracelens] could not sanitize {trace}: {exc}", flush=True)
        return trace
    return destination


def _failure_detail(returncode: int, output: str) -> str:
    """Return a one-line explanation for a failed TraceLens invocation."""
    lines = [line for line in output.strip().splitlines() if line.strip()]
    return lines[-1] if lines else f"exit code {returncode}"


def _has_no_gpu_events(output: str) -> bool:
    """Return True when TraceLens refused a trace for carrying no GPU activity."""
    return _NO_GPU_EVENTS_ERROR in output


def _report_stem(path: str, root: str) -> str:
    """Return a filesystem-safe, collision-resistant name for a trace's reports."""
    relative = os.path.relpath(path, root)
    for suffix in (".json.gz", ".pt.trace.json", ".pftrace", ".json"):
        if relative.endswith(suffix):
            relative = relative[: -len(suffix)]
            break
    return re.sub(r"[^A-Za-z0-9._-]+", "_", relative).strip("_") or "trace"


def _pytorch_args(
    trace: str, out_base: str, gpu_arch: Optional[str], extra: Sequence[str]
) -> List[str]:
    # --short_kernel_study is deliberately not requested. On some real traces its
    # sheets have MultiIndex columns, and TraceLens writes every sheet with
    # `index=False`, which pandas refuses: "Writing to Excel with MultiIndex
    # columns and no index ('index'=False) is not yet implemented". That loses the
    # whole workbook after the CSVs are already written. Pass it back through the
    # analyzer's trailing extra args if you want those sheets.
    # https://github.com/AMD-AGI/TraceLens/issues/938
    args = [
        "--profile_json_path",
        trace,
        "--output_xlsx_path",
        f"{out_base}.xlsx",
        "--output_csvs_dir",
        f"{out_base}_csv",
        "--enable_kernel_summary",
    ]
    if gpu_arch:
        args += ["--gpu_arch_platform", gpu_arch]
    return args + list(extra)


def _rocprof_args(trace: str, out_base: str, extra: Sequence[str]) -> List[str]:
    # --short_kernel_study is safe to keep here, unlike for the PyTorch report:
    # this generator writes either the CSVs or the workbook, never both, so with
    # a CSV directory requested it never reaches the Excel writer that the flag's
    # MultiIndex sheets break. That also means --output_xlsx_path is ignored.
    return [
        "--profile_json_path",
        trace,
        "--output_xlsx_path",
        f"{out_base}.xlsx",
        "--output_csvs_dir",
        f"{out_base}_csv",
        "--kernel_details",
        "--short_kernel_study",
        *extra,
    ]


def _pftrace_jobs(
    trace: str, out_base: str, extra: Sequence[str]
) -> List[Tuple[str, List[str]]]:
    """Return the three complementary pftrace reports for one trace."""
    return [
        (
            "TraceLens_generate_perf_report_pftrace_hip_activity",
            [
                "--trace_path",
                trace,
                "--output_csvs_dir",
                f"{out_base}_activity_csv",
                "--output_md_path",
                f"{out_base}_activity.md",
                *extra,
            ],
        ),
        (
            "TraceLens_generate_perf_report_pftrace_hip_api",
            [
                "--trace_path",
                trace,
                "--output_xlsx_path",
                f"{out_base}_hip_api.xlsx",
                "--output_csvs_dir",
                f"{out_base}_hip_api_csv",
                *extra,
            ],
        ),
        (
            "TraceLens_generate_perf_report_pftrace_memory_copy",
            [
                "--trace_path",
                trace,
                "--output_xlsx_path",
                f"{out_base}_memory_copy.xlsx",
                "--output_csvs_dir",
                f"{out_base}_memory_copy_csv",
                *extra,
            ],
        ),
    ]


def _rank_regex() -> str:
    """Return the rank-extraction regex covering rank-labelled trace filenames.

    Matches the torch.profiler default (``..._rank0_...``) and the ``rank[N]``
    form written by ``tensorboard_trace_handler``. Traces captured on demand
    through dynolog are named after the process id instead, and carry no rank.
    """
    return r"rank[\[\-_/]?(?P<rank>\d+)"


def _is_rank_labelled(trace: str) -> bool:
    """Return True when the rank of ``trace`` can be read from its filename."""
    return re.search(_rank_regex(), os.path.basename(trace)) is not None


def _collective_args(
    traces: Sequence[str], out_base: str, world_size: int, extra: Sequence[str]
) -> List[str]:
    # TraceLens takes a glob rather than a list of traces, so scope it to the tree
    # the per-rank traces were found in; a wider one sweeps up unrelated JSON, and
    # rocprofv3 results are hundreds of megabytes each.
    directory = os.path.commonpath([os.path.dirname(t) for t in traces])
    return [
        "--trace_glob",
        os.path.join(directory, "**", "*.json*"),
        "--rank_regex",
        _rank_regex(),
        "--world_size",
        str(world_size),
        "--output_xlsx_path",
        f"{out_base}.xlsx",
        "--output_csvs_dir",
        f"{out_base}_csv",
        "--use_multiprocessing",
        *extra,
    ]


def analyze(
    root: str,
    output_dir: str,
    mode: str = "auto",
    python: Optional[str] = None,
    gpu_arch: Optional[str] = None,
    world_size: int = 0,
    max_traces: int = 0,
    extra_args: Sequence[str] = (),
) -> Dict[str, object]:
    """Generate TraceLens reports for every supported trace under ``root``.

    Args:
        root: Directory to search for trace artifacts.
        output_dir: Directory that receives the generated reports.
        mode: ``auto`` to analyze every discovered kind, or one of ``pytorch``,
            ``rocprof``, ``pftrace``, ``collective`` to restrict the run.
        python: Interpreter that has TraceLens installed. Defaults to
            ``$TRACELENS_VENV/bin/python3`` when set, else the current one.
        gpu_arch: Bundled TraceLens GPU arch name (e.g. ``MI300X``) enabling
            roofline bound classification on PyTorch reports.
        world_size: Rank count for the multi-rank collective report. When 0, it
            is inferred from the number of discovered PyTorch traces.
        max_traces: Cap on traces analyzed per kind. 0 means no cap.
        extra_args: Extra flags forwarded verbatim to every TraceLens command.

    Returns:
        A summary dict with ``results``, ``unsupported``, and counters.
    """
    root = os.path.abspath(root)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    interpreter = _resolve_python(python)
    traces, unsupported = discover_traces(root, exclude_dirs=[output_dir])

    if max_traces > 0:
        traces = {kind: paths[:max_traces] for kind, paths in traces.items()}

    wanted = {
        "pytorch": {KIND_PYTORCH},
        "rocprof": {KIND_ROCPROF_JSON},
        "pftrace": {KIND_PFTRACE},
        "collective": {KIND_PYTORCH},
        "auto": {KIND_PYTORCH, KIND_ROCPROF_JSON, KIND_PFTRACE},
    }[mode]

    # Scratch directory for sanitized trace copies, created on first need.
    sanitize_workspace: List[Optional[str]] = [None]

    jobs: List[Tuple[str, str, str, List[str]]] = []
    for kind, paths in sorted(traces.items()):
        if kind not in wanted:
            continue
        for trace in paths:
            stem = _report_stem(trace, root)
            out_base = os.path.join(output_dir, stem)
            readable = _sanitized_trace(trace, kind, sanitize_workspace)
            if kind == KIND_PYTORCH and mode != "collective":
                jobs.append(
                    (
                        trace,
                        kind,
                        "TraceLens_generate_perf_report_pytorch",
                        _pytorch_args(readable, out_base, gpu_arch, extra_args),
                    )
                )
            elif kind == KIND_ROCPROF_JSON:
                jobs.append(
                    (
                        trace,
                        kind,
                        "TraceLens_generate_perf_report_rocprof",
                        _rocprof_args(readable, out_base, extra_args),
                    )
                )
            elif kind == KIND_PFTRACE:
                for tool, args in _pftrace_jobs(readable, out_base, extra_args):
                    jobs.append((trace, kind, tool, args))

    # A multi-rank collective report needs at least two per-rank PyTorch traces,
    # and TraceLens reads each trace's rank from its filename.
    pytorch_traces = traces.get(KIND_PYTORCH, [])
    ranked = [trace for trace in pytorch_traces if _is_rank_labelled(trace)]
    unrankable: List[Tuple[str, str]] = []
    ranks = world_size or len(ranked)
    if mode in ("auto", "collective") and len(ranked) > 1 and ranks > 1:
        jobs.append(
            (
                f"{len(ranked)} per-rank traces",
                KIND_PYTORCH,
                "TraceLens_generate_multi_rank_collective_report_pytorch",
                _collective_args(
                    ranked,
                    os.path.join(output_dir, "multi_rank_collective"),
                    ranks,
                    extra_args,
                ),
            )
        )
    elif mode in ("auto", "collective") and len(pytorch_traces) > 1:
        unrankable.append(
            (
                f"{len(pytorch_traces)} PyTorch traces",
                "the collective report needs the rank in each trace's filename, "
                "and none of these carry one. Traces captured on demand through "
                "dynolog are named after the process id.",
            )
        )

    results: List[Dict[str, str]] = []
    for trace, kind, tool, args in jobs:
        print(f"[tracelens] {tool}: {trace}", flush=True)
        code, output = _run(_build_command(interpreter, tool, args))
        if code == 0:
            status, detail, produced = "SUCCESS", "", os.path.relpath(output_dir, root)
        elif _has_no_gpu_events(output):
            # Nothing was wrong with the analysis, and nothing was produced.
            print(f"[tracelens] skipping {trace}: {_NO_GPU_EVENTS_REASON}", flush=True)
            status, detail, produced = "SKIPPED", _NO_GPU_EVENTS_REASON, ""
        else:
            status = "FAILURE"
            detail = _failure_detail(code, output)
            produced = os.path.relpath(output_dir, root)
        results.append(
            {
                "trace_file": (
                    os.path.relpath(trace, root) if os.path.exists(trace) else trace
                ),
                "kind": kind,
                "tracelens_tool": tool,
                "status": status,
                "output": produced,
                "detail": detail,
            }
        )

    if sanitize_workspace[0] is not None:
        shutil.rmtree(sanitize_workspace[0], ignore_errors=True)

    for path, reason in unsupported:
        print(f"[tracelens] skipping {path}: {reason}", flush=True)
        results.append(
            {
                "trace_file": os.path.relpath(path, root),
                "kind": KIND_UNSUPPORTED,
                "tracelens_tool": "",
                "status": "SKIPPED",
                "output": "",
                "detail": reason,
            }
        )

    for label, reason in unrankable:
        print(f"[tracelens] skipping the collective report: {reason}", flush=True)
        results.append(
            {
                "trace_file": label,
                "kind": KIND_PYTORCH,
                "tracelens_tool": (
                    "TraceLens_generate_multi_rank_collective_report_pytorch"
                ),
                "status": "SKIPPED",
                "output": "",
                "detail": reason,
            }
        )

    summary_csv = os.path.join(output_dir, "tracelens_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_CSV_FIELDS))
        writer.writeheader()
        writer.writerows(results)

    return {
        "root": root,
        "output_dir": output_dir,
        "mode": mode,
        "python": interpreter,
        "summary_csv": summary_csv,
        "discovered": {kind: len(paths) for kind, paths in sorted(traces.items())},
        "succeeded": sum(1 for r in results if r["status"] == "SUCCESS"),
        "failed": sum(1 for r in results if r["status"] == "FAILURE"),
        "skipped": sum(1 for r in results if r["status"] == "SKIPPED"),
        "results": results,
    }


def compare(
    reports: Sequence[str],
    output: str,
    names: Sequence[str] = (),
    python: Optional[str] = None,
) -> Dict[str, object]:
    """Diff two or more TraceLens reports into a single comparison workbook."""
    interpreter = _resolve_python(python)
    args: List[str] = [*reports, "-o", output]
    if names:
        args += ["--names", *names]
    code, out = _run(
        _build_command(interpreter, "TraceLens_compare_perf_reports_pytorch", args)
    )
    return {
        "reports": list(reports),
        "output": output,
        "status": "SUCCESS" if code == 0 else "FAILURE",
        "detail": "" if code == 0 else out.strip(),
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TraceLens reports for madengine trace artifacts."
    )
    parser.add_argument(
        "--root", default=".", help="Directory to search for traces (default: .)"
    )
    parser.add_argument(
        "--output-dir",
        default="tracelens_output",
        help="Directory for generated reports (default: tracelens_output)",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "pytorch", "rocprof", "pftrace", "collective"],
        help="Restrict analysis to one trace kind (default: auto)",
    )
    parser.add_argument(
        "--python", default=None, help="Interpreter that has TraceLens installed"
    )
    parser.add_argument(
        "--gpu-arch",
        default=None,
        help="TraceLens GPU arch platform for roofline bound classification",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=0,
        help="Rank count for the collective report (default: number of traces)",
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=0,
        help="Cap traces analyzed per kind (default: no cap)",
    )
    parser.add_argument(
        "--json-summary", default=None, help="Write the run summary as JSON here"
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="List discovered traces without running TraceLens",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        metavar="REPORT",
        help="Compare existing TraceLens reports instead of analyzing traces",
    )
    parser.add_argument(
        "--compare-output",
        default="tracelens_comparison.xlsx",
        help="Output workbook for --compare (default: tracelens_comparison.xlsx)",
    )
    parser.add_argument(
        "--compare-names", nargs="+", default=(), help="Display tags for --compare"
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Extra flags forwarded verbatim to every TraceLens command",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.compare:
        summary: Dict[str, object] = compare(
            args.compare, args.compare_output, args.compare_names, args.python
        )
        failed = summary["status"] != "SUCCESS"
    elif args.discover_only:
        traces, unsupported = discover_traces(args.root, exclude_dirs=[args.output_dir])
        for kind, paths in sorted(traces.items()):
            for path in paths:
                print(f"{kind}\t{path}")
        for path, reason in unsupported:
            print(f"{KIND_UNSUPPORTED}\t{path}\t{reason}")
        summary = {
            "discovered": {kind: len(paths) for kind, paths in sorted(traces.items())},
            "unsupported": [{"path": p, "reason": r} for p, r in unsupported],
        }
        failed = False
    else:
        summary = analyze(
            root=args.root,
            output_dir=args.output_dir,
            mode=args.mode,
            python=args.python,
            gpu_arch=args.gpu_arch,
            world_size=args.world_size,
            max_traces=args.max_traces,
            extra_args=args.extra_args,
        )
        if not summary["results"]:
            print(
                "[tracelens] No supported trace artifacts found under "
                f"{os.path.abspath(args.root)}. Stack a profiling tool such as "
                "torch_profiler_dynolog, rocprofv3_lightweight, or "
                "rocprofv3_perfetto with the tracelens tool.",
                flush=True,
            )
        failed = bool(summary["failed"])

    if args.json_summary:
        with open(args.json_summary, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
