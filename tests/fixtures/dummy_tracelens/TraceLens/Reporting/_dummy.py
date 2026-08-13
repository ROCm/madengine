"""Shared implementation behind every dummy TraceLens report generator.

Each entry point declares the flags madengine's analyzer is expected to pass. A
flag that madengine drops or renames therefore fails the test suite rather than
surfacing as an obscure TraceLens usage error at profiling time. Invocations are
appended to ``$DUMMY_TRACELENS_LOG`` as JSON lines so tests can assert which
report generator ran for which trace.

This encodes madengine's side of the contract; it cannot verify that upstream
TraceLens still accepts these flags. Bumping the pinned TraceLens revision needs
a real run against real traces.

Set ``DUMMY_TRACELENS_FAIL`` to a comma-separated list of entry-point names (or
``all``) to make those reports fail, which is how tests cover failure handling.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import argparse
import glob
import json
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence

# Per entry point: value-taking flags madengine always passes, bare switches it
# always passes, value-taking flags it passes conditionally, and which flag
# carries the input trace so the dummy can check it was given a real file.
_SPECS: Dict[str, Dict[str, object]] = {
    "TraceLens_generate_perf_report_pytorch": {
        "required": ("--profile_json_path", "--output_xlsx_path", "--output_csvs_dir"),
        "switches": ("--enable_kernel_summary", "--short_kernel_study"),
        "optional": ("--gpu_arch_platform",),
        "input_file": "profile_json_path",
    },
    "TraceLens_generate_perf_report_rocprof": {
        "required": ("--profile_json_path", "--output_xlsx_path", "--output_csvs_dir"),
        "switches": ("--kernel_details", "--short_kernel_study"),
        "input_file": "profile_json_path",
    },
    "TraceLens_generate_perf_report_pftrace_hip_activity": {
        "required": ("--trace_path", "--output_csvs_dir", "--output_md_path"),
        "input_file": "trace_path",
    },
    "TraceLens_generate_perf_report_pftrace_hip_api": {
        "required": ("--trace_path", "--output_xlsx_path", "--output_csvs_dir"),
        "input_file": "trace_path",
    },
    "TraceLens_generate_perf_report_pftrace_memory_copy": {
        "required": ("--trace_path", "--output_xlsx_path", "--output_csvs_dir"),
        "input_file": "trace_path",
    },
    "TraceLens_generate_multi_rank_collective_report_pytorch": {
        "required": (
            "--trace_glob",
            "--rank_regex",
            "--world_size",
            "--output_xlsx_path",
            "--output_csvs_dir",
        ),
        "switches": ("--use_multiprocessing",),
        "input_glob": "trace_glob",
    },
}

_FORCED_FAILURE_EXIT_CODE = 3


def _parser(entry_point: str) -> argparse.ArgumentParser:
    spec = _SPECS[entry_point]
    parser = argparse.ArgumentParser(prog=entry_point)
    for flag in spec["required"]:  # type: ignore[union-attr]
        parser.add_argument(flag, required=True)
    for flag in spec.get("optional", ()):  # type: ignore[union-attr]
        parser.add_argument(flag)
    for flag in spec.get("switches", ()):  # type: ignore[union-attr]
        parser.add_argument(flag, action="store_true", required=True)
    return parser


def _fail(message: str) -> int:
    print(f"dummy TraceLens: {message}", file=sys.stderr)
    return 1


def _forced_failure(entry_point: str) -> bool:
    requested = os.environ.get("DUMMY_TRACELENS_FAIL", "")
    wanted = {name.strip() for name in requested.split(",") if name.strip()}
    return "all" in wanted or entry_point in wanted


def _check_input_file(path: str) -> Optional[int]:
    if not os.path.isfile(path):
        return _fail(f"input trace {path} does not exist")
    if os.path.getsize(path) == 0:
        return _fail(f"input trace {path} is empty")
    if path.endswith(".json"):
        # TraceLens loads traces with orjson, which rejects the whole document
        # when any byte in it is not valid UTF-8.
        try:
            with open(path, "rb") as handle:
                handle.read().decode("utf-8")
        except UnicodeDecodeError:
            return _fail(
                "orjson.JSONDecodeError: str is not valid UTF-8: surrogates not "
                "allowed: line 1 column 1 (char 0)"
            )
    return None


def _check_input_glob(pattern: str, world_size: str) -> Optional[int]:
    matches = [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]
    if len(matches) < 2:
        return _fail(
            f"a collective report needs at least two per-rank traces; "
            f"{pattern} matched {len(matches)}"
        )
    if int(world_size) < 2:
        return _fail(f"--world_size must be at least 2, got {world_size}")
    return None


def _write_reports(entry_point: str, parsed: argparse.Namespace) -> List[str]:
    """Write the report files the flags promise, returning their paths."""
    written: List[str] = []
    for dest, value in sorted(vars(parsed).items()):
        if not isinstance(value, str):
            continue
        if dest.endswith("_xlsx_path"):
            _write_text(value, f"dummy TraceLens workbook from {entry_point}\n")
            written.append(value)
        elif dest.endswith("_md_path"):
            _write_text(value, f"# dummy TraceLens report from {entry_point}\n")
            written.append(value)
        elif dest.endswith("_csvs_dir"):
            os.makedirs(value, exist_ok=True)
            summary = os.path.join(value, "kernel_summary.csv")
            _write_text(summary, "kernel,duration_us\ndummy_gemm_kernel,42\n")
            written.append(summary)
    return written


def _write_text(path: str, text: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _log(entry_point: str, parsed: argparse.Namespace, extra: Sequence[str]) -> None:
    log_path = os.environ.get("DUMMY_TRACELENS_LOG", "")
    if not log_path:
        return
    record = {
        "entry_point": entry_point,
        # "-c" when madengine fell back to importing the module, the console
        # script path when it found the installed entry point.
        "argv0": sys.argv[0],
        "args": vars(parsed),
        "extra": list(extra),
        "cwd": os.getcwd(),
    }
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _report(entry_point: str, argv: Optional[Sequence[str]]) -> int:
    # Unrecognised flags are recorded rather than rejected: madengine forwards
    # user-supplied TraceLens flags verbatim, and tests assert on that.
    parsed, extra = _parser(entry_point).parse_known_args(argv)
    _log(entry_point, parsed, extra)

    spec = _SPECS[entry_point]
    if "input_file" in spec:
        failure = _check_input_file(getattr(parsed, str(spec["input_file"])))
        if failure is not None:
            return failure
    if "input_glob" in spec:
        failure = _check_input_glob(
            getattr(parsed, str(spec["input_glob"])), parsed.world_size
        )
        if failure is not None:
            return failure

    if _forced_failure(entry_point):
        print(f"dummy TraceLens: forced failure for {entry_point}", file=sys.stderr)
        return _FORCED_FAILURE_EXIT_CODE

    for path in _write_reports(entry_point, parsed):
        print(f"dummy TraceLens: wrote {path}")
    return 0


def main_for(entry_point: str) -> Callable[[Optional[Sequence[str]]], int]:
    """Return the ``main`` for one dummy report generator.

    Like the console scripts pip generates for the real package, ``main``
    returns an exit code rather than raising ``SystemExit``.
    """

    def main(argv: Optional[Sequence[str]] = None) -> int:
        return _report(entry_point, argv)

    main.__name__ = "main"
    main.__doc__ = f"Dummy implementation of {entry_point}."
    return main


def compare_main(argv: Optional[Sequence[str]] = None) -> int:
    """Dummy implementation of ``TraceLens_compare_perf_reports_pytorch``."""
    entry_point = "TraceLens_compare_perf_reports_pytorch"
    parser = argparse.ArgumentParser(prog=entry_point)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--names", nargs="+", default=[])
    parsed, extra = parser.parse_known_args(argv)
    _log(entry_point, parsed, extra)

    if len(parsed.reports) < 2:
        return _fail("comparing reports needs at least two inputs")
    for report in parsed.reports:
        if not os.path.exists(report):
            return _fail(f"report {report} does not exist")
    if parsed.names and len(parsed.names) != len(parsed.reports):
        return _fail(f"got {len(parsed.names)} names for {len(parsed.reports)} reports")

    if _forced_failure(entry_point):
        print(f"dummy TraceLens: forced failure for {entry_point}", file=sys.stderr)
        return _FORCED_FAILURE_EXIT_CODE

    _write_text(parsed.output, f"dummy TraceLens comparison of {parsed.reports}\n")
    print(f"dummy TraceLens: wrote {parsed.output}")
    return 0
