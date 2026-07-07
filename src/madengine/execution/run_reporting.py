#!/usr/bin/env python3
"""Shared run-reporting helpers (performance extraction, status, perf.csv).

Pure-Python helpers used by execution backends to turn a run log into a
performance/metric pair, a SUCCESS/FAILURE status, and rows in ``perf.csv`` /
``perf_super``. These are Docker-independent so both the container and
bare-metal backends can share the same reporting semantics.

The performance-extraction and status logic mirror the inline behavior in
``ContainerRunner.run_container`` so results are consistent across backends.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import os
import re
import typing

from madengine.deployment.base import PERFORMANCE_LOG_PATTERN
from madengine.execution.container_runner_helpers import (
    log_text_has_error_pattern,
    resolve_log_error_scan_config,
)
from madengine.reporting.update_perf_csv import update_perf_csv
from madengine.reporting.update_perf_super import (
    update_perf_super_csv,
    update_perf_super_json,
)
from madengine.utils.path_utils import scripts_base_dir_from

# HuggingFace Trainer emits "'train_samples_per_second': 4.23"; used as a
# fallback when the canonical "performance: <n> <metric>" line is absent.
_HF_PERF_PATTERN = r"train_samples_per_second[\'\"\s:=]+([0-9][0-9.eE+-]*)"

# Benign log substrings that must not be treated as errors (mirrors
# ContainerRunner.run_container).
_DEFAULT_BENIGN_SUBSTRINGS: typing.Tuple[str, ...] = (
    "Failed to establish connection to the metrics exporter agent",
    "RpcError: Running out of retries to initialize the metrics agent",
    "Metrics will not be exported",
    "FutureWarning",
    "Opened result file:",
    "SQLite3 generation ::",
    "rocpd_op:",
    "rpd_tracer:",
)

_DEFAULT_BENIGN_REGEXES: typing.Tuple[str, ...] = (
    r"^E[0-9]{8}.*generateRocpd\.cpp",
    r"^W[0-9]{8}.*simple_timer\.cpp",
    r"^W[0-9]{8}.*generateRocpd\.cpp",
    r"^E[0-9]{8}.*tool\.cpp",
    r"\[rocprofv3\]",
)


def extract_performance_from_log(
    log_file_path: str,
) -> typing.Tuple[typing.Optional[str], typing.Optional[str]]:
    """Extract ``(performance, metric)`` from a run log file.

    Tries the canonical ``performance: <value>[<unit>][,] <metric>`` pattern
    first, then the HuggingFace ``train_samples_per_second`` fallback.

    Args:
        log_file_path: Path to the run log file.

    Returns:
        A ``(performance, metric)`` tuple; either element may be None.
    """
    if not log_file_path or not os.path.exists(log_file_path):
        return None, None

    with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
        log_content = f.read()

    match = re.search(PERFORMANCE_LOG_PATTERN, log_content)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    hf_match = re.search(_HF_PERF_PATTERN, log_content)
    if hf_match:
        return hf_match.group(1).strip(), "samples_per_second"

    return None, None


def log_has_errors(
    log_file_path: str,
    model_info: typing.Dict,
    additional_context: typing.Optional[typing.Dict] = None,
) -> bool:
    """Scan a run log for error patterns (respecting benign exclusions).

    Args:
        log_file_path: Path to the run log file.
        model_info: Model definition dict (may override scan config).
        additional_context: Additional context (may override scan config).

    Returns:
        True if an error pattern was found and scanning is enabled.
    """
    scan_enabled, error_patterns, extra_benign = resolve_log_error_scan_config(
        model_info, additional_context
    )
    if not scan_enabled or not log_file_path or not os.path.exists(log_file_path):
        return False

    benign_substrings = list(_DEFAULT_BENIGN_SUBSTRINGS) + list(extra_benign)

    with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
        log_scan_text = f.read()

    for pattern in error_patterns:
        if log_text_has_error_pattern(
            log_scan_text,
            pattern,
            benign_substrings,
            _DEFAULT_BENIGN_REGEXES,
        ):
            print(f"Found error pattern '{pattern}' in logs")
            return True
    return False


def determine_status(
    log_file_path: str,
    performance: typing.Optional[str],
    model_info: typing.Dict,
    additional_context: typing.Optional[typing.Dict] = None,
) -> str:
    """Determine SUCCESS/FAILURE from log errors and presence of performance.

    Mirrors ContainerRunner.run_container status logic: a run is SUCCESS when it
    has a performance metric and no error patterns; worker nodes
    (``MAD_COLLECT_METRICS=false``) and deferred-collection runs
    (``skip_perf_collection``) succeed without a local metric.

    Args:
        log_file_path: Path to the run log file.
        performance: Extracted performance value (or None).
        model_info: Model definition dict.
        additional_context: Additional context dict.

    Returns:
        "SUCCESS" or "FAILURE".
    """
    additional_context = additional_context or {}

    has_errors = log_has_errors(log_file_path, model_info, additional_context)
    has_performance = bool(
        performance and str(performance).strip() and str(performance).strip() != "N/A"
    )
    is_worker_node = os.environ.get("MAD_COLLECT_METRICS", "true").lower() == "false"

    if has_errors:
        return "FAILURE"
    if has_performance:
        return "SUCCESS"
    if is_worker_node:
        return "SUCCESS"
    if additional_context.get("skip_perf_collection", False):
        return "SUCCESS"
    return "FAILURE"


def resolve_multiple_results_path(
    multiple_results: str, model_dir: str
) -> typing.Optional[str]:
    """Resolve a multiple_results CSV path: try cwd then model_dir.

    Args:
        multiple_results: Relative CSV filename from the model card.
        model_dir: Directory the model ran in.

    Returns:
        First existing path, or None.
    """
    if not multiple_results:
        return None
    if os.path.isfile(multiple_results):
        return multiple_results
    candidate = os.path.join(model_dir, multiple_results)
    if os.path.isfile(candidate):
        return candidate
    return None


def write_perf_records(
    run_details: typing.Dict,
    model_info: typing.Dict,
    perf_csv_path: str,
    status: str,
    model_dir: str = ".",
) -> None:
    """Write perf.csv and perf_super records for a completed run.

    Handles the multiple_results (CSV) case and the single-result case, matching
    ContainerRunner.run_container's reporting. Failures write an exception row.

    Args:
        run_details: Run details dict (from a create_run_details_dict-style call).
        model_info: Model definition dict.
        perf_csv_path: Path to perf.csv.
        status: Run status ("SUCCESS"/"FAILURE").
        model_dir: Directory the model ran in (for multiple_results resolution).
    """
    scripts_path = model_info.get("scripts", "")
    scripts_base_dir = scripts_base_dir_from(scripts_path)

    multiple_results = (model_info.get("multiple_results") or "").strip()
    resolved_multiple = (
        resolve_multiple_results_path(multiple_results, model_dir)
        if multiple_results
        else None
    )

    if resolved_multiple and status == "SUCCESS":
        common_info = run_details.copy()
        for key in ("model", "performance", "metric", "status"):
            common_info.pop(key, None)
        with open("common_info.json", "w") as f:
            json.dump(common_info, f)

        update_perf_csv(
            multiple_results=resolved_multiple,
            perf_csv=perf_csv_path,
            model_name=run_details["model"],
            common_info="common_info.json",
        )
        try:
            num_entries = update_perf_super_json(
                multiple_results=resolved_multiple,
                perf_super_json="perf_super.json",
                model_name=run_details["model"],
                common_info="common_info.json",
                scripts_base_dir=scripts_base_dir,
            )
            update_perf_super_csv(
                perf_super_json="perf_super.json",
                perf_super_csv="perf_super.csv",
                num_entries=num_entries,
            )
        except Exception as e:
            print(f"⚠️  Warning: Could not update perf_super files: {e}")
        return

    # Single-result path.
    with open("perf_entry.json", "w") as f:
        json.dump(run_details, f)

    if status == "SUCCESS":
        update_perf_csv(single_result="perf_entry.json", perf_csv=perf_csv_path)
    else:
        update_perf_csv(exception_result="perf_entry.json", perf_csv=perf_csv_path)

    try:
        if status == "SUCCESS":
            num_entries = update_perf_super_json(
                single_result="perf_entry.json",
                perf_super_json="perf_super.json",
                scripts_base_dir=scripts_base_dir,
            )
        else:
            num_entries = update_perf_super_json(
                exception_result="perf_entry.json",
                perf_super_json="perf_super.json",
                scripts_base_dir=scripts_base_dir,
            )
        update_perf_super_csv(
            perf_super_json="perf_super.json",
            perf_super_csv="perf_super.csv",
            num_entries=num_entries,
        )
    except Exception as e:
        print(f"⚠️  Warning: Could not update perf_super files: {e}")
