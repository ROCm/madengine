"""Shared helpers for run-details / perf row construction.

Centralizes pipeline and build_number from environment so container_runner
and deployment (e.g. kubernetes) use the same values. Also provides
flatten_tags_in_place for consistent tags handling.
"""

import os
from typing import Any, Dict


def get_pipeline() -> str:
    """Return pipeline name from environment (e.g. CI pipeline)."""
    return os.environ.get("pipeline", "")


def get_build_number() -> str:
    """Return build number from environment (e.g. CI BUILD_NUMBER)."""
    return os.environ.get("BUILD_NUMBER", "0")


def flatten_tags_in_place(record: Dict[str, Any]) -> None:
    """Flatten 'tags' in a run-details record in place.

    If record['tags'] is a list, convert it to a comma-separated string.
    Matches behavior expected by perf CSV and update_perf_csv.
    """
    if "tags" not in record:
        return
    if isinstance(record["tags"], list):
        record["tags"] = ",".join(str(item) for item in record["tags"])
