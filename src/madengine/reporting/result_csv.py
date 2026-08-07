"""Recognise a results CSV by the shape of its header, to explain a missing metric.

A model card declares where its results CSV lives (``multiple_results``), and that
declaration stays the only thing a run reports from. The field is optional, though, and
its value has to agree with a path a script in another repository builds by hand, so when
the two disagree a run has to say something better than "metric not found in expected
format": a results CSV is one whose header carries ``model``, ``performance`` and
``metric``, in any order and with any number of extra columns, and a run that read no
metric can at least name the files beside it that look like one.

Nothing here chooses a file to report from. A CSV nobody declared is named in a message
and never read for a verdict, so a typo in a card and a card that declares nothing read
differently while both still fail.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import csv
import os
import typing
from pathlib import Path

#: A results CSV is recognised by these three columns, whatever else it carries.
REQUIRED_COLUMNS = ("model", "performance", "metric")

#: madengine's own outputs satisfy the predicate by construction, so naming one as a
#: candidate would point a reader at a run's own previous verdict.
OWN_OUTPUT_NAMES = frozenset({"perf.csv"})
OWN_OUTPUT_PREFIXES = ("perf_super", "perf_entry")


def normalise_columns(fieldnames: typing.Optional[typing.Iterable[str]]) -> typing.List[str]:
    """Column names as they are compared: stripped of space, quotes and case."""
    return [
        (name or "").strip().strip('"').strip("'").strip().lower()
        for name in (fieldnames or [])
    ]


def read_columns(path: typing.Union[str, Path]) -> typing.Optional[typing.List[str]]:
    """The normalised header of *path*, or None when it cannot be read as CSV."""
    try:
        # utf-8-sig so a byte-order mark does not hide behind the first column name.
        with open(path, "r", newline="", encoding="utf-8-sig", errors="ignore") as handle:
            for row in csv.reader(handle):
                return normalise_columns(row)
    except (OSError, csv.Error):
        return None
    return None


def missing_columns(path: typing.Union[str, Path]) -> typing.Optional[typing.List[str]]:
    """Which required columns *path* lacks; None when it is not readable as CSV."""
    columns = read_columns(path)
    if columns is None:
        return None
    return [name for name in REQUIRED_COLUMNS if name not in columns]


def has_result_shape(path: typing.Union[str, Path]) -> bool:
    """True when *path* parses as CSV and its header carries all required columns."""
    return missing_columns(path) == []


def is_own_output(path: typing.Union[str, Path]) -> bool:
    """True for the files madengine writes itself (perf.csv and the super/entry family)."""
    name = os.path.basename(str(path)).lower()
    return name in OWN_OUTPUT_NAMES or name.startswith(OWN_OUTPUT_PREFIXES)


def count_rows(path: typing.Union[str, Path]) -> typing.Tuple[int, int]:
    """(rows with a non-empty ``performance``, total rows) in *path*."""
    with_metric = 0
    total = 0
    try:
        with open(path, "r", newline="", encoding="utf-8-sig", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            reader.fieldnames = normalise_columns(reader.fieldnames)
            if "performance" not in (reader.fieldnames or []):
                return (0, 0)
            for row in reader:
                total += 1
                value = row.get("performance") or ""
                if str(value).strip():
                    with_metric += 1
    except (OSError, csv.Error):
        return (0, 0)
    return (with_metric, total)


def metric_rejection_reason(path: typing.Union[str, Path]) -> typing.Optional[str]:
    """Why *path* yields no metric, or None when it does.

    This is the question to ask about the file a model card named: it was declared, so it
    is trusted to be the results CSV, and all that is left to check is whether a metric
    can be read out of it. A card that reports through a two-column CSV keeps working,
    which is why the three-column shape is not required here.
    """
    columns = read_columns(path)
    if columns is None:
        return "not readable as CSV"
    if "performance" not in columns:
        found = ", ".join(columns) if columns else "(no header)"
        return f"no 'performance' column; found: {found}"
    if count_rows(path)[0] == 0:
        return "every row has an empty 'performance' value"
    return None


def suggest_candidates(
    search_dirs: typing.Sequence[typing.Union[str, Path]],
) -> typing.List[Path]:
    """CSVs lying directly in *search_dirs* whose header says they are results files.

    Depth 1 only, and only ever to phrase a message: a training run can leave hundreds of
    CSVs behind, and walking the tree would turn a diagnostic into a scan.
    """
    found: typing.Dict[str, Path] = {}
    for directory in search_dirs:
        if directory is None:
            continue
        as_path = Path(directory)
        if not as_path.is_dir():
            continue
        for entry in sorted(as_path.glob("*.csv")):
            real = os.path.realpath(str(entry))
            if real in found or not entry.is_file():
                continue
            if is_own_output(entry) or not has_result_shape(entry):
                continue
            found[real] = entry
    return list(found.values())


def suggestion_lines(
    search_dirs: typing.Sequence[typing.Union[str, Path]], limit: int = 3
) -> typing.List[str]:
    """Lines naming the results-looking CSVs beside a run, or saying there were none."""
    where = ", ".join(str(Path(d)) for d in search_dirs if d) or "(nowhere to look)"
    candidates = suggest_candidates(search_dirs)
    if not candidates:
        return [f"No CSV with model, performance and metric columns in: {where}"]
    lines = [
        "These files look like results CSVs; declare one in the model card's "
        "multiple_results:"
    ]
    lines.extend(f"  {path}" for path in candidates[:limit])
    remaining = len(candidates) - limit
    if remaining > 0:
        lines.append(f"  ... and {remaining} more")
    return lines
