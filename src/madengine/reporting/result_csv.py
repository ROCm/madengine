"""Find the results CSV a model wrote, by the shape of its header.

A model card can declare where its results CSV lives (``multiple_results``), but the
field is optional and its value has to agree with a path a script in another repository
builds by hand. When the two disagree madengine used to fall back to scraping the log,
find nothing there either, and record an empty FAILURE row for a run that had measured
perfectly well. The shape of the file is the sturdier contract: a results CSV is one
whose header carries ``model``, ``performance`` and ``metric``, in any order and with
any number of extra columns. A declared file that exists still wins; discovery only
answers when the declaration is absent or does not resolve.
"""

import csv
import os
import typing
from pathlib import Path

#: A results CSV is recognised by these three columns, whatever else it carries.
REQUIRED_COLUMNS = ("model", "performance", "metric")

#: madengine's own outputs satisfy the predicate by construction, so discovering one
#: would feed a run its own previous verdict back as input.
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

    This is the question to ask about a file the model card named: it was declared, so it
    is trusted to be the results CSV, and all that is left to check is whether a metric
    can be read out of it.
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


def rejection_reason(path: typing.Union[str, Path]) -> typing.Optional[str]:
    """Why *path* cannot be *discovered* as a results CSV, or None when it can.

    Stricter than :func:`metric_rejection_reason` on purpose: nobody pointed at this file,
    so it has to look like a results CSV on its own -- all three columns, and a number in
    at least one row -- before madengine reads a run's verdict out of it.
    """
    if is_own_output(path):
        return "madengine's own output"
    missing = missing_columns(path)
    if missing is None:
        return "not readable as CSV"
    if missing:
        columns = read_columns(path) or []
        found = ", ".join(columns) if columns else "(no header)"
        return f"header lacks {', '.join(missing)}; found: {found}"
    return metric_rejection_reason(path)


def rank_candidates(
    candidates: typing.Sequence[typing.Union[str, Path]]
) -> typing.List[Path]:
    """Candidates best first: most measured rows, then most rows, then newest, then name.

    In a multi-node run every node copies its own CSV into the collection directory and
    only some of them observed the final throughput, so the file with the most non-empty
    ``performance`` rows is the one carrying the measurement. The remaining keys exist so
    that two runs over the same inputs pick the same file: newest first, and failing that
    the order the caller offered them in, which is the order the directories were searched.
    """

    def sort_key(entry: typing.Tuple[int, Path]) -> typing.Tuple:
        position, candidate = entry
        with_metric, total = count_rows(candidate)
        try:
            mtime = os.path.getmtime(candidate)
        except OSError:
            mtime = 0.0
        return (-with_metric, -total, -mtime, position)

    numbered = list(enumerate(Path(c) for c in candidates))
    return [candidate for _, candidate in sorted(numbered, key=sort_key)]


def select_best(
    candidates: typing.Sequence[typing.Union[str, Path]]
) -> typing.Optional[Path]:
    """The best of *candidates*, or None when there are none."""
    ranked = rank_candidates(candidates)
    return ranked[0] if ranked else None


def _staleness_reason(
    path: typing.Union[str, Path], min_mtime: typing.Optional[float]
) -> typing.Optional[str]:
    """"Written before this run started", when that is what happened."""
    if min_mtime is None:
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return "not readable"
    # A second of slack: file timestamps and the clock this run started by do not always
    # come from the same source, and a file written in the first moments of the run is
    # this run's.
    if mtime < min_mtime - 1.0:
        return "written before this run started"
    return None


class Discovery(typing.NamedTuple):
    """What a search found, and enough of what it discarded to explain itself."""

    winner: typing.Optional[Path]
    candidates: typing.List[Path]
    rejected: typing.List[typing.Tuple[Path, str]]
    searched: typing.List[Path]
    seen: int


def discover(
    search_dirs: typing.Sequence[typing.Union[str, Path]],
    excluded: typing.Sequence[typing.Union[str, Path]] = (),
    min_mtime: typing.Optional[float] = None,
) -> Discovery:
    """Look for a results CSV directly inside each of *search_dirs*.

    Depth 1 only: a training run can leave hundreds of CSVs behind and walking the tree
    would turn a diagnostic into a scan. Directories are searched in the order given,
    duplicates of the same file are collapsed, and *excluded* paths are skipped along
    with madengine's own outputs.

    *min_mtime* is how a shared working directory stays safe: several models run one after
    another in the same place, and the one that ran before this one left its results CSV
    behind. A file that was not written during this run cannot be this run's result.
    """
    excluded_real = {os.path.realpath(str(path)) for path in excluded}
    searched: typing.List[Path] = []
    seen_files: typing.Dict[str, Path] = {}

    for directory in search_dirs:
        if directory is None:
            continue
        as_path = Path(directory)
        if not as_path.is_dir():
            continue
        real_dir = os.path.realpath(str(as_path))
        if real_dir in {os.path.realpath(str(d)) for d in searched}:
            continue
        searched.append(as_path)
        for entry in sorted(as_path.glob("*.csv")):
            if not entry.is_file():
                continue
            real = os.path.realpath(str(entry))
            if real in excluded_real or real in seen_files:
                continue
            seen_files[real] = entry

    candidates: typing.List[Path] = []
    rejected: typing.List[typing.Tuple[Path, str]] = []
    for entry in seen_files.values():
        reason = _staleness_reason(entry, min_mtime) or rejection_reason(entry)
        if reason is None:
            candidates.append(entry)
        else:
            rejected.append((entry, reason))

    ranked = rank_candidates(candidates)
    return Discovery(
        winner=ranked[0] if ranked else None,
        candidates=ranked,
        rejected=rejected,
        searched=searched,
        seen=len(seen_files),
    )


def describe(discovery: Discovery, limit: int = 4) -> typing.List[str]:
    """A few lines saying where the search looked and why each file was discarded."""
    where = ", ".join(str(path) for path in discovery.searched) or "(no directory to search)"
    lines = [f"Searched for a results CSV in: {where}", f"CSV files seen: {discovery.seen}"]
    for path, reason in discovery.rejected[:limit]:
        lines.append(f"  rejected {path}: {reason}")
    remaining = len(discovery.rejected) - limit
    if remaining > 0:
        lines.append(f"  ... and {remaining} more rejected the same way")
    return lines
