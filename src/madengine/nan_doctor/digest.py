#!/usr/bin/env python3
"""The capture's own story from a run's logs, in a few dozen lines (debug-nan 5-6).

    capture_digest.py <log | .json | .json.gz | run dir> ... [--width N] [--max N]

Prints the [op-nan] lines -- armed, op filter, pending, MASKED, captured,
settle, nan origin, census, deferred-check stats -- and the smoke test's loop
locator ([op-nan-locate], locate/sitecustomize.py) from wherever the workload
put them: a plain log, or a harness's JSON facts file (log text embedded in
JSON strings, e.g. a facts/logs.json.gz).  A directory is searched for
*.log, *.txt, *.json and *.json.gz.  A log file with no [op-nan] lines of its own is
followed to the run dir it names (`finalized run facts complete: <dir>`, `run_dir=<dir>`)
or a sibling facts/ dir -- the capture's lines live there for such harnesses.  Identical lines are printed once (a line
repeated per step or per rank would otherwise dominate), only the last
"Progress" line is kept, and every line is cut to --width characters.  A run
whose capture never constructed (no [op-nan] line beyond the locator's) is
called out: the wired loop is not on the executed path, or the NAN_* env did
not reach the process.

Exists because reading these by hand -- zgrep on gzipped JSON, unescaping, then
filtering the runtime's warnings out of an 8,000-line log -- cost more tokens
per question than the answer was worth.
"""

import argparse
import gzip
import json
import os
import re
import sys

TAG = re.compile(r"\[(?:op|gemm)-nan")  # gemm-nan: logs from before the rename
LOCATE_TAG = re.compile(r"\[(?:op|gemm)-nan-locate\]")
NEVER = (
    "!! CAPTURE NEVER CONSTRUCTED: the wired loop is not on the executed path, or the "
    "NAN_* env did not reach the process"
)


def _text(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as f:
        return f.read()


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def _lines(path):
    raw = _text(path)
    if path.endswith((".json", ".json.gz")):
        try:
            for s in _strings(json.loads(raw)):
                yield from s.splitlines()
            return
        except ValueError:
            pass
    yield from raw.splitlines()


def _files(target):
    if os.path.isfile(target):
        return [target]
    out = []
    for root, _, names in os.walk(target):
        for n in sorted(names):
            if n.endswith((".log", ".txt", ".json", ".json.gz")):
                out.append(os.path.join(root, n))
    return out


RUN_DIR_RE = re.compile(
    r"(?:finalized run facts complete|run[ _-]?dir|RUN_DIR|output dir)[:=]\s*(/\S+)"
)


def linked_run_dirs(path):
    """Run dirs a log names (a harness that keeps its own logs elsewhere, e.g.
    `finalized run facts complete: <dir>`), plus a sibling facts/ dir; existing dirs only.
    """
    out = []
    try:
        for line in _lines(path):
            for m in RUN_DIR_RE.finditer(line):
                d = m.group(1).rstrip("'\",.;")
                if os.path.isdir(d) and d not in out:
                    out.append(d)
    except OSError:
        pass
    sib = os.path.join(os.path.dirname(os.path.abspath(path)), "facts")
    if os.path.isdir(sib) and sib not in out:
        out.append(sib)
    return out


def digest(path, width, cap):
    seen, keep, last_progress, total = set(), [], None, 0
    for line in _lines(path):
        m = TAG.search(line)
        if not m:
            continue
        line = line[m.start() :].rstrip()
        total += 1
        if "] Progress step" in line:
            last_progress = line
            continue
        if line in seen:
            continue
        seen.add(line)
        keep.append(line)
    if last_progress:
        keep.append(last_progress)
    if not keep:
        return None
    never = all(LOCATE_TAG.match(line) for line in keep)
    out = ["== %s  (%d [op-nan] lines, %d distinct)" % (path, total, len(keep))]
    for line in keep[:cap]:
        out.append(line if len(line) <= width else line[: width - 3] + "...")
    if len(keep) > cap:
        out.append("... %d more distinct line(s); raise --max" % (len(keep) - cap))
    if never:
        out.append(NEVER + " -- the locator's NEXT line names the loop to wire")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--width", type=int, default=200)
    ap.add_argument("--max", type=int, default=60, help="distinct lines per source")
    a = ap.parse_args()
    found = False
    for t in a.paths:
        hit = False
        for f in _files(t):
            d = digest(f, a.width, a.max)
            if d:
                print(d)
                hit = found = True
        if not hit and os.path.isfile(t):
            for rd in linked_run_dirs(t):
                print(
                    "(no [op-nan] lines in %s; following the run dir it names: %s)"
                    % (t, rd)
                )
                for f in _files(rd):
                    d = digest(f, a.width, a.max)
                    if d:
                        print(d)
                        found = True
    if not found:
        print("no [op-nan] lines in %s" % " ".join(a.paths))
        print(
            NEVER
            + " -- or the run died before the loop.  Re-run the smoke test with the "
            "locator on PYTHONPATH (step4-capture.md 4.2) to see which loop runs."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
