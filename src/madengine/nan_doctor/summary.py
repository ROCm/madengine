"""Summarise ONE attempt of a node-wide capsule replay (debug-nan step 5b).

    madengine nan summary <attempt_dir>

<attempt_dir> holds one log per copy (g<gpu>_k<copy>.<host>.log), written by
`madengine nan replay`.  Prints a line per copy, the [C1] walk from the first copy that
has one (it is deterministic, so every copy prints the same table), totals, and finally
exactly one

    ATTEMPT_RESULT=REPRODUCED|CLEAN|NOT_SET_UP|OOM|ERROR

line, which the concurrency ladder reads.  The same text is written to
<attempt_dir>/summary.txt.

Per copy:  NAN      the loop counted >= 1 non-finite run/launch
           CLEAN    the loop ran and counted 0
           STOPPED  0 hits, and the node-wide stop file ended it: not a clean copy
           OOM      the copy died out of device memory
           ERROR    anything else (no loop line) -- read the log
A copy is never scored from its exit code: the chain loop exits 1 when clean.  The attempt
is NOT_SET_UP when the futility stop ended it (FUTILE in the attempt dir) with no hit: the
replay did not recreate the model's trigger.
"""

# flake8: noqa: E741,E731,E402  (terse variable names, kept from the original scripts)

import glob
import io
import os
import re
import sys
from contextlib import redirect_stdout

# Lines that are never part of a verdict: ROCm/aiter/Python runtime warnings.
NOISE_RE = re.compile(
    r"^\[hsa-hotswap|^\[aiter\] (WARNING|import)|UserWarning|^ *warnings\.warn|NUMA balancing"
    r"|^rocblaslt (error|warning)|^hipModuleLoad failed| error: file not found$"
)
LOOP_RE = re.compile(r"non-finite on (\d+)/(\d+) (?:chain runs|launches)")
OOM_RE = re.compile(r"OutOfMemoryError|out of memory|hipErrorOutOfMemory")


def _read(path):
    with open(path, errors="replace") as f:
        return f.read().splitlines()


def _last(pattern, lines):
    hits = [m.group(0) for l in lines for m in [re.search(pattern, l)] if m]
    return hits[-1] if hits else None


def summarize(attempt_dir):
    """Return (text, result)."""
    d = attempt_dir
    out = []
    w = out.append
    logs = sorted(glob.glob(os.path.join(d, "g*_k*.log")))
    if not logs:
        return "no copy logs in %s\nATTEMPT_RESULT=ERROR\n" % d, "ERROR"
    stop = os.path.join(d, "STOP")
    futile = os.path.join(d, "FUTILE")
    n = {"NAN": 0, "CLEAN": 0, "OOM": 0, "ERROR": 0}
    bad = runs = 0
    all_lines = {}
    w("=== copies (%s) ===" % d)
    for f in logs:
        lines = all_lines[f] = _read(f)
        loop = None
        for l in lines:
            m = LOOP_RE.search(l)
            if m:
                loop = m
        verdict = _last(r"^\[verdict\] .*", lines)
        if loop:
            b, r = int(loop.group(1)), int(loop.group(2))
            bad += b
            runs += r
            if b > 0:
                st = "NAN"
                n["NAN"] += 1
            elif os.path.exists(stop):
                st = "STOPPED"
            else:
                st = "CLEAN"
                n["CLEAN"] += 1
        elif any(OOM_RE.search(l) for l in lines):
            st = "OOM"
            n["OOM"] += 1
        else:
            st = "ERROR"
            n["ERROR"] += 1
        w(
            "  %-24s %-5s %-40s %s"
            % (os.path.basename(f), st, loop.group(0) if loop else "-", verdict or "-")
        )

    shown = False
    for f in logs:
        lines = all_lines[f]
        if any(l.startswith("[C1] chain walk") for l in lines):
            w("=== [C1] walk (deterministic; from %s) ===" % os.path.basename(f))
            seg, on = [], False
            for l in lines:
                if l.startswith("[C1] chain walk"):
                    on = True
                if on:
                    seg.append(l)
                    if l.startswith("[verdict]"):
                        break
            dropped = 0
            for l in seg:
                if NOISE_RE.search(l):
                    dropped += 1
                else:
                    w(l)
            if dropped:
                w(
                    "  (%d runtime-noise line(s) dropped; full text in %s)"
                    % (dropped, os.path.basename(f))
                )
            shown = True
            break
    if not shown:
        w("=== [C1] walk: none (no chain in the capsule, or no copy reached it) ===")

    flat = [l for f in logs for l in all_lines[f]]
    envs = {}
    for l in flat:
        if l.startswith("env:"):
            envs[l] = envs.get(l, 0) + 1
    w("=== env banners ===")
    if envs:
        for l, c in sorted(envs.items()):
            w("%7d %s" % (c, l))
    else:
        w("  (none)")
    build = sorted({l for l in flat if l.startswith("captured build:")})
    if build:
        w(build[0])
    for l in sorted({l for l in flat if l.startswith("operands:")}):
        w(l)
    for l in sorted({l for l in flat if l.startswith("[verdict-flag]")}):
        w(l)

    w(
        "=== totals: copies NAN %d / CLEAN %d / OOM %d / ERROR %d; non-finite %d/%d ==="
        % (n["NAN"], n["CLEAN"], n["OOM"], n["ERROR"], bad, runs)
    )
    if os.path.exists(futile):
        w(
            "=== futility stop (%s): 0 non-finite in the runs the model needed many times over -- "
            "the replay is not recreating the model's trigger (NOT_SET_UP, not a negative) ==="
            % _first_line(stop)
        )
    elif os.path.exists(stop):
        w(
            "=== stopped early (%s): rate = non-finite / runs actually done; stopping on a hit count "
            "biases it slightly high (~1/hits) ===" % _first_line(stop)
        )
    if n["NAN"] > 0:
        res = "REPRODUCED"
    elif n["OOM"] > 0:
        res = "OOM"
    elif os.path.exists(futile) and n["ERROR"] == 0:
        res = "NOT_SET_UP"
    elif n["CLEAN"] > 0:
        res = "CLEAN"
    else:
        res = "ERROR"
    if res in ("REPRODUCED", "CLEAN"):
        # The loop's first-non-finite tally and the [C1]-proposed verdict, so reading an
        # attempt needs this file only.
        from . import verdict

        buf = io.StringIO()
        with redirect_stdout(buf):
            verdict.attempt(d, res == "REPRODUCED")
        text = buf.getvalue()
        if text:
            w(text.rstrip("\n"))
    w("ATTEMPT_RESULT=%s" % res)
    return "\n".join(out) + "\n", res


def _first_line(path):
    try:
        return _read(path)[0]
    except (OSError, IndexError):
        return "?"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0].startswith("-"):
        print(__doc__.strip().split("\n\n")[0], file=sys.stderr)
        return 2
    d = argv[0]
    if not os.path.isdir(d):
        print("no such directory: %s" % d, file=sys.stderr)
        return 2
    text, _ = summarize(d)
    sys.stdout.write(text)
    try:
        with open(os.path.join(d, "summary.txt"), "w") as f:
            f.write(text)
    except OSError as e:
        print("could not write summary.txt: %s" % e, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
