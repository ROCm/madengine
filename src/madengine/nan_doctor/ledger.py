#!/usr/bin/env python3
"""Append to a debug-nan <LEDGER> and do its timing arithmetic (SKILL.md rule 7).

    L=<ARTIFACT_DIR>/ledger.md
    madengine nan ledger $L start step4
    madengine nan ledger $L note  step4 rate=3/8 (host, image, ...)
    madengine nan ledger $L job   step4 370877 capture wave, 16 trials
    madengine nan ledger $L wait  step4 start confirm wave geometry
    madengine nan ledger $L wait  step4 end
    madengine nan ledger $L end   step4          # prints the one-liner for the user
    madengine nan ledger $L table                # step-9 per-step table

Every line is `YYYY-MM-DD HH:MM:SS  <step>  <text>` (`date '+%F %T'`).  `end`
finds the step's latest `start`, sums the `waiting-user` pairs in between, and
splits the visit into queue / compute / waiting-user / other from `sacct` for every
`job=<id>` recorded in the visit (compute = union of the jobs' run intervals, so
overlapping jobs count once).  Without sacct (docker) queue and compute print n/a
and stay in `other`.  Steps 5b-8 are tagged per group: `step6[G1]`.
"""

import collections
import datetime as dt
import re
import subprocess
import sys

FMT = "%Y-%m-%d %H:%M:%S"
LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s+(\S+)\s+(.*)$")


def now():
    return dt.datetime.now().replace(microsecond=0)


def hms(sec):
    sec = int(round(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return ("%dh%02dm%02ds" % (h, m, s)) if h else ("%dm%02ds" % (m, s))


def unhms(txt):
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", txt)
    if not m or not txt:
        return None
    return (
        int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + int(m.group(3) or 0)
    )


def lines(path):
    out = []
    with open(path) as f:
        for raw in f:
            m = LINE.match(raw.rstrip("\n"))
            if m:
                out.append(
                    (dt.datetime.strptime(m.group(1), FMT), m.group(2), m.group(3))
                )
    return out


def append(path, step, text):
    ln = "%s  %s  %s" % (now().strftime(FMT), step, text)
    with open(path, "a") as f:
        f.write(ln + "\n")
    print(ln)


def sacct(jobs):
    """{job: (submit, start, end)} or None if sacct is unavailable."""
    try:
        out = subprocess.run(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-o",
                "JobID,Submit,Start,End",
                "-j",
                ",".join(jobs),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    res = {}
    for row in out.splitlines():
        f = row.split("|")
        if len(f) != 4:
            continue
        t = []
        for x in f[1:]:
            try:
                t.append(dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S"))
            except ValueError:  # Unknown: not started / still running
                t.append(None)
        res[f[0]] = t
    return res


def union(ivs, lo, hi):
    """Seconds covered by the intervals, clipped to [lo, hi], overlaps once."""
    tot, cur = 0.0, None
    for a, b in sorted((max(a, lo), min(b, hi)) for a, b in ivs):
        if b <= a:
            continue
        if cur and a <= cur[1]:
            cur = (cur[0], max(cur[1], b))
        else:
            if cur:
                tot += (cur[1] - cur[0]).total_seconds()
            cur = (a, b)
    if cur:
        tot += (cur[1] - cur[0]).total_seconds()
    return tot


def visit(entries, step):
    """Entries of the step's latest visit (from its last `start`)."""
    idx = [
        i
        for i, (_, s, txt) in enumerate(entries)
        if s == step and txt.split()[:1] == ["start"]
    ]
    if not idx:
        sys.exit("no `%s start` line in the ledger" % step)
    return entries[idx[-1] :]


def end(path, step):
    entries = lines(path)
    v = visit(entries, step)
    t0, t1 = v[0][0], now()
    total = (t1 - t0).total_seconds()
    wait, wstart = 0.0, None
    jobs = []
    for ts, s, txt in v:
        if s != step:
            continue
        if txt.startswith("waiting-user start"):
            wstart = ts
        elif txt.startswith("waiting-user end") and wstart:
            wait += (ts - wstart).total_seconds()
            wstart = None
        jobs += re.findall(r"\bjob=(\d+)", txt)
    jobs = list(dict.fromkeys(jobs))
    # A bare `step7` visit that wraps per-group `step7[G1]` visits: those report their
    # own time on their own rows, so this one keeps only what lies outside them (the
    # entry questions) -- otherwise the table counts the group loop twice.
    nested = [
        (m, unhms(m.group(1)))
        for ts, s, txt in v
        if "[" not in step and s.startswith(step + "[") and txt.startswith("end")
        for m in [re.search(r"\btotal=(\S+)", txt)]
        if m
    ]
    inner = sum(sec or 0 for _, sec in nested)
    total = max(total - inner, 0)
    parts = ["total=%s" % hms(total)]
    queue = compute = None
    notes = ["excludes %s of nested group visit(s)" % hms(inner)] if nested else []
    if jobs:
        acct = sacct(jobs)
        if acct is None:
            notes.append("queue/compute n/a (no sacct)")
        else:
            qiv, civ = [], []
            for j in jobs:
                t = acct.get(j)
                if not t or t[0] is None:
                    notes.append("job %s unknown" % j)
                    continue
                sub, st, en = t
                if st is None:  # still queued
                    qiv.append((sub, t1))
                    continue
                qiv.append((sub, st))
                civ.append((st, en or t1))  # en None = still running: up to now
            # only what falls inside this visit counts, overlaps once -- a job that
            # spans several steps (the capture-to-reproducer job) is split by step
            queue, compute = union(qiv, t0, t1), union(civ, t0, t1)
    parts += [
        "queue=%s" % (hms(queue) if queue is not None else "n/a"),
        "compute=%s" % (hms(compute) if compute is not None else "n/a"),
        "waiting-user=%s" % hms(wait),
        "other=%s" % hms(max(total - (queue or 0) - (compute or 0) - wait, 0)),
        "jobs=%s" % (",".join(jobs) or "none"),
    ]
    append(
        path,
        step,
        "end  " + "  ".join(parts) + ("  (%s)" % "; ".join(notes) if notes else ""),
    )
    d = dict(p.split("=", 1) for p in parts)
    print(
        "TELL USER: %s done in %s (queue %s, compute %s, waiting for you %s, other %s)."
        % (
            re.sub(r"^step", "Step ", step),
            d["total"],
            d["queue"],
            d["compute"],
            d["waiting-user"],
            d["other"],
        )
    )


def table(path):
    rows = collections.OrderedDict()
    for _, s, txt in lines(path):
        if txt.split()[:1] != ["end"]:
            continue
        r = rows.setdefault(s, collections.Counter())
        r["visits"] += 1
        for k, val in re.findall(
            r"\b(total|queue|compute|waiting-user|other)=(\S+)", txt
        ):
            sec = unhms(val)
            if sec is not None:
                r[k] += sec
    cols = ["total", "queue", "compute", "waiting-user", "other"]
    print("| step | visits | total | queue | compute | waiting for user | other |")
    print("|---|---|---|---|---|---|---|")
    tot = collections.Counter()
    for s, r in rows.items():
        print(
            "| %s | %d | %s |" % (s, r["visits"], " | ".join(hms(r[c]) for c in cols))
        )
        tot.update(r)
    print(
        "| **total** | %d | %s |"
        % (tot["visits"], " | ".join(hms(tot[c]) for c in cols))
    )
    print(
        "(steps with no `end` line -- skipped or still open -- are not listed; add them by hand)"
    )


PER_GROUP = re.compile(r"^step(5b|6|7|8)$")


def warn_untagged(path, step):
    """Steps 5b-8 are timed per group (`step7[G1]`).  A bare visit next to tagged ones
    holds the whole group loop in its `other` column and reads as work in the table."""
    if PER_GROUP.match(step):
        print(
            "ledger: WARNING %s is a per-group step -- start %s[G<i>] instead (SKILL.md "
            "rule 7)" % (step, step),
            file=sys.stderr,
        )
        return
    base = step.split("[")[0]
    if base != step and PER_GROUP.match(base):
        entries = lines(path)
        starts = [
            i
            for i, (_, s_, t) in enumerate(entries)
            if s_ == base and t.split()[:1] == ["start"]
        ]
        ends = [
            i
            for i, (_, s_, t) in enumerate(entries)
            if s_ == base and t.split()[:1] == ["end"]
        ]
        if starts and (not ends or ends[-1] < starts[-1]):
            print(
                "ledger: WARNING a bare %s visit is still open -- `end %s` it (or it counts "
                "the whole %s loop twice)" % (base, base, step),
                file=sys.stderr,
            )


def main():
    a = sys.argv[1:]
    if len(a) < 2:
        sys.exit(__doc__)
    path, cmd, rest = a[0], a[1], a[2:]
    if cmd == "table":
        return table(path)
    if not rest:
        sys.exit(__doc__)
    step, text = rest[0], " ".join(rest[1:])
    if cmd == "start":
        warn_untagged(path, step)
        append(path, step, ("start  " + text).rstrip())
    elif cmd == "note":
        append(path, step, text)
    elif cmd == "job":
        append(path, step, ("job=%s  %s" % (rest[1], " ".join(rest[2:]))).rstrip())
    elif cmd == "wait":
        append(
            path, step, ("waiting-user %s  %s" % (rest[1], " ".join(rest[2:]))).rstrip()
        )
    elif cmd == "end":
        end(path, step)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
