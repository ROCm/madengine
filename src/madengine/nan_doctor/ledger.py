#!/usr/bin/env python3
"""Append to a mad-debug-numeric <LEDGER> and do its timing arithmetic (SKILL.md rule 7).

    L=<ARTIFACT_DIR>/ledger.md
    madengine nan ledger $L start step4
    madengine nan ledger $L note  step4 rate=3/8 (host, image, ...)
    madengine nan ledger $L job   step4 370877 capture wave, 16 trials
    madengine nan ledger $L wait  step4 start confirm wave geometry
    madengine nan ledger $L wait  step4 end
    madengine nan ledger $L end   step4          # prints the one-liner for the user
    madengine nan ledger $L skip  step2 the user chose to skip it   # a step never run: no visit
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
import os
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
    if not os.path.exists(path):  # a new ledger: the first `start` creates it
        return out
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
    print("ledgered: " + ln)  # a confirmation, not a second copy of the ledger line


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
            qiv, civ, running = [], [], []
            for j in jobs:
                t = acct.get(j)
                if not t or t[0] is None:
                    notes.append("job %s unknown" % j)
                    continue
                sub, st, en = t
                if st is None:  # never started: queued up to its cancel, or now
                    qiv.append((sub, en or t1))
                    if en:
                        notes.append("job %s never ran" % j)
                    continue
                qiv.append((sub, st))
                civ.append((st, en or t1))  # en None = still running: up to now
                if en is None:
                    running.append(j)
            # only what falls inside this visit counts, overlaps once -- a job that
            # spans several steps (the capture-to-reproducer job) is split by step.
            # Queue = waiting with NONE of the visit's jobs computing: a job held or
            # pinned to a busy node while others ran is not time lost to the queue
            # (a held job and stopped pinned ones once put 22 min of queue on a
            # step whose other jobs ran throughout).
            compute = union(civ, t0, t1)
            queue = max(union(qiv + civ, t0, t1) - compute, 0)
            if running:
                # e.g. an image build submitted in step 1 and still building when step 1
                # ended: its later compute falls in no step's row
                notes.append("job(s) %s still running" % ",".join(running))
                print(
                    "WARNING job(s) %s still running: the rest of their compute is in no "
                    "step's row unless the step that waits on them records them: "
                    "madengine nan ledger %s job <that step> <id>"
                    % (",".join(running), path)
                )
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
            re.sub(
                r"^step(\d+)([a-z]*)",
                lambda m: "Step %s%s" % (m.group(1), m.group(2).upper()),
                step,
            ),
            d["total"],
            d["queue"],
            d["compute"],
            d["waiting-user"],
            d["other"],
        )
    )


def table(path):
    rows, skipped = collections.OrderedDict(), collections.OrderedDict()
    for _, s, txt in lines(path):
        if txt.split()[:1] == ["skipped"]:
            skipped[s] = txt.split(None, 1)[1].strip() if len(txt.split()) > 1 else ""
            continue
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
    for s in sorted(set(rows) | set(skipped), key=step_key):
        if s in rows:
            r = rows[s]
            print(
                "| %s | %d | %s |"
                % (s, r["visits"], " | ".join(hms(r[c]) for c in cols))
            )
            tot.update(r)
        else:
            print(
                "| %s | 0 | *skipped*%s | - | - | - | - |"
                % (s, ": " + skipped[s] if skipped[s] else "")
            )
    out = outside_compute(path)
    if out:
        # a job's compute while no step was open (an image build that ends between step 1
        # and step 4) gets its own row and counts in the total
        print(
            "| *outside any step* | - | %s | - | %s | - | - |"
            % (hms(out[0]), hms(out[0]))
        )
        tot["total"] += out[0]
        tot["compute"] += out[0]
    print(
        "| **total** | %d | %s |"
        % (tot["visits"], " | ".join(hms(tot[c]) for c in cols))
    )
    print(
        "(steps with neither an `end` nor a `skip` line -- still open -- are not listed)"
    )
    ov = overlaps(path)
    if ov:
        print(
            "ledger: WARNING overlapping visits (their times add up twice in the total): %s"
            % "; ".join("%s / %s %s" % o for o in ov[:6])
        )


def outside_compute(path):
    """(seconds, jobs) of recorded jobs' compute that fell in no step's visit, or None
    (none, or no sacct)."""
    entries = lines(path)
    open_, ivs, jobs = {}, [], []
    for ts, s, txt in entries:
        w = txt.split()[:1]
        if w == ["start"]:
            open_[s] = ts
        elif w == ["end"] and s in open_:
            ivs.append((open_.pop(s), ts))
        jobs += re.findall(r"\bjob=(\d+)", txt)
    jobs = list(dict.fromkeys(jobs))
    if not jobs or not ivs:
        return None
    acct = sacct(jobs)
    if not acct:
        return None
    civ = [(t[1], t[2]) for t in acct.values() if t and t[1] and t[2]]
    if not civ:
        return None
    lo, hi = min(a for a, _ in civ), max(b for _, b in civ)
    merged = []
    for a, b in sorted(ivs):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    outside = union(civ, lo, hi) - sum(union(civ, a, b) for a, b in merged)
    return (outside, jobs) if outside >= 60 else None


def step_key(s):
    """step4 < step5 < step5a < step5b[G1] < step6[G1] < ... (was: first appearance)."""
    m = re.match(r"step(\d+)([a-z]*)(?:\[(.*)\])?", s)
    return (int(m.group(1)), m.group(2), m.group(3) or "") if m else (99, s, "")


def overlaps(path):
    """(step a, step b, 'from-to') for visits of different steps that overlap in time."""
    open_, iv = {}, []
    for ts, s, txt in lines(path):
        w = txt.split()[:1]
        if w == ["start"]:
            open_[s] = ts
        elif w == ["end"] and s in open_:
            iv.append((s, open_.pop(s), ts))
    out = []
    for i, (a, a0, a1) in enumerate(iv):
        for b, b0, b1 in iv[i + 1 :]:
            if a != b and a0 < b1 and b0 < a1:
                out.append(
                    (
                        a,
                        b,
                        "%s-%s"
                        % (
                            max(a0, b0).strftime("%H:%M:%S"),
                            min(a1, b1).strftime("%H:%M:%S"),
                        ),
                    )
                )
    return out


PER_GROUP = re.compile(r"^step(5b|6|7|8)$")


def warn_untagged(path, step):
    """Steps 5b-8 are timed per group (`step7[G1]`).  A bare visit next to tagged ones
    holds the whole group loop in its `other` column and reads as work in the table."""
    if PER_GROUP.match(step):
        # refused, not warned after the fact: the visit would already be in the ledger
        sys.exit(
            "ledger: %s is a per-group step -- `start %s[G<i>]` (or `[native]`); nothing "
            "written (SKILL.md rule 7)" % (step, step)
        )
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


def warn_not_started(path, step, cmd):
    """A note/job for a step with no open visit: its time lands in no row (or the
    previous step's), so the table under-reports it."""
    if not step.startswith("step") or step == "step0":
        return  # a job's own tag (capture_to_repro); step 0 has no visit
    entries = lines(path)
    starts = [
        i
        for i, (_, s_, t) in enumerate(entries)
        if s_ == step and t.split()[:1] == ["start"]
    ]
    # A note on a step that has run (a result read after its job closed the step, a
    # resume note) has no time to lose: only a job, or a note for a step never started,
    # is worth a warning
    if cmd == "note" and starts:
        return
    ends = [
        i
        for i, (_, s_, t) in enumerate(entries)
        if s_ == step and t.split()[:1] == ["end"]
    ]
    if not starts or (ends and ends[-1] > starts[-1]):
        print(
            "ledger: WARNING %s for %s, which has no open visit -- `start %s` first, or the "
            "table does not time this work" % (cmd, step, step),
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
    # one spelling per step: SKILL.md says "5N", step5n-native.md logs `step5n`; an agent wrote
    # `step5N` and the report missed the fallback's lines
    step = re.sub(
        r"^step(\d+)([A-Za-z]+)",
        lambda m: "step%s%s" % (m.group(1), m.group(2).lower()),
        rest[0],
    )
    text = " ".join(rest[1:])
    if cmd == "start":
        warn_untagged(path, step)
        append(path, step, ("start  " + text).rstrip())
    elif cmd == "note":
        warn_not_started(path, step, cmd)
        append(path, step, text)
    elif cmd == "job":
        warn_not_started(path, step, cmd)
        append(path, step, ("job=%s  %s" % (rest[1], " ".join(rest[2:]))).rstrip())
    elif cmd == "wait":
        append(
            path, step, ("waiting-user %s  %s" % (rest[1], " ".join(rest[2:]))).rstrip()
        )
    elif cmd == "end":
        end(path, step)
    elif cmd == "skip":
        append(path, step, ("skipped  " + text).rstrip())
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
