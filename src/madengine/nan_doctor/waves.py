#!/usr/bin/env python3
"""Summarise one wave of model trials (mad-debug-nan steps 2-4) in a few lines.

    madengine nan waves <log or dir> ... --signal '<NAN_SIGNAL>' \
        [--error '<the model's own error line>'] [--capsules <CAPSULE_DIR>] [--name REGEX]

Reads every trial log of a wave and prints, instead of the logs themselves:
one line per trial (GPU, arm, verdict, last step, NaN step, s/step, flags),
per-arm and per-GPU tallies, and ALERTS for the readings the skill warns about:
a GPU that disagrees with the rest, a trial that lost its co-tenant (step rate
jumps late), an all-ERROR wave whose logs stop together (external kill), OOMs.

Model-independent by design: the NaN signal is REQUIRED (it is <NAN_SIGNAL>), the
model's own error-line regex is optional (--error; e.g. 'Error at step \\d+' for a
loop that swallows exceptions), step lines match any 'step <n>'.  Trial identity
comes from the file name via --name (default `<gpu>_<trial>[.<arm>].*.log`, the
examples' launcher shape) and, failing that, `HIP_VISIBLE_DEVICES=<n>` inside the
log; names that do not match are reported, never silently mis-attributed.  A directory of
`attempt-NN/run.log` (sequential attempts on one GPU, e.g. run_it.sh-style launchers) is
detected: trial = NN, gpu = 0, arm and outcome from `attempts.tsv` beside it, the model's
own log in the tsv's run_dir is read too, and capsules are looked up in `<CAPSULE_DIR>/attempt-NN`.  Capture trials (arm `cap`) are judged
by their capsule directory when --capsules is given, never by the log: the
capture's exception makes the model print its own NaN line.  A multi-rank job writes
rank<N>_*.pt beside manifest.json at the root of <CAPSULE_DIR> and has no per-trial dir:
its one trial is credited with those.

Verdicts: NAN (signal seen) | CAPSULE (capsules on disk) | NAN_NO_CAPSULE (a capture
trial whose model printed the signal but whose capture wrote nothing: a capture miss,
counted as a hit for the rate) | CLEAN (ran to the end,
no signal) | ERROR (error line, no signal -- e.g. a step-0 OOM swallowed by the
model) | STOPPED (killed before a verdict by the early stop: capture trials always, controls with
--stopped-early) | DIED (no verdict at all: killed, walltime, still running).
"""

# flake8: noqa: E741,E731,E402  (terse variable names, kept from the original scripts)

import argparse
import collections
import datetime as dt
import glob
import os
import re

ANSI = re.compile(r"\x1b\[[0-9;]*m")  # torchtitan & co colour their log lines
import sys

TS = re.compile(r"(\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d)")


def parse_ts(line):
    m = TS.search(line[:40])
    if m:
        return dt.datetime.strptime(m.group(1).replace("T", " "), "%Y-%m-%d %H:%M:%S")
    return None


ATTEMPT = re.compile(r"^attempt-(\d+)$")


def attempt_layout(d):
    """{run.log path: row} for a sequential-attempt wave (`<d>/attempt-NN/run.log`, as
    run_it.sh-style launchers write; one GPU, attempts one after another), with the
    launcher's `<d>/attempts.tsv` row (attempt, arm, outcome, exit, ..., run_dir) when present.
    """
    logs = sorted(glob.glob(os.path.join(d, "attempt-*", "run.log")))
    if not logs:
        return {}
    rows = {}
    tsv = os.path.join(d, "attempts.tsv")
    if os.path.exists(tsv):
        lines = open(tsv).read().splitlines()
        hdr = lines[0].split("\t") if lines else []
        for l in lines[1:]:
            rows[l.split("\t")[0]] = dict(zip(hdr, l.split("\t")))
    return {lg: rows.get(os.path.basename(os.path.dirname(lg)), {}) for lg in logs}


def read_trial(path, a, row=None):
    name = os.path.basename(path)
    m = re.match(a.name, name)
    t = dict(
        path=path,
        name=name,
        gpu=None,
        trial=None,
        arm=None,
        nan=False,
        err=None,
        err_step=None,
        oom=False,
        save_fail=False,
        tb=False,
        rc=None,
        steps=[],
        last_ts=None,
        capsules=None,
    )
    if m:
        gd = m.groupdict()
        t.update({k: gd.get(k) for k in ("gpu", "trial", "arm")})
    extra = []
    if row is not None:  # sequential-attempt layout: identity from the dir + tsv
        n = ATTEMPT.match(os.path.basename(os.path.dirname(path)))
        t.update(
            trial=n.group(1) if n else None,
            gpu="0",
            arm=row.get("arm") or None,
            name=os.path.join(os.path.basename(os.path.dirname(path)), name),
        )
        rd = row.get("run_dir") or ""
        if rd and os.path.isdir(
            rd
        ):  # the harness keeps the model's own log in its run dir
            extra = sorted(glob.glob(os.path.join(rd, "nodes", "*", "*", "run.log")))
    sig = re.compile(a.signal) if a.signal else None
    step = re.compile(a.step)
    above = None
    if a.signal_above:  # NAME=VALUE: a magnitude signal (exploding values)
        name, _, lim = a.signal_above.partition("=")
        above = (
            re.compile(
                r"%s\s*[:=]\s*([-+0-9.eE]+|nan|inf|-inf)" % re.escape(name.strip()),
                re.I,
            ),
            float(lim),
        )
    err = re.compile(a.error) if a.error else None
    for lp in [path] + extra:
        with open(lp, errors="replace") as f:
            for line in f:
                line = ANSI.sub("", line)
                if above is not None:
                    m = above[0].search(line)
                    if m:
                        try:
                            v = float(m.group(1))
                        except ValueError:
                            v = float("nan")
                        if not v <= above[1]:  # above the limit, or non-finite
                            t["nan"] = True
                ts = parse_ts(line)
                if ts:
                    t["last_ts"] = ts
                if t["gpu"] is None:
                    g = re.search(r"HIP_VISIBLE_DEVICES=(\d+)", line)
                    if g:
                        t["gpu"] = g.group(1)
                if sig is not None and sig.search(line):
                    t["nan"] = True
                if t["err"] is None and err is not None:
                    e = err.search(line)
                    if e:
                        t["err"] = line.strip()[e.start() : e.start() + 90]
                        n = re.search(r"\d+", e.group(0))
                        t["err_step"] = int(n.group(0)) if n else None
                s = step.search(line)
                if s:
                    t["steps"].append((int(s.group(1)), ts))
                if "out of memory" in line or "OutOfMemoryError" in line:
                    t["oom"] = True
                if "cannot pickle" in line or "capsule not picklable" in line:
                    t["save_fail"] = (
                        True  # the capture caught a NaN but torch.save failed
                    )
                if line.startswith("Traceback"):
                    t["tb"] = True
                r = re.match(r"TRIAL_RC=(\d+)", line)
                if r:
                    t["rc"] = int(r.group(1))
    if row:  # the launcher's own outcome, when the logs are silent
        out, ex = (row.get("outcome") or "").lower(), row.get("exit")
        if out == "nan" and not t["nan"]:
            t["nan"] = True
            t["err"] = t["err"] or "attempts.tsv: outcome nan"
        elif out == "pass" and t["rc"] is None:
            t["rc"] = 0
        elif out in ("crash", "error") and not t["err"] and not t["nan"]:
            t["err"] = "attempts.tsv: outcome %s (exit %s)" % (out, ex)
    if a.capsules and t["trial"] is not None:
        d = capsule_dir(a, t["trial"])
        t["capsules"] = len(glob.glob(os.path.join(d, "**", "*.pt"), recursive=True))
        t["capsule_dir_found"] = os.path.isdir(d)
        if not os.path.isdir(d) and os.path.exists(
            os.path.join(a.capsules, "manifest.json")
        ):
            # multi-rank job (step4-distributed.md): the job is ONE trial and its ranks write
            # rank<N>_*.pt beside manifest.json at the root -- scored NAN_NO_CAPSULE before.
            t["capsules"] = len(glob.glob(os.path.join(a.capsules, "*.pt")))
    return t


# Per-trial capsule dir names launchers use: the examples' trial_<n>, attempt-<n> (run_it.sh-style),
# t<n>, or the bare trial number.
SUBDIR_GUESSES = ("trial_{trial}", "t{trial}", "attempt-{trial}", "{trial}")


def capsule_dir(a, trial):
    """<CAPSULE_DIR>/<subdir> for one trial; --capsule-subdir auto tries SUBDIR_GUESSES, first existing wins."""
    if a.capsule_subdir != "auto":
        return os.path.join(a.capsules, a.capsule_subdir.format(trial=trial))
    for g in SUBDIR_GUESSES:
        d = os.path.join(a.capsules, g.format(trial=trial))
        if os.path.isdir(d):
            return d
    return os.path.join(a.capsules, SUBDIR_GUESSES[0].format(trial=trial))


def rate(steps):
    """(s/step overall, ratio of early to late s/step) from (step, ts) pairs."""
    pts = [(s, ts) for s, ts in steps if ts]
    if len(pts) < 4:
        return None, None
    span = lambda p: (p[-1][1] - p[0][1]).total_seconds() / max(p[-1][0] - p[0][0], 1)
    q = len(pts) // 4
    early, late = span(pts[: 2 * q]), span(pts[-q:])
    return span(pts), (early / late if late > 0 else None)


def verdict(t, a):
    if t["arm"] in a.capture_arms and t["capsules"] is not None:
        if not t["capsules"] and t["rc"] in (None, 130, 137, 143) and not t["err"]:
            return "STOPPED"  # killed before capturing (early stop, walltime)
        if t["capsules"]:
            return "CAPSULE"
        if t["nan"]:
            return "NAN_NO_CAPSULE"  # the model saw it; the capture did not write it
        return "ERROR" if t["err"] and "nan" not in t["err"].lower() else "NONE"
    if t["nan"]:
        return "NAN"
    if t["err"] or t["oom"]:
        return "ERROR"
    if t["rc"] == 0 or (
        t["steps"] and a.last_step is not None and t["steps"][-1][0] >= a.last_step
    ):
        return "CLEAN"
    if a.stopped_early and t["rc"] in (None, 130, 137, 143):
        return "STOPPED"  # a control the early stop killed: no verdict, not a death
    return "DIED"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("paths", nargs="+", help="trial logs, or directories of *.log")
    p.add_argument(
        "--signal",
        help="<NAN_SIGNAL> regex (the model's own NaN line); this or --signal-above",
    )
    p.add_argument(
        "--signal-above",
        metavar="NAME=VALUE",
        help='a magnitude signal: a trial counts as NAN when a logged "NAME: <number>" '
        "exceeds VALUE or is non-finite (e.g. grad_norm=100; ANSI colours stripped)",
    )
    p.add_argument(
        "--error",
        default=None,
        help="the model's own error-line regex, first number = step (default: none)",
    )
    p.add_argument(
        "--step", default=r"(?i)\bstep (\d+)\b", help="step-line regex, group 1 = step"
    )
    p.add_argument(
        "--name",
        default=r"^(?P<gpu>\d+)_(?P<trial>\d+)(?:\.(?P<arm>\d*[A-Za-z][\w-]*))?\.",
        help="file-name regex with named groups gpu, trial, arm",
    )
    p.add_argument("--capsules", help="<CAPSULE_DIR>; capture trials are judged by it")
    p.add_argument(
        "--capsule-subdir",
        default="auto",
        help="per-trial dir under --capsules, e.g. 'trial_{trial}'; auto tries %s"
        % ", ".join(SUBDIR_GUESSES),
    )
    p.add_argument("--capture-arms", default="cap", type=lambda s: set(s.split(",")))
    p.add_argument(
        "--last-step", type=int, help="final step index; a trial reaching it is CLEAN"
    )
    p.add_argument(
        "--stopped-early",
        action="store_true",
        help="the wave was stopped early on purpose: a killed trial of any arm is STOPPED, not DIED",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="tallies and alerts only, no per-trial lines",
    )
    a = p.parse_args()
    if not (a.signal or a.signal_above):
        p.error("one of --signal / --signal-above is required")

    files, rows = [], {}
    for x in a.paths:
        lay = attempt_layout(x) if os.path.isdir(x) else {}
        if lay:
            rows.update(lay)
            files += list(lay)
            if a.capsule_subdir == "auto":
                a.capsule_subdir = "attempt-{trial}"
        else:
            files += (
                sorted(glob.glob(os.path.join(x, "*.log"))) if os.path.isdir(x) else [x]
            )
    if not files:
        sys.exit("no logs found")
    trials = [read_trial(f, a, rows.get(f)) for f in files]
    key = lambda t: (
        int(t["trial"]) if (t["trial"] or "").isdigit() else 1 << 30,
        t["name"],
    )
    trials.sort(key=key)

    alerts = []
    unnamed = [t for t in trials if t["trial"] is None]
    if unnamed:
        alerts.append(
            "%d log name(s) do not match --name (e.g. %s): trial/arm unknown, so capsules "
            "cannot be attributed per trial -- pass --name with (?P<gpu>)(?P<trial>)(?P<arm>) "
            "groups, or read the capsule count below"
            % (len(unnamed), unnamed[0]["name"])
        )
        if a.capsules:
            n = len(glob.glob(os.path.join(a.capsules, "**", "*.pt"), recursive=True))
            print(
                "== capsules in %s: %d (not attributable to trials)" % (a.capsules, n)
            )
    by_arm = collections.defaultdict(collections.Counter)
    by_gpu = collections.defaultdict(collections.Counter)
    nan_steps = collections.Counter()
    for t in trials:
        t["v"] = verdict(t, a)
        arm = t["arm"] or "-"
        by_arm[arm][t["v"]] += 1
        by_gpu[t["gpu"] or "?"][t["v"]] += 1
        if t["v"] == "NAN" and t["err_step"] is not None:
            nan_steps[t["err_step"]] += 1
        sps, ratio = rate(t["steps"])
        t["sps"] = sps
        flags = []
        if ratio and ratio > 3:
            flags.append("ORPHAN?(late steps %.0fx faster)" % ratio)
            alerts.append(
                "trial %s: late steps %.0fx faster -- lost its co-tenant? drop it from the "
                "denominator if so (step2-rate.md)" % (t["trial"], ratio)
            )
        if t["oom"]:
            flags.append("OOM")
        if t["tb"]:
            flags.append("traceback")
        t["flags"] = flags
        if not a.quiet:
            print(
                "trial %-3s gpu %-2s %-4s %-7s last_step=%-5s nan_step=%-5s s/step=%-6s %s%s"
                % (
                    t["trial"] or "?",
                    t["gpu"] or "?",
                    arm,
                    t["v"],
                    t["steps"][-1][0] if t["steps"] else "-",
                    (
                        t["err_step"]
                        if t["v"] == "NAN" and t["err_step"] is not None
                        else "-"
                    ),
                    "%.2f" % sps if sps else "-",
                    " ".join(flags),
                    ("  capsules=%d" % t["capsules"]) if t["capsules"] else "",
                )
            )
            if t["v"] == "ERROR" and t["err"]:
                print("    err: %s" % t["err"])

    print(
        "== by arm:   "
        + "; ".join("%s %s" % (arm, dict(c)) for arm, c in sorted(by_arm.items()))
    )
    for arm, c in sorted(by_arm.items()):
        hit = c["NAN"] + c["CAPSULE"] + c["NAN_NO_CAPSULE"]
        n = sum(c.values()) - c["DIED"] - c["ERROR"] - c["STOPPED"]
        print(
            "   %-4s rate %d/%d (excl. %d ERROR, %d DIED, %d STOPPED)"
            % (arm, hit, n, c["ERROR"], c["DIED"], c["STOPPED"])
        )
    print(
        "== by GPU:   "
        + "  ".join(
            "%s:%s" % (g, ",".join("%s=%d" % kv for kv in sorted(c.items())))
            for g, c in sorted(by_gpu.items())
        )
    )
    if nan_steps:
        print(
            "== NaN step: "
            + " ".join("%s x%d" % kv for kv in sorted(nan_steps.items()))
        )
    sp = sorted(t["sps"] for t in trials if t["sps"])
    if sp:
        print(
            "== s/step:   min %.2f  median %.2f  max %.2f"
            % (sp[0], sp[len(sp) // 2], sp[-1])
        )

    # a GPU whose trials all disagree with the rest -- within one arm only: arms
    # differ by design (capture vs control, injected vs not)
    for arm in sorted({t["arm"] or "-" for t in trials}):
        ga = collections.defaultdict(collections.Counter)
        for t in trials:
            if (t["arm"] or "-") == arm and t["v"] not in ("STOPPED", "DIED", "ERROR"):
                ga[t["gpu"] or "?"][t["v"] in ("NAN", "CAPSULE", "NAN_NO_CAPSULE")] += 1
        if len(ga) > 2:
            good = [g for g, c in ga.items() if c[True] and not c[False]]
            bad = [g for g, c in ga.items() if c[False] and not c[True]]
            if good and bad and len(bad) < len(good):
                alerts.append(
                    "arm %s, GPU(s) %s: 0 hits while %d GPUs hit every trial -- partition the "
                    "denominator, probe those GPUs alone (step2-rate.md)"
                    % (arm, ",".join(sorted(bad)), len(good))
                )
    if not a.capsules and any(t["arm"] in a.capture_arms for t in trials):
        alerts.append(
            "capture trials judged from their logs -- pass --capsules <CAPSULE_DIR>; the "
            "capture's own exception makes the model print its NaN line"
        )
    if a.capsules and not os.path.exists(os.path.join(a.capsules, "manifest.json")):
        caps = [t for t in trials if t["arm"] in a.capture_arms]
        on_disk = len(glob.glob(os.path.join(a.capsules, "**", "*.pt"), recursive=True))
        if caps and on_disk and not any(t.get("capsule_dir_found") for t in caps):
            alerts.append(
                "%d capsule(s) under %s, but no capture trial has a per-trial dir (tried %s) -- the "
                "verdicts above ignore them: pass --capsule-subdir <pattern with {trial}>"
                % (
                    on_disk,
                    a.capsules,
                    (
                        ", ".join(SUBDIR_GUESSES)
                        if a.capsule_subdir == "auto"
                        else a.capsule_subdir
                    ),
                )
            )
    missed = [t for t in trials if t["v"] == "NAN_NO_CAPSULE"]
    if missed:
        alerts.append(
            "%d capture trial(s) printed the NaN signal but wrote no capsule -- a capture miss, not a "
            "clean trial: capture_digest.py on one log; `pending:` / `MASKED` lines with no capsule = "
            "settle mode never fired (NAN_CAPTURE_SETTLE=0, step5-read.md)"
            % len(missed)
        )
    errs = [t for t in trials if t["v"] in ("ERROR", "DIED")]
    if errs and len(errs) == len(trials):
        ends = [t["last_ts"] for t in trials if t["last_ts"]]
        if ends and (max(ends) - min(ends)).total_seconds() < 30:
            alerts.append(
                "every trial ERROR/DIED and all logs stop within 30 s -- external kill or "
                "walltime, not the model (waves.md)"
            )
    if any(t["oom"] for t in trials):
        alerts.append(
            "OOM in %d trial(s) -- exclude them; see step4-capture-memory.md if capturing"
            % sum(t["oom"] for t in trials)
        )
    lost = [t for t in trials if t["save_fail"]]
    if lost:
        alerts.append(
            "%d trial(s) hit a capsule save error (cannot pickle / not picklable) -- "
            "a caught NaN may be lost or saved with leaves replaced; check the "
            "capsule's `unpicklable` list (capture_digest.py)" % len(lost)
        )
    for x in alerts:
        print("!! " + x)


if __name__ == "__main__":
    main()
