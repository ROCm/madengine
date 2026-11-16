#!/usr/bin/env python3
"""Write <ARTIFACT_DIR>/native/handoff.md: what the capsule path tried, for step 5N (debug-nan).

    madengine nan handoff <ARTIFACT_DIR> [--logs <dir with <job>.out files>]

Step 5N starts where the capsule path stopped; its failures are evidence.  This
collects them from the files the ladder already writes -- the ledger, the wave
summaries, the capture trial logs, the grouping, every 5b attempt's summary.txt and
one copy log, and each 5c/5d pressure sweep's result table (found in --logs by the
job ids the ledger recorded) -- and writes one file:

  * Trigger      why 5N was entered
  * Facts        image actually used, precision, model command, when the model NaNs
                 (capture vs control), groups that share one op geometry
  * Capture      the census of NaN-creating sites, aggregated over trials
  * Contention   the capture's load beside every replay's: processes on the GPU,
                 what they run, memory -- LIGHTER flags a load far below the capture's
  * Attempts     one row per replay: load, n/N, verdict, walk
  * Ruled out    every clean experiment with the smallest rate it excludes (95%):
                 a null shorter than 3/rate rules nothing out at that rate
  * Implications PROPOSED readings, each with its source; the agent confirms or
                 strikes them (step5n-native.md N1) -- the script does not decide
  * Reusable     capsule per group, co-tenant variants, model hipBLASLt log

Model-independent; reads only what the ladder writes.
"""

# flake8: noqa: E741,E731,E402  (terse variable names, kept from the original scripts)

import argparse
import ast
import collections
import glob
import json
import os
import re
import statistics

LEDGER_LINE = re.compile(r"^(\S+ \S+)\s+(\S+)\s+(.*)$")
KIND = {
    "full": "fwd+bwd+optimizer",
    "fwdbwd": "fwd+bwd",
    "fwd": "fwd only",
    "adamw": "optimizer only",
    "copy": "HBM copy",
    "triad": "HBM triad",
    "gemm": "big GEMM",
    "mix": "copy+GEMM",
    "bust": "same GEMM, cache-busting",
}
NO_BACKWARD = (
    "fwd only",
    "optimizer only",
    "HBM copy",
    "HBM triad",
    "big GEMM",
    "copy+GEMM",
    "same GEMM, cache-busting",
)


def read(p):
    try:
        return open(p, errors="replace").read()
    except OSError:
        return ""


def first(lines, pat):
    return next((l.strip() for l in lines if re.search(pat, l)), "")


def excl(runs):
    """Smallest per-run rate a 0/runs null excludes at 95% (rule of three)."""
    return 3.0 / runs if runs else None


def fmt_rate(r):
    return "-" if r is None else "%.1e" % r


def attempt(summary_path, A):
    d = os.path.dirname(summary_path)
    s = read(summary_path).splitlines()
    rel = os.path.relpath(d, A)
    g = re.search(r"(?:^|/)(?:replay/|verify-|repro-)(G\d+)", rel)
    c = re.search(r"(?:^|/)c(\d+)\.(\d+)\.([^/]+)$", rel)
    tot = first(s, r"^=== totals")
    nf = re.search(r"non-finite (\d+)/(\d+)", tot)
    gpus = len({m.group(1) for l in s for m in [re.match(r"\s+g(\d+)_k\d+\.", l)] if m})
    walk = [l for l in s if re.match(r"\s+\d+\s+aten\.", l)]
    walk_verdict = [l.strip() for l in s if l.startswith("[verdict]")]
    copy = sorted(glob.glob(os.path.join(d, "g*_k*.log")))[:1]
    cl = read(copy[0]).splitlines() if copy else []
    return dict(
        rel=rel,
        group=g.group(1) if g else "?",
        conc=c.group(1) if c else "?",
        job=c.group(2) if c else "?",
        host=c.group(3) if c else "?",
        gpus=gpus,
        hits=int(nf.group(1)) if nf else None,
        runs=int(nf.group(2)) if nf else None,
        result=first(s, r"^ATTEMPT_RESULT=").split("=", 1)[-1],
        futility=first(s, r"futility stop"),
        first_wrong=first(s, r"-> first op to go wrong"),
        tally=first(s, r"first op to go non-finite"),
        terminal=first(s, r"terminal fp64 ref"),
        diverged=[l.strip() for l in walk if "DIVERGED" in l],
        terminal_idx=walk[-1].split()[0] if walk else None,
        mutated=[l.strip() for l in walk if "MUTATED AFTER PRODUCTION" in l],
        walk=walk_verdict[-1] if walk_verdict else "",
        env=first(s, r"env:"),
        precision=first(cl, r"^precision:"),
        under=first(cl, r"captured under:"),
        sif=next(
            (
                m.group(1)
                for l in cl
                for m in [re.search(r"apptainer exec .* (\S+\.sif)\b", l)]
                if m
            ),
            "",
        ),
        build=first(cl, r"^captured build:"),
    )


def pressure_rows(ledger, logs):
    """5c/5d sweep rows from <logs>/*.<job>.out, by the job ids the ledger recorded."""
    rows = []
    for l in ledger:
        m = LEDGER_LINE.match(l)
        if not (m and re.match(r"step5[cd]", m.group(2))):
            continue
        j = re.search(r"\bjob=(\d+)", m.group(3))
        for f in (
            sorted(glob.glob(os.path.join(logs, "*.%s.out" % j.group(1)))) if j else []
        ):
            t = read(f)
            if "=== Results" not in t:
                continue
            for r in t.split("=== Results", 1)[1].splitlines()[1:]:
                x = re.match(
                    r"\s*GPU (\d+)\s+(\S+)\s+non-finite on (\d+)/(\d+).*?\[(.*?)\]\s*(.*)$",
                    r,
                )
                if not x:
                    continue
                cmd = x.group(6).strip()
                mode = re.search(r"--mode (\w+)", cmd)
                procs = re.search(r"--procs (\d+)", cmd)
                gb = re.search(r"([\d.]+) GB taken", x.group(5))
                real = "cotenant_loop" in cmd or "cotenant.sh" in cmd
                tool = (
                    "model"
                    if real
                    else next(
                        (
                            os.path.basename(t_)
                            for t_ in cmd.split()
                            if t_.endswith(".py")
                        ),
                        "?",
                    )
                )
                rows.append(
                    dict(
                        job=j.group(1),
                        gpu=x.group(1),
                        verdict=x.group(2),
                        hits=int(x.group(3)),
                        runs=int(x.group(4)),
                        cmd=cmd,
                        tool=tool,
                        procs=int(procs.group(1)) if procs else (None if real else 1),
                        kind=(
                            "the real model"
                            if real
                            else KIND.get(mode.group(1) if mode else "", "?")
                        ),
                        gb=float(gb.group(1)) if gb else None,
                    )
                )
    return rows


def capture_facts(A):
    """Census rows, captured ops and contention, aggregated over the capture trials' logs."""
    logs = []
    for w in sorted(glob.glob(os.path.join(A, "wave.*.out"))):
        logs += re.findall(r"(/\S+?\.cap\.\S+?\.log)", read(w))
    # docker / hand-written launchers: the trial logs themselves (<gpu>_<trial>.cap.<job>.<host>.log) anywhere
    # under the artifact dir, e.g. wave/
    logs += sorted(glob.glob(os.path.join(A, "**", "*.cap.*.log"), recursive=True))
    sites = collections.OrderedDict()
    captured = collections.Counter()
    held, procs, cpgs = [], [], []
    armed = 0
    for lg in dict.fromkeys(logs):
        t = read(lg)
        if (
            "nan-capture armed" not in t
        ):  # killed before arming (early stop): says nothing
            continue
        armed += 1
        cpgs += [
            int(x)
            for x in re.findall(r"nan-capture armed.*?containers_per_gpu=(\d+)", t)
        ]
        lines = t.splitlines()
        for i, l in enumerate(lines):
            m = re.search(
                r"\]\s+(survived|MASKED|pending|captured|settled)\s+(\S+)\s+(\d+)\s+(\d+)\.\.(\d+)\s+(.*)$",
                l,
            )
            if m:
                s = sites.setdefault(
                    (m.group(1), m.group(6).strip()),
                    dict(trials=0, hits=0, lo=10**9, hi=-1),
                )
                s["trials"] += 1
                s["hits"] += int(m.group(3))
                s["lo"] = min(s["lo"], int(m.group(4)))
                s["hi"] = max(s["hi"], int(m.group(5)))
            m = re.search(
                r"\]\s+op=(\S+)\s+(?:op|gemm) call", l
            )  # gemm: pre-rename logs
            if m:
                # several processes write one log: the problem line may be dozens of lines on
                prob = next(
                    (
                        re.sub(r".*hipBLASLt problem:\s*", "", x)
                        for x in lines[i : i + 80]
                        if "hipBLASLt problem" in x
                    ),
                    "",
                )
                captured[(m.group(1), prob.strip())] += 1
            m = re.search(
                r"contention: (\d+) process\(es\) on this device.*?(?:([\d.]+) GB held by others)?$",
                l,
            )
            if m:
                procs.append(int(m.group(1)))
                if m.group(2):
                    held.append(float(m.group(2)))
    return dict(
        armed=armed,
        sites=sites,
        captured=captured,
        held=held,
        procs=procs,
        cpg=max(cpgs) if cpgs else None,
    )


def source_line(A, site):
    """' -- <the code at file:line>' for a census site, from the model source copy in <A>/src."""
    m = re.search(r"([\w.-]+\.py):(\d+)", site)
    if not m:
        return ""
    for f in glob.glob(os.path.join(A, "src", "**", m.group(1)), recursive=True)[:1]:
        lines = read(f).splitlines()
        i = int(m.group(2)) - 1
        if 0 <= i < len(lines):
            return "` -- `%s" % lines[i].strip()[:70]
    return ""


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("artifact_dir")
    p.add_argument(
        "--logs",
        default=os.environ.get("NAN_LOGS"),
        help="dir with <job>.out files (default: $NAN_LOGS, else <ARTIFACT_DIR>)",
    )
    p.add_argument("--out", default=None)
    a = p.parse_args()
    A = os.path.abspath(a.artifact_dir)
    ledger = read(os.path.join(A, "ledger.md")).splitlines()
    text = [
        LEDGER_LINE.match(l).group(3) if LEDGER_LINE.match(l) else l for l in ledger
    ]
    atts = [
        attempt(f, A)
        for f in sorted(glob.glob(os.path.join(A, "**", "summary.txt"), recursive=True))
        if "/native/" not in f and "/blind/" not in f
    ]
    sweeps = pressure_rows(ledger, a.logs or A)
    cap = capture_facts(A)
    grp = read(os.path.join(A, "groups_priority.txt")).strip()
    pts = sorted(glob.glob(os.path.join(A, "capsules", "**", "*.pt"), recursive=True))
    out = []
    w = out.append
    imp = []  # (proposal, source)

    w("# Step 5N handoff -- %s\n" % os.path.basename(A))
    w(
        "_Written by native_handoff.py from the capsule path's artifacts. `Implications` are "
        "PROPOSED: confirm or strike each before N3 (step5n-native.md)._\n"
    )

    # ---------------------------------------------------------------- trigger
    w("## Trigger\n")
    trig = [t for t in text if "NATIVE_TRIGGER=" in t] or [
        t for t in text if "CAPTURE_TO_REPRO_RESULT=" in t or "FUSED_RESULT=" in t
    ]  # FUSED_RESULT: pre-rename ledgers
    w(
        "\n".join("* " + t for t in trig[-3:])
        or "* (no NATIVE_TRIGGER or CAPTURE_TO_REPRO_RESULT line in the ledger)"
    )

    # ---------------------------------------------------------------- facts
    w("\n## Facts\n")
    a0 = next((t for t in atts if t["sif"] or t["precision"]), None)
    if a0 and a0["sif"]:
        w(
            "* image actually used (5b copy log): `%s`%s -- not necessarily the launcher's default"
            % (a0["sif"], "; %s" % a0["build"] if a0["build"] else "")
        )
    else:
        img = [t for t in text if re.search(r"\b(image|SIF|IMAGE)\b", t)]
        w("* image (ledger): %s" % (img[-1][:200] if img else "TODO -- no image line"))
    if a0 and a0["precision"]:
        w(
            "* precision at capture: `%s`"
            % a0["precision"].replace("precision:", "").strip()
        )
    if a0 and a0["under"]:
        w(
            "* model command at capture: `%s`"
            % a0["under"].replace("captured under:", "").strip()[:250]
        )
    for f in sorted(glob.glob(os.path.join(A, "wave_summary.*.txt"))):
        arms = collections.defaultdict(list)
        for l in read(f).splitlines():
            m = re.match(
                r"trial\s+\d+\s+gpu\s+\d+\s+(cap|ctl)\s+(CAPSULE|NAN)\s+last_step=(\d+)",
                l,
            )
            if m:
                arms[m.group(1)].append(int(m.group(3)))
        for arm, st in sorted(arms.items()):
            w(
                "* model NaN step, %s arm (`%s`): %d trial(s), steps %d..%d, median %g"
                % (
                    "capture" if arm == "cap" else "control",
                    os.path.basename(f),
                    len(st),
                    min(st),
                    max(st),
                    statistics.median(st),
                )
            )
    try:
        groups = json.load(open(os.path.join(A, "groups.json")))["groups"]
    except Exception:
        groups = []
    by_key = collections.defaultdict(list)
    for i, g in enumerate(groups):
        ident = g.get("identity") or {}
        if isinstance(ident, str):
            try:
                ident = ast.literal_eval(ident)
            except Exception:
                ident = {}
        by_key[ident.get("geometry_key", "?")].append(
            "G%d %s (%s)" % (i + 1, ident.get("op", "?"), g.get("count", "?"))
        )
    for k, gs in by_key.items():
        if len(gs) > 1:
            w(
                "* **same op geometry `%s`**: %s -- one kernel; a candidate for one covers both"
                % (k, ", ".join(gs))
            )
    km = [t for t in text if re.search(r"KERNEL=|KERNEL_(MISMATCH|MATCH|UNKNOWN)", t)]
    if km:
        w("\n".join("* " + t[:200] for t in km[-2:]))

    # ---------------------------------------------------------------- capture
    w("\n## Capture (census over %d armed capture trial(s))\n" % cap["armed"])
    if cap["sites"]:
        w("| status | site | trials | hits | steps |")
        w("|---|---|---|---|---|")
        for (st, site), s in cap["sites"].items():
            w(
                "| %s | `%s` | %d | %d | %d..%d |"
                % (
                    st,
                    site[:90] + source_line(A, site),
                    s["trials"],
                    s["hits"],
                    s["lo"],
                    s["hi"],
                )
            )
    for (op, prob), n in cap["captured"].most_common():
        w("* captured %dx: `%s`  %s" % (n, op, prob))
    if not cap["armed"]:
        w(
            "(no armed capture log found via wave.*.out or **/*.cap.*.log -- run capture_digest.py on a capture trial log by hand)"
        )
    masked = [site for (st, site) in cap["sites"] if st == "MASKED"]
    if masked and not cap["captured"]:
        imp.append(
            (
                "No capsule, but the hook saw MASKED non-finite sites: start N2/N3 from those ops (%s)."
                % ", ".join("`%s`" % m[:60] for m in masked[:3]),
                "capture census",
            )
        )

    # ---------------------------------------------------------------- contention
    w("\n## Contention: the capture's load beside every replay's\n")
    cpg = (
        cap["cpg"]
        or next(
            (
                int(m.group(1))
                for t in text
                for m in [
                    re.search(r"CONTAINERS_PER_GPU=(\d+)|containers/GPU=(\d+)", t)
                ]
                if m and m.group(1)
            ),
            None,
        )
        or 1
    )
    cap_others = (max(cap["procs"]) * cpg - 1) if cap["procs"] else None
    held_med = statistics.median(cap["held"]) if cap["held"] else None
    w(
        "LIGHTER = fewer other processes than the capture, or no backward pass. Memory is shown, not judged: "
        "a co-tenant far smaller than the capture has reproduced (a training step's kernel mix, not its "
        "footprint, was the trigger once).\n"
    )
    w(
        "| load | other processes on the GPU | what they run | memory they hold | result |"
    )
    w("|---|---|---|---|---|")
    if cap["procs"]:
        w(
            "| **capture** (the model) | %s | the model's training step | %s | NaN |"
            % (
                "%d (%d process(es)/container x %d container(s)/GPU, minus itself)"
                % (cap_others, max(cap["procs"]), cpg),
                (
                    "%.1f-%.1f GB (median %.1f)"
                    % (min(cap["held"]), max(cap["held"]), held_med)
                    if held_med
                    else "?"
                ),
            )
        )
    for t in atts:
        w(
            "| 5b %s `%s` | %s copies of the capsule | the captured op%s, nothing else | ~1 GB each | %s/%s %s |"
            % (
                t["group"],
                os.path.basename(t["rel"]),
                int(t["conc"]) - 1 if t["conc"].isdigit() else "?",
                " chain" if t["walk"] else "",
                t["hits"],
                t["runs"],
                t["result"],
            )
        )
    lighter = []
    for r in sweeps:
        light = []
        if cap_others and r["procs"] is not None and r["procs"] < cap_others:
            light.append("%d vs %d processes" % (r["procs"], cap_others))
        if r["kind"] in NO_BACKWARD:
            light.append("no backward pass")
        if light and r["hits"] == 0:
            lighter.append(light)
        w(
            "| 5d `%s` | %s | %s | %s GB | %d/%d%s |"
            % (
                r["tool"],
                r["procs"] if r["procs"] is not None else "?",
                r["kind"],
                "%.1f" % r["gb"] if r["gb"] is not None else "?",
                r["hits"],
                r["runs"],
                " **LIGHTER: %s**" % "; ".join(light) if light else "",
            )
        )
    clean = [r for r in sweeps if r["hits"] == 0]
    if clean and len(lighter) == len(clean):
        imp.append(
            (
                "Every clean co-tenant load was LIGHTER than the capture's (%s): no clean load had as many "
                "neighbours running a backward pass as the capture did, so its zeros do not clear the load. "
                "Start from at least the capture's number of neighbours per GPU, each running forward + "
                "backward (the model's step, or a stand-in with its op types and sizes); reduce the kind of load after."
                % "; ".join(sorted({x for l in lighter for x in l}))[:220],
                "Contention table",
            )
        )

    # ---------------------------------------------------------------- attempts
    w("\n## Attempts (capsule replays)\n")
    if atts:
        w(
            "| group | attempt | host | copies/GPU x GPUs | non-finite / runs | result | first op to go wrong | walk verdict |"
        )
        w("|---|---|---|---|---|---|---|---|")
        for t in atts:
            w(
                "| %s | `%s` | %s | %s x %d | %s | %s | %s | %s |"
                % (
                    t["group"],
                    t["rel"],
                    t["host"],
                    t["conc"],
                    t["gpus"],
                    "%s/%s" % (t["hits"], t["runs"]) if t["runs"] is not None else "?",
                    t["result"] or "?",
                    t["first_wrong"].replace("-> first op to go wrong: ", "") or "-",
                    t["walk"] or "-",
                )
            )
        for t in atts:
            extra = [
                x for x in (t["futility"], t["tally"], t["terminal"], t["env"]) if x
            ]
            if extra:
                w("\n`%s`:\n\n" % t["rel"] + "\n".join("    " + x[:220] for x in extra))
    else:
        w("(no 5b attempt on disk)")

    # ---------------------------------------------------------------- ruled out
    w(
        "\n## Ruled out (clean experiments; a candidate must differ, or run past the bound)\n"
    )
    w(
        '"excludes" = the smallest per-run rate a 0/N null rules out at 95% (3/N). A rarer trigger is NOT '
        "ruled out.\n"
    )
    ruled = []
    for t in atts:
        if t["hits"] == 0 and t["runs"]:
            ruled.append(
                "capsule %s replay (%s), copies only, %s x %d GPUs, 0/%d -> excludes >= %s (%s)"
                % (
                    t["group"],
                    "chain" if t["walk"] else "op",
                    t["conc"],
                    t["gpus"],
                    t["runs"],
                    fmt_rate(excl(t["runs"])),
                    t["result"] or "?",
                )
            )
    for r in clean:
        ruled.append(
            "capsule op beside %s x `%s` (%s), 0/%d -> excludes >= %s (job %s)"
            % (
                r["procs"] if r["procs"] is not None else "?",
                r["tool"],
                r["kind"],
                r["runs"],
                fmt_rate(excl(r["runs"])),
                r["job"],
            )
        )
    w("\n".join("* " + r for r in ruled) or "* (nothing ruled out on disk)")
    if clean:
        imp.append(
            (
                "Each null bounds a rate, not a load: 0/1M excludes only >= 3e-6 per run, and a load one step "
                "lighter than a reproducing one can fail 10-100x less often. Rerun a near-miss load (e.g. "
                "the heaviest clean one) longer before treating it as clean.",
                "Ruled out",
            )
        )

    # ---------------------------------------------------------------- implications
    if any(t["result"] == "NOT_SET_UP" for t in atts):
        t = next(t for t in atts if t["result"] == "NOT_SET_UP")
        imp.insert(
            0,
            (
                "The capsule's exact values, op and copies-only load never fail where the model needed far "
                "fewer runs (NOT_SET_UP): the missing ingredient is the load, the kernel mix or state around "
                "the op -- not the values. A copies-only single-op candidate repeats this experiment; start "
                "from the op beside the model's step (top-down, or bottom-up beside a training-step "
                "stand-in).",
                t["rel"],
            ),
        )
    for t in atts:
        up = [d for d in t["diverged"] if d.split()[0] != t["terminal_idx"]]
        if up:
            imp.append(
                (
                    "An op UPSTREAM of the captured op DIVERGED from the record on identical inputs (%s): the "
                    "fault may start there -- include it and its producer in the first candidate."
                    % re.sub(r"\s+", " ", up[0])[:80],
                    t["rel"],
                )
            )
            break
        if t["diverged"]:
            imp.append(
                (
                    "The captured op itself DIVERGED from its record on identical inputs (%s): no upstream "
                    "suspect, but the fault also corrupts finite values -- compare every hit against fp64, "
                    "not only isfinite." % re.sub(r"\s+", " ", t["diverged"][0])[:90],
                    t["rel"],
                )
            )
            break
    for t in atts:
        nv = [m for m in t["mutated"] if not re.search(r"\bview\b", m)]
        if nv:
            imp.append(
                (
                    "An operand was MUTATED AFTER PRODUCTION by a non-view op (%s): the consumer may have read "
                    "a value the capsule did not store -- keep the in-place writes around the op."
                    % re.sub(r"\s+", " ", nv[0])[:80],
                    t["rel"],
                )
            )
            break
    if any(t["result"] == "PROPAGATED" or "PROPAGATED" in t["walk"] for t in atts):
        imp.append(
            (
                "PROPAGATED: the capsule's op was handed non-finite input -- move the suspect to its producer "
                "(the chain names it).",
                "summary.txt",
            )
        )
    if grp and all(
        re.search(r"FALSE_POSITIVE|PROPAGATED", l) for l in grp.splitlines()
    ):
        imp.append(
            (
                "Every group is an allocator/view false positive or PROPAGATED: the op that first went bad was "
                "not captured -- upstream, or not watched (op filter). A PROPAGATED chain names its producer.",
                "groups_priority.txt",
            )
        )
    if any("KERNEL_MISMATCH" in t for t in text):
        imp.append(
            (
                "The replay ran a different hipBLASLt solution than the model: pin the model's solution in "
                "every candidate.",
                "ledger KERNEL_MISMATCH",
            )
        )
    hit = [r for r in sweeps if r["hits"]]
    if hit:
        imp.insert(
            0,
            (
                "A co-tenant DID reproduce the capsule (%s): start from that load, not from copies."
                % "; ".join(
                    "%d/%d beside %s x %s"
                    % (r["hits"], r["runs"], r["procs"], r["kind"])
                    for r in hit[:3]
                ),
                "pressure sweep",
            ),
        )
    if any(t["tally"] for t in atts):
        t = next(t for t in atts if t["tally"])
        imp.append(
            (
                "The replay loop named the first op to go non-finite (%s): that op is the bottom-up suspect."
                % t["tally"][:120],
                t["rel"],
            )
        )
    if any(t["hits"] == 0 for t in atts) and not sweeps:
        imp.append(
            (
                "Copies-only replays were clean and no pressure sweep ran: the load is unexplored -- "
                "step5-pressure.md before or as the first candidates.",
                "attempts",
            )
        )
    if pts and not atts:
        imp.append(
            (
                "%d capsule(s) exist but none was replayed: the captured op, its exact views/strides and "
                "precision (capsule_info.py) are the bottom-up suspect; the capture's contention is the load "
                "to start from." % len(pts),
                "capsules/",
            )
        )
    w(
        "\n## Implications (PROPOSED -- confirm or strike each, then ledger the kept ones)\n"
    )
    w(
        "\n".join(
            "%d. %s  \n   _source: %s_" % (i + 1, p_, s_)
            for i, (p_, s_) in enumerate(imp)
        )
        or "(none proposed -- read the attempts yourself)"
    )

    # ---------------------------------------------------------------- reusable
    w("\n## Reusable\n")
    reuse = []
    for l in grp.splitlines() if grp else []:
        f = l.split()
        if len(f) >= 3 and f[0].startswith("G"):
            reuse.append(
                "capsule %s (%s): `%s`" % (f[0], f[3] if len(f) > 3 else "?", f[2])
            )
    if pts and not grp:
        reuse.append("capsule (ungrouped): `%s`" % os.path.relpath(pts[0], A))
    for f in sorted(glob.glob(os.path.join(A, "pressure", "*variants*"))):
        reuse.append("co-tenant variants: `%s`" % os.path.relpath(f, A))
    hb = [
        f
        for f in sorted(
            glob.glob(os.path.join(A, "**", "*hipblaslt*.log"), recursive=True)
        )
        if "/native/" not in f
    ][:1]
    reuse += ["model hipBLASLt log: `%s`" % os.path.relpath(f, A) for f in hb]
    w("\n".join("* " + r for r in reuse) or "* (none)")

    path = a.out or os.path.join(A, "native", "handoff.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(out) + "\n")
    print(
        "HANDOFF %s (%d attempts, %d sweep rows, %d ruled out, %d implications, %d lines)"
        % (path, len(atts), len(sweeps), len(ruled), len(imp), len(out))
    )


if __name__ == "__main__":
    main()
