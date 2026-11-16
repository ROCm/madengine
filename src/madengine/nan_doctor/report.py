#!/usr/bin/env python3
"""Draft <ARTIFACT_DIR>/report.md for mad-debug-nan step 9 from the skill's own artifacts.

    madengine nan report <ARTIFACT_DIR> [--out report.md]

Fills every section step9-report.md requires that is a fact on disk -- the per-step
time table (ledger.py), the rate lines from the ledger, each wave's summary, the
grouping, every 5b attempt (copies per GPU x GPUs, runs done, hits, early stop,
env verdict, captured build, capture-time contention), each tarball's sha256 and
proposed verdict, the groups never replayed, and the step-5N native fallback
(trigger, candidates, jobs, verdict from the ledger), the step-7 root cause and step-8
fix (ROOT_CAUSE.md, fix/ tarball, their ledger lines) -- and leaves `TODO(agent)` where
judgement is needed (verdict per group, what it means, caveats, next steps).
Model-independent: it reads only what ledger.py, wave_summary.py,
compare_nan_capsules.py, the node launchers and make_repro_readme.py write.
"""

# flake8: noqa: E741,E731,E402  (terse variable names, kept from the original scripts)

import argparse
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def read(p):
    try:
        return open(p, errors="replace").read()
    except OSError:
        return ""


def group_of(d):
    """'G1' for replay/G1/c2.* and verify-G1/repro-G1/out/c2.*; else the parent dir."""
    m = re.search(r"(?:^|/)(?:replay/|verify-|repro-)(G\d+)(?:/|$)", d)
    return m.group(1) if m else os.path.basename(os.path.dirname(d))


def attempt_facts(d):
    s = read(os.path.join(d, "summary.txt")).splitlines()
    tot = next((l for l in s if l.startswith("=== totals")), "")
    m = re.search(r"non-finite (\d+)/(\d+)", tot)
    copies = [l for l in s if re.match(r"\s+g\d+_k\d+\.", l)]
    gpus = len({re.match(r"\s+g(\d+)_", l).group(1) for l in copies})
    c = re.search(r"/c(\d+)\.\d+\.([\w-]+)$", d.rstrip("/"))
    res = next((l.split("=", 1)[1] for l in s if l.startswith("ATTEMPT_RESULT=")), "?")
    env = sorted(
        {re.sub(r"^\s*\d+\s+env:\s*", "", l).strip() for l in s if "env:" in l}
    )
    stop = next((l for l in s if l.startswith("=== stopped early")), "")
    prop = next(
        (
            re.sub(r"^=== proposed verdict: (.*) -- confirm.*", r"\1", l)
            for l in s
            if l.startswith("=== proposed verdict")
        ),
        "",
    )
    first = next(
        (
            re.sub(
                r"^=== first op to go non-finite \(all copies\): (.*) ===$", r"\1", l
            )
            for l in s
            if l.startswith("=== first op to go non-finite")
        ),
        "",
    )
    log = sorted(glob.glob(os.path.join(d, "g*_k*.log")))[:1]
    lines = read(log[0]).splitlines() if log else []
    build = next(
        (l.split(":", 1)[1].strip() for l in lines if l.startswith("captured build:")),
        "?",
    )
    cont = []
    for i, l in enumerate(lines):
        if l.startswith("contention:"):
            cont = [l.split(":", 1)[1].strip()] + [
                x.strip() for x in lines[i + 1 : i + 2] if x.startswith(" ")
            ]
    hits, runs = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    # The load beside each replay copy: a co-tenant (CAPSULE_COTENANT, 5d), else none
    # but the copies themselves. Without it a co-tenant rate reads like a solo rate.
    cot = next(
        (
            l.split(":", 1)[1].strip()
            for l in lines
            if l.strip().startswith("co-tenant:")
        ),
        "",
    )
    cot = re.sub(r"^python3 -m nan_doctor_rt\.(\S+)", r"\1", cot)
    cot = re.sub(
        r" --(hidden|inter|heads|kv-heads|head-dim|layers|vocab|seq|batch|qkv-bias) \S+",
        "",
        cot,
    )
    return dict(
        dir=d,
        load=cot or "replay copies only",
        conc=c.group(1) if c else "?",
        host=c.group(2) if c else "?",
        gpus=gpus,
        copies=len(copies),
        hits=hits,
        runs=runs,
        result=res,
        env="; ".join(env) or "?",
        build=build,
        contention=" / ".join(cont) or "?",
        proposed=prop,
        first=first,
        stop=re.sub(r"^=== stopped early \(([^)]*)\).*", r"\1", stop) if stop else "",
    )


def _ledger_table(path):
    import io
    from contextlib import redirect_stdout

    from . import ledger

    buf, old = io.StringIO(), sys.argv
    sys.argv = ["ledger", path, "table"]
    try:
        with redirect_stdout(buf):
            ledger.main()
    except SystemExit:
        pass
    finally:
        sys.argv = old
    return buf.getvalue()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("artifact_dir")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    A = os.path.abspath(a.artifact_dir)
    led = os.path.join(A, "ledger.md")
    ledger = read(led).splitlines()
    out = []
    w = out.append
    w("# NaN investigation report -- %s\n" % os.path.basename(A))
    w(
        "_Drafted by make_report.py from the ledger and artifacts; every `TODO(agent)` needs "
        "judgement. Numbers below are copied from files, not from memory._\n"
    )

    w(
        "## Verdict\n\nTODO(agent): one paragraph per group -- verdict (step5-read.md table), the op, "
        "and what it means; plus the step-5N verdict if it ran (a native reproducer, not a capsule). "
        "Use the per-rung wording in step9-report.md. Then the three outcomes, one line each: "
        'reproducer (step 6), root cause (step 7), fix (step 8) -- never "reproducer done" as "fixed".\n'
    )

    w("## Time per step\n")
    if os.path.exists(led):
        t = _ledger_table(led)
        ended = set(
            re.findall(r"^\S+ \S+\s+(step\d+)\S*\s+end\b", "\n".join(ledger), re.M)
        )
        skipped = []
        for st in ["step%d" % i for i in range(2, 9)]:
            if st in ended:
                continue
            hits = [
                l
                for l in ledger
                if re.search(r"\b%s\b[^;(]{0,12}\b(skip\w*|SKIP|NO|declined)\b" % st, l)
            ]
            # the step's own line first ("<ts>  step2  skipped ..."), then a mention elsewhere (step 1's answers)
            hits.sort(key=lambda l: not re.match(r"^\S+ \S+\s+%s\b" % st, l))
            why = [re.sub(r"^\S+ \S+\s+\S+\s+", "", l)[:160] for l in hits]
            if (
                not why
                and st == "step3"
                and any(re.search(r"(?i)\bstep2\b.*\bskip", l) for l in ledger)
            ):
                why = ["follows from the step-2 skip (no rate to raise)"]
            skipped.append(
                "* %s: *skipped* -- %s" % (st, why[0] if why else "TODO(agent) why")
            )
        w(t.strip() + "\n\nNot run:\n\n" + ("\n".join(skipped) or "* (none)") + "\n")

    w("## Where the rate came from\n")
    rate = [
        l
        for l in ledger
        if re.search(r"\bstep[24]\b", l) and re.search(r"rate|skip|wave", l)
    ]
    w("\n".join("    " + l for l in rate[-8:]) or "    (no rate line in the ledger)")
    for f in sorted(glob.glob(os.path.join(A, "wave_summary.*.txt"))):
        s = [
            l
            for l in read(f).splitlines()
            if l.startswith(("   ", "== by GPU", "== NaN step", "!!"))
        ]
        w("\n`%s`:\n\n" % os.path.basename(f) + "\n".join("    " + l for l in s))
    if any(re.search(r"(?i)\bstep2\b.*\bskip", l) for l in ledger):
        ctl = [
            l.strip()
            for f in glob.glob(os.path.join(A, "wave_summary.*.txt"))
            for l in read(f).splitlines()
            if re.match(r"\s+ctl\s+rate", l)
        ]
        if (
            not ctl
        ):  # no saved wave_summary file: the step-4 ledger line that quotes the control arm
            ctl = [
                re.sub(r"^\S+ \S+\s+\S+\s+", "", l)[:200]
                for l in ledger
                if re.search(r"\sstep4\s", l) and re.search(r"\bctl\b.*\d+/\d+", l)
            ]
        if ctl:
            w(
                "\nStep 2 was skipped: the only model-level rate is the step-4 control arm (%s), "
                "in that wave's environment.\n" % "; ".join(ctl)
            )
        else:
            w(
                "\nStep 2 was skipped and the capture wave had no control arm: steps 2-4 measured no "
                "model-level rate. Step 8's stock arm, if it ran, is the only uncaptured measurement "
                "(see Fix). TODO(agent): quote it, or say there is none.\n"
            )
    else:
        w(
            "\nTODO(agent): measured in step 2 (with environment), or user-supplied (provisional)?\n"
        )

    w("## Image, build and replay environment\n")
    # `madengine nan replay` writes replay/<G>/; verifying a package
    # replays from the extracted tarball, so its attempts live in verify-<G>/.../out/.
    atts = sorted(
        glob.glob(os.path.join(A, "replay", "*", "c*.*.*"))
        + glob.glob(os.path.join(A, "verify-*", "*", "out", "c*.*.*"))
    )
    facts = [
        attempt_facts(d) for d in atts if os.path.exists(os.path.join(d, "summary.txt"))
    ]
    imgs = sorted(
        {
            m.group(1)
            for f in glob.glob(os.path.join(A, "repro-*", "README.md"))
            for m in [re.search(r"\* (?:Image|Environment[^:]*): `([^`]+)`", read(f))]
            if m
        }
    )
    kerns = {}
    for k in sorted(glob.glob(os.path.join(A, "replay", "*", "kernel.*.txt"))):
        kf = dict(
            re.match(r"(\w+):\s*(.*)", l).groups()
            for l in read(k).splitlines()
            if re.match(r"\w+:", l)
        )
        kerns[os.path.basename(os.path.dirname(k))] = kf
    inj = sorted({kf.get("injected", "?").split(" --")[0] for kf in kerns.values()})
    native_v = [
        re.search(r"VERDICT (NATIVE_\w+)", l).group(1)
        for l in ledger
        if re.search(r"VERDICT NATIVE_\w+", l)
    ]
    if inj:
        injected = ", ".join(inj)
    elif (
        native_v
    ):  # no capsule kernel to enumerate: the native verdict names the owner instead
        injected = (
            "n/a (native reproducer, %s: see Native reproducer / Root cause)"
            % native_v[-1]
        )
    else:
        injected = "unknown (no kernel identified)"
    w("* image: %s -- NaN-injected: %s" % (", ".join(imgs) or "TODO(agent)", injected))
    for b in sorted({f["build"] for f in facts}):
        w("* captured build: `%s`" % b)
    for e in sorted({f["env"] for f in facts}):
        w("* env banner: `%s`" % e)
    w("")

    if kerns:
        w("## Kernel\n")
        for g, kf in sorted(kerns.items()):
            w(
                "* %s: solution_index %s, %s -- `%s`"
                % (
                    g,
                    kf.get("solution_index", "?"),
                    kf.get("arch", "?"),
                    kf.get("kernel", "?"),
                )
            )
            w("  * bench: `%s`" % kf.get("bench", "?"))
        idx = {kf.get("solution_index") for kf in kerns.values()}
        if len(kerns) > 1 and len(idx) == 1:
            w(
                "\nEvery identified group dispatches the SAME kernel (solution_index %s)."
                % idx.pop()
            )
        w("")
    w("## Replays (5b) and the contention behind every rate\n")
    w(
        "| group | attempt | host | copies/GPU x GPUs | load beside each copy | non-finite / runs | early stop | result | proposed verdict | first non-finite | capture-time contention |"
    )
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for f in facts:
        g = group_of(f["dir"])
        w(
            "| %s | `%s` | %s | %s x %d | %s | %d/%d | %s | %s | %s | %s | %s |"
            % (
                g,
                os.path.basename(f["dir"]),
                f["host"],
                f["conc"],
                f["gpus"],
                f["load"],
                f["hits"],
                f["runs"],
                f["stop"] or "no",
                f["result"],
                f["proposed"] or "-",
                f["first"] or "-",
                f["contention"],
            )
        )
    w(
        "\nA one-copy `0/N` on a contention-gated fault is an experiment not set up, not a clean result.\n"
    )

    w("## Groups\n")
    pr = read(os.path.join(A, "groups_priority.txt")).splitlines()
    replayed = {group_of(f["dir"]) for f in facts}
    for l in pr:
        g = l.split()[0]
        l = re.sub(r"\S*/%s/" % re.escape(os.path.basename(A)), "<ARTIFACT_DIR>/", l)
        w(
            "* %s%s"
            % (l, "" if g in replayed else "  -- **not replayed**: TODO(agent) why")
        )
    w("")

    w("## Reproducer tarballs\n")
    last, fixes = {}, {}  # the latest line per tarball wins
    for l in ledger:
        m = re.search(r"TARBALL (\w+)\s+(\S+\.tar\.gz)\s+size=(\d+)\s*(.*)", l)
        if m:
            # a step-8 fix package is not a reproducer: listed under its own heading
            (fixes if re.search(r"\sstep8\S*\s", l) else last)[m.group(2)] = m
    fmt = lambda d: "\n".join(
        "* `%s`  sha256 `%s`  %s B  %s"
        % (m.group(2), m.group(1), m.group(3), m.group(4))  # noqa: E731
        for _, m in sorted(d.items())
    )
    w(fmt(last) or "* none")
    if fixes:
        w("\nFix tarballs (step 8, not reproducers):\n")
        w(fmt(fixes))
    w(
        "\nTODO(agent): confirm each README `CHECK` line was resolved; say which groups were folded.\n"
    )

    native = [
        re.sub(r"^\S+ \S+\s+", "", l)
        for l in ledger
        if re.search(r"\bstep(?:5n|6\[native\])\s", l)
    ]
    if native or os.path.isdir(os.path.join(A, "native")):
        w("## Native reproducer (step 5N)\n")
        trig = [l for l in native if "NATIVE_TRIGGER=" in l]
        w(
            "* trigger: %s"
            % (
                trig[-1].split(None, 1)[1]
                if trig
                else "TODO(agent) -- no NATIVE_TRIGGER line"
            )
        )
        cands = sorted(
            glob.glob(os.path.join(A, "native", "repro_*.py")),
            key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p)],
        )
        if os.path.exists(os.path.join(A, "native", "handoff.md")):
            w(
                "* handoff from the capsule path: `native/handoff.md` (KEEP/STRIKE lines below)"
            )
        w(
            "* candidates on disk: %s"
            % (", ".join("`%s`" % os.path.basename(c) for c in cands) or "none")
        )
        rows = [
            l.split(None, 1)[1]
            for l in native
            if not re.match(r"\S+\s+(start|end)\b", l) and "NATIVE_TRIGGER=" not in l
        ]
        w(
            "\nLedger (evidence, candidates, jobs, verdict, verification):\n\n"
            + ("\n".join("    " + r for r in rows) or "    (none)")
        )
        hits = sorted(glob.glob(os.path.join(A, "native", "hit_*.pt")))
        w(
            "\n* saved hits: %d%s"
            % (len(hits), " (e.g. `%s`)" % os.path.basename(hits[0]) if hits else "")
        )
        w(
            "\nTODO(agent): the 5N verdict (step5n-native.md N6), the minimal candidate and what it "
            "needs (load, components), and every removal or load that ran clean -- the trigger's bounds.\n"
        )

    for st, d, f, title in (
        ("step7", "rootcause", "ROOT_CAUSE.md", "Root cause (step 7)"),
        ("step8", "fix", "README.md", "Fix (step 8)"),
    ):
        rows = [
            l.split(None, 1)[1]
            for l in ledger
            if re.search(r"\b%s\b" % st, l)
            and not re.match(r"\S+\s+\S+\s+\S+\s+(start|end)\b", l)
        ]
        doc = os.path.join(A, d, f)
        w("## %s\n" % title)
        w(
            "* `%s`: %s"
            % (os.path.join(d, f), "on disk" if os.path.exists(doc) else "missing")
        )
        on_disk = len(glob.glob(os.path.join(A, d, "iter-*.md")))
        in_ledger = len(
            set(
                re.findall(
                    r"\b%s(?:\[[^\]]*\])?\s+iter (\d+)\b" % st, "\n".join(ledger)
                )
            )
        )
        w(
            "* iterations on disk: %d%s"
            % (
                on_disk,
                (
                    ""
                    if in_ledger in (0, on_disk)
                    else " -- **the ledger has %d**: one `iter-<k>.md` per iteration (%s), write the missing one(s)"
                    % (
                        in_ledger,
                        "step7-rootcause.md" if st == "step7" else "step8-fix.md",
                    )
                ),
            )
        )
        w(
            "\nLedger (budget, kind, iterations, outcome):\n\n"
            + ("\n".join("    " + r for r in rows) or "    (none -- skipped?)")
        )
        w(
            "\nTODO(agent): %s\n"
            % (
                "the root cause (where, mechanism, trigger, causal experiment) and which track found it, "
                "or NO_ROOT_CAUSE with excluded and still-open hypotheses, or why skipped."
                if st == "step7"
                else "the fix (change, fix or workaround, validation: control n/N, patched 0/N with expected hits, "
                "load, values, cost, tarball sha256), or NO_FIX with candidates tried, or why skipped."
            )
        )

    capped = [
        "* 5b %s stopped at concurrency %s (the first that reproduced)%s"
        % (
            group_of(f["dir"]),
            f["conc"],
            "; early stop at %s" % f["stop"] if f["stop"] else "",
        )
        for f in facts
        if f["result"] == "REPRODUCED"
    ]
    w(
        "## Capped, skipped or folded\n\n"
        + "\n".join(capped)
        + "\n\nTODO(agent): anything else capped, sampled or folded -- and on what basis.\n"
    )
    w("## Next steps\n\nTODO(agent)\n")
    path = a.out or os.path.join(A, "report.md")
    if os.path.exists(path) and not a.out:
        path = os.path.join(A, "report.draft.md")
    open(path, "w").write("\n".join(out) + "\n")
    print("DRAFT %s (%d TODO)" % (path, sum(l.count("TODO(agent)") for l in out)))


if __name__ == "__main__":
    main()
