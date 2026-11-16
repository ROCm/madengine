#!/usr/bin/env python3
"""Write README.md for a step-6 reproducer, tar it, and print its sha256 (debug-nan).

    madengine nan readme --repro <ARTIFACT_DIR>/repro-G1 \
        --attempt <ARTIFACT_DIR>/replay/G1/c2.<job>.<host> --groups <ARTIFACT_DIR>/groups.json \
        [--image IMAGE] [--pack] [--ledger <LEDGER> --step 'step6[G1]']

Everything mechanical comes from the artifacts, so nobody transcribes it: the op and its
geometry key, the capsule's name/size/sha256, the `captured build:` lines, the
`env:` verdict, the reproducing rate with copies per GPU and GPU count, the one
command, and the "one copy alone means nothing" warning.

The operand-validation sentence is PROPOSED from the [C1] walk in the attempt's
summary.txt (replay_verdict.propose) and marked `<!-- CHECK -->`: KERNEL_BUG when
the terminal op's fp64 reference on the replay's own operands is finite, or its
reference is finite and no ancestor DIVERGED; KERNEL_BUG_UNVERIFIED when the
terminal op has no reference (`unavailable` / `-`) or an ancestor DIVERGED (trailing
view rows alias the op before them and are not the terminal); a
deterministic walk (terminal non-finite, nothing DIVERGED) is flagged for
step5-replay-guardrail.md.  The agent confirms it (step6-finish.md) -- the script
does not decide verdicts.  The command and environment line follow the run.sh shipped.

--pack writes <repro>.tar.gz beside the directory and prints `sha256 path`;
--ledger appends one line with path, size and sha256.  --repack re-tars an edited
README (no rewrite) and ledgers the new sha256: `--repro <dir> --repack [--ledger ..]`.
The image defaults to $IMAGE, else run.sh's `IMAGE=${IMAGE:-...}`; a kernel.*.txt
beside the attempt (written by the replay ladder) becomes the README's Kernel section.
"""

# flake8: noqa: E741,E731,E402  (terse variable names, kept from the original scripts)

import argparse
import ast
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

from .verdict import (
    c1_rows,
    c1_terminal,
    op_terminal,
)
from .verdict import propose as _propose  # one copy of the rules

CONTENTION = (
    "**One copy alone reports `0/N` and means nothing.** It reproduced with {conc} copies "
    "per GPU, which are each other's co-tenants; a contention-gated fault needs that "
    "load. Do not close this after a single-process run."
)
EVERY_RUN = (
    "**Every run failed** with {conc} copies per GPU ({rate}): this is not a rare race "
    "waiting for a co-tenant, and one copy is expected to reproduce it too "
    "(`CAPSULE_CONCURRENCY=1 bash run.sh`)."
)
COTENANT = (
    "**The replay alone does not reproduce this.** Each GPU runs one replay beside a "
    "co-tenant workload -- `{cot}` -- which `run.sh` starts and stops with it. Replayed "
    "alone, or with copies of itself as the only load, the same capsule ran clean: the "
    "fault needs that kind of device load (mad-debug-nan step5-pressure.md). Do not close "
    "this after a run without the co-tenant."
)
SOLO = (
    "It reproduced with **one copy per GPU** -- no co-tenant needed at this rate, which "
    "points to a deterministic or load-independent fault rather than a race."
)

KB = (
    "The fp64 CPU reference returns finite from these operands and every upstream op "
    "reproduces its recorded value, so a correct kernel does not produce this NaN."
)
KB_TERMINAL = (
    "On the very operands this replay hands the op, its fp64 CPU reference is finite "
    "and fits the output dtype, so no correct kernel returns a non-finite value "
    "from them -- whatever produced those operands."
)
KB_SYNTH = (
    "On these operands -- synthesized from the capsule's per-tensor statistics, "
    "since the capsule holds no floating-point values -- the fp64 CPU reference is finite and "
    "fits the output dtype, so a correct kernel returns no non-finite value from them. "
    "Upstream ops were not compared with recorded values (there are none), and this "
    "says nothing about the operands the model actually used."
)
SYNTH_NOTE = (
    "\n**The capsule holds no floating-point values** -- shapes, strides, alignment "
    "and a few per-tensor statistics only; integer tensors (sequence offsets, "
    "indices) keep their recorded values. Every replay runs on look-alike float "
    "operands drawn from those statistics (`--synthesize`, which `run.sh` passes); a "
    "fault that needs the real float values will not reproduce from it.\n"
)
SYNTH_DERIVED = (
    "\nDerived operands (the attention forward's `out` and `softmax_lse`) are not drawn "
    "from statistics: the replay recomputes them with the forward from the synthesized "
    "q/k/v, so they are consistent and a fix can be validated on this package.\n"
)
SYNTH_INCONSISTENT = (
    "\n**Derived operands are inconsistent**: `out`/`softmax_lse` were drawn from "
    "their statistics independently of q/k/v (the forward could not be re-run), and a "
    "correct kernel can overflow on such operands. This package shows the NaN but "
    "cannot validate a fix; check a fix on consistent operands "
    "(`python3 -m nan_doctor_rt.attn_consistent_check`).\n"
)
UNV = (
    "This is the recorded dependency chain; the terminal op has no CPU reference (or an "
    "upstream op did not reproduce its recorded value), so its operands are reproduced "
    "rather than verified. The op named here is where the NaN first APPEARS, not "
    "necessarily where the fault is: an upstream op that writes finite but wrong values "
    "(a race) hands them on, and the op that then overflows is the one blamed. "
    "`python3 -m nan_doctor_rt.replay_nan_op --chain --chain-iters N --contend C --chain-repeat-check` names "
    "the first op whose output stopped repeating in the non-finite runs."
)


HELD = (
    "\n* **`{var}` is held back by the replay's env restore** (`env:` banner `held back:`) and "
    "`run.sh` does not pass it: if the workload set it (e.g. a library path the model needs), "
    "add `--env {var}=<the workload's value>` to the replay command in `run.sh`, or the replay runs without it."
)


def cotenant_of(val):
    """The co-tenant command, or None: "none"/empty = copies are the only load, so the README
    gets no "the replay alone does not reproduce this" paragraph."""
    return None if (val or "").strip().lower() in ("", "none", "0", "off") else val


def held_back(logs, repro):
    """Vars a copy log's env banner lists as `held back:` that run.sh does not export."""
    run_sh = os.path.join(repro, "run.sh")
    txt = open(run_sh).read() if os.path.exists(run_sh) else ""
    out = []
    for lg in logs[:1]:
        for l in open(lg, errors="replace").read().splitlines():
            m = re.match(r"\s*held back:\s*([A-Za-z_][A-Za-z0-9_]*)", l)
            if (
                m
                and m.group(1) not in out
                and not re.search(
                    r"(^|[\s;])(export\s+)?%s=" % re.escape(m.group(1)), txt, re.M
                )
            ):
                out.append(m.group(1))
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def propose(rows, terminal=None):
    verdict, why = _propose(rows, terminal)
    kb = KB_TERMINAL if "same operands" in why else KB
    return (
        verdict,
        {"KERNEL_BUG": kb, "KERNEL_BUG_UNVERIFIED": UNV}.get(verdict, ""),
        why,
    )


def image_from_run_sh(repro):
    m = (
        re.search(
            r"IMAGE=\$\{IMAGE:-([^}]+)\}", open(os.path.join(repro, "run.sh")).read()
        )
        if os.path.exists(os.path.join(repro, "run.sh"))
        else None
    )
    return m.group(1) if m else None


def pack(repro, ledger, step, verdict):
    tgz = repro.rstrip("/") + ".tar.gz"
    subprocess.check_call(
        [
            "tar",
            "-czf",
            tgz,
            "-C",
            os.path.dirname(repro),
            "--exclude",
            "out",
            os.path.basename(repro),
        ]
    )
    line = "%s  %s  size=%d" % (sha256(tgz), tgz, os.path.getsize(tgz))
    print("SHA256 " + line)
    if ledger and os.path.exists(ledger):
        from . import ledger as _ledger

        _ledger.append(
            ledger,
            step,
            "TARBALL %s %s"
            % (line.replace(os.path.dirname(os.path.dirname(tgz)) + "/", ""), verdict),
        )


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="madengine nan readme", description=__doc__.split("\n")[0]
    )
    p.add_argument("--repro", required=True)
    p.add_argument(
        "--attempt",
        help="the reproducing attempt dir (summary.txt + copy logs); not with --repack",
    )
    p.add_argument("--groups", help="groups.json, for the op and its geometry")
    p.add_argument(
        "--image",
        default=os.environ.get("IMAGE"),
        help="the image it reproduced in (default: $IMAGE; required)",
    )
    p.add_argument("--pack", action="store_true")
    p.add_argument(
        "--repack", action="store_true", help="re-tar the edited README; write nothing"
    )
    p.add_argument(
        "--cotenant",
        default=os.environ.get("CAPSULE_COTENANT"),
        help="the co-tenant command the reproducer runs beside the replay "
        "(default $CAPSULE_COTENANT); none = copies are the only load",
    )
    p.add_argument(
        "--mode",
        default=os.environ.get("CAPSULE_MODE", "chain"),
        help="chain or op, as the reproducing ladder ran (default $CAPSULE_MODE)",
    )
    p.add_argument("--ledger")
    p.add_argument("--step", default="step6")
    a = p.parse_args(argv)
    a.cotenant = cotenant_of(a.cotenant)
    repro = os.path.abspath(a.repro)
    if a.repack:
        # Label by the step first: a step-8 fix/ has no capsule either, and is not 5N's.
        if a.step.startswith("step8"):
            what = "fix package (step 8), README by hand"
        elif not glob.glob(os.path.join(repro, "capsule", "*.pt")):
            what = "native package (step 5N), README by hand"
        else:
            what = "repacked after README edit (CHECK resolved)"
        pack(repro, a.ledger, a.step, what)
        return
    if not a.attempt:
        sys.exit("--attempt is required (unless --repack)")
    a.image = a.image or image_from_run_sh(repro)
    if not a.image:
        sys.exit("--image (or $IMAGE, or IMAGE=${IMAGE:-...} in run.sh) is required")
    caps = glob.glob(os.path.join(repro, "capsule", "*.pt"))
    if len(caps) != 1:
        sys.exit("expected exactly one capsule in %s/capsule" % repro)
    cap = caps[0]
    summ = open(os.path.join(a.attempt, "summary.txt")).read().splitlines()
    tot = next((l for l in summ if l.startswith("=== totals")), "")
    m = re.search(r"non-finite (\d+)/(\d+)", tot)
    bad, runs = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    copies = len([l for l in summ if re.match(r"\s+g\d+_k\d+\.", l)])
    gpus = len(
        {
            re.match(r"\s+g(\d+)_", l).group(1)
            for l in summ
            if re.match(r"\s+g\d+_k\d+\.", l)
        }
    )
    conc = re.search(r"/c(\d+)\.", a.attempt.rstrip("/") + "/")
    conc = int(conc.group(1)) if conc else (copies // max(gpus, 1))
    stopped = next((l for l in summ if l.startswith("=== stopped early")), "")
    envl = next(
        (re.sub(r"^\d+\s+", "", l.strip()) for l in summ if "env:" in l),
        "env: (no banner)",
    )
    envl = re.sub(r"\s+", " ", envl)
    # captured build: the line plus its indented continuation, from one copy log
    build, key = [], ""
    for lg in sorted(glob.glob(os.path.join(a.attempt, "g*_k*.log")))[:1]:
        lines = open(lg, errors="replace").read().splitlines()
        for i, l in enumerate(lines):
            if l.startswith(
                ("captured on:", "captured build:", "contention:", "operands:")
            ):
                build.append(l)
                for x in lines[i + 1 : i + 8]:  # its indented continuation only
                    if (
                        not x.startswith("  ")
                        or x.strip().endswith(":")
                        or ":  " in x[:18]
                    ):
                        break
                    build.append(x)
            if l.startswith("hipBLASLt key:") and not key:
                key = l.split(":", 1)[1].strip()
    held = held_back(sorted(glob.glob(os.path.join(a.attempt, "g*_k*.log"))), repro)
    op = geom = ""
    if a.groups:
        rep = os.path.basename(cap)
        for g in json.load(open(a.groups))["groups"]:
            if os.path.basename(g["representative"]) == rep:
                ident = g["identity"]
                if isinstance(ident, str):
                    try:
                        ident = ast.literal_eval(ident)
                    except Exception:
                        ident = {}
                op, geom = ident.get("op", ""), ident.get("geometry_key", "")
    rows = c1_rows(summ)
    terminal = c1_terminal(summ)
    if not rows:  # single-op replay (CAPSULE_MODE=op): the op's own fp64 check
        for lg in sorted(glob.glob(os.path.join(a.attempt, "g*_k*.log"))):
            terminal = op_terminal(open(lg, errors="replace").read().splitlines())
            if terminal:
                break
    verdict, sentence, why = propose(rows, terminal)
    synth = any("SYNTHETIC" in b for b in build)
    inconsistent = any("SYNTHETIC-INCONSISTENT" in b for b in build)
    derived = any("SYNTHETIC (derived" in b for b in build)
    if inconsistent and verdict == "KERNEL_BUG":
        verdict = "KERNEL_BUG_UNVERIFIED"
    if synth and verdict == "KERNEL_BUG":
        sentence = KB_SYNTH
    kern = sorted(
        glob.glob(
            os.path.join(
                os.path.dirname(os.path.abspath(a.attempt.rstrip("/"))), "kernel.*.txt"
            )
        )
    )
    kfacts = (
        dict(
            re.match(r"(\w+):\s*(.*)", l).groups()
            for l in open(kern[-1]).read().splitlines()
            if re.match(r"\w+:", l)
        )
        if kern
        else {}
    )
    kernel_md = ""
    if kfacts.get("kernel", "?")[:1] not in ("?", ""):
        kernel_md = (
            "## Kernel\n\n* solution_index %s, %s\n* `%s`\n* image NaN-injected: %s\n"
            % (
                kfacts.get("solution_index", "?"),
                kfacts.get("arch", "?"),
                kfacts["kernel"],
                kfacts.get("injected", "?"),
            )
        )
        if kfacts.get("bench", "").startswith("hipblaslt-bench"):
            kernel_md += (
                "* Without the capsule (one process; it may need co-tenants to fail):\n\n"
                "  ```bash\n  %s\n  ```\n" % kfacts["bench"]
            )
        kernel_md += "\n"
    chain = " -> ".join(r["op"] for r in rows) or op or "?"
    unit = "launches of the op alone" if a.mode == "op" else "chain runs"
    rate = "%d/%d %s (%.2g per run)" % (bad, runs, unit, bad / runs) if runs else "n/a"
    size = os.path.getsize(cap)

    readme = """# NaN reproducer: {op}{geom}

## What this is

A capsule recorded from a live run of the user's workload at the moment `{op}`
produced a non-finite output, plus the scripts to replay it **without the model, dataset or
training code**.
{synth_note}
* {replayed}: `{chain}`{key}
* Capsule: `capsule/{capname}` ({mb:.1f} MB) -- sha256 `{csha}`
* Captured on / build / contention:
{build}
* Image: `{image}`
* Replay environment: `{envl}`{held}

## How the operands were validated

<!-- CHECK: proposed {verdict} from the [C1] walk ({why}); confirm against step5-read.md -->
{sentence}

{kernel_md}## The one command

```bash
bash run.sh            # {override}
```

It starts **{conc} {copies} of the replay on every GPU** of the node and prints
`LADDER_RESULT=...`; per-copy logs and `summary.txt` land in `out/`. Measured here:
**{rate}** with {conc} {copies} on each of {gpus} GPU{gs}{stop}.

{contention}
""".format(
        op=op or rows[-1]["op"] if rows else op,
        geom=(" " + geom) if geom else "",
        chain=chain,
        replayed=(
            "Op replayed (alone, no chain)" if a.mode == "op" else "Chain replayed"
        ),
        key=("\n* hipBLASLt problem: `%s`" % key) if key else "",
        capname=os.path.basename(cap),
        mb=size / 1e6,
        csha=sha256(cap),
        build="\n".join("  `%s`" % b.strip() for b in build) or "  (not printed)",
        image=a.image,
        envl=envl,
        held="".join(HELD.format(var=v) for v in held),
        override="IMAGE=..., NGPUS=..., CAPSULE_ITER_COUNT=... to override",
        kernel_md=kernel_md,
        synth_note=(
            SYNTH_NOTE
            + (SYNTH_INCONSISTENT if inconsistent else SYNTH_DERIVED if derived else "")
            if synth
            else ""
        ),
        verdict=verdict,
        why=why,
        sentence=sentence,
        conc=conc,
        gpus=gpus,
        rate=rate,
        copies="copy" if conc == 1 else "copies",
        gs="" if gpus == 1 else "s",
        contention=(
            COTENANT.format(cot=a.cotenant)
            if a.cotenant
            else (
                EVERY_RUN.format(conc=conc, rate=rate)
                if runs and bad == runs and conc > 1
                else CONTENTION.format(conc=conc) if conc > 1 else SOLO
            )
        ),
        stop=(
            " (attempt stopped early at its hit target; the rate is over runs done)"
            if stopped
            else ""
        ),
    )
    with open(os.path.join(repro, "README.md"), "w") as f:
        f.write(readme)
    print("README %s/README.md  proposed=%s (%s)" % (repro, verdict, why))
    if verdict.startswith("FP32") or "DETERMINISTIC" in why:
        print("DO_NOT_SHIP %s" % why)
    if a.pack:
        pack(repro, a.ledger, a.step, "proposed=%s (README auto; CHECK)" % verdict)


if __name__ == "__main__":
    sys.exit(main())
