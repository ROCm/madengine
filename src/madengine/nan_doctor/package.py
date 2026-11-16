"""Step 6: build a standalone reproducer directory from a reproducing replay attempt.

    madengine nan package --group G1 --artifact-dir <dir> --image <image> \
        --capsule <capsule.pt> --attempt <attempt dir> --groups groups.json \
        [--concurrency C] [--iters N] [--mode chain|op] [--cotenant CMD] [--extra-args "..."] \
        [--env K=V ...] [--pack] [--ledger LEDGER --step 'step6[G1]']

Writes <artifact-dir>/repro-<group>/:

  run.sh            the one command; CAPSULE_ITER_COUNT, CAPSULE_CONCURRENCY, NGPUS, IMAGE override
  README.md         from the attempt's artifacts (see `madengine nan readme --help`); the proposed
                    operand-validation sentence is marked <!-- CHECK --> for the agent to confirm
  capsule/<pt>      the capsule the attempt replayed
  host/nan_doctor/  this package (standard library only): `python3 -m nan_doctor replay ...`
  runtime/nan_doctor_rt/   the replay's import closure, bind-mounted into the container

Ship exactly what the reproducing attempt ran: whole chain, its concurrency and iterations,
its co-tenant.  A reproducer needs docker, python3 and the image -- not madengine and not a
checkout.  It is bound to the GPU architecture it was captured on.

--pack also writes <repro>.tar.gz and prints `SHA256 <sha>  <path>  size=<n>`; with --ledger
it appends the tarball line.  Verify by extracting the tarball ELSEWHERE (not under /tmp) and
running its run.sh: a reproducing run (n/N > 0 near the ledger's rate) is the verification.
"""

import argparse
import os
import re
import shlex
import shutil
import sys

from . import repro_readme, runner

# What replay_nan_op needs to run (plus the two co-tenant loads), relative to nan_doctor_rt/.
CLOSURE = [
    "__init__.py",
    "op_nan_capture.py",
    "replay_nan_op.py",
    "kernel_patch.py",
    "kernel_mutations.py",
    "replay_hipblaslt_log_with_nan.py",
    "transformer_hog.py",
    "pressure_hog.py",
    "shim/hsa_patch_shim.c",
    "shim/build_hsa_shim.sh",
]

RUN_SH = """#!/bin/bash
# Reproducer: replays one captured capsule under load on every GPU of this node, without the
# model, its data or its code.  Needs docker, python3 and the image.  Bound to the GPU
# architecture it was captured on.
#   IMAGE=...  NGPUS=...  CAPSULE_ITER_COUNT=...  CAPSULE_CONCURRENCY=...  to override
set -uo pipefail
cd -- "$(dirname -- "$0")"
IMAGE=${{IMAGE:-{image}}}
export CAPSULE_ITER_COUNT=${{CAPSULE_ITER_COUNT:-{iters}}} CAPSULE_CONCURRENCY=${{CAPSULE_CONCURRENCY:-{conc}}}
PYTHONPATH="$PWD/host${{PYTHONPATH:+:$PYTHONPATH}}" exec python3 -m nan_doctor replay "capsule/{capname}" \\
  --image "$IMAGE" --out-dir "$PWD/out" --runtime-dir "$PWD/runtime" --mode {mode}{cotenant}{extra}{envs} "$@"
"""


def build(repro, capsule, image, iters, conc, mode, cotenant, extra_args, env):
    if os.path.exists(repro):
        shutil.rmtree(repro)
    os.makedirs(os.path.join(repro, "capsule"))
    cap = os.path.join(repro, "capsule", os.path.basename(capsule))
    shutil.copyfile(capsule, cap)
    here = os.path.dirname(os.path.abspath(__file__))
    shutil.copytree(
        here,
        os.path.join(repro, "host", "nan_doctor"),
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    rt = os.path.join(runner.runtime_parent(), "nan_doctor_rt")
    for rel in CLOSURE:
        src, dst = os.path.join(rt, rel), os.path.join(
            repro, "runtime", "nan_doctor_rt", rel
        )
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    run_sh = os.path.join(repro, "run.sh")
    with open(run_sh, "w") as f:
        f.write(
            RUN_SH.format(
                image=image,
                iters=iters,
                conc=conc,
                capname=os.path.basename(capsule),
                mode=mode,
                cotenant=(" --cotenant %s" % shlex.quote(cotenant)) if cotenant else "",
                extra=(
                    (" --extra-args %s" % shlex.quote(extra_args)) if extra_args else ""
                ),
                envs="".join(
                    " --env %s" % shlex.quote("%s=%s" % kv) for kv in env.items()
                ),
            )
        )
    os.chmod(run_sh, 0o755)
    return cap


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="madengine nan package",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--group", required=True, help="G1, G2, ... (names the directory repro-<group>)"
    )
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--image", required=True, help="the image it reproduced in")
    ap.add_argument("--capsule", required=True, help="the capsule the attempt replayed")
    ap.add_argument(
        "--attempt", required=True, help="the reproducing attempt directory"
    )
    ap.add_argument("--groups", help="groups.json, for the op and its geometry")
    ap.add_argument(
        "--concurrency",
        type=int,
        help="the reproducing concurrency (default: read from the attempt directory name)",
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=int(os.environ.get("CAPSULE_ITER_COUNT") or 1000000),
    )
    ap.add_argument(
        "--mode",
        default=os.environ.get("CAPSULE_MODE", "chain"),
        choices=["chain", "op"],
    )
    ap.add_argument("--cotenant", default=os.environ.get("CAPSULE_COTENANT", ""))
    ap.add_argument("--extra-args", default=os.environ.get("CAPSULE_EXTRA_ARGS", ""))
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--ledger")
    ap.add_argument("--step")
    a = ap.parse_args(argv)
    if not os.path.isfile(a.capsule):
        print("capsule not readable: %s" % a.capsule, file=sys.stderr)
        return 2
    m = re.search(r"/c(\d+)\.", os.path.abspath(a.attempt).rstrip("/") + "/")
    conc = a.concurrency or (int(m.group(1)) if m else None)
    if not conc:
        print(
            "--concurrency is required when the attempt directory is not named c<N>.<tag>.<host>",
            file=sys.stderr,
        )
        return 2
    repro = os.path.join(os.path.abspath(a.artifact_dir), "repro-%s" % a.group)
    env = dict(kv.partition("=")[::2] for kv in a.env)
    cap = build(
        repro,
        os.path.abspath(a.capsule),
        a.image,
        a.iters,
        conc,
        a.mode,
        a.cotenant.strip() if a.cotenant else "",
        a.extra_args.strip(),
        env,
    )
    rd = [
        "--repro",
        repro,
        "--attempt",
        a.attempt,
        "--image",
        a.image,
        "--mode",
        a.mode,
        "--cotenant",
        a.cotenant or "none",
        "--step",
        a.step or "step6[%s]" % a.group,
    ]
    if a.groups:
        rd += ["--groups", a.groups]
    if a.ledger:
        rd += ["--ledger", a.ledger]
    if a.pack:
        rd += ["--pack"]
    print(
        "PACKAGE %s (capsule %s, concurrency %d, %d iters)"
        % (repro, os.path.basename(cap), conc, a.iters)
    )
    return repro_readme.main(rd) or 0


if __name__ == "__main__":
    sys.exit(main())
