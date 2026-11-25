#!/usr/bin/env python3
"""6: does the reproducer reproduce on OTHER nodes?  A reproducer that reproduced only on the
node step 5b ran on looks complete and is not; this runs the extracted package on K other SLURM
nodes at once, each on its own copy, as a vendor would (from the package's directory, in an empty
environment), and scores each node from the package's own result line.

    madengine nan verify-nodes script --package DIR --out DIR [--nodes 2] [--exclude HOST]
        [--script run.sh] [--partition P] [--time 01:00:00] [--env K=V ...] [--submit]
    madengine nan verify-nodes score <OUT>

`script` writes <OUT>/verify_nodes.slurm (one srun task per node) and <OUT>/verify_node.sh (one
node's run; standalone, so it also runs without SLURM) and prints the sbatch command; --submit
submits it (`VERIFY_NODES=SUBMITTED job=<id>`, or `VERIFY_NODES=NO_SLURM` and exit 3 on a host
without sbatch).  Per node:
<OUT>/<host>.log, the package's output.  Kept from the environment: HOME USER LOGNAME PATH LANG
TERM TMPDIR and every --env (an override the package's README documents, e.g. IMAGE=...).

`score` reads each <host>.log's last LADDER_RESULT= / REPRO_RESULT= / NATIVE_TOTAL= line (a native
package: reproduced when hits > 0).  Last line:
    NODES_REPRODUCED=<k>/<K> hosts=<host>:<REPRODUCED|NOT_REPRODUCED|ERROR>,... [tag]
tag: NODE_DEPENDENT (some nodes, no ERROR: the node is a variable of the bug -- ledger it, say it
in the README, step 7 reads it) or REPRODUCES_NOWHERE_ELSE (no node, no ERROR).
"""

import argparse
import glob
import os
import re
import shlex
import shutil
import subprocess
import sys

KEEP = ("HOME", "USER", "LOGNAME", "PATH", "LANG", "TERM", "TMPDIR")
RESULT = re.compile(r"^(LADDER_RESULT|REPRO_RESULT|NATIVE_TOTAL)=.*", re.M)

NODE_SH = """#!/bin/bash
# One node of `madengine nan verify-nodes`: copy the package, run its script from that copy in an
# empty environment, log to <out>/<host>.log.
set -uo pipefail
h=${{SLURMD_NODENAME:-$(hostname -s)}}
d={out}/pkg.$h
rm -rf "$d" && cp -a {package} "$d" && chmod -R a+rwX "$d" ||
  {{ echo "NODE_DONE $h rc=2"; exit 2; }}
keep=()
for v in {keep}; do [ -n "${{!v:-}}" ] && keep+=("$v=${{!v}}"); done
cd -- "$d" && env -i "${{keep[@]}}" {extra} bash ./{script} > {out}/$h.log 2>&1
rc=$?
echo "PACKAGE_RC=$rc" >> {out}/$h.log
echo "NODE_DONE $h rc=$rc"
"""

SBATCH = """#!/bin/bash
#SBATCH --job-name=nan-verify-nodes
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time={time}
#SBATCH --output={out_raw}/verify_nodes.%j.out
{extra}# independent runs, not one multi-rank job: no node may be ended because another finished
srun --ntasks-per-node=1 --wait=0 bash {out}/verify_node.sh
{python} -m madengine.nan_doctor.verify_nodes score {out}
"""


def score_line(text):
    """REPRODUCED | NOT_REPRODUCED | ERROR, and the result line it came from."""
    hits = [m.group(0) for m in RESULT.finditer(text)]
    line = hits[-1] if hits else ""
    if line.startswith("NATIVE_TOTAL="):
        n = re.match(r"NATIVE_TOTAL=(\d+)/", line)
        return ("REPRODUCED" if n and int(n.group(1)) > 0 else "NOT_REPRODUCED"), line
    v = re.sub(r"^[A-Z_]+=([A-Z_]+).*", r"\1", line) if line else ""
    if v == "REPRODUCED":
        return v, line
    if v in ("NOT_REPRODUCED", "CLEAN", "NOT_SET_UP"):  # NOT_SET_UP: ran clean, stopped
        return "NOT_REPRODUCED", line
    return "ERROR", line


def score(out):
    logs = sorted(glob.glob(os.path.join(out, "*.log")))
    if not logs:
        print("NODES_REPRODUCED=0/0 hosts= (no <host>.log in %s)" % out)
        return 1
    k = errs = 0
    res = []
    for f in logs:
        host = os.path.basename(f)[: -len(".log")]
        with open(f, errors="replace") as fh:
            v, line = score_line(fh.read())
        k += v == "REPRODUCED"
        errs += v == "ERROR"
        res.append("%s:%s" % (host, v))
        print("  %s: %s" % (host, line or "no result line -- read " + f))
    n = len(logs)
    tag = ""
    if not errs and 0 < k < n:
        tag = " NODE_DEPENDENT"
    elif not errs and k == 0:
        tag = " REPRODUCES_NOWHERE_ELSE"
    print("NODES_REPRODUCED=%d/%d hosts=%s%s" % (k, n, ",".join(res), tag))
    return 0


def write_script(a):
    package = os.path.realpath(a.package)
    if not os.path.isfile(os.path.join(package, a.script)):
        sys.exit("no %s in %s" % (a.script, package))
    for kv in a.env:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", kv):
            sys.exit("--env takes K=V: %r" % kv)
    out = os.path.realpath(a.out)
    os.makedirs(out, exist_ok=True)
    q = shlex.quote
    node = os.path.join(out, "verify_node.sh")
    with open(node, "w") as f:
        f.write(
            NODE_SH.format(
                out=q(out),
                package=q(package),
                keep=" ".join(KEEP),
                extra=" ".join(q(kv) for kv in a.env),
                script=q(a.script),
            )
        )
    extra = ""
    if a.exclude:
        extra += "#SBATCH --exclude=%s\n" % a.exclude
    if a.partition:
        extra += "#SBATCH --partition=%s\n" % a.partition
    sb = os.path.join(out, "verify_nodes.slurm")
    with open(sb, "w") as f:
        f.write(
            SBATCH.format(
                nodes=a.nodes,
                time=a.time,
                out=q(out),
                out_raw=out,
                extra=extra,
                python=q(sys.executable),
            )
        )
    print("wrote %s and %s" % (sb, node))
    if not a.submit:
        print(
            "submit: sbatch %s   (then: madengine nan verify-nodes score %s)"
            % (sb, out)
        )
        return 0
    if not shutil.which("sbatch"):
        print("VERIFY_NODES=NO_SLURM (no sbatch on this host)")
        return 3
    r = subprocess.run(["sbatch", "--parsable", sb], capture_output=True, text=True)
    if r.returncode != 0:
        print("VERIFY_NODES=ERROR sbatch: %s" % r.stderr.strip()[-300:])
        return 1
    print(
        "VERIFY_NODES=SUBMITTED job=%s out=%s" % (r.stdout.strip().split(";")[0], out)
    )
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="madengine nan verify-nodes",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("script", help="write the sbatch job for K other nodes")
    s.add_argument("--package", required=True, help="the extracted package dir")
    s.add_argument("--out", required=True, help="where the logs and scripts go")
    s.add_argument("--nodes", type=int, default=2)
    s.add_argument("--exclude", default="", help="the host step 5b ran on")
    s.add_argument("--script", default="run.sh")
    s.add_argument("--partition", default="")
    s.add_argument("--time", default="01:00:00")
    s.add_argument("--env", action="append", default=[], metavar="K=V")
    s.add_argument("--submit", action="store_true", help="submit the job with sbatch")
    c = sub.add_parser("score", help="score the per-node logs")
    c.add_argument("out")
    a = p.parse_args(argv)
    if a.cmd == "script":
        return write_script(a)
    return score(a.out)


if __name__ == "__main__":
    sys.exit(main())
