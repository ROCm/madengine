#!/usr/bin/env python3
"""R3 (debug-nan, world size > 1): is a collective itself wrong?

A capture run with NAN_CAPTURE_RECORD_COLLECTIVE=<group>:<seq>[:<step>] (group and seq from
the job manifest's earliest arrival) leaves, on every member rank,
<capsule dir>/collective_g<group>_s<seq>.rank<r>.pt: the collective's inputs before the call
and its outputs after it landed.

    python3 -m nan_doctor_rt.collective_r3 check  <dir> --group G --seq S          # no GPU
    torchrun --nproc_per_node=<group size> collective_r3.py replay <dir> --group G --seq S [--iters 200]

check   recomputes, in fp64 from ALL ranks' recorded inputs, what a correct collective must
        have produced on each rank, and compares with what each rank recorded:
          COLLECTIVE_FAULT ranks [..]   output wrong while every input was finite
          INPUT_NONFINITE  ranks [..]   the NaN came in on those ranks' inputs (look there)
          CORRECT                       outputs match the inputs
replay  re-runs the same collective on the recorded inputs --iters times on this machine's
        GPUs and counts outputs that differ from the fp64 expectation (an intermittent
        transport fault reproduces here, without the model).

Covers c10d allreduce_ / _allgather_base_ / _reduce_scatter_base_ / alltoall_base_ and the
_c10d_functional all_reduce / all_gather_into_tensor / reduce_scatter_tensor /
all_to_all_single.  A c10d reduction's op (a ScriptObject) is not recorded: check tries
sum / avg / max / min and takes the one most ranks agree with.
"""
import argparse
import glob
import os
import sys

import torch


def load(d, group, seq):
    recs = {}
    for f in glob.glob(os.path.join(d, "**", "collective_g%s_s%d.rank*.pt" % (group, seq)), recursive=True):
        r = torch.load(f, map_location="cpu", weights_only=False)
        recs[r["rank"]] = r
    if not recs:
        sys.exit("no collective_g%s_s%d.rank*.pt under %s" % (group, seq, d))
    ranks = next(iter(recs.values())).get("group_ranks") or sorted(recs)
    missing = [r for r in ranks if r not in recs]
    if missing:
        sys.exit("records missing for ranks %s (every member of the group must record)" % missing)
    return [recs[r] for r in ranks], ranks


def kind(op):
    for k in ("all_to_all", "alltoall", "reduce_scatter", "allgather", "all_gather", "allreduce", "all_reduce"):
        if k in op:
            return {"alltoall": "all_to_all", "allgather": "all_gather",
                    "allreduce": "all_reduce"}.get(k, k)
    return None


def io(rec):
    """(input, recorded output, extra) of one rank's record."""
    op, a, ao, out = rec["op"], rec["args_pre"], rec["args_post"], rec["out_post"]
    k = kind(op)
    if op.startswith("c10d."):
        if k == "all_reduce":
            return a[0][0], ao[0][0], {}
        if k == "all_to_all":
            return a[1], ao[0], {"out_split": list(a[3] or []), "in_split": list(a[4] or [])}
        return a[1], ao[0], {}                          # _allgather_base_ / _reduce_scatter_base_
    o = out[0] if isinstance(out, (list, tuple)) else out
    if k == "all_reduce":
        return a[0], o, {"reduce_op": a[1]}
    if k == "reduce_scatter":
        return a[0], o, {"reduce_op": a[1]}
    if k == "all_to_all":
        return a[0], o, {"out_split": list(a[1] or []), "in_split": list(a[2] or [])}
    return a[0], o, {}


_REDUCE = {"sum": lambda xs: torch.stack(xs).sum(0), "avg": lambda xs: torch.stack(xs).mean(0),
           "max": lambda xs: torch.stack(xs).amax(0), "min": lambda xs: torch.stack(xs).amin(0)}


def expected(recs, reduce_name):
    """[expected output per rank (fp64)] for the group, given a reduction name."""
    ins = [io(r)[0].double() for r in recs]
    k, n = kind(recs[0]["op"]), len(recs)
    if k == "all_reduce":
        e = _REDUCE[reduce_name](ins)
        return [e] * n
    if k == "all_gather":
        e = torch.cat(ins, 0)
        return [e] * n
    if k == "reduce_scatter":
        return list(_REDUCE[reduce_name](ins).chunk(n, 0))
    if k == "all_to_all":
        out = []
        for dst in range(n):
            parts = []
            for src, r in enumerate(recs):
                split = io(r)[2]["in_split"] or [ins[src].shape[0] // n] * n
                parts.append(ins[src].split(split, 0)[dst])
            out.append(torch.cat(parts, 0))
        return out
    raise SystemExit("unsupported collective %s" % recs[0]["op"])


def tol(dtype):
    return {torch.bfloat16: 2e-2, torch.float16: 5e-3}.get(dtype, 1e-4)


def agrees(got, exp):
    g = got.double()
    if g.shape != exp.shape:
        return False
    fin = torch.isfinite(exp)
    if not torch.equal(torch.isfinite(g), fin):
        return False
    scale = max(float(exp[fin].abs().max()) if fin.any() else 0.0, 1e-30)
    return float((g[fin] - exp[fin]).abs().max() if fin.any() else 0.0) <= tol(got.dtype) * scale


def reductions(recs):
    k = kind(recs[0]["op"])
    if k not in ("all_reduce", "reduce_scatter"):
        return [None]
    named = io(recs[0])[2].get("reduce_op")
    return [named.lower()] if isinstance(named, str) and named.lower() in _REDUCE else list(_REDUCE)


def check(d, group, seq):
    recs, ranks = load(d, group, seq)
    best = None
    for red in reductions(recs):
        exp = expected(recs, red)
        ok = [agrees(io(r)[1], e) for r, e in zip(recs, exp)]
        if best is None or sum(ok) > sum(best[1]):
            best = (red, ok, exp)
    red, ok, exp = best
    bad_in = [ranks[i] for i, r in enumerate(recs) if not torch.isfinite(io(r)[0].float()).all()]
    wrong = [ranks[i] for i, o in enumerate(ok) if not o]
    print("collective %s #%d of group %s (ranks %s)%s" % (recs[0]["op"], seq, group, ranks,
                                                          ", reduction %s" % red if red else ""))
    for i, r in enumerate(recs):
        x, y, _ = io(r)
        print("  rank %-3s input finite=%-5s output finite=%-5s %s"
              % (ranks[i], bool(torch.isfinite(x.float()).all()), bool(torch.isfinite(y.float()).all()),
                 "OK" if ok[i] else "DIFFERS from the reduction of the recorded inputs"))
    if bad_in:
        print("VERDICT INPUT_NONFINITE ranks %s -- the NaN came in on the inputs: look upstream there" % bad_in)
    elif wrong:
        print("VERDICT COLLECTIVE_FAULT ranks %s -- every input finite, output wrong: the collective "
              "(RCCL / transport) is at fault on those ranks" % wrong)
    else:
        print("VERDICT CORRECT -- outputs match the recorded inputs")
    return 1 if (wrong and not bad_in) else 0


def replay(d, group, seq, iters):
    import torch.distributed as dist
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    recs, ranks = load(d, group, seq)
    if world != len(recs):
        sys.exit("launch with --nproc_per_node=%d (the group size)" % len(recs))
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group("nccl" if cuda else "gloo")
    dev = torch.device("cuda") if cuda else torch.device("cpu")
    red = None
    for cand in reductions(recs):                       # the reduction the records agree with
        exp = expected(recs, cand)
        if all(agrees(io(r)[1], e) for r, e in zip(recs, exp)) or red is None:
            red = cand
    exp = expected(recs, red)[rank]
    x, y0, extra = io(recs[rank])
    k = kind(recs[0]["op"])
    ops = {"sum": dist.ReduceOp.SUM, "avg": dist.ReduceOp.AVG, "max": dist.ReduceOp.MAX,
           "min": dist.ReduceOp.MIN}
    bad = 0
    for _ in range(iters):
        xi = x.to(dev)
        if k == "all_reduce":
            out = xi.clone()
            if red == "avg" and not cuda:               # gloo has no AVG
                dist.all_reduce(out)
                out /= world
            else:
                dist.all_reduce(out, op=ops[red])
        elif k == "all_gather":
            out = torch.empty((xi.shape[0] * world,) + tuple(xi.shape[1:]), dtype=xi.dtype, device=dev)
            dist.all_gather_into_tensor(out, xi)
        elif k == "reduce_scatter":
            out = torch.empty((xi.shape[0] // world,) + tuple(xi.shape[1:]), dtype=xi.dtype, device=dev)
            dist.reduce_scatter_tensor(out, xi, op=ops[red])
        else:
            out = torch.empty_like(y0, device=dev)
            dist.all_to_all_single(out, xi, extra.get("out_split") or None, extra.get("in_split") or None)
        if not agrees(out.cpu(), exp):
            bad += 1
    tot = torch.tensor([bad], device=dev)
    dist.all_reduce(tot)
    if rank == 0:
        print("R3 replay %s #%d of group %s x%d: wrong outputs on %d rank-iterations (all ranks)%s"
              % (recs[0]["op"], seq, group, iters, int(tot), "  [REPRODUCED]" if int(tot) else ""))
    print("  rank %d: %d/%d wrong" % (rank, bad, iters), flush=True)
    dist.destroy_process_group()
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("check", "replay"))
    ap.add_argument("dir")
    ap.add_argument("--group", required=True)
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()
    rc = check(a.dir, a.group, a.seq) if a.mode == "check" else replay(a.dir, a.group, a.seq, a.iters)
    sys.stdout.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
