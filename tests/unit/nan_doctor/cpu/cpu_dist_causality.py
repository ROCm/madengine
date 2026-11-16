#!/usr/bin/env python3
"""CPU test of cross-rank causality in op_nan_capture (needs torch, no GPU; spawns 4
gloo processes).

    python3 tests/debug_nan/test_dist_causality.py

Rank 1 makes a NaN locally at step 2 (full_like: no math origin), all_reduce carries
it to every rank, and each rank writes it into a persistent tensor.  Expected:

* rank 1's capsule: made locally (inputs finite), not arrived;
* ranks 0, 2, 3: capsule inputs_finite=False, arrived_via the all_reduce, and the SAME
  (group, seq) on every rank -- the n-th collective of a group names one event;
* manifest.json (rank 0, at sync_stop): origin_ranks == [1], three arrivals;
* compare_nan_capsules.py --job prints the verdict and the arrivals.
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

RT_PARENT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'src', 'madengine', 'scripts', 'common', 'tools', 'nan_doctor'))
sys.path.insert(0, RT_PARENT)

FAILS = []


def check(name, cond, detail=''):
    print('%s  %s%s' % ('PASS' if cond else 'FAIL', name, ('  -- ' + str(detail)) if detail and not cond else ''))
    if not cond:
        FAILS.append(name)


def worker(rank, world, d, port, how="full_like"):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from nan_doctor_rt import op_nan_capture as C
    cap = C.OpNanCapture(out=d, settle=True, chain_depth=2, exclude_ops=C.ALLOC_EXCLUDE_OPS,
                         record_data=True, verbose=False)
    w = torch.ones(8)                                   # persistent, per rank
    try:
        with cap:
            for step in range(6):
                cap.settle_step()
                cap.sync_stop()
                x = torch.randn(8)
                if rank == 1 and step == 2:
                    if how == "math":
                        x = torch.sqrt(-torch.ones_like(x))  # made here, origin math
                    else:
                        x = torch.full_like(x, float("nan"))   # made here
                g = x * 2
                dist.all_reduce(g)
                w.add_(g)
    except RuntimeError as e:
        if "[op-nan]" not in str(e):
            raise
    dist.destroy_process_group()


def main():
    d = tempfile.mkdtemp()
    os.chmod(d, 0o777)
    mp.spawn(worker, args=(4, d, 29561), nprocs=4, join=True)
    from nan_doctor_rt import op_nan_capture as C                          # noqa: F401  (upgrade path)
    caps = {}
    for f in glob.glob(os.path.join(d, "*.pt")):
        c = torch.load(f, weights_only=False)
        caps[c["dist"]["rank"]] = c
    check("every rank wrote a capsule", sorted(caps) == [0, 1, 2, 3], sorted(caps))
    c1 = caps.get(1, {})
    dc1 = c1.get("dist_causality") or {}
    check("rank 1: made locally (inputs finite, not arrived)",
          c1.get("inputs_finite") is True and dc1.get("arrived_via") is None,
          (c1.get("op"), c1.get("inputs_finite"), dc1.get("arrived_via")))
    others = [caps[r] for r in (0, 2, 3) if r in caps]
    arr = [(c.get("dist_causality") or {}).get("arrived_via") for c in others]
    check("ranks 0, 2, 3: inputs_finite=False and arrived via the all_reduce",
          len(others) == 3 and all(c["inputs_finite"] is False for c in others)
          and all(a and "allreduce" in a["op"] for a in arr), arr)
    check("the arrival names the same (group, seq) on every receiving rank",
          len(set((a["group"], a["seq"]) for a in arr if a)) == 1, arr)
    mans = glob.glob(os.path.join(d, "manifest*.json"))
    check("one manifest written", len(mans) == 1, mans)
    if mans:
        doc = json.load(open(mans[0]))
        v = doc["verdict"]
        check("manifest: origin_ranks == [1]", v["origin_ranks"] == [1], v)
        check("manifest: three arrivals", len(v["arrivals"]) == 3, v["arrivals"])
    out = subprocess.run([sys.executable, os.path.join(RT_PARENT, 'nan_doctor_rt', 'compare_nan_capsules.py'),
                          d, "--job"], capture_output=True, text=True).stdout
    check("compare --job prints the verdict and the arrivals",
          "origin: rank(s) [1]" in out and "arrived through a collective" in out, out[-600:])
    # A math-origin NaN (sqrt(-x)) escaping through the collective is still this rank's:
    # written, whatever the rule for allocator-garbage candidates (seen in a GPU DDP injection run).
    d2 = tempfile.mkdtemp()
    os.chmod(d2, 0o777)
    mp.spawn(worker, args=(4, d2, 29566, "math"), nprocs=4, join=True)
    c1 = [torch.load(f, weights_only=False) for f in glob.glob(os.path.join(d2, "rank1_*.pt"))]
    check("math-origin NaN escaping a collective: rank 1 capsule written, origin math",
          len(c1) == 1 and (c1[0].get("nan_origin") or {}).get("class") == "math",
          [(c.get("op"), (c.get("nan_origin") or {}).get("class")) for c in c1])
    print("\n%d failure(s)" % len(FAILS))
    sys.stdout.flush()
    os._exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
