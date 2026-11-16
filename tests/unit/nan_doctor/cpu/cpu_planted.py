#!/usr/bin/env python3
"""CPU tests: planted failure classes from REVISED.md's test matrix (cases 5-8), checked
against the evidence the capture writes (needs torch, no GPU; spawns 4 gloo processes).

    python3 tests/debug_nan/test_planted.py

5 collective   one rank's all_reduce output is corrupted after the collective (a
               transport fault): no rank MAKES a NaN; the only arrival is that rank's.
6 parallelism  one rank skips its gradient all_reduce (a dropped reduction): no NaN at
               all; watch_replicas() stops the job, manifest verdict PARALLELISM.
7 dynamics     lr far too large: values overflow; the capsule's origin is `overflow`
               (fp64 re-run finite but out of range): a correct op, no faulty kernel.
8 data         a NaN batch from outside the dispatcher (a loader thread): the capsule's
               dirty operand is a chain root no op produced -> DATA.
"""
import glob
import json
import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils._python_dispatch import _disable_current_modes

RT_PARENT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'src', 'madengine', 'scripts', 'common', 'tools', 'nan_doctor'))
sys.path.insert(0, RT_PARENT)

FAILS = []


def check(name, cond, detail=''):
    print('%s  %s%s' % ('PASS' if cond else 'FAIL', name, ('  -- ' + str(detail)) if detail and not cond else ''))
    if not cond:
        FAILS.append(name)


def new_cap(C, d, **kw):
    return C.OpNanCapture(out=d, settle=True, chain_depth=2, exclude_ops=C.ALLOC_EXCLUDE_OPS,
                          record_data=True, verbose=False, **kw)


def worker(rank, world, d, port, case):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from nan_doctor_rt import op_nan_capture as C
    cap = new_cap(C, d)
    torch.manual_seed(0)
    w = torch.ones(8)                                   # replicated parameter
    if case == "parallelism":
        cap.watch_replicas({"w": w})
    try:
        with cap:
            for step in range(6):
                cap.settle_step()
                cap.sync_stop()
                g = torch.randn(8, generator=torch.Generator().manual_seed(step * 10 + rank))
                red = g.clone()
                dist.all_reduce(red)                    # every rank takes part ...
                red.div_(world)
                if not (case == "parallelism" and rank == 1 and step == 2):
                    g = red                             # ... rank 1 drops the result once
                if case == "collective" and rank == 2 and step == 2:
                    with _disable_current_modes():
                        g[3] = float("nan")            # the transport wrote garbage
                w.sub_(0.01 * g)
    except RuntimeError as e:
        if "[op-nan]" not in str(e):
            raise
    dist.destroy_process_group()


def manifest(d):
    m = glob.glob(os.path.join(d, "manifest*.json"))
    return json.load(open(m[0])) if m else None


def single(case):
    from nan_doctor_rt import op_nan_capture as C
    d = tempfile.mkdtemp()
    cap = new_cap(C, d)
    # w = 1 is off the solution w = 0.5 (y = x / 2): the residual is non-zero
    w, x, y = torch.full((64,), 1.0), torch.linspace(-2, 2, 64), torch.linspace(-1, 1, 64)
    lr = 1e3 if case == "dynamics" else 1e-3
    try:
        with cap:
            for step in range(40):
                cap.settle_step()
                if case == "data" and step == 3:
                    with _disable_current_modes():       # a loader thread, unseen
                        x = torch.linspace(-2, 2, 64)
                        x[7] = float("nan")
                r = w * x - y
                grad = 2 * r * x
                w = w - lr * grad
                _ = float(w.abs().max())
    except RuntimeError as e:
        if "[op-nan]" not in str(e):
            raise
    caps = glob.glob(os.path.join(d, "*.pt"))
    return torch.load(caps[0], weights_only=False) if caps else None


def main():
    for case, port in (("collective", 29571), ("parallelism", 29581)):
        d = tempfile.mkdtemp()
        os.chmod(d, 0o777)
        mp.spawn(worker, args=(4, d, port, case), nprocs=4, join=True)
        doc = manifest(d)
        v = (doc or {}).get("verdict") or {}
        if case == "collective":
            check("5 collective: no rank made the NaN", doc is not None and v.get("origin_ranks") == [],
                  v)
            check("5 collective: the only arrival is rank 2's (the corrupted output)",
                  [a["rank"] for a in v.get("arrivals", [])] == [2], v.get("arrivals"))
        else:
            check("6 parallelism: verdict PARALLELISM, no NaN anywhere",
                  doc is not None and "PARALLELISM" in v.get("note", "")
                  and not glob.glob(os.path.join(d, "*.pt")), v.get("note"))
            div = v.get("parallelism") or {}
            check("6 parallelism: names the tensor and the step after the dropped reduction",
                  div.get("tensor") == "w" and div.get("step") == 4, div)
    c = single("dynamics")
    o = (c or {}).get("nan_origin") or {}
    check("7 dynamics: capsule origin is overflow (a correct op), no wrong-value hit",
          c is not None and o.get("class") == "overflow" and not c.get("verify"),
          (c or {}).get("op"), )
    c = single("data")
    check("8 data: inputs_finite=False and the dirty operand is a chain root no op produced",
          c is not None and c["inputs_finite"] is False
          and "outside_window" in set((c.get("truncation") or {}).values()),
          (c or {}).get("op"))
    print("\n%d failure(s)" % len(FAILS))
    sys.stdout.flush()
    os._exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
