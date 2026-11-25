#!/usr/bin/env python3
"""CPU test of R3 (needs torch, no GPU; spawns 4 gloo processes).

    python3 tests/debug_nan/test_collective_r3.py

Each step every rank all_reduces a gradient.  At step 2 a dispatch mode BELOW the capture
corrupts rank 2's all_reduce output inside the collective call (a transport fault).  The
capture records that collective (NAN_CAPTURE_RECORD_COLLECTIVE semantics) on every rank;
collective_r3.py check must say COLLECTIVE_FAULT ranks [2]; replay of the recorded inputs
(no corruption now) must find every output correct.
"""
import os
import subprocess
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils._python_dispatch import TorchDispatchMode

RT_PARENT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'src', 'madengine', 'scripts', 'common', 'tools', 'nan_doctor'))
sys.path.insert(0, RT_PARENT)
R3 = os.path.join(RT_PARENT, 'nan_doctor_rt', 'collective_r3.py')
FAILS = []


def check(name, cond, detail=''):
    print('%s  %s%s' % ('PASS' if cond else 'FAIL', name, ('  -- ' + str(detail)) if detail and not cond else ''))
    if not cond:
        FAILS.append(name)


class CorruptAllReduce(TorchDispatchMode):
    def __init__(self, rank):
        super().__init__()
        self.rank, self.n = rank, 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if str(func) == "c10d.allreduce_.default":
            self.n += 1
            if self.rank == 2 and self.n == 6:           # the test's all_reduce at step 2
                # gloo returns before the reduction is done: corrupt only the FINISHED output, or the
                # NaN can still enter the sum and reach every rank (flaky: ranks [0, 1, 2, 3])
                work = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
                if hasattr(work, "wait"):
                    work.wait()
                args[0][0][3] = float("nan")
        return out


def worker(rank, world, d, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from nan_doctor_rt import op_nan_capture as C
    # per step: sync_stop's flag all_reduce (odd seq), then the gradient's (even): step 2 -> seq 6
    cap = C.OpNanCapture(out=d, settle=True, chain_depth=2, exclude_ops=C.ALLOC_EXCLUDE_OPS,
                         record_data=True, verbose=False, record_collectives="0:6")
    w = torch.ones(8)
    try:
        with CorruptAllReduce(rank), cap:
            for step in range(5):
                cap.settle_step()
                cap.sync_stop()
                g = torch.randn(8, generator=torch.Generator().manual_seed(step * 10 + rank))
                dist.all_reduce(g)
                w.sub_(0.01 * g)
    except RuntimeError as e:
        if "[op-nan]" not in str(e):
            raise
    dist.destroy_process_group()


def main():
    d = tempfile.mkdtemp()
    os.chmod(d, 0o777)
    mp.spawn(worker, args=(4, d, 29591), nprocs=4, join=True)
    files = sorted(f for f in os.listdir(d) if f.startswith("collective_"))
    check("every rank recorded collective #6 of group 0", len(files) == 4, files)
    out = subprocess.run([sys.executable, R3, "check", d, "--group", "0", "--seq", "6"],
                         capture_output=True, text=True).stdout
    check("check: COLLECTIVE_FAULT on rank 2 only", "VERDICT COLLECTIVE_FAULT ranks [2]" in out, out)
    out = subprocess.run(["torchrun", "--nproc_per_node=4", "--master_port=29601", R3, "replay", d,
                          "--group", "0", "--seq", "6", "--iters", "20"],
                         capture_output=True, text=True)
    check("replay of the recorded inputs (no fault now): every output correct",
          "wrong outputs on 0 rank-iterations" in out.stdout, (out.stdout + out.stderr)[-500:])
    print("\n%d failure(s)" % len(FAILS))
    sys.stdout.flush()
    os._exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
