#!/usr/bin/env python3
"""CPU tests for tensor subclasses and uninitialized memory in op_nan_capture (needs
torch, no GPU; uses a 1-rank gloo group for DTensor).

    python3 tests/debug_nan/test_subclass_uninit.py

* DTensor: the capture defers wrapper subclasses, so it sees DTensor's local ops as
  plain tensors (a torchtitan capture once died on `.tolist()` of a DTensor),
  passes DTensor's fake-tensor sharding propagation through, still checks
  nn.Parameter ops itself, and captures a NaN written into a DTensor's local storage.
* Uninitialized memory (settle mode): a NaN that is only allocator garbage (FSDP2's
  partly written all-gather buffer) is MASKED as `uninitialized` and is not an escape
  when handed to a collective; a computed NaN in the same position still is.
"""
import glob
import os
import sys
import tempfile

import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

RT_PARENT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'src', 'madengine', 'scripts', 'common', 'tools', 'nan_doctor'))
sys.path.insert(0, RT_PARENT)
from nan_doctor_rt import op_nan_capture as C  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    print('%s  %s%s' % ('PASS' if cond else 'FAIL', name, ('  -- ' + str(detail)) if detail and not cond else ''))
    if not cond:
        FAILS.append(name)


os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29541")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
dist.init_process_group("gloo", rank=0, world_size=1)
from torch.distributed.device_mesh import init_device_mesh  # noqa: E402
from torch.distributed.tensor import Replicate, Shard, distribute_tensor  # noqa: E402

MESH = init_device_mesh("cpu", (1,))


class SpyCapture(C.OpNanCapture):
    """The capture, recording (op, argument type names) of every op it watched (its
    op counter moved); deferred and passed-through calls are not recorded."""
    seen = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        n0 = self.n
        r = super().__torch_dispatch__(func, types, args, kwargs)
        if self.n != n0:
            if self.seen is None:
                self.seen = []
            self.seen.append((str(func), frozenset(type(a).__name__ for a in C._tensors((args, kwargs)))))
        return r


def capsules(d):
    return sorted(glob.glob(os.path.join(d, "*.pt")))


def masked(d):
    return sorted(glob.glob(os.path.join(d, "*.masked-*")))


def new_cap(d, cls=C.OpNanCapture, **kw):
    kw.setdefault("settle", True)
    kw.setdefault("chain_depth", 2)
    kw.setdefault("exclude_ops", C.ALLOC_EXCLUDE_OPS)
    kw.setdefault("record_data", True)
    kw.setdefault("verbose", False)
    return cls(out=d, **kw)


# ---- DTensor: no crash, local ops seen, Parameter still checked ------------------
with tempfile.TemporaryDirectory() as d:
    cap = new_cap(d, cls=SpyCapture)
    w = torch.nn.Parameter(torch.randn(4, 4))
    err = None
    try:
        with cap:
            for _ in range(3):
                cap.settle_step()
                x = distribute_tensor(torch.randn(8, 4), MESH, [Shard(0)])
                y = (x @ x.t()).redistribute(MESH, [Replicate()]).sum()
                # an op DTensor shards by tracing its decomposition on meta tensors
                # (torchtitan's tokens_per_expert // 2 failed under the capture)
                _ = (x.to(torch.int64) // 2).sum()
                _ = (w * 2).sum().item()
                _ = y.item()
            cap.finish()
    except Exception as e:                              # noqa: BLE001
        err = e
    check("DTensor ops run under the capture without error", err is None, repr(err))
    check("no capsule from a clean DTensor run", not capsules(d), capsules(d))
    seen = cap.seen or []
    types_seen = set().union(*[t for _, t in seen]) if seen else set()
    check("the local mm reached the capture as a plain-tensor op",
          any(op == "aten.mm.default" and ts <= {"Tensor"} for op, ts in seen),
          sorted(set(op for op, _ in seen))[:15])
    check("the capture never handled a DTensor or fake tensor itself",
          not (types_seen & {"DTensor", "FakeTensor"}), types_seen)
    check("nn.Parameter ops are still handled (checked) by the capture",
          "Parameter" in types_seen, types_seen)
    check("subclass kind: DTensor deferred, Parameter checked, Tensor plain",
          C._subclass_kind((type(x),)) == "subclass"
          and C._subclass_kind((torch.nn.Parameter,)) is None
          and C._subclass_kind((torch.Tensor,)) is None)

# ---- DTensor: a NaN written into persistent local storage is captured ------------
with tempfile.TemporaryDirectory() as d:
    cap = new_cap(d)
    x = distribute_tensor(torch.ones(8, 4), MESH, [Shard(0)])     # outlives the steps
    raised = None
    try:
        with cap:
            cap.settle_step()
            x.mul_(float("nan"))
            cap.settle_step()
    except RuntimeError as e:
        raised = e
    caps = capsules(d)
    check("NaN written into a DTensor's storage is captured (world size 1 raises)",
          raised is not None and len(caps) == 1, (raised, caps))
    if caps:
        c = torch.load(caps[0], weights_only=False)
        check("the capsule's op is the local in-place op, not a DTensor-level one",
              c["op"].startswith("aten.mul_"), c["op"])

# ---- uninitialized memory is not an escape ---------------------------------------
def garbage_buffer(n):
    """torch.empty seen by the capture, then NaN bytes it does not see being written:
    what a reused allocator block can hold."""
    buf = torch.empty(n)
    with _disable_current_modes():
        buf.fill_(float("nan"))
    return buf


with tempfile.TemporaryDirectory() as d:
    cap = new_cap(d)
    raised = None
    try:
        with cap:
            for _ in range(3):
                cap.settle_step()
                buf = garbage_buffer(16)
                part = buf.clone()                 # reads garbage: dirty raw input
                part[:8].copy_(torch.ones(8))      # FSDP copy-in: a partial write
                dist.all_reduce(part)              # handed to a collective
            cap.settle_step()
            cap.finish()
    except RuntimeError as e:
        raised = e
    check("garbage handed to a collective is not captured", raised is None and not capsules(d),
          (raised, capsules(d)))
    ms = masked(d)
    check("garbage is saved once as a masked file with origin uninitialized",
          len(ms) == 1 and torch.load(ms[0], weights_only=False)["nan_origin"]["class"]
          == "uninitialized", ms)

# ---- garbage through a functional all-gather (FSDP2's path) -------------------------
import torch.distributed._functional_collectives as FC  # noqa: E402


def gather_and_read(fill):
    d = tempfile.mkdtemp()
    cap = new_cap(d)
    raised = None
    try:
        with cap:
            for _ in range(3):
                cap.settle_step()
                buf = garbage_buffer(16) if fill is None else torch.empty(16)
                if fill is not None:
                    buf.copy_(fill)                # a visible write of a real value
                part = buf.clone()
                g = FC.wait_tensor(FC.all_gather_tensor(part, 0, dist.group.WORLD))
                _ = g[:8].sum().item()             # reads the gathered buffer
                _ = g.sum().item()                 # ... including its garbage
            cap.settle_step()
            cap.finish()
    except RuntimeError as e:
        raised = e
    return raised, capsules(d)


raised, caps = gather_and_read(None)
check("garbage through all_gather + wait + reads is not captured",
      raised is None and not caps, (raised, caps))
raised, caps = gather_and_read(torch.full((16,), float("nan")))
check("a real NaN through the same path is still captured", raised is not None and len(caps) == 1,
      (raised, caps))

# ---- a NaN another rank sent: copied = not this rank's; computed with = captured ---
def received_then(op):
    d = tempfile.mkdtemp()
    cap = new_cap(d)
    keep = torch.ones(16)                          # outlives the steps
    raised = None
    try:
        with cap:
            cap.settle_step()
            buf = torch.ones(16)
            dist.all_reduce(buf)                   # what "another rank" sent ...
            with _disable_current_modes():
                buf[5] = float("nan")              # ... held a NaN (its garbage or bug)
            op(keep, buf)
            cap.settle_step()
            cap.finish()
    except RuntimeError as e:
        raised = e
    return raised, capsules(d), cap._first_arrival


raised, caps, fa = received_then(lambda keep, buf: keep.copy_(buf))
check("received NaN only COPIED into persistent storage: not captured, arrival recorded",
      raised is None and not caps and fa is not None and "allreduce" in fa["op"], (raised, caps, fa))
raised, caps, fa = received_then(lambda keep, buf: keep.mul_(buf))
c = torch.load(caps[0], weights_only=False) if caps else {}
check("received NaN entering arithmetic: captured, marked arrived_via the collective",
      raised is not None and len(caps) == 1
      and ((c.get("dist_causality") or {}).get("arrived_via") or {}).get("op", "").startswith("c10d.allreduce"),
      (raised, caps, (c.get("dist_causality") or {}).get("arrived_via")))

# ---- controls: a computed NaN in the same place still escapes ---------------------
with tempfile.TemporaryDirectory() as d:
    cap = new_cap(d)
    raised = None
    try:
        with cap:
            cap.settle_step()
            buf = torch.empty(16)                  # raw
            src = torch.zeros(8) / torch.zeros(8)  # a computed NaN (0/0)
            buf[:8].copy_(src)                     # written into the raw buffer
            buf[8:].copy_(torch.ones(8))
            dist.all_reduce(buf)
            cap.settle_step()
    except RuntimeError as e:
        raised = e
    check("a computed NaN copied into a raw buffer still escapes via the collective",
          raised is not None and len(capsules(d)) == 1, (raised, capsules(d)))

with tempfile.TemporaryDirectory() as d:
    cap = new_cap(d)
    p = torch.ones(16)                             # persistent
    raised = None
    try:
        with cap:
            cap.settle_step()
            buf = garbage_buffer(16)
            p.copy_(buf)                           # garbage reaching persistent state
            cap.settle_step()
    except RuntimeError as e:
        raised = e
    check("garbage written into a tensor that outlives the step is captured",
          raised is not None and len(capsules(d)) == 1, (raised, capsules(d)))

with tempfile.TemporaryDirectory() as d:
    cap = new_cap(d)
    raised = None
    try:
        with cap:
            cap.settle_step()
            buf = torch.empty(16)
            buf.fill_(1.0)                         # a full write: initialized now
            with _disable_current_modes():
                buf[3] = float("nan")              # an unseen kernel writes a NaN later
            out = buf.clone()
            dist.all_reduce(out)
            cap.settle_step()
    except RuntimeError as e:
        raised = e
    check("after a full write the buffer is no longer raw: its NaN escapes",
          raised is not None and len(capsules(d)) == 1, (raised, capsules(d)))

# ---- higher-order operators run (unchecked) instead of failing ----------------------
with tempfile.TemporaryDirectory() as d:
    cap = new_cap(d)
    err = None
    try:
        from torch._higher_order_ops.cond import cond
        with cap:
            cap.settle_step()
            r = cond(torch.tensor(True), lambda x: x + 1, lambda x: x - 1, (torch.ones(3),))
            cap.settle_step()
        ok = torch.equal(r, torch.full((3,), 2.0))
    except ImportError:
        ok, err = True, "no cond HOP in this build"
    except Exception as e:                              # noqa: BLE001
        ok, err = False, e
    check("a higher-order operator (cond) runs under the capture", ok, repr(err))

# ---- a compiled entry point: fails under the mode, runs once suspended -------------
class Block(object):
    _compiled = staticmethod(torch.compile(lambda x: torch.sin(x) * 2, backend="eager",
                                           fullgraph=True))

    @staticmethod
    def run(x):
        return Block._compiled(x)


sys.modules[__name__].Block = Block
with tempfile.TemporaryDirectory() as d:
    def attempt():
        torch._dynamo.reset()
        cap = new_cap(d)
        try:
            with cap:
                cap.settle_step()
                return Block.run(torch.ones(4)), None
        except Exception as e:                          # noqa: BLE001
            return None, e
    r0, e0 = attempt()
    check("control: a fullgraph compiled call fails under the capture",
          e0 is not None and "dispatch mode" in str(e0), repr(e0)[:200])
    done = C.suspend_capture_in("%s:Block.run" % __name__)
    r1, e1 = attempt()
    check("suspended: the same call runs and is correct",
          e1 is None and r1 is not None and torch.allclose(r1, torch.sin(torch.ones(4)) * 2),
          repr(e1)[:200])
    check("suspend_capture_in reports the path and keeps it a staticmethod",
          done == ["%s:Block.run" % __name__]
          and isinstance(vars(Block)["run"], staticmethod))

dist.destroy_process_group()
print("\n%d failure(s)" % len(FAILS))
sys.exit(1 if FAILS else 0)
