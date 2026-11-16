#!/usr/bin/env python3
"""CPU tests for the capsule save path and mutation selection (needs torch, no GPU).

    python3 tests/debug_nan/test_capture_save.py

* op_nan_capture._emit: a capsule holding an unpicklable leaf (a module, as an op's
  chain once did -- 5 of 6 capsules of a real run were lost to
  `TypeError: cannot pickle 'module' object`) is still written, with the leaf
  replaced by a marker and named in capsule["unpicklable"]; a clean capsule is
  written unchanged.
* kernel_mutations.kernel_mutations(only=...): mutator names and single mutation
  names (`lgkmcnt0_each@0x...`) both select.
* contention: torch.compile's inductor compile workers hold /dev/kfd but launch nothing;
  they are not counted in gpu_procs, and replay discounts them in older capsules that
  counted them (gpu_procs=34 beside one trainer).
"""
import os
import sys
import tempfile
import types

import torch

RT_PARENT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'src', 'madengine', 'scripts', 'common', 'tools', 'nan_doctor'))
sys.path.insert(0, RT_PARENT)
from nan_doctor_rt import op_nan_capture as C  # noqa: E402
from nan_doctor_rt import kernel_mutations as KM  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    print('%s  %s%s' % ('PASS' if cond else 'FAIL', name, ('  -- ' + detail) if detail and not cond else ''))
    if not cond:
        FAILS.append(name)


BASE = {"format": 2, "op": "aten.mm.default", "op_call_index": 1, "nan_found_in": "return",
        "mutation": None, "inputs_finite": True, "geometry": "g"}

# ---- _emit: unpicklable leaves --------------------------------------------------
with tempfile.TemporaryDirectory() as d:
    out = os.path.join(d, "c.pt")
    stub = types.SimpleNamespace(rank=0, out=out, verbose=False)
    cap = dict(BASE, t=torch.ones(2),
               kwargs={"dtype": torch.float32, "mod": torch, "nested": [1, (os, 2.0)]})
    C.OpNanCapture._emit(stub, cap)
    check('capsule with a module leaf is written', os.path.exists(out))
    c = torch.load(out, weights_only=False)
    check('unpicklable leaves are named', len(c.get("unpicklable", [])) == 2, repr(c.get("unpicklable")))
    check('leaf replaced by a marker', c["kwargs"]["mod"].get("unpicklable") == "module")
    check('nested leaf replaced', c["kwargs"]["nested"][1][0].get("unpicklable") == "module")
    check('picklable values kept', c["kwargs"]["dtype"] is torch.float32 and c["kwargs"]["nested"][1][1] == 2.0)
    check('tensors kept', torch.equal(c["t"], torch.ones(2)))
    check('no tmp file left', not os.path.exists(out + ".tmp"))

    C.OpNanCapture._emit(stub, dict(BASE))
    check('clean capsule has no unpicklable key', "unpicklable" not in torch.load(out, weights_only=False))

# ---- kernel_mutations(only=...) ------------------------------------------------
saved = dict(KM.MUTATORS)
try:
    KM.MUTATORS.clear()
    M = KM.Mutation

    def each(spec):
        return [M("each@0x%X" % a, "site %X" % a, "addrs", "lgkmcnt0", {"addrs": [a]}) for a in (1, 2, 3)]

    def whole(spec):
        return [M("whole", "all", "lgkm_wait", "lgkmcnt0", {})]
    KM.MUTATORS["each"], KM.MUTATORS["whole"] = each, whole
    KM._validate = lambda m, name: None                     # these stubs are not real payloads
    names = lambda only: [m.name for m in KM.kernel_mutations(None, only=only)]  # noqa: E731
    check('mutator name selects all its mutations', names(["each"]) == ["each@0x1", "each@0x2", "each@0x3"])
    check('mutation name selects one', names(["each@0x2"]) == ["each@0x2"])
    check('names mix', names(["each@0x3", "whole", "each@0x1"]) == ["each@0x1", "each@0x3", "whole"])
    check('mutator name wins over its sites', names(["each@0x2", "each"]) == ["each@0x1", "each@0x2", "each@0x3"])
    for bad in (["nope"], ["each@0x9"]):
        try:
            names(bad)
            check('rejects %s' % bad, False)
        except ValueError:
            check('rejects %s' % bad, True)
finally:
    KM.MUTATORS.clear()
    KM.MUTATORS.update(saved)

# ---- contention: torch.compile helpers are not co-tenants ------------------------
from nan_doctor_rt import replay_nan_op as RP  # noqa: E402

W = ('/opt/venv/bin/python3 /opt/venv/lib/python3.12/site-packages/torch/_inductor/'
     'compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool')
check('compile worker is a helper', C._is_compile_helper(W))
check('trainer is not a helper', not C._is_compile_helper('python3 /mnt/x/train.py --steps 50'))
saved_scan, saved_anc = C._gpu_proc_scan, C._launcher_ancestors
try:
    C._gpu_proc_scan = lambda: [(10, True, 'python3 train.py')] + [(20 + i, True, W) for i in range(33)]
    C._launcher_ancestors = lambda scan=None: []
    check('gpu_procs skips compile workers', C._gpu_procs() == 1, str(C._gpu_procs()))
finally:
    C._gpu_proc_scan, C._launcher_ancestors = saved_scan, saved_anc
old = {"contention": {"gpu_procs": 34, "containers_per_gpu": 1,
                      "gpu_proc_list": [(10, True, 'python3 train.py')] + [(20 + i, True, W) for i in range(33)]}}
check('replay discounts helpers in old capsules', RP.capture_gpu_procs(old)[0] == 1, str(RP.capture_gpu_procs(old)))
new = {"contention": {"gpu_procs": 2, "containers_per_gpu": 1, "gpu_proc_helpers_excluded": True,
                      "gpu_proc_list": [(10, True, 'python3 train.py'), (11, True, 'python3 train.py')]
                                       + [(20, True, W)]}}
check('replay leaves new capsules alone', RP.capture_gpu_procs(new)[0] == 2, str(RP.capture_gpu_procs(new)))

print('\n%d failure(s)' % len(FAILS) + (': ' + ', '.join(FAILS) if FAILS else ''))
sys.exit(1 if FAILS else 0)
