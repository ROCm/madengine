#!/usr/bin/env python3
"""CPU tests for the wrong-finite-value checks of op_nan_capture (needs torch, no GPU).

    python3 tests/debug_nan/test_wrong_value.py

* Clean GEMMs (mm, addmm, bmm, baddbmm in fp32 and bf16, many shapes) never fail the
  Cauchy-Schwarz bound or the reference check: no capsule (false-positive control).
* A GEMM whose output is corrupted to a wrong FINITE value is captured with origin
  `wrong_value`: by the bound when the value is impossible, by the reference when it
  is merely wrong.  The corruption comes from a dispatch mode below the capture that
  touches only the dtype under test, so the higher-precision reference stays clean.
* _grouped_mm_ref matches a per-group loop (skipped where the build has no CPU
  _grouped_mm).
"""
import glob
import os
import sys
import tempfile

import torch
from torch.utils._python_dispatch import TorchDispatchMode

RT_PARENT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'src', 'madengine', 'scripts', 'common', 'tools', 'nan_doctor'))
sys.path.insert(0, RT_PARENT)
from nan_doctor_rt import op_nan_capture as C  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    print('%s  %s%s' % ('PASS' if cond else 'FAIL', name, ('  -- ' + str(detail)) if detail and not cond else ''))
    if not cond:
        FAILS.append(name)


class Corrupt(TorchDispatchMode):
    """Below the capture: add `delta` to element [0, 0] of every `dtype` mm output from
    call `start` on (a wrong value the kernel 'computed')."""
    def __init__(self, dtype, delta, start=0):
        super().__init__()
        self.dtype, self.delta, self.start, self.calls = dtype, delta, start, 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if str(func) == "aten.mm.default" and out.dtype == self.dtype:
            self.calls += 1
            if self.calls > self.start:
                out = out.clone()
                out.view(-1)[0] += self.delta
        return out


def run(dtype, n, corrupt=None, **kw):
    """n mm calls under the capture; returns (capsules, exception)."""
    d = tempfile.mkdtemp()
    kw.setdefault("verbose", False)
    cap = C.OpNanCapture(out=d, include_only_op_classes="gemm", **kw)
    g = torch.Generator().manual_seed(0)
    err = None
    try:
        if corrupt is not None:
            with corrupt, cap:
                for i in range(n):
                    a = torch.randn(16 + i % 5, 32, generator=g).to(dtype)
                    b = torch.randn(32, 24, generator=g).to(dtype)
                    a @ b
        else:
            with cap:
                for i in range(n):
                    a = torch.randn(16 + i % 5, 32, generator=g).to(dtype)
                    b = torch.randn(32, 24, generator=g).to(dtype)
                    a @ b
    except RuntimeError as e:
        err = e
    return sorted(glob.glob(os.path.join(d, "*.pt"))), err


# ---- clean controls: bounds and reference never fire --------------------------------
g = torch.Generator().manual_seed(1)
worst = 0.0
for dt in (torch.float32, torch.bfloat16):
    for (m, k, n) in ((7, 3, 5), (64, 512, 33), (1, 4096, 1), (128, 64, 128)):
        a = (torch.randn(m, k, generator=g) * 10).to(dt)
        b = (torch.randn(k, n, generator=g) * 0.1).to(dt)
        bias = torch.randn(n, generator=g).to(dt)
        for func, args in ((torch.ops.aten.mm.default, (a, b)),
                           (torch.ops.aten.addmm.default, (bias, a, b))):
            c = func(*args)
            worst = max(worst, C._gemm_bound_ratio(func, args, {}, c))
        x, y = a.unsqueeze(0).repeat(3, 1, 1), b.unsqueeze(0).repeat(3, 1, 1)
        c = torch.ops.aten.bmm.default(x, y)
        worst = max(worst, C._gemm_bound_ratio(torch.ops.aten.bmm.default, (x, y), {}, c))
check("bound ratio <= 1 on clean mm/addmm/bmm (fp32, bf16)", worst <= 1.0 + 0.02, worst)

for dt in (torch.float32, torch.bfloat16):
    caps, err = run(dt, 300, verify_every=1, bounds=True)
    check("no capsule on 300 clean %s mm calls with bounds + reference on every call"
          % str(dt).replace("torch.", ""), not caps and err is None, (caps, err))

# ---- an impossible value: the bound fires --------------------------------------------
caps, err = run(torch.float32, 50, corrupt=Corrupt(torch.float32, 1e6, start=10), bounds=True)
c = torch.load(caps[0], weights_only=False) if caps else {}
check("impossible fp32 value captured by the bound", len(caps) == 1 and err is not None
      and c.get("verify", {}).get("kind") == "bound"
      and c.get("nan_origin", {}).get("class") == "wrong_value", (caps, err, c.get("verify")))

# ---- a wrong but possible value: only the reference sees it -----------------------
# Element [0,0] of a 32-term dot product of N(0,1) entries: |C| ~ 6, bound ~ 32; +3 is
# within the bound but ~50% relative error at that element (~15% of max|C|).
caps, err = run(torch.bfloat16, 60, corrupt=Corrupt(torch.bfloat16, 3.0, start=40),
                verify_every=1, bounds=True)
c = torch.load(caps[0], weights_only=False) if caps else {}
v = c.get("verify", {})
check("wrong-but-bounded bf16 value captured by the reference check",
      len(caps) == 1 and v.get("kind") == "reference", (caps, err, v))
check("its site had clean calls before it (learned tolerance in use)",
      v.get("site_calls", 0) > 40 and v.get("site_checked", 0) > 16, v)

caps, err = run(torch.bfloat16, 60, corrupt=Corrupt(torch.bfloat16, 3.0, start=40),
                verify_every=0, bounds=True)
check("the same wrong value passes with the reference off (bounds alone miss it)",
      not caps, caps)

# ---- sampling: verify_every=8 checks calls 1, 9, 17, ... of each site ------------------
caps, err = run(torch.bfloat16, 60, corrupt=Corrupt(torch.bfloat16, 3.0, start=41),
                verify_every=8)
c = torch.load(caps[0], weights_only=False) if caps else {}
check("sampled reference check catches a persistent fault at its next sample",
      len(caps) == 1 and c["verify"]["site_calls"] in (49,), c.get("verify"))

# ---- the fp32 floor follows the GEMM's compute precision (TF32 on: 1e-2) -------------
# A ~0.3% error at element [0,0] (2e-3 of max|C|): wrong for true fp32, normal for TF32.
caps_hi, _ = run(torch.float32, 30, corrupt=Corrupt(torch.float32, 0.02, start=0), verify_every=1)
torch.backends.cuda.matmul.allow_tf32 = True
caps_tf, _ = run(torch.float32, 30, corrupt=Corrupt(torch.float32, 0.02, start=0), verify_every=1)
torch.backends.cuda.matmul.allow_tf32 = False
check("fp32 floor: a 2e-3 error is flagged at full fp32, not with TF32 allowed",
      len(caps_hi) == 1 and not caps_tf, (caps_hi, caps_tf))

# ---- _grouped_mm reference -----------------------------------------------------------
try:
    a = torch.randn(10, 16, dtype=torch.bfloat16)        # strides: multiples of 16 B
    b = torch.randn(3, 16, 8, dtype=torch.bfloat16)
    offs = torch.tensor([4, 4, 9], dtype=torch.int32)
    out = torch._grouped_mm(a, b, offs=offs)
    have = True
except Exception as e:                                  # noqa: BLE001
    have, why = False, e
if have:
    ref, region = C._grouped_mm_ref((a, b, offs), {}, torch.float32)
    loop = torch.zeros(10, 8)
    loop[0:4] = a[0:4].float() @ b[0].float()
    loop[4:9] = a[4:9].float() @ b[2].float()
    check("_grouped_mm_ref (rows grouped) matches the loop on the valid rows",
          torch.allclose(ref[region], loop[:9]) and region[0] == slice(0, 9))
    err = C._ref_rel_err(torch.ops.aten._grouped_mm.default, (a, b, offs), {}, out)
    check("clean _grouped_mm has a small reference error", err is not None and err < 3e-2, err)
else:
    print("SKIP  _grouped_mm not available on CPU here (%s)" % str(why)[:80])

print("\n%d failure(s)" % len(FAILS))
sys.exit(1 if FAILS else 0)
