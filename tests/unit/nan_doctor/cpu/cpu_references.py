#!/usr/bin/env python3
"""CPU tests for replay_nan_op.py's fp64 references (needs torch, no GPU).

    python3 tests/debug_nan/test_references.py

The attention-backward math references against autograd through an fp64 softmax
attention (causal both alignments, GQA, sq != sk), and the chain reference's
placement-kwarg rewrite that used to send the "CPU fp64" reference to the GPU.
"""
import os
import sys

import torch

RT_PARENT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'src', 'madengine', 'scripts', 'common', 'tools', 'nan_doctor'))
sys.path.insert(0, RT_PARENT)
from nan_doctor_rt import replay_nan_op as R  # noqa: E402  (importable: parses no argv when imported)

FAILS = []


def check(name, cond, detail=''):
    print('%s  %s%s' % ('PASS' if cond else 'FAIL', name, ('  -- ' + detail) if detail and not cond else ''))
    if not cond:
        FAILS.append(name)


def autograd_attn(q, k, v, do, scale, causal, offset):
    """(b, h, s, d) fp64 attention via autograd -> (out, lse, dq, dk, dv)."""
    q, k, v = (t.clone().requires_grad_() for t in (q, k, v))
    g = q.shape[1] // k.shape[1]
    kq, vq = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    s = scale * q @ kq.transpose(-1, -2)
    if causal:
        sq, sk = q.shape[2], k.shape[2]
        keep = torch.arange(sk)[None, :] <= torch.arange(sq)[:, None] + offset
        s = s.masked_fill(~keep, float('-inf'))
    lse = torch.logsumexp(s, -1)
    o = torch.softmax(s, -1) @ vq
    o.backward(do)
    return o.detach(), lse.detach(), q.grad, k.grad, v.grad


def close(a, b):
    return torch.allclose(a, b, rtol=1e-9, atol=1e-9)


torch.manual_seed(0)
for causal in (False, True):
    for hq, hk in ((4, 4), (4, 2)):
        b, sq, sk, d = 2, 5, 9, 8
        q, do = torch.randn(b, hq, sq, d, dtype=torch.float64), torch.randn(b, hq, sq, d, dtype=torch.float64)
        k, v = torch.randn(b, hk, sk, d, dtype=torch.float64), torch.randn(b, hk, sk, d, dtype=torch.float64)
        scale = d ** -0.5
        # flash-attn: (b, s, h, d), causal aligned bottom-right
        o, lse, dq, dk, dv = autograd_attn(q, k, v, do, scale, causal, sk - sq)
        n = {'dout': do.transpose(1, 2), 'q': q.transpose(1, 2), 'k': k.transpose(1, 2),
             'v': v.transpose(1, 2), 'out': o.transpose(1, 2), 'softmax_lse': lse,
             'dq': None, 'dk': None, 'dv': None, 'dropout_p': 0.0, 'softmax_scale': scale,
             'causal': causal, 'window_size_left': -1, 'window_size_right': 0 if causal else -1,
             'softcap': 0.0, 'alibi_slopes': None, 'deterministic': False, 'rng_state': None}
        R._schema_args = lambda func, a, kw: kw            # bypass the schema: no flash_attn here
        got = R._ref_flash_attn_backward(None, (), n)
        want = (dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2))
        check('flash_attn bwd ref causal=%s heads=%d/%d' % (causal, hq, hk),
              all(close(x, y) for x, y in zip(got, want)))
        # SDPA flash: (b, h, s, d), causal aligned top-left
        o, lse, dq, dk, dv = autograd_attn(q, k, v, do, scale, causal, 0)
        n = {'grad_out': do, 'query': q, 'key': k, 'value': v, 'out': o, 'logsumexp': lse,
             'cum_seq_q': None, 'cum_seq_k': None, 'dropout_p': 0.0, 'is_causal': causal, 'scale': scale}
        got = R._ref_sdpa_flash_backward(None, (), n)
        check('sdpa flash bwd ref causal=%s heads=%d/%d' % (causal, hq, hk),
              all(close(x, y) for x, y in zip(got, (dq, dk, dv))))

# unmodelled variants are refused, never approximated
try:
    R._ref_flash_attn_backward(None, (), dict(n, dropout_p=0.1, softcap=0.0))
    check('dropout refused', False)
except NotImplementedError:
    check('dropout refused', True)

# placement kwargs: a recorded _to_copy(dtype=float32, device=cuda) stays CPU fp64
a = R._cpu64({'dtype': torch.float32, 'device': torch.device('cuda', 0), 'layout': torch.strided})
check('cpu64 rewrites device and float dtype',
      a['device'] == torch.device('cpu') and a['dtype'] == torch.float64 and a['layout'] == torch.strided)
check('cpu64 keeps int dtype', R._cpu64(torch.int64) == torch.int64)
t = R._cpu64(torch.ones(2, dtype=torch.bfloat16))
check('cpu64 tensor -> cpu float64', t.dtype == torch.float64 and t.device.type == 'cpu')

# _grouped_mm: no fp64 kernel; the math reference is one matmul per group
a = torch.randn(10, 16, dtype=torch.bfloat16)
b = torch.randn(3, 16, 8, dtype=torch.bfloat16)
offs = torch.tensor([4, 4, 9], dtype=torch.int32)
ref = R._reference_call('aten._grouped_mm.default', torch.ops.aten._grouped_mm.default,
                        (a, b, offs), {})
loop = torch.zeros(10, 8, dtype=torch.float64)
loop[0:4] = a[0:4].double() @ b[0].double()
loop[4:9] = a[4:9].double() @ b[2].double()
check('_grouped_mm fp64 reference = per-group matmul (empty group 1 skipped)',
      ref.dtype == torch.float64 and torch.allclose(ref, loop))

print('\n%d failure(s)' % len(FAILS) + (': ' + ', '.join(FAILS) if FAILS else ''))
sys.exit(1 if FAILS else 0)
