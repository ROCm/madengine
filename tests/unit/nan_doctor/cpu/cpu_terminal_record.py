#!/usr/bin/env python3
"""CPU tests for the terminal checks of replay_nan_op.py (needs torch, no GPU).

    python3 tests/unit/nan_doctor/cpu/cpu_terminal_record.py

A capsule's chain can end in view rows recorded after the op the capsule is named
for -- e.g. an aten.t of another tensor right after a weight-grad mm in a backward.
The terminal fp64 reference and the failure signature must judge the capsule's own
op, not chain[-1]: a reference of the trailing view is finite and turned a correct
fp16 overflow into a KERNEL_BUG verdict.
"""
import os
import sys

import torch

RT_PARENT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'src', 'madengine', 'scripts', 'common', 'tools', 'nan_doctor'))
sys.path.insert(0, RT_PARENT)
FAILS = []


def check(name, cond, detail=''):
    print('%s  %s%s' % ('PASS' if cond else 'FAIL', name, ('  -- ' + str(detail)) if detail and not cond else ''))
    if not cond:
        FAILS.append(name)


def load():
    from nan_doctor_rt import replay_nan_op
    replay_nan_op._import_torch()
    return replay_nan_op


def row(R, op, args, produces, view=False):
    return {'op': op, 'args': [{R._CHAIN_REF: k} for k in args], 'kwargs': {}, 'produces': produces,
            'call_index': 7, 'is_view': view, 'needs_rng': False, 'rng': None, 'version_ok': True}


for tag in ('replay_nan_op',):
    R = load()
    mm = row(R, 'aten.mm.default', ['a', 'b'], ['c'])
    t = row(R, 'aten.t.default', ['w'], ['wt'], view=True)
    cap = {'op': 'aten.mm.default', 'op_call_index': 7, 'chain': [mm, t], 'pool': {}, 'roots': ['a', 'b', 'w'],
           'out_dtype': torch.float16}

    # the record the terminal checks use
    check('%s: trailing view -> the capsule\'s own op' % tag, R._terminal_record(cap) is mm)
    mm2 = dict(mm, produces=['c2'])
    check('%s: own op twice -> the last of them' % tag,
          R._terminal_record(dict(cap, chain=[mm, mm2, t])) is mm2)
    other = row(R, 'aten.addmm.default', ['a', 'b'], ['d'])
    check('%s: no row of the own op -> last non-view row' % tag,
          R._terminal_record(dict(cap, op='aten.bmm.default', chain=[other, t])) is other)
    check('%s: chain ending in the op itself is unchanged' % tag,
          R._terminal_record(dict(cap, chain=[t, mm])) is mm)

    # end to end: an fp16 mm whose true result exceeds fp16 max, followed by a view of a
    # finite tensor. The correct verdict is NON-FINITE (a correct overflow, not a kernel bug).
    a = torch.full((4, 8), 300.0, dtype=torch.float16)
    b = torch.full((8, 4), 300.0, dtype=torch.float16)          # true value 720000 > 65504
    env = {'a': a, 'b': b, 'w': torch.ones(3, 2, dtype=torch.float16)}
    got = R._terminal_reference(cap, dict(env))
    check('%s: overflowing mm + trailing view -> NON-FINITE reference' % tag,
          got is not None and got[0] == 'NON-FINITE', got)

print('\n%d failure(s)' % len(FAILS))
sys.exit(1 if FAILS else 0)
