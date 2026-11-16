#!/usr/bin/env python3
"""No-GPU regression tests for the debug-nan helper scripts.

    python3 tests/debug_nan/test_tools.py

Covers find_capture_points.py (synthetic framework fixtures + the repo's two real
examples), wave_summary.py (log-name shapes, per-arm GPU alert, STOPPED), the
verdict proposal in make_repro_readme.py, ledger.py (interval union, no sacct),
and make_report.py.
"""
import datetime as dt
import importlib.util
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'src'))
FIX = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'fixtures', 'nan_doctor'))
PKG = 'madengine.nan_doctor'
ENV = dict(os.environ, PYTHONPATH=SRC + os.pathsep + os.environ.get('PYTHONPATH', ''))
FAILS = []


def check(name, cond, detail=''):
    print('%s  %s%s' % ('PASS' if cond else 'FAIL', name, ('  -- ' + detail) if detail and not cond else ''))
    if not cond:
        FAILS.append(name)


def run(cmd, *args, env=None):
    """`madengine nan <cmd> ...` through the package, standard library only."""
    return subprocess.run([sys.executable, '-m', PKG, cmd] + list(args), capture_output=True, text=True,
                          env=env or ENV).stdout


def load(name):
    import importlib
    sys.path.insert(0, SRC)
    return importlib.import_module('madengine.nan_doctor.' + name)


def scan(path):
    out = run('capture-points', path)
    top = re.search(r'LOOP\s+L(\d+)\s+#1 ', out)
    return out, int(top.group(1)) if top else None


# ---- find_capture_points ---------------------------------------------------
cases = [  # file, expected top loop line (None = framework-owned), must-contain categories
    ('accelerate_loop.py', 13, ['FANOUT', 'NANCHECK']),
    ('ddp_torchrun.py', 15, ['DIST', 'extra CUDA streams']),
    ('deepspeed_loop.py', 7, ['engine.step']),
    ('hf_trainer.py', None, ['HOOK', 'SWALLOW', 'host threads']),
    ('lightning_module.py', None, ['HOOK', 'LAZY', 'NANCHECK']),
]
for f, line, must in cases:
    out, top = scan(os.path.join(FIX, f))
    check('scan %s top loop' % f, top == line, 'got %s' % top)
    for m in must:
        check('scan %s finds %s' % (f, m), m in out)
    if line is None:
        check('scan %s says callback' % f, 'wire the capture as a callback' in out)
out = run('capture-points', os.path.join(FIX, 'launch_torchrun.sh'))
check('scan launcher torchrun', 'distributed launch' in out and 'env amplifier' in out)

# ---- wave_summary ---------------------------------------------------------
with tempfile.TemporaryDirectory() as d:
    caps = os.path.join(d, 'caps')
    def log(name, body):
        open(os.path.join(d, name), 'w').write(body)
    t0 = '2025-01-02 10:00:%02d,000 INFO step %d\n'
    for trial, gpu, arm, txt in [(0, 0, 'hb1', 'SUCCESS: NaN reproduced\nTRIAL_RC=0\n'),
                                 (1, 1, 'hb1', 'SUCCESS: NaN reproduced\nTRIAL_RC=0\n'),
                                 (2, 2, 'hb1', 'SUCCESS: NaN reproduced\nTRIAL_RC=0\n'),
                                 (3, 3, 'hb0', 'TRIAL_RC=0\n'), (4, 4, 'hb0', 'TRIAL_RC=0\n'),
                                 (5, 5, 'hb0', 'TRIAL_RC=0\n')]:
        log('%d_%d.%s.999.log' % (gpu, trial, arm), ''.join(t0 % (s, s) for s in range(5)) + txt)
    out = run('waves', d, '--signal', 'SUCCESS: NaN reproduced', '--quiet')
    check('wave arms hb1/hb0 kept apart', 'hb1  rate 3/3' in out and 'hb0  rate 0/3' in out, out)
    check('wave no cross-arm GPU alert', '!!' not in out, out)
    log('weird-name.log', 'TRIAL_RC=0\n')
    out = run('waves', d, '--signal', 'x', '--quiet')
    check('wave reports unmatched names', 'do not match --name' in out)
with tempfile.TemporaryDirectory() as d:
    caps = os.path.join(d, 'caps')
    os.makedirs(os.path.join(caps, 'trial_0'))
    open(os.path.join(caps, 'trial_0', 'rank0_x.pt'), 'w').write('x')
    open(os.path.join(d, '0_0.cap.1.log'), 'w').write('INFO step 1\n')            # captured
    open(os.path.join(d, '1_1.cap.1.log'), 'w').write('INFO step 30\n')           # killed: no RC
    out = run('waves', d, '--signal', 'x', '--capsules', caps, '--quiet')
    check('wave capture STOPPED excluded', 'cap  rate 1/1' in out and '1 STOPPED' in out, out)
with tempfile.TemporaryDirectory() as d:
    # multi-rank job (MoE): capsules + manifest at the root, no trial dir
    caps = os.path.join(d, 'caps')
    os.makedirs(caps)
    for f in ('manifest.json', 'rank4_a.pt', 'rank5_b.pt'):
        open(os.path.join(caps, f), 'w').write('x')
    open(os.path.join(d, '0_0.cap.1.log'), 'w').write('grad_norm: nan\nTRIAL_RC=1\n')
    out = run('waves', d, '--signal', 'grad_norm: +nan', '--capsules', caps)
    check('wave multi-rank capsules at root count', 'NAN_NO_CAPSULE' not in out and 'capsules=2' in out, out)

# ---- verdict proposal ------------------------------------------------------
mk = load('repro_readme')
R = lambda op, fin, vs, ref: dict(idx='1', op=op, finite=fin, vs=vs, ref=ref, note='')
for name, rows, want in [
        ('finite ref, nothing diverged', [R('permute', True, 'match', 'finite'), R('mm', True, 'DIVERGED', 'finite')], 'KERNEL_BUG'),
        ('no reference (SDPA bwd)', [R('sdpa', True, 'match', 'finite'), R('sdpa_bwd', True, 'DIVERGED', '-')], 'KERNEL_BUG_UNVERIFIED'),
        ('ancestor diverged', [R('a', True, 'DIVERGED', 'finite'), R('mm', True, 'DIVERGED', 'finite')], 'KERNEL_BUG_UNVERIFIED'),
        ('fp64 ref non-finite', [R('sub', True, 'match', 'finite'), R('sqrt', False, 'match', 'NON-FINITE')], 'FP32_OVERFLOW?'),
        ('deterministic walk', [R('mm', False, 'match', 'finite')], 'KERNEL_BUG_UNVERIFIED'),
        ('no walk', [], 'KERNEL_BUG_UNVERIFIED')]:
    got = mk.propose(rows)
    check('propose %s' % name, got[0] == want, '%s (%s)' % (got[0], got[2]))
check('propose deterministic flagged', 'DETERMINISTIC' in mk.propose([R('mm', False, 'match', 'finite')])[2])
# 'unavailable' is no reference at all -- never an overflow (it used to read as one)
for name, rows, term, want in [
        ('unavailable ref is unverified, not overflow',
         [R('empty_like', True, '-', 'unavailable'), R('fa_bwd', False, 'match', 'unavailable')], None,
         'KERNEL_BUG_UNVERIFIED'),
        ('terminal ref on replay operands finite', [R('mm', True, 'match', 'finite'),
                                                     R('fa_bwd', False, 'match', 'finite')], 'finite', 'KERNEL_BUG'),
        ('terminal finite beats ancestor DIVERGED', [R('mm', True, 'DIVERGED', 'finite'),
                                                     R('fa_bwd', False, 'match', 'unavailable')], 'finite', 'KERNEL_BUG'),
        ('terminal ref on replay operands non-finite', [R('fa_bwd', False, 'match', 'finite')], 'NON-FINITE',
         'FP32_OVERFLOW?')]:
    got = mk.propose(rows, term)
    check('propose %s' % name, got[0] == want, '%s (%s)' % (got[0], got[2]))
# An MoE walk: the culprit GEMM is followed by a transpose view; the GEMM
# must be the terminal, not its own 'ancestor DIVERGED'.
RN = lambda op, fin, vs, ref, note: dict(idx='1', op=op, finite=fin, vs=vs, ref=ref, note=note)
moe = [RN('aten.index.Tensor', True, 'match', 'finite', ''),
       RN('aten._grouped_mm.default', True, 'match', 'finite', 'ref: math'),
       RN('aten.mul.Tensor', True, 'match', 'finite', ''),
       RN('aten.transpose.int', True, 'match', 'finite', 'view'),
       RN('aten._grouped_mm.default', False, 'DIVERGED', 'finite',
          'max|delta|=3.39e+38, out[0], WRONG VALUE: 0.84 vs higher precision, ref: math'),
       RN('aten.transpose.int', True, 'match', 'finite', 'view')]
got = mk.propose(moe, 'finite')
check('propose trailing view: culprit is the terminal', got[0] == 'KERNEL_BUG' and 'ancestor' not in got[2],
      '%s (%s)' % (got[0], got[2]))
got = mk.propose([RN('a', True, 'DIVERGED', 'finite', '')] + moe[1:], None)
check('propose trailing view keeps a real ancestor DIVERGED', got[0] == 'KERNEL_BUG_UNVERIFIED'
      and 'ancestor DIVERGED: a' in got[2], '%s (%s)' % (got[0], got[2]))
vd = load('verdict')
with tempfile.TemporaryDirectory() as d:
    lg = os.path.join(d, 'kernel.log')
    open(lg, 'w').write('  solution_index: 14013\n'
                        '# aten._grouped_mm: no grouped problem in hipblaslt-bench -- x\n'
                        '# group rows 573: group(s) 7\nhipblaslt-bench --api_method c -m 256 -n 573 -k 768\n'
                        '# group rows 13803: group(s) 10\nhipblaslt-bench --api_method c -m 256 -n 13803 -k 768\n')
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        vd.kernel(lg)
    out = buf.getvalue()
    check('kernel summary lists every grouped bench line', '-n 573 -k 768' in out
          and '[group rows 13803: group(s) 10] hipblaslt-bench' in out, out)
check('split_terminal skips trailing views', vd.split_terminal(moe)[0]['op'] == 'aten._grouped_mm.default'
      and len(vd.split_terminal(moe)[1]) == 4)
check('c1_terminal parses the walk line',
      vd.c1_terminal(['  terminal fp64 ref on replay operands: finite  max|.|=3 fits torch.bfloat16']) == 'finite')

# ---- make_report group naming -------------------------------------------------
mr = load('report')
check('report group from replay dir', mr.group_of('/a/replay/G2/c3.1.h') == 'G2')
check('report group from verify dir', mr.group_of('/a/verify-G1/repro-G1/out/c2.1.h') == 'G1')

# ---- ledger -----------------------------------------------------------------
lg = load('ledger')
T = lambda m: dt.datetime(2026, 10, 30, 10, m)
check('ledger union overlaps once', lg.union([(T(0), T(10)), (T(5), T(15))], T(0), T(30)) == 900)
check('ledger union clips to visit', lg.union([(T(0), T(10))], T(5), T(30)) == 300)
with tempfile.TemporaryDirectory() as d:
    L = os.path.join(d, 'ledger.md')
    open(L, 'w').write('# t\n2025-01-02 10:00:00  step5  start\n2025-01-02 10:00:01  step5  job=123  x\n')
    env = dict(os.environ, PATH=d)                    # no sacct on PATH: the docker case
    out = run('ledger', L, 'end', 'step5', env=env)
    check('ledger end without sacct', 'queue=n/a' in out and 'TELL USER' in out, out)
    out = run('ledger', L, 'table')
    check('ledger table parses its own end line', '| step5 | 1 |' in out, out)

# sequential attempts (run_it.sh-style): attempt-NN/run.log + attempts.tsv; the model's
# NaN line only in the run dir the tsv names
with tempfile.TemporaryDirectory() as d:
    rd = os.path.join(d, 'rundir')
    os.makedirs(os.path.join(rd, 'nodes', 'h', 'train'))
    open(os.path.join(rd, 'nodes', 'h', 'train', 'run.log'), 'w').write('NonFiniteDetected: step 9\n')
    rows = ['attempt\tarm\toutcome\texit\tminutes\trun_dir']
    for n, arm, outc, rdir in [(1, 'fix', 'pass', ''), (2, 'stock', 'nan', rd), (3, 'stock', 'crash', '')]:
        os.makedirs(os.path.join(d, 'attempt-%02d' % n))
        open(os.path.join(d, 'attempt-%02d' % n, 'run.log'), 'w').write('harness output\n')
        rows.append('attempt-%02d\t%s\t%s\t0\t4\t%s' % (n, arm, outc, rdir))
    open(os.path.join(d, 'attempts.tsv'), 'w').write('\n'.join(rows) + '\n')
    out = run('waves', d, '--signal', 'NonFiniteDetected', '--quiet')
    check('wave attempt layout: arms from tsv', 'fix  rate 0/1' in out and 'stock rate 1/1' in out, out)
    check('wave attempt layout: no DIED / unmatched', 'DIED' not in out.split('rate')[0] and 'do not match' not in out, out)

# ---- make_repro_readme: --cotenant none ---------------------------------------
mr = load('repro_readme')
with tempfile.TemporaryDirectory() as d:
    for val, want in (('none', None), ('', None), (None, None), ('NONE', None), ('python3 hog.py', 'python3 hog.py')):
        check('readme cotenant %r -> %r' % (val, want), mr.cotenant_of(val) == want)
    rp = os.path.join(d, 'repro')
    os.makedirs(rp)
    open(os.path.join(rp, 'run.sh'), 'w').write('export RUNNER=venv\n')
    lg = os.path.join(d, 'g0_k0.h.log')
    open(lg, 'w').write('  held back:      HIPBLASLT_TENSILE_LIBPATH -- overlay\nenv: matches capture\n')
    check('readme held-back var not exported -> named', mr.held_back([lg], rp) == ['HIPBLASLT_TENSILE_LIBPATH'])
    open(os.path.join(rp, 'run.sh'), 'w').write('export HIPBLASLT_TENSILE_LIBPATH=/x\n')
    check('readme held-back var exported -> silent', mr.held_back([lg], rp) == [])

# ---- capture_digest: a log that names its run dir ----------------------------
with tempfile.TemporaryDirectory() as d:
    rd = os.path.join(d, 'run')
    os.makedirs(rd)
    open(os.path.join(rd, 'model.log'), 'w').write('[op-nan] nan-capture armed at step() 1\n')
    lg = os.path.join(d, 'run.log')
    open(lg, 'w').write('[harness] finalized run facts complete: %s\n' % rd)
    out = run('digest', lg)
    check('digest follows the named run dir', 'nan-capture armed' in out and 'NEVER' not in out, out)
    open(lg, 'w').write('nothing here\n')
    out = run('digest', lg)
    check('digest alarm when nothing reachable', 'NEVER' in out, out)

# ---- capture_preflight: a run script that greps no NaN line ------
with tempfile.TemporaryDirectory() as d:
    rs = os.path.join(d, 'run.sh')
    open(rs, 'w').write("grep -o 'step: *[0-9]* *loss: *[^ ]* *grad_norm: *[^ ]*' log\n")
    os.makedirs(os.path.join(d, 'src'))
    open(os.path.join(d, 'src', 'trainer.py'), 'w').write(
        'raise RuntimeError(\n    f"Loss is not finite (global_avg_loss={x})")\n')
    out = run('preflight', rs, '--root', d)
    check('preflight: no NaN grep -> no false NOT RAISED', 'NOT RAISED' not in out
          and 'greps no NaN line' in out and 'trainer.py:2' in out, out)

# ---- capture_preflight: pgrep patterns are not NaN signals --------------------
with tempfile.TemporaryDirectory() as d:
    rs = os.path.join(d, 'run.sh')
    open(rs, 'w').write("pgrep -af 'harness\\.suite\\.worker|torchrun|run_auto\\.sh'\n"
                        "zgrep -qE 'NonFiniteDetected|max_seqlens_q must be provided' x.gz\n")
    alts = load('preflight').signal_alternatives(rs, None)
    check('preflight skips pgrep pattern', 'NonFiniteDetected' in alts
          and not any('torchrun' in x or 'worker' in x for x in alts), str(alts))

# ---- make_report: no control arm, ledger vs iter files -----------
with tempfile.TemporaryDirectory() as A2:
    os.makedirs(os.path.join(A2, 'rootcause'))
    open(os.path.join(A2, 'rootcause', 'iter-1.md'), 'w').write('# iter 1\n')
    open(os.path.join(A2, 'ledger.md'), 'w').write(
        '2025-01-09 18:32:40  step1  answers: step2 SKIP\n'
        '2025-01-09 19:04:45  step7[G1]  iter 1: x\n'
        '2025-01-09 19:10:16  step7[G1]  iter 2: y\n')
    run('report', A2, '--out', os.path.join(A2, 'r.md'))
    rep = open(os.path.join(A2, 'r.md')).read()
    check('report: no control arm is not called a control arm', 'had no control arm' in rep
          and 'control arm (no control arm)' not in rep, rep[:600])
    check('report: ledger iterations vs files flagged', 'the ledger has 2' in rep)


print('\n%d failure(s)' % len(FAILS) + (': ' + ', '.join(FAILS) if FAILS else ''))
sys.exit(1 if FAILS else 0)
