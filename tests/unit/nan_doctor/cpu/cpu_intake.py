#!/usr/bin/env python3
"""No-GPU tests for scripts/internal/intake.py (the step-1 intake contract).

    python3 tests/debug_nan/test_intake.py
"""
import copy
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'src')))
from madengine.nan_doctor import intake  # noqa: E402

FIX = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'fixtures', 'nan_doctor', 'intake_mad.json'))
FAILS = []


def check(name, cond, detail=''):
    print('%s  %s%s' % ('PASS' if cond else 'FAIL', name, ('  -- ' + str(detail)) if detail and not cond else ''))
    if not cond:
        FAILS.append(name)


good = json.load(open(FIX))
n, errs = intake.validate(good)
check('fixture is valid', not errs, errs)
check('defaults filled: nan_signal', n and n['nan_signal']['regex'] == intake.DEFAULT_NAN_SIGNAL)
check('defaults filled: preferences', n and n['preferences']['capture_data'] == 'shapes'
      and n['preferences']['fix'] is False and n['preferences']['groups'] == 'largest')
check('default signal matches a NaN loss line',
      bool(__import__('re').search(intake.DEFAULT_NAN_SIGNAL, 'step 12 | loss: nan | lr 1e-4')))
check('default signal ignores a clean loss line',
      not __import__('re').search(intake.DEFAULT_NAN_SIGNAL, 'step 12 | loss: 2.31 | grad_norm: 0.9'))


def bad(name, mutate, needle):
    d = copy.deepcopy(good)
    mutate(d)
    _, e = intake.validate(d)
    check('rejects ' + name, any(needle in x for x in e), e)


bad('missing run', lambda d: d.pop('run'), 'run: required')
bad('unknown field', lambda d: d.update(colour='red'), 'colour: unknown field')
bad('wrong schema', lambda d: d.update(schema=2), 'schema:')
bad('bool as int', lambda d: d.update(world_size=True), 'world_size: must be int')
bad('k8s target', lambda d: d.update(deploy_target='k8s'), 'deploy_target')
bad('tools in context', lambda d: d['additional_context'].update(tools=[]), "must not carry 'tools'")
bad('madengine without tag', lambda d: d.pop('model_tag'), 'model_tag: required')
bad('non-madengine run cmd', lambda d: d.update(run='bash run.sh'), "'madengine run ...'")
bad('bad regex', lambda d: d.update(nan_signal={'regex': '('}), 'does not compile')
bad('fix without rootcause', lambda d: d['preferences'].update(fix=True), 'needs preferences.rootcause')
bad('unknown preference', lambda d: d['preferences'].update(speed='max'), 'unknown preference')
bad('bad groups value', lambda d: d['preferences'].update(groups='some'), 'must be one of')
bad('bad priors value', lambda d: d['preferences'].update(priors='some'), 'must be one of')
check('defaults filled: priors', n and n['preferences']['priors'] == 'use')

d = copy.deepcopy(good)
d.update(env='docker', run='bash run.sh')
d.pop('model_tag')
_, e = intake.validate(d)
check('docker env needs no model_tag', not e, e)

cli = [sys.executable, '-m', 'madengine.nan_doctor', 'intake']
r = subprocess.run(cli + [FIX], capture_output=True, text=True,
                   env=dict(os.environ, PYTHONPATH=os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'src'))))
check('CLI valid: exit 0 and KEY=VALUE', r.returncode == 0 and 'MODEL_TAG=dummy/dummy_nan' in r.stdout
      and 'PREF_ROOTCAUSE=false' in r.stdout and 'PREF_PRIORS=use' in r.stdout, r.stdout + r.stderr)
r = subprocess.run(cli + [os.devnull], capture_output=True, text=True)
check('CLI invalid: exit 2 and INTAKE_ERROR', r.returncode == 2 and 'INTAKE_ERROR' in r.stdout, r.stdout)
r = subprocess.run(cli + ['--schema'], capture_output=True, text=True)
check('CLI --schema lists every field', all(k in r.stdout for k in intake.FIELDS), r.stdout)

print('\n%s' % ('ALL PASS' if not FAILS else 'FAILED: %s' % ', '.join(FAILS)))
sys.exit(1 if FAILS else 0)
