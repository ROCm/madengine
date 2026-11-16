#!/bin/bash
# mad-debug-nan step 8: validate env-setting workarounds on the step-6 reproducer -- stock first
# (the control: it must reproduce), then each "VAR=VALUE" arm, every arm in its own process,
# the chain loop at the same iteration count.  No --restore-env: it would put the capture's
# environment back, i.e. UNSET the TENSILE_* / HIPBLASLT_* variable under test.
#
#   bash env_workaround_validate.sh <capsule.pt> "TENSILE_FIXED_WGMXCC=1 TENSILE_STREAMK_DYNAMIC_GRID=0" [iters] [-- extra replay args]
#
# Per arm: ARM_ENV (what the process saw -- a dropped variable reads as a clean pass), the
# chain count, the walk's WRONG VALUE line, and with PERF_CMD set, its output (a timing
# command run under the same env, e.g. "python3 my_timing.py").  Run under <GPU_EXEC>;
# PYTHONPATH must reach nan_doctor_rt (`madengine nan exec` sets it).  Example of a result:
# stock 2000/2000, TENSILE_FIXED_WGMXCC=1 0/2000, TENSILE_STREAMK_DYNAMIC_GRID=0 0/2000.
set -u
CAP=${1:?capsule}; ARMS=${2:?'"VAR=VALUE ..." arms'}; ITERS=${3:-2000}; shift $(( $# < 3 ? $# : 3 ))
[ "${1:-}" = "--" ] && shift
for arm in STOCK=1 $ARMS; do
  echo "== ARM $arm"
  env "$arm" python3 -c 'import os; print("ARM_ENV", {k: v for k, v in os.environ.items()
      if k.startswith(("TENSILE_", "HIPBLASLT_", "TORCH_BLAS", "TORCH_ROCM", "HSA_"))})'
  env "$arm" python3 -m replay_nan_op "$CAP" --chain --chain-iters "$ITERS" --seconds 600 "$@" 2>&1 |
    grep -E 'WRONG VALUE|non-finite on|terminal fp64|first op to go|^\[verdict\]' | head -8
  [ -n "${PERF_CMD:-}" ] && env "$arm" bash -c "$PERF_CMD" 2>&1 | tail -3
done
echo ENV_VALIDATE_DONE
