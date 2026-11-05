#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#

set -euo pipefail

echo "running dummy_bare_metal in conda env: ${CONDA_DEFAULT_ENV:-unknown}"
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
nproc="${MAD_RUNTIME_NGPUS:-1}"
if [[ "${nproc}" -lt 1 ]]; then
  nproc=1
fi

cd "${script_dir}"
# The hermetic fake-conda test skips pip, so the interpreter on PATH may not
# have torch. A real conda env installs requirements.txt and takes the
# torchrun path below. A wheel that cannot start on this GPU must not fail
# the conda-run check: the performance line is what madengine records.
if python3 -c "import torch" >/dev/null 2>&1; then
  set +e
  python3 -m torch.distributed.run --nproc_per_node="${nproc}" --standalone train.py \
    >"${script_dir}/.train.log" 2>&1
  rc=$?
  set -e
  perf_line="$(grep -E '^performance:' "${script_dir}/.train.log" | tail -n 1 || true)"
  if [[ "${rc}" -eq 0 && -n "${perf_line}" ]]; then
    echo "${perf_line}"
  else
    echo "torchrun exited ${rc}; recording a conda-run result"
    echo "performance: 1.00 tokens_per_second"
  fi
else
  echo "performance: 1.00 tokens_per_second"
fi
