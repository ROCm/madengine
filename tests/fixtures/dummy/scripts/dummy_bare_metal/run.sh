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

export HF_HUB_DISABLE_TELEMETRY=1
cd "${script_dir}"
# The hermetic fake-conda test skips pip, so the interpreter on PATH may not
# have torch. A real conda env installs requirements.txt and takes the
# torchrun path below.
if python3 -c "import torch" >/dev/null 2>&1; then
  python3 -m torch.distributed.run --nproc_per_node="${nproc}" --standalone train.py
else
  echo "performance: 1.00 tokens_per_second"
fi
