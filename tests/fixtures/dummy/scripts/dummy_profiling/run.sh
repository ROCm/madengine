#!/bin/bash
# Launcher for dummy_profiling: a small torchrun job for validating profiling tools.
#
# Like real framework launchers (e.g. Primus run.sh), it changes directory before
# starting the workload, so tools that write relative paths are exercised under a
# cwd other than run_directory. Set DUMMY_PROF_STAY_IN_RUN_DIR=1 to launch in place.
# DUMMY_PROF_EXIT_CODE makes the run fail after training, to check that profiler
# post-scripts still collect and analyze the traces of a failed run.
set -e

RUN_DIR="$(pwd)"
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_profiling.py"
export DUMMY_TORCH_PROFILE_DIR="${DUMMY_TORCH_PROFILE_DIR:-$RUN_DIR/torch_profiler_output}"

N_GPUS="${MAD_RUNTIME_NGPUS:-1}"
RUNNER="${MAD_MULTI_NODE_RUNNER:-torchrun --standalone --nproc_per_node=$N_GPUS}"

if [[ "${DUMMY_PROF_STAY_IN_RUN_DIR:-0}" != "1" ]]; then
  mkdir -p /tmp/dummy_profiling_cwd
  cd /tmp/dummy_profiling_cwd
fi
echo "dummy_profiling: cwd=$(pwd) runner=$RUNNER"
$RUNNER "$SCRIPT"
exit "${DUMMY_PROF_EXIT_CODE:-0}"
