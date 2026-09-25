#!/usr/bin/env bash
# cvs_health_gate.sh -- gate a madengine run on a cvs cluster-health check.
#
# Usage (as a madengine pre_scripts entry):
#   {"path": "scripts/common/pre_scripts/cvs_health_gate.sh",
#    "args": "monitor check_cluster_health --cluster_file /path/to/cluster.json"}
#
# Requires `cvs` (https://github.com/ROCm/cvs) to already be installed inside
# the model's Docker image, along with any SSH credentials cvs needs to reach
# the target cluster nodes -- madengine does not install or provision cvs.
#
# Forwards cvs's own exit code: a non-zero exit here aborts the madengine run
# before the model script executes (existing pre_scripts behavior).
set -euo pipefail

if ! command -v cvs >/dev/null 2>&1; then
  echo "cvs_health_gate: 'cvs' CLI not found in PATH. This pre_script requires" >&2
  echo "cvs_health_gate: cvs (https://github.com/ROCm/cvs) to be installed in the model image." >&2
  exit 1
fi

if [ "$#" -eq 0 ]; then
  echo "cvs_health_gate: usage: cvs_health_gate.sh <cvs-subcommand-and-args...>" >&2
  echo "cvs_health_gate: example: cvs_health_gate.sh monitor check_cluster_health --cluster_file /workspace/cluster.json" >&2
  exit 1
fi

echo "cvs_health_gate: running 'cvs $*'"
set +e
cvs "$@"
status=$?
set -e

if [ "$status" -ne 0 ]; then
  echo "cvs_health_gate: cvs exited with status ${status}; aborting run" >&2
fi
exit "$status"
