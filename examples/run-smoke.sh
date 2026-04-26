#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAKEFILE_PATH="${ROOT_DIR}/examples/Makefile.smoke"

usage() {
  cat <<'EOF'
Usage:
  examples/run-smoke.sh <command> MODEL_DIR=<path> MODEL_TAG=<tag> [extra make vars...]

Commands:
  slurm         Run SLURM smoke (build + run)
  k8s           Run Kubernetes smoke (build + run)
  verify-slurm  Verify SLURM smoke artifacts
  verify-k8s    Verify Kubernetes smoke artifacts

Examples:
  examples/run-smoke.sh slurm MODEL_DIR=/path/to/model MODEL_TAG=dummy
  examples/run-smoke.sh k8s MODEL_DIR=/path/to/model MODEL_TAG=dummy
  examples/run-smoke.sh verify-slurm
  examples/run-smoke.sh verify-k8s
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

cmd="$1"
shift

case "${cmd}" in
  slurm)
    exec make -f "${MAKEFILE_PATH}" smoke-slurm "$@"
    ;;
  k8s)
    exec make -f "${MAKEFILE_PATH}" smoke-k8s "$@"
    ;;
  verify-slurm)
    exec make -f "${MAKEFILE_PATH}" smoke-slurm-verify "$@"
    ;;
  verify-k8s)
    exec make -f "${MAKEFILE_PATH}" smoke-k8s-verify "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage
    exit 2
    ;;
esac
