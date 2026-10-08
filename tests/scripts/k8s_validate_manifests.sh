#!/usr/bin/env bash
# Optional static validation of rendered Kubernetes YAML (debug output).
# Install kubeconform: https://github.com/yannh/kubeconform
# Usage (from madengine repo root):
#   MODEL_DIR=... madengine run ... --additional-context '{"debug": true}'
#   ./tests/scripts/k8s_validate_manifests.sh ./k8s_manifests

set -euo pipefail
DIR="${1:-./k8s_manifests}"
if ! command -v kubeconform >/dev/null 2>&1; then
  echo "kubeconform not installed; skipping validation."
  exit 0
fi
if [[ ! -d "$DIR" ]]; then
  echo "Directory not found: $DIR"
  exit 1
fi
exec kubeconform -summary -strict "$DIR"
