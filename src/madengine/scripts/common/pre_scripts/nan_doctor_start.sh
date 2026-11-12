#!/usr/bin/env bash
#
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# nan_doctor tool: arm nan-doctor's NaN capture in an unmodified PyTorch workload.
#
# Installs the vendored capture runtime (tools/nan_doctor/, synced from
# nan-doctor; see its VERSION) into site-packages with a .pth hook, so every
# Python process of the run -- every torchrun rank included -- captures the
# first NaN-producing op as a capsule under NAN_CAPTURE_DIR. Activation needs
# NAN_CAPTURE_DIR and NAN_CAPTURE_AUTO=optimizer, both set by tools.json.
# post_scripts/nan_doctor_collect.sh removes the hook and summarizes.
#
# Fails the run when the capture cannot be armed: a NaN hunt that silently
# runs unwatched costs a full model run and proves nothing.

set -e

fail() {
    echo "nan_doctor: $*" >&2
    exit 1
}

RUNTIME=""
for d in scripts/common/tools/nan_doctor ../scripts/common/tools/nan_doctor /myworkspace/scripts/common/tools/nan_doctor; do
    if [ -f "$d/install_autocapture.py" ]; then
        RUNTIME="$d"
        break
    fi
done
[ -n "$RUNTIME" ] || fail "capture runtime not found (scripts/common/tools/nan_doctor)"
[ -n "${NAN_CAPTURE_DIR:-}" ] || fail "NAN_CAPTURE_DIR is not set"
[ "${NAN_CAPTURE_AUTO:-}" = "optimizer" ] || fail "NAN_CAPTURE_AUTO must be 'optimizer' (got '${NAN_CAPTURE_AUTO:-}')"
python3 -c 'import torch' >/dev/null 2>&1 || fail "torch is not importable with python3 in this image; nan_doctor captures PyTorch workloads only"

# The capture asserts the directory exists and is writable by whatever uid the ranks run as.
mkdir -p "$NAN_CAPTURE_DIR"
chmod -R 777 "$(dirname "$NAN_CAPTURE_DIR")" 2>/dev/null || true

site=$(python3 "$RUNTIME/install_autocapture.py") || fail "install_autocapture.py failed"
echo "$site" > /tmp/nan_doctor_autocapture.site

# Prove the hook loads in a fresh interpreter, without starting a capture.
python3 -c 'import sys; sys.exit(0 if "nan_doctor_autocapture" in sys.modules else 1)' \
    || fail "the .pth hook in ${site} did not load"

echo "✓ nan_doctor: capture armed (runtime $(head -n 1 "$RUNTIME/VERSION" 2>/dev/null || echo unknown), site ${site}, capsules -> ${NAN_CAPTURE_DIR})"
