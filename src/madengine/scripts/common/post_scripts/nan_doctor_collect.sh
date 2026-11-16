#!/usr/bin/env bash
#
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# nan_doctor tool: remove the capture hook and summarize the capsules.
#
# Runs after the model, also when the model failed (a NaN usually ends the run).
# Capsules are already in NAN_CAPTURE_DIR (under /myworkspace/nan_doctor_output),
# which the host and the SLURM per-node collection pick up; this writes
# summary.txt beside them. Never fails the run.

set +e

site_file=/tmp/nan_doctor_autocapture.site
if [ -f "$site_file" ]; then
    site=$(cat "$site_file")
    rm -rf "${site}/nan_doctor_rt" "${site}/nan_doctor_rt.pth"
    rm -f "$site_file"
fi

CAPS=${NAN_CAPTURE_DIR:-/myworkspace/nan_doctor_output/capsules}
OUT=$(dirname "$CAPS")
mkdir -p "$OUT"
n=$(find "$CAPS" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l)
manifest=no
ls "$CAPS"/manifest*.json >/dev/null 2>&1 && manifest=yes
{
    echo "capsules=$n"
    echo "manifest=$manifest"
    echo "capsule_dir=$CAPS"
    echo "host=$(hostname)"
} > "$OUT/summary.txt"
chmod -R a+rwX "$OUT" 2>/dev/null
echo "nan_doctor: $n capsule(s) in $CAPS (manifest=$manifest); summary in $OUT/summary.txt"
exit 0
