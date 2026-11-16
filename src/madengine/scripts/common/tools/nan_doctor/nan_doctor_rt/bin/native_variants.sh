#!/bin/bash
# mad-debug-nan step 5N (N5): one variant per GPU, all concurrent, one node -- a candidate
# and its controls / reductions in a single job.
#
#   bash native_variants.sh <ARTIFACT_DIR>/native/repro_<k>.py <variants file>
#
# Variants file: line i (0-based; `#` and blank lines skipped) runs on GPU i as
#   <copies>|<env VAR=VALUE ...>|<candidate args>|<optional co-tenant command>
# e.g.  3|||                                        the candidate, 3 copies
#       1|||                                        solo
#       3|TORCH_BLAS_PREFER_HIPBLASLT=0||           a library toggle
#       1||--mode op|python3 .../transformer_hog.py --mode fwdbwd --procs 2
# The co-tenant starts once on its GPU before the copies and is killed after them.
# Runs in the candidate's directory; logs in out.<jobid>.<host>/.  Candidate contract
# as native_run.sh.  Prints one `VARIANT_TOTAL gpu=<i> <hits>/<runs> results=<n>
# variant=<line>` per GPU, `TRACEBACK <log>` for any crash, then `VARIANTS_DONE`.
#
# Run it under all GPUs of a node, inside the model's image:
#   madengine nan exec --image <image> --gpus all -- bash \$NAN_DOCTOR_RT/bin/native_variants.sh <cand> <file>
set -uo pipefail
cand=$(readlink -f -- "$1"); vf=$(readlink -f -- "$2")
cd -- "$(dirname -- "$cand")" || exit 2
out=out.${SLURM_JOB_ID:-$(date +%s).$$}.$(hostname -s)
mkdir -p "$out"
pids=(); cpids=(); g=0
while IFS='|' read -r copies envs args cot; do
  [[ -z "$copies" || "$copies" == \#* ]] && continue
  if [[ -n "${cot:-}" ]]; then
    HIP_VISIBLE_DEVICES=$g setsid bash -c "$cot" >"$out/cotenant_g$g.log" 2>&1 &
    cpids+=($!)
  fi
  for ((c = 0; c < copies; c++)); do
    env HIP_VISIBLE_DEVICES=$g $envs python3 "$cand" --copy "$c" --gpu "$g" $args >"$out/g${g}_c${c}.log" 2>&1 &
    pids+=($!)
  done
  echo "$copies|$envs|$args|${cot:-}" >"$out/variant_g$g.txt"
  g=$((g + 1))
done <"$vf"
wait "${pids[@]}"
for p in "${cpids[@]}"; do kill -- -"$p" 2>/dev/null; done
for ((i = 0; i < g; i++)); do
  v=$(cat "$out/variant_g$i.txt")
  n=${v%%|*}; logs=(); for ((c = 0; c < n; c++)); do logs+=("$out/g${i}_c${c}.log"); done
  r=$(awk -F'[=/ ]' '/^NATIVE_RESULT=/ {h += $2; n += $3; k++} END {printf "%d/%d results=%d", h, n, k}' "${logs[@]}")
  echo "VARIANT_TOTAL gpu=$i $r variant=$v"
done
grep -l Traceback "$out"/g*_c*.log | sed 's/^/TRACEBACK /'
echo "VARIANTS_DONE out=$out"
