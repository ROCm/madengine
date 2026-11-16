#!/bin/bash
# mad-debug-nan step 5N (N4): run one native candidate under load on every GPU of the node.
#
#   bash native_run.sh <ARTIFACT_DIR>/native/repro_<k>.py [args for the candidate...]
#
# Starts COPIES copies of the candidate per GPU (HIP_VISIBLE_DEVICES=<g>, each given
# --copy <c> --gpu <g>), plus COTENANT once per GPU when set, waits for the copies,
# kills the co-tenants, and sums the candidates' last lines.  Runs in the candidate's
# directory; per-copy logs in out.<jobid>.<host>/ beside it.
#
# Candidate contract (step5n-native.md N3): accepts --copy/--gpu, prints `HIT ...`
# lines and ends with `NATIVE_RESULT=<hits>/<runs> ...`.
# Env: COPIES (per GPU, default 3 = capture's processes per GPU + 1), NGPUS (8),
#      COTENANT (command, started once per GPU; killed at the end),
#      COTENANT_WARMUP (seconds before the copies start, default 30).
# Last line: NATIVE_TOTAL=<hits>/<runs> copies=<c>x<gpus> out=<dir>
#
# Run it under all GPUs of a node, inside the model's image:
#   madengine nan exec --image <image> --gpus all -- bash \$NAN_DOCTOR_RT/bin/native_run.sh <candidate>
set -uo pipefail
cand=$(readlink -f -- "$1"); shift
cd -- "$(dirname -- "$cand")" || exit 2
COPIES=${COPIES:-3}; NGPUS=${NGPUS:-8}
out=out.${SLURM_JOB_ID:-$(date +%s).$$}.$(hostname -s)
mkdir -p "$out"
cot=()
if [[ -n "${COTENANT:-}" ]]; then
  for ((g = 0; g < NGPUS; g++)); do
    HIP_VISIBLE_DEVICES=$g setsid bash -c "$COTENANT" >"$out/cotenant_g$g.log" 2>&1 &
    cot+=($!)
  done
  sleep "${COTENANT_WARMUP:-30}"
fi
pids=()
for ((g = 0; g < NGPUS; g++)); do
  for ((c = 0; c < COPIES; c++)); do
    HIP_VISIBLE_DEVICES=$g python3 "$cand" --copy "$c" --gpu "$g" "$@" >"$out/g${g}_c${c}.log" 2>&1 &
    pids+=($!)
  done
done
wait "${pids[@]}"
for p in "${cot[@]}"; do kill -- -"$p" 2>/dev/null; done
grep -h '^HIT ' "$out"/g*_c*.log | head -20
grep -L '^NATIVE_RESULT=' "$out"/g*_c*.log | sed 's/^/NO_RESULT /'
grep -l Traceback "$out"/g*_c*.log | sed 's/^/TRACEBACK /'
awk -F'[=/ ]' '/^NATIVE_RESULT=/ {h += $2; n += $3} END {printf "NATIVE_TOTAL=%d/%d copies=%sx%s out=%s\n", h, n, c, g, o}' \
  c="$COPIES" g="$NGPUS" o="$out" "$out"/g*_c*.log
