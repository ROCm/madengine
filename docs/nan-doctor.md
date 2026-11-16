# nan-doctor: find where a NaN starts, and hand over a reproducer

`madengine nan` finds the first operation that produced a NaN in a PyTorch workload on AMD Instinct
GPUs, replays it without the model, and packages a standalone reproducer. It is the deterministic
part of an investigation -- every command prints a verdict or a count, never an opinion -- so an
agent (MAD's `/mad-debug-nan` skill and `mad-nan-hunt` workflow) or a person can drive it.

```
 model run ──(tools: nan_doctor)──▶ capsules ──group──▶ groups ──replay──▶ verdict ──package──▶ reproducer
   madengine run                                5a          5b                          6
```

| Term | Meaning |
|---|---|
| **Capsule** | A saved record of the first NaN-producing op, with its upstream chain and operand shapes (values with `NAN_CAPTURE_DATA=1`) |
| **Group** | Capsules judged to share one cause; `G1` is the largest |
| **Replay** | Re-running a capsule on the GPUs without the model: `REPRODUCED`, `NOT_REPRODUCED` or `NOT_SET_UP` |
| **Reproducer** | A directory (`run.sh`, README, capsule, a copy of the replay code) that triggers the NaN with no model, data or checkout. Bound to the GPU architecture it was captured on |

## Capture: the `nan_doctor` tool

```bash
madengine run --tags <model> --additional-context '{"tools": [{"name": "nan_doctor"}]}'
```

See [profiling.md](profiling.md#nan_doctor---nan-capture-for-pytorch-workloads). Capsules land in
`nan_doctor_output/capsules/` (on SLURM: `node_<N>/nan_doctor_output/`, also for failed nodes).

## Commands

All commands run on the host with the standard library only (no torch); what needs torch runs in the
model's image through `madengine nan exec`.

| Command | Step | Does |
|---|---|---|
| `exec` | -- | Run a command in the model's image on the GPUs (`--image` or `--manifest build_manifest.json`). The `nan_doctor_rt` runtime is mounted at `/opt/nan_doctor` and on `PYTHONPATH` |
| `group` | 5a | Group the capsules in the image; write `groups.json` and the priority-ordered `groups_priority.txt` |
| `replay` | 5b | Replay one capsule under load on every GPU; the concurrency ladder; ends with `LADDER_RESULT=` |
| `summary` | 5b | Summarise one replay attempt directory (`ATTEMPT_RESULT=`) |
| `package` | 6 | Build `repro-<group>/`; `--pack` writes the tarball and prints its sha256 |
| `readme` | 6 | Write or re-pack the reproducer README |
| `report` | 9 | Draft the investigation report from the artifact directory |
| `intake` | 1 | Validate an intake file (the answers to "how does this workload run") |
| `ledger` | -- | Append-only investigation journal: `start`, `note`, `job`, `wait`, `end`, `table` |
| `verdict`, `waves`, `handoff`, `digest`, `kernel-match` | | Readers for replay and capture logs |
| `framework-check`, `gpu-check`, `preflight`, `capture-points`, `wire` | 0-4 | Gates and checks before a capture; `wire` is the source-wiring fallback for loops the tool cannot see |

`madengine nan <command> --help` lists a command's options.

```bash
# group the capsules, replay the largest group, package it
madengine nan group nan_doctor_output/capsules --manifest build_manifest.json --out-dir art
madengine nan replay <capsule of G1> --manifest build_manifest.json --out-dir art/replay/G1
madengine nan package --group G1 --artifact-dir art --image <image> \
    --capsule <capsule> --attempt art/replay/G1/c3.<pid>.<host> --groups art/groups.json --pack
```

## The replay ladder

`replay` starts at the capture's processes per GPU + 1 copies per GPU, so the replay is at least as
loaded as the model was, and steps down one copy at a time on out-of-memory. It stops every copy
together once the node has `--stop-hits` (10) non-finite runs, or declares `NOT_SET_UP` after
`--futility` (100) times the capsule's op call index of clean runs: the replay is then not
recreating the model's trigger, which more runs cannot fix. A clean result is a bound, not proof.
The docker flags (`--ipc=host`, `--shm-size`, memlock, seccomp) shape the timing being replayed and
are not tunable.

## Reproducer

`package` ships exactly what the reproducing attempt ran: whole chain, its concurrency and
iteration count, its co-tenant. `run.sh` needs docker, python3 and the image -- not madengine. A
reproducing run (`n/N > 0` near the original rate) from an extracted copy is the verification; a
`0/N` is a failed build. The README's operand-validation sentence is proposed from the replay and
marked `<!-- CHECK -->` for a person to confirm.

## Limits

- PyTorch workloads on AMD Instinct MI300X or newer (gfx942, gfx950); other GPUs get a warning.
- Local Docker, and SLURM where madengine launches the workload. Models with their own multi-node
  launchers (`slurm_multi`) do not get the tool on their ranks. K8s is not supported yet.
- No GPU lock is taken here; callers that need one (MAD's `mad-run.sh`) hold it around the command.
