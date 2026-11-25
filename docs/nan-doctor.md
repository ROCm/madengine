# Numeric Doctor: find where a NaN starts, and hand over a reproducer

Numeric Doctor (formerly nan-doctor) is integrated into madengine as the `madengine nan` commands
and the `nan_doctor` tool; the command, tool and `NAN_*` variable names are unchanged.

`madengine nan` finds the first operation that produced a NaN in a PyTorch workload on AMD Instinct
GPUs, replays it without the model, and packages a standalone reproducer. It is the deterministic
part of an investigation -- every command prints a verdict or a count, never an opinion -- so an
agent (MAD's `/mad-debug-numeric` skill and `mad-nan-hunt` workflow) or a person can drive it.

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
| `exec` | -- | Run a command in the model's image on the GPUs (`--image` or `--manifest build_manifest.json`). The `nan_doctor_rt` runtime is mounted at `/opt/nan_doctor` and this host package at `/opt/nan_doctor_host`, both on `PYTHONPATH` (`python3 -m nan_doctor gpu-check --torch` runs in the image) |
| `group` | 5a | Group the capsules in the image; write `groups.json` and the priority-ordered `groups_priority.txt` |
| `replay` | 5b | Replay one capsule under load on every GPU; the concurrency ladder; ends with `LADDER_RESULT=` |
| `summary` | 5b | Summarise one replay attempt directory (`ATTEMPT_RESULT=`) |
| `package` | 6 | Build `repro-<group>/`; `--pack` writes the tarball and prints its sha256 |
| `readme` | 6 | Write or re-pack the reproducer README |
| `report` | 9 | Draft the investigation report from the artifact directory; `--fill <answers.md>` replaces its numbered `TODO(agent)` lines |
| `intake` | 1 | Validate an intake file (the answers to "how does this workload run") |
| `ledger` | -- | Append-only investigation journal: `start`, `note`, `job`, `wait`, `end`, `skip`, `table`, `status`. Every line rewrites `status.json` beside the ledger (the reproducer, root cause, fix, current step and job, one summary line) |
| `verify-nodes` | 6 | Run the extracted reproducer on K other SLURM nodes, each from its own copy in an empty environment: `script` writes the sbatch job, `score` prints `NODES_REPRODUCED=k/K` (`NODE_DEPENDENT` when only some nodes reproduce) |
| `arms` | 2 | Differential mode, for a model that prints nothing when it is wrong: `detect` the framework, list the `arms` (A, A2, the conservative reference REF, one `flip_<switch>` per switch; `--format context --out-dir DIR` prints one `--additional-context` per arm), `list` the switch catalogue |
| `diff` | 2 | Compare the arms' tap outputs: `ND_DIFF verdict=` `AGREE`, `DIVERGE`, `NONDETERMINISTIC`, `INPUTS_DIFFER` or `NO_OUTPUTS`, and `ND_DIFF_SWITCH=` names the switch that removes a divergence; `--modules` names the first module that disagrees (`ND_DIFF_LOCATE`) |
| `power` | 7-8 | How many runs an experiment needs from the reproducer's rate (`RUNS=`); `--clean K/N` judges a clean result (`POWER=`) |
| `verdict`, `waves`, `handoff`, `digest`, `kernel-match` | | Readers for replay and capture logs |
| `framework-check`, `gpu-check`, `preflight`, `capture-points`, `wire` | 0-4 | Gates and checks before a capture; `wire` is the source-wiring fallback for loops the tool cannot see |

`madengine nan <command> --help` lists a command's options.

The container package `nan_doctor_rt` also carries the capture modes added in
nan-doctor after the training-ladder port. They are not separate `madengine nan`
commands. A run selects one by putting its directory on `PYTHONPATH` through `docker_env_vars`, as
the absolute path inside the container, where the MAD root is `/myworkspace`
(`{"docker_env_vars": {"PYTHONPATH": "/myworkspace/scripts/common/tools/nan_doctor/nan_doctor_rt/diff_tap"}}`
is what `madengine nan arms ... --format context` prints):

| Mode | Path under `nan_doctor_rt` |
|---|---|
| vLLM workers: tensor and pipeline parallel, multi-node (each rank tagged `pp<i>/tp<j>`); TP 1 with the engine in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) | `vllm_capture/` |
| vLLM CUDA graphs | `vllm_graphs/` |
| SGLang schedulers: tensor, pipeline and data parallel (one capsule dir per replica) | `sglang_capture/` |
| SGLang CUDA graphs: off without an edit, and a graph-replay check | `sglang_graphs/` |
| A server (`vllm serve`, `sglang.launch_server`) as one trial | `bin/serve_run.sh`, with `serve_client.py` |
| Inductor / `torch.compile` kernels | `inductor_capsule/`, replay with `python3 -m nan_doctor_rt.replay_inductor_kernel` |
| One saved graph piece | `python3 -m nan_doctor_rt.replay_graph_piece` |
| Uninitialized-memory check | `python3 -m nan_doctor_rt.alloc_poison` |

`nd`, Slurm site profiles, and Apptainer launchers are not part of this package.

### Twin references (wrong finite values)

A NaN is not the only failure: an op can return a finite but wrong value. With
`NAN_CAPTURE_TWIN_EVERY=N` in `docker_env_vars`, the capture recomputes every Nth call of an op that has a
reference ("twin": the framework's own PyTorch path, an alternative kernel, or an fp64 recompute of an
Inductor kernel) and captures the first call whose output differs by more than the dtype's floor. The
capsule's origin is `wrong_value` and its verify kind `twin`; `NAN_CAPTURE_TWIN_OPS=<op,...>` limits the
ops. It is off unless the variable is set. `tests/unit/nan_doctor/gpu/gpu_twins_capture.py <work dir>`
shows it on a planted wrong-value op (one GPU, run in any PyTorch image with the repo mounted).

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
Without a co-tenant the ladder first runs one copy alone on GPU 0 (`--deterministic-probe`, 100 runs):
a fault that hits every run there is deterministic and ends the ladder `REPRODUCED at concurrency 1`.
Runs in the first `--futility-grace` seconds of an attempt (180 with a co-tenant, which loads the
device before it computes) do not count toward the futility stop.
The docker flags (`--ipc=host`, `--shm-size`, memlock, seccomp) shape the timing being replayed and
are not tunable. A run over several GPUs also gets `--pid=host`, so the ranks of two concurrent
multi-GPU trials never share PIDs during RCCL setup.

## Differential mode

For silently wrong outputs (no NaN, no signal line), the same model runs once per arm with
`nan_doctor_rt/diff_tap` recording what it returned, and `diff` compares the arms:

```bash
madengine nan arms arms --framework vllm --format context --out-dir nd_diff |
while IFS=$'\t' read -r arm ctx; do madengine run --tags <model> --additional-context "$ctx" </dev/null; done
madengine nan diff --a nd_diff/A/outputs.jsonl --a2 nd_diff/A2/outputs.jsonl --ref nd_diff/REF/outputs.jsonl \
    --arm attention=nd_diff/flip_attention/outputs.jsonl --json nd_diff/diff.json
```

## Reproducer

`package` ships exactly what the reproducing attempt ran: whole chain, its concurrency and
iteration count, its co-tenant. `run.sh` needs docker, python3 and the image -- not madengine. A
reproducing run (`n/N > 0` near the original rate) from an extracted copy is the verification; a
`0/N` is a failed build. The README's operand-validation sentence is proposed from the replay and
marked `<!-- CHECK -->` for a person to confirm.

On SLURM, check that the reproducer is not tied to the node it was built on:

```bash
madengine nan verify-nodes script --package <extracted dir> --out art/verify_nodes --nodes 2 \
    --exclude <host> --submit                    # or sbatch art/verify_nodes/verify_nodes.slurm
madengine nan verify-nodes score art/verify_nodes   # after the job: NODES_REPRODUCED=k/K ...
```

## Limits

- PyTorch workloads on AMD Instinct MI300X or newer (gfx942, gfx950); other GPUs get a warning.
- Local Docker, and SLURM where madengine launches the workload. Models with their own multi-node
  launchers (`slurm_multi`) do not get the tool on their ranks. K8s is not supported yet.
- No GPU lock is taken here; callers that need one (MAD's `mad-run.sh`) hold it around the command.
