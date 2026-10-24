# Bare-Metal Execution

Run a model on this machine inside a conda prefix. There is no Docker build and no container. The prefix takes the place of the image: one prefix per model, created during build and entered with `conda run -p` during the run.

Bare metal is single-node and local. It does not submit to SLURM, Spur, Kubernetes, or llm-d.

## Select it

`platform` is a string on `--additional-context`. Omitted, it stays Docker. `"bare_metal"` selects conda. Conda settings live in a `conda` object.

```bash
madengine run --tags my_model --live-output \
  --additional-context '{
    "platform": "bare_metal",
    "conda": {
      "conda_bin": "/path/to/micromamba"
    }
  }'
```

The same object works with `madengine build`. A run that finds no manifest builds the conda prefix first, then executes.

These forms are rejected before any env is created:

| Form | Result |
|------|--------|
| `"platform": "bare_metal"` plus a `conda` object | Bare-metal run |
| `platform` omitted | Docker, as before |
| A top-level `bare_metal` object | Rejected. That key is retired. Use the string above |
| A `conda` object while `platform` is `docker` or omitted | Rejected |
| `"platform": "bare_metal"` together with `slurm`, `k8s`, `kubernetes`, or `llm_d` | Rejected |
| Any other `platform` string | Rejected |

`--config platform=bare_metal` is still rejected. The Hydra `platform` group only accepts `docker`. Pass bare metal through `--additional-context` or `--additional-context-file`.

## Prerequisites

- A conda, mamba, or micromamba executable already on `PATH`, or `conda.conda_bin` pointing at one. madengine does not download a conda implementation.
- A working GPU driver and runtime on the host. The build validates the installation before it solves an environment, because the workload runs on this machine.
- The model script and its dependency files (`requirements.txt`, `environment.yml`, `setup.sh`) on the host. `MODEL_DIR` is copied the same way as a Docker run.

Search order when `conda_bin` is unset: `micromamba`, then `mamba`, then `conda`.

## What build and run do

**Build** discovers the model, creates or updates the prefix, installs the card's dependencies, and writes `build_manifest.json`. The manifest entry has an empty `docker_image`, `bare_metal: true`, and the card fields the run needs, including `distributed`.

**Run** reads that manifest, wraps the model script in `conda run -p <prefix>`, and writes `perf.csv`. micromamba streams by default. conda and mamba get `--no-capture-output` so the log is not swallowed.

Prefixes live under `~/.cache/madengine/envs/<name>`, not in the user's interactive conda root. A same-named env the user created by hand is not reused. The name is a single path segment: slashes and `..` are flattened so `conda_env` cannot escape the cache.

## Model card

Put the dependency recipe on the card. A `conda` block on the command line overrides the same field for that invocation.

```json
{
  "name": "pyt_train_example_bare_metal",
  "scripts": "scripts/example/train_bare_metal.sh",
  "n_gpus": "-1",
  "conda_env": "mad_pyt_train_example_bare_metal",
  "python_version": "3.12",
  "requirements_file": "scripts/example/requirements.txt",
  "setup_script": "scripts/example/setup.sh",
  "distributed": {
    "launcher": "torchrun",
    "nnodes": 1
  }
}
```

| Card field | Role |
|------------|------|
| `conda_env` | Prefix name. Default is `mad_<model name>` with `/` replaced by `_` |
| `python_version` | Interpreter requested by `conda create` when there is no `environment_file` |
| `environment_file` | A conda env YAML. Used instead of `conda create` + `python_version` |
| `requirements_file` | `pip install -r` inside the prefix |
| `setup_script` | `bash` inside the prefix after requirements. Use it for an editable install or a wheel pin. A step that must run on every invocation belongs in `pre_scripts` |
| `rocm` | Optional ROCm pip wheels. See below |
| `distributed.launcher` | Recorded on the result row. The script starts the launcher |
| `n_gpus` | How many GPUs this run may see. `"-1"` means every GPU visible to the process |
| `scripts` | The script `conda run` executes |

Vendor-specific files follow the Dockerfile suffix rule. For `environment.yml` on an AMD host, `environment.amd.yml` is used when that file exists. The same applies to `requirements_file` and `setup_script`, with `.nvidia.` for NVIDIA.

## `conda` block

```json
{
  "platform": "bare_metal",
  "conda": {
    "conda_bin": "/home/user/.cache/madengine/micromamba/micromamba",
    "conda_env": "mad_my_model",
    "env_root": "/scratch/madengine/envs",
    "reuse_env": true,
    "cleanup_env": false
  }
}
```

| Key | Default | Meaning |
|-----|---------|---------|
| `conda_bin` | search `PATH` | conda, mamba, or micromamba executable |
| `conda_env` | card, then `mad_<name>` | Prefix name. The command-line value wins over the card |
| `env_root` | `~/.cache/madengine/envs` | Directory that holds prefixes |
| `reuse_env` | `true` | Keep an existing prefix when its dependency stamp still matches |
| `cleanup_env` | `false` | Remove the prefix after the run |
| `environment_file`, `requirements_file`, `python_version`, `setup_script`, `rocm` | card | Same fields as the card. A value here wins for this invocation |
| `gpu_vendor`, `gpu_arch` | detected host | Picks the vendor file suffix and the ROCm wheel index |

`reuse_env` does not skip a prefix whose inputs changed. The stamp is a hash of the environment file, requirements file, setup script, and the ROCm wheel inputs. A changed file forces create-or-update, pip, the ROCm install, and `setup_script` together. `reuse_env: false` rebuilds every time. The stamp lives in `~/.cache/madengine/envhash/`.

## GPUs and the launcher

There is no `docker run --device`. madengine exports `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` on AMD, or `CUDA_VISIBLE_DEVICES` on NVIDIA:

- `n_gpus: "-1"` leaves a mask the scheduler already exported, and sets `MAD_RUNTIME_NGPUS` to the number of devices in that mask, or to the detected system count when no mask is set.
- A positive `n_gpus` takes the first N devices from an existing mask. With no mask, it uses `0..N-1`.

`torchrun --nproc_per_node` rejects `-1`. Read `MAD_RUNTIME_NGPUS` in the model script and pass that count. madengine does not wrap the script in `torchrun`, `mpirun`, or `srun`. The script is the launcher.

The result row still names the workload launcher. Precedence is `launcher.type`, then `distributed.launcher` in the run context, then `distributed.launcher` on the card, then `MAD_LAUNCHER_TYPE` / `MAD_LAUNCHER`. An empty value, or `conda` / `docker` / `native`, is recorded as `native`.

| Column | Bare-metal value |
|--------|------------------|
| `platform` | `bare_metal` |
| `deployment_type` | `local` (this machine; the column stays the scheduler) |
| `launcher` | The workload launcher, or `native` |
| `docker_image` | empty |

`performance:` lines in the log and a `multiple_results` CSV are collected the same way as a local Docker run.

## ROCm wheels

Set `rocm.enabled` to pip-install a ROCm userspace, and optionally torch, into the prefix before `requirements_file`.

```json
"rocm": {
  "enabled": true,
  "index_url": "auto",
  "packages": ["rocm[libraries,devel]"],
  "torch": true
}
```

`index_url: "auto"` uses `https://rocm.nightlies.amd.com/v2/<gfx arch>/` from the detected architecture (`gfx942` on MI300X). Set `index_url` to a full URL when detection is not available. `torch: true` then installs `torch` and `torchvision` from that same index.

A requirements file that pins its own ROCm torch can leave `rocm.enabled` unset and install through `requirements_file` and `setup_script` instead.

## Pre-scripts and post-scripts

`pre_scripts` and `post_scripts` from `--additional-context` still run, inside the prefix. `scripts/common/` is copied in for the run and removed afterwards, same as Docker.
