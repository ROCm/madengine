# Configuration Guide

Complete guide to configuring madengine for various use cases and environments.

## Configuration Methods

### 1. Inline JSON String

```bash
madengine run --tags model \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
```

### 2. Configuration File

```bash
madengine run --tags model --additional-context-file config.json
```

**config.json:**
```json
{
  "gpu_vendor": "AMD",
  "guest_os": "UBUNTU"
}
```

## Default Configuration Values

madengine provides sensible defaults for common AMD/Ubuntu workflows:

| Field | Default Value | Customization |
|-------|---------------|---------------|
| `gpu_vendor` | `AMD` | Set to `NVIDIA` for NVIDIA GPUs |
| `guest_os` | `UBUNTU` | Set to `CENTOS` for CentOS containers |

### When Defaults Apply

Defaults are applied during the **build** command when fields are not explicitly provided:

```bash
# Uses defaults: {"gpu_vendor": "AMD", "guest_os": "UBUNTU"}
madengine build --tags model

# Explicit override
madengine build --tags model \
  --additional-context '{"gpu_vendor": "NVIDIA", "guest_os": "CENTOS"}'
```

When defaults are applied, you'll see an informative message:

```
ℹ️  Using default values for build configuration:
   • gpu_vendor: AMD (default)
   • guest_os: UBUNTU (default)

💡 To customize, use --additional-context '{"gpu_vendor": "NVIDIA", "guest_os": "CENTOS"}'
```

### Partial Configuration

You can provide one field and let the other default:

```bash
# Override only gpu_vendor (guest_os defaults to UBUNTU)
madengine build --tags model \
  --additional-context '{"gpu_vendor": "NVIDIA"}'

# Override only guest_os (gpu_vendor defaults to AMD)
madengine build --tags model \
  --additional-context '{"guest_os": "CENTOS"}'
```

### Production Recommendations

For production deployments:
- ✅ **DO** explicitly specify all configuration values
- ✅ **DO** use configuration files for reproducibility
- ⚠️ **AVOID** relying on defaults in automated workflows

### Run Command Behavior

The **run** command does NOT require these values because it can detect GPU vendor at runtime.
Defaults only apply to the **build** command where Dockerfile selection requires them.

## Run phase: log error pattern scan

After a successful container run, madengine may scan the **run log file** for fixed substrings (for example `RuntimeError:`, `OutOfMemoryError`, `Traceback (most recent call last)`)—intended as a safety net when logs show obvious Python or OOM errors. If a match is found **and no valid performance metrics were extracted**, the run is marked `FAILURE`.

If valid performance metrics *were* extracted, a pattern match no longer fails the run: the log scan cannot distinguish madengine/framework diagnostics from a model's own generated stdout, so a generative model whose output happens to contain a banned substring (e.g. an LLM writing `"ValueError:"` in a code sample) no longer produces a false `FAILURE` (see ROCM-27774). The match is still printed (in yellow) for triage visibility.

Some suites (for example layer unit tests) intentionally print benign `RuntimeError:` text while pytest still passes, with no performance metrics to fall back on. In those cases you can **disable** the scan or **narrow** what counts as an error.

Keys can be set in `--additional-context` / `--additional-context-file`, or on the **model** entry in `models.json` (same keys). **Runtime context overrides the model** when both are set.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `log_error_pattern_scan` | bool or string/number (coerced) | `true` | If `false`, skip substring-based log failure detection entirely (rely on exit codes and other signals). |
| `log_error_benign_patterns` | array of strings | `[]` | Extra lines to **exclude** before matching (appended to built-in exclusions such as ROCProf/metrics noise). Model list is merged first, then context list. |
| `log_error_patterns` | array of strings (non-empty) | (built-in list) | If set, **replaces** the default pattern list. Use only when you need a custom allowlist of failure substrings. |

**Example — disable scan for a tag (pytest is authoritative):**

```bash
madengine run --tags my_unit_test_suite \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU", "log_error_pattern_scan": false}'
```

**Example — extra benign substrings (prefer stable strings from real logs):**

```json
{
  "gpu_vendor": "AMD",
  "guest_os": "UBUNTU",
  "log_error_benign_patterns": [
    "expected benign fragment from workload log"
  ]
}
```

Disabling the scan does **not** change performance metric extraction from the log; it only affects the post-hoc grep used to set `has_errors` for status.

## System environment collection (rocEnvTool)

Before each container run, madengine appends `scripts/common/pre_scripts/run_rocenv_tool.sh` to the model's pre-scripts. It captures a CSV snapshot of the container's environment (OS, CPU, GPU, ROCm/CUDA, packages, env vars, NUMA). The output ends up at `<model_name>_env.csv` in the workspace root.

Two collection modes, selected via `rocenv_mode` in `--additional-context`:

| `rocenv_mode` | Sections collected | In-container tool requirements |
|---|---|---|
| `"lite"` (default) | OS, CPU, GPU, memory, ROCm/CUDA info, packages, env vars, NUMA balancing | Already present in standard ROCm/CUDA base images |
| `"full"` | Lite **plus** `hardware_information` (lshw), `bios_settings` (dmidecode), `dmsg_gpu_drm_atom_logs` (dmesg), `amdgpu_modinfo` (modinfo) | Auto-installed best-effort if missing |

**Full-mode auto-install** uses the `guest_os`-native package manager:

- `UBUNTU` → `apt-get install -y -qq lshw dmidecode kmod util-linux`
- `CENTOS` → `microdnf` / `dnf` / `yum install -y …` (first one found)

The install is **best-effort**: if the package manager is missing, the network is unreachable, or the container lacks permissions, the pre-script logs a warning and continues — the affected sections are simply omitted from the CSV. Empty tool output (for example, `dmesg` in a container without `CAP_SYSLOG`, or `dmidecode` without `/dev/mem` access) is also handled gracefully — the section is skipped and the rest of the CSV still parses.

**Examples:**

```bash
# Default (lite): low overhead, works in any ROCm/CUDA base image
madengine run --tags model

# Full: include hardware/BIOS/dmesg/amdgpu modinfo
madengine run --tags model \
  --additional-context '{"rocenv_mode": "full"}'

# CentOS guest with full mode (auto-install uses microdnf/dnf/yum)
madengine run --tags model \
  --additional-context '{"guest_os": "CENTOS", "rocenv_mode": "full"}'

# Skip system-env collection entirely (e.g. GPU-binding smoke tests)
madengine run --tags model \
  --additional-context '{"generate_sys_env_details": false}'
```

`guest_os` from `--additional-context` (default `UBUNTU`) controls which package manager is used. The same value is also exported into the container as `MAD_GUEST_OS`, so the pre-script picks the correct package manager without re-detecting from `/etc/os-release`.

Unknown `rocenv_mode` values fall back to `lite` with a warning.

## Basic Configuration

**gpu_vendor** (case-insensitive):
- `"AMD"` - AMD ROCm GPUs
- `"NVIDIA"` - NVIDIA CUDA GPUs

**guest_os** (case-insensitive):
- `"UBUNTU"` - Ubuntu Linux
- `"CENTOS"` - CentOS Linux

### ROCm path (run only)

**Host** (where `madengine` runs validation): by default, the ROCm root is **auto-detected** (traditional `/opt/rocm`, [TheRock](https://github.com/ROCm/TheRock) `rocm-sdk` / manifest layout, or `ROCM_PATH`-like env hints). Set `MAD_AUTO_ROCM_PATH=0` to skip auto and use only legacy resolution (`ROCM_PATH` then `/opt/rocm`).

**Overrides** (recommended for CI):

- **Additional context (host):** top-level `"MAD_ROCM_PATH": "/path/to/host/rocm"` — controls where madengine looks for host GPU tools (`rocminfo`, `amd-smi`, etc.).
- **Additional context (container):** `"docker_env_vars": { "ROCM_PATH": "/path/inside/image" }` — sets the in-container `ROCM_PATH` for Docker runs. If omitted, at `run` time madengine uses the image OCI `Env` (`ROCM_PATH` / `ROCM_HOME`) if present, then an in-container probe, then defaults to `/opt/rocm`. The host-resolved path is **not** mirrored into the container.

These two keys are independent, allowing host and container to use different ROCm installations without confusion.

Precedence (host): top-level `MAD_ROCM_PATH` → auto-detect (unless disabled) → `ROCM_PATH` → `/opt/rocm`.

Precedence (container, **local Docker `run`**, **AMD**): explicit `ROCM_PATH` in `docker_env_vars` → image OCI `Env` (`ROCM_PATH` / `ROCM_HOME`) → in-image probe → default `/opt/rocm` with a warning. Implemented in `ContainerRunner.run_container` after the run image is resolved.

This applies to the run phase; build uses build-only context (no GPU detection) but still honors `MAD_ROCM_PATH` in context when set.

At the start of each container run, a **Run Phase Environment** table is printed showing host vs container installation type (`apt install` or `therock`), ROCm/CUDA root, and version side-by-side. See [Run phase environment table](usage.md#run-phase-environment-table).

## Build Configuration

### Batch Manifest

Use batch manifest files for selective builds with per-model configuration:

```bash
madengine build --batch-manifest batch.json \
  --registry my-registry.com \
  --additional-context-file config.json
```

**Batch manifest structure** (`batch.json`):

```json
[
  {
    "model_name": "model1",
    "build_new": true,
    "registry": "registry1.io",
    "registry_image": "namespace/model1"
  },
  {
    "model_name": "model2",
    "build_new": false,
    "registry": "registry2.io",
    "registry_image": "namespace/model2"
  }
]
```

**Fields:**
- `model_name` (string, required): Model tag to include
- `build_new` (boolean, optional, default: `false`): Whether to build this model
  - `true`: Build the model from source
  - `false`: Reference existing image without rebuilding
- `registry` (string, optional): Per-model registry override
- `registry_image` (string, optional): Custom registry image name/namespace

**Key Behaviors:**
- Only models with `"build_new": true` are built
- Models with `"build_new": false` are included in output manifest without building
- Per-model `registry` overrides the global `--registry` flag
- Cannot use `--batch-manifest` and `--tags` together (mutually exclusive)

**Use Case - CI/CD Incremental Builds:**

```json
[
  {"model_name": "changed_model", "build_new": true},
  {"model_name": "stable_model1", "build_new": false},
  {"model_name": "stable_model2", "build_new": false}
]
```

This allows you to rebuild only changed models while maintaining references to existing stable images in a single manifest.

## Docker Configuration

### Environment Variables

Pass environment variables to containers:

```json
{
  "docker_env_vars": {
    "HSA_ENABLE_SDMA": "0",
    "PYTORCH_TUNABLEOP_ENABLED": "1",
    "NCCL_DEBUG": "INFO"
  }
}
```

### Pre-Built Container Images

`MAD_CONTAINER_IMAGE` runs the model in an image you already have, **skipping the
build phase entirely**. madengine validates the image exists locally (pulling it
if not) and writes a synthetic `build_manifest.json` for it:

```bash
madengine run --tags my_model \
  --additional-context "{'MAD_CONTAINER_IMAGE': 'rocm/pytorch:custom-tag'}"
```

`--tags` is required in this mode — without it madengine has no models to map
onto the image and fails with a configuration error. Note this is an
`--additional-context` key, not an environment variable.

### Custom Base Image

To keep the normal build but change the image the Dockerfile's `FROM` line
resolves to, override `BASE_DOCKER` as a build argument:

```json
{
  "docker_build_arg": {
    "BASE_DOCKER": "rocm/pytorch:rocm6.1_ubuntu22.04_py3.10"
  }
}
```

### Build Arguments

Pass build-time variables:

```json
{
  "docker_build_arg": {
    "ROCM_VERSION": "6.1",
    "PYTHON_VERSION": "3.10",
    "CUSTOM_ARG": "value"
  }
}
```

### Mount Host Directories

Mount host directories inside containers:

```json
{
  "docker_mounts": {
    "/data-inside-container": "/data-on-host",
    "/models": "/home/user/models"
  }
}
```

### Select GPUs and CPUs

Specify GPU and CPU subsets:

```json
{
  "docker_gpus": "0,2-4,7",
  "docker_cpus": "0-15,32-47"
}
```

Format: Comma-separated list with hyphen ranges.

## Performance Configuration

### Timeout Settings

Set a per-model timeout (seconds) in `models.json`. Omit the field to get the
7200s (2 hour) default. Use `0` (or any non-positive value) to run the model
without a timeout:

```json
{
  "timeout": 7200
}
```

Or use the command-line option, which overrides the model's timeout:

```bash
madengine run --tags model --timeout 7200
```

Full precedence rules, including the `-1` sentinel and how the timeout is
applied on SLURM and Kubernetes, are in
[Usage — Custom Timeouts](usage.md#custom-timeouts).

### Local Data Mirroring

Force local data caching:

```json
{
  "mirrorlocal": "/tmp/local_mirror"
}
```

Or use command-line option:

```bash
madengine run --tags model --force-mirror-local /tmp/mirror
```

## Kubernetes Deployment

### Minimal Configuration

```json
{
  "k8s": {
    "gpu_count": 1
  }
}
```

Automatically applies (see presets under `src/madengine/deployment/presets/k8s/`):
- Namespace: `default`
- Resource limits based on GPU count
- Image pull policy: `Always` (base default)
- Service account: `default`
- GPU vendor detection from context
- `k8s.secrets` defaults (see below)

### Full Configuration

```json
{
  "k8s": {
    "gpu_count": 2,
    "namespace": "ml-team",
    "gpu_vendor": "AMD",
    "memory": "32Gi",
    "memory_limit": "64Gi",
    "cpu": "16",
    "cpu_limit": "32",
    "image_pull_policy": "Always",
    "ttl_seconds_after_finished": null,
    "allow_privileged_profiling": null,
    "secrets": {
      "strategy": "from_local_credentials",
      "image_pull_secret_names": ["my-registry-secret"],
      "runtime_secret_name": null
    }
  }
}
```

**K8s Options:**
- `gpu_count` - Number of GPUs (required)
- `namespace` - Kubernetes namespace (default: `default`)
- `gpu_vendor` - GPU vendor override (auto-detected from context)
- `memory` - Memory request (default: auto-scaled by GPU count)
- `memory_limit` - Memory limit (default: 2× memory request)
- `cpu` - CPU cores request (default: auto-scaled by GPU count)
- `cpu_limit` - CPU cores limit (default: 2× CPU request)
- `image_pull_policy` - `Always`, `IfNotPresent`, or `Never`
- `ttl_seconds_after_finished` - Optional Job TTL in seconds (auto-delete finished Job); `null` to omit
- `allow_privileged_profiling` - `null` means enable elevated `securityContext` when tools/profiling are configured; `true`/`false` to force
- `secrets.strategy` - `from_local_credentials` (default): create `Secret` objects from local `credential.json` at deploy time; `existing`: only reference pre-created Secrets; `omit`: no runtime Secret from client
- `secrets.image_pull_secret_names` - Extra pull secret names (strings) merged with any created from `credential.json` when using `from_local_credentials`
- `secrets.runtime_secret_name` - Required for `existing` (pre-created opaque Secret with key `credential.json`); optional for `omit` if you still mount a runtime Secret

**Cluster and scheduling keys:**
- `kubeconfig` - Path to kubeconfig (default: `~/.kube/config`)
- `gpu_resource_name` - GPU resource name (default: `amd.com/gpu`; use `nvidia.com/gpu` for NVIDIA)
- `node_selector` - Label selectors for pod placement (default: `{}`)
- `tolerations` - Tolerations for tainted nodes (default: `[]`)
- `backoff_limit` - Job retry attempts before marking failed (default: `3`)
- `output_dir` - Directory for generated manifests (default: `./k8s_manifests`)

**Storage keys:**
- `data_pvc` - Name of an existing PVC to use for data, skipping auto-creation
- `storage_class` - Broad fallback StorageClass for both the shared-data PVC and the single-node results PVC
- `nfs_storage_class` / `data_storage_class` - RWX class for shared data and multi-node results
- `single_node_results_storage_class` / `multi_node_results_storage_class` - Fine-grained results-PVC overrides (`local_path_storage_class` is the legacy single-node fallback)
- `results_storage_size` / `data_storage_size` - PVC sizes (defaults: `10Gi` / `100Gi`)
- `recreate_shared_data_pvc` - Delete and recreate `madengine-shared-data` before use. **Destroys existing data** — back up first; intended for migrating an RWO PVC to RWX

Multi-node jobs require an RWX results StorageClass; madengine warns when one is
not set. The full key reference, PVC access-mode matrix, and preset defaults are
in [examples/k8s-configs/README.md](../examples/k8s-configs/README.md).

### Multi-Node Kubernetes

```json
{
  "k8s": {
    "gpu_count": 8
  },
  "distributed": {
    "launcher": "torchrun",
    "nnodes": 2,
    "nproc_per_node": 4
  }
}
```

## SLURM Deployment

### Basic Configuration

```json
{
  "slurm": {
    "partition": "gpu",
    "gpus_per_node": 4,
    "time": "02:00:00"
  }
}
```

### Full Configuration

```json
{
  "slurm": {
    "partition": "gpu",
    "account": "research_group",
    "qos": "normal",
    "gpus_per_node": 8,
    "nodes": 2,
    "nodelist": "node01,node02",
    "time": "24:00:00",
    "exclusive": true,
    "constraint": "mi300x",
    "exclude": "node07,node09",
    "modules": ["rocm/6.2"],
    "network_interface": "ib0",
    "output_dir": "./slurm_results"
  }
}
```

**Note:** `nodelist` is optional; omit it to let SLURM choose nodes. When set, the job runs only on the listed nodes and node health preflight is skipped.

**SLURM Options:**
- `partition` - SLURM partition name (required)
- `account` - Billing account
- `qos` - Quality of Service
- `gpus_per_node` - GPUs per node (default: 8)
- `nodes` - Number of nodes (default: 1)
- `nodelist` - Comma-separated node names to run on (e.g. `"node01,node02"`); when set, job is restricted to these nodes and automatic node health preflight is skipped
- `reservation` - SLURM reservation name; forwarded to srun health/cleanup commands and SBATCH directives
- `exclusive` - Exclusive node access (default: `true`)
- `time` - Wall time limit HH:MM:SS (default: `24:00:00`)
- `constraint` - SBATCH `--constraint` feature expression
- `exclude` - Comma-separated nodes to exclude; node health preflight appends to this list
- `modules` - Array of environment modules to `module load` in the job (default: `[]`)
- `network_interface` - Interface exported as `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` (e.g. `ib0`)
- `output_dir` - Directory for SLURM `.out`/`.err` files (default: `./slurm_results`)
- `skip_gpus_directive` - Omit the `#SBATCH --gpus-per-node` directive (default: `false`). Set `true` on clusters that expose no GPU GRES and reject any job script carrying it; allocation then relies on `exclusive` / `nproc_per_node`.

Node health preflight, shared-storage, and results-collection keys (`enable_node_check`,
`auto_cleanup_nodes`, `allow_submit_without_clean_nodes`, `verbose_node_check`,
`shared_workspace`, `results_dir`) are documented in
[examples/slurm-configs/README.md](../examples/slurm-configs/README.md).

### Multi-Node SLURM

```json
{
  "slurm": {
    "partition": "gpu",
    "nodes": 4,
    "gpus_per_node": 8,
    "time": "48:00:00"
  },
  "distributed": {
    "launcher": "torchrun",
    "nnodes": 4,
    "nproc_per_node": 8
  }
}
```

## Distributed Training

### Launcher Configuration

```json
{
  "distributed": {
    "launcher": "torchrun",
    "nnodes": 2,
    "nproc_per_node": 4,
    "master_port": 29500
  }
}
```

**Launcher Options:**
- `launcher` - Framework name (required)
- `nnodes` - Number of nodes
- `nproc_per_node` - Processes/GPUs per node
- `master_port` - Master communication port (default: 29500)

**Supported Launchers:**
- `torchrun` - PyTorch DDP/FSDP
- `deepspeed` - ZeRO optimization
- `megatron-lm` - Large transformers (K8s + SLURM)
- `torchtitan` - LLM pre-training
- `primus` - Primus unified pretrain
- `vllm` - LLM inference
- `sglang` - Structured generation
- `sglang-disagg` - Disaggregated SGLang
- `slurm_multi` / `slurm-multi` - Self-managed multi-container topologies (SLURM only)

See [Launchers Guide](launchers.md) for details.

### TorchTitan Configuration

```json
{
  "distributed": {
    "launcher": "torchtitan",
    "nnodes": 4,
    "nproc_per_node": 8
  },
  "env_vars": {
    "TORCHTITAN_TENSOR_PARALLEL_SIZE": "8",
    "TORCHTITAN_PIPELINE_PARALLEL_SIZE": "4",
    "TORCHTITAN_FSDP_ENABLED": "1"
  }
}
```

### vLLM Configuration

```json
{
  "distributed": {
    "launcher": "vllm",
    "nnodes": 2,
    "nproc_per_node": 4
  }
}
```

`nproc_per_node` is exported into the container as `VLLM_TENSOR_PARALLEL_SIZE`; there is no separate `vllm.*` config block for tensor/pipeline parallel sizing.

## Profiling Configuration

### Basic Profiling

```json
{
  "tools": [
    {"name": "rocprof"}
  ]
}
```

### Custom Tool Configuration

```json
{
  "tools": [
    {
      "name": "rocprof",
      "cmd": "rocprof --timestamp on",
      "env_vars": {
        "NCCL_DEBUG": "INFO"
      }
    }
  ]
}
```

### Multiple Tools (Stackable)

```json
{
  "tools": [
    {"name": "rocprof"},
    {"name": "miopen_trace"},
    {"name": "rocblas_trace"}
  ]
}
```

**Available Tools:**
- `rocprof` - GPU profiling
- `rpd` - ROCm Profiler Data
- `rocblas_trace` - rocBLAS library tracing
- `miopen_trace` - MIOpen library tracing
- `tensile_trace` - Tensile library tracing
- `rccl_trace` - RCCL communication tracing
- `gpu_info_power_profiler` - Power consumption profiling
- `gpu_info_vram_profiler` - VRAM usage profiling

See [Profiling Guide](profiling.md) for details.

## Pre/Post Execution Scripts

Run scripts before and after model execution:

```json
{
  "pre_scripts": [
    {
      "path": "scripts/common/pre_scripts/setup.sh",
      "args": "-v"
    }
  ],
  "encapsulate_script": "scripts/common/wrapper.sh",
  "post_scripts": [
    {
      "path": "scripts/common/post_scripts/cleanup.sh",
      "args": "-r"
    }
  ]
}
```

## Model Arguments

Pass arguments to model execution script:

```json
{
  "model_args": "--model_name_or_path bigscience/bloom --batch_size 32"
}
```

## Data Provider Configuration

Configure in `data.json` (MAD package root):

```json
{
  "model_data": {
    "nas": {"path": "/home/datum", "mirrorlocal": "/tmp/local_mirror"},
    "minio": {"path": "s3://datasets/datum"},
    "aws": {"path": "s3://datasets/datum"}
  }
}
```

## Credential Configuration

Configure in `credential.json` (MAD package root):

```json
{
  "dockerhub": {
    "username": "your_username",
    "password": "your_token",
    "repository": "myorg"
  },
  "PUBLIC_GITHUB_ROCM_KEY": {
    "username": "github_username",
    "token": "github_token"
  },
  "MAD_AWS_S3": {
    "USERNAME": "aws_access_key",
    "PASSWORD": "aws_secret_key"
  }
}
```

### Environment Variable Override

```bash
export MAD_DOCKERHUB_USER=myusername
export MAD_DOCKERHUB_PASSWORD=mytoken
export MAD_DOCKERHUB_REPO=myorg
```

### Registry Authentication

madengine reuses an existing `docker login` — including an organization access
token (OAT) — rather than requiring credentials to be duplicated into
`credential.json`. Ambient credentials are read from
`${DOCKER_CONFIG:-~/.docker}/config.json` exactly as the Docker CLI reads them,
covering `auths` entries, `credHelpers`, and `credsStore`.

Blank values are treated as **not configured**, not as credentials. A placeholder
entry such as `{"username": "", "password": ""}` will never override or break a
working `docker login`.

| Existing `docker login` | Credentials in `credential.json` / env | Behavior |
|---|---|---|
| no  | yes (non-blank) | `docker login` with the configured credentials |
| yes | yes (non-blank) | `docker login` with the configured credentials (explicit wins) |
| yes | absent or blank | Reuse the existing login; no `docker login` is run |
| no  | absent or blank | Push fails with an actionable error; pull warns and continues |

Before `docker build`, madengine logs in to the base image's registry only when
that registry has no existing login and usable credentials are configured, so a
node authenticated with an OAT is never re-authenticated.

Relevant environment variables:

- `DOCKER_CONFIG` — directory holding `config.json` (default `~/.docker`)
- `MAD_SKIP_DOCKER_LOGIN=1` — never run `docker login`; always defer to the
  credentials the machine already has

If a base image pull is denied, madengine distinguishes the two causes:
credentials were rejected (authentication — supply credentials or run
`docker login`), versus credentials were accepted but the registry granted no
pull scope for that repository (`insufficient_scope` — an authorization problem,
where the access token needs to be scoped to the repository and re-running
`docker login` will not help).

## Configuration Priority

For Kubernetes/SLURM deployments:
1. CLI overrides (`--additional-context`) - Highest
2. User config file (`--additional-context-file`)
3. Profile presets (single-gpu/multi-gpu/multi-node)
4. GPU vendor presets (AMD/NVIDIA optimizations)
5. Base defaults (k8s/defaults.json)
6. Environment variables
7. Built-in fallbacks - Lowest

## Complete Examples

### Local GPU Development

```json
{
  "gpu_vendor": "AMD",
  "guest_os": "UBUNTU",
  "docker_gpus": "0",
  "docker_env_vars": {
    "PYTORCH_TUNABLEOP_ENABLED": "1"
  }
}
```

### Kubernetes Single-GPU

```json
{
  "k8s": {
    "gpu_count": 1,
    "namespace": "dev"
  }
}
```

### Kubernetes Multi-GPU Training

```json
{
  "k8s": {
    "gpu_count": 4,
    "memory": "64Gi",
    "cpu": "32"
  },
  "distributed": {
    "launcher": "torchrun",
    "nnodes": 1,
    "nproc_per_node": 4
  }
}
```

### SLURM Multi-Node

```json
{
  "slurm": {
    "partition": "gpu",
    "nodes": 8,
    "gpus_per_node": 8,
    "time": "72:00:00",
    "account": "research_proj"
  },
  "distributed": {
    "launcher": "deepspeed",
    "nnodes": 8,
    "nproc_per_node": 8
  }
}
```

### Production with Profiling

```json
{
  "k8s": {
    "gpu_count": 2,
    "namespace": "production",
    "memory": "32Gi"
  },
  "tools": [
    {"name": "rocprof"},
    {"name": "gpu_info_power_profiler"}
  ],
  "docker_env_vars": {
    "NCCL_DEBUG": "INFO",
    "PYTORCH_TUNABLEOP_ENABLED": "1"
  }
}
```

## Troubleshooting

### Configuration Not Applied

```bash
# Verify configuration is valid JSON
python -m json.tool config.json

# Use verbose logging
madengine run --tags model \
  --additional-context-file config.json \
  --verbose
```

### Environment Variables Not Set

```bash
# Check environment variables
env | grep MAD

# Verify Docker receives env vars
docker inspect container_name | grep -A 10 Env
```

### GPU Vendor Auto-Detection

madengine auto-detects GPU vendor if not specified:
- Looks for ROCm drivers → AMD
- Looks for CUDA drivers → NVIDIA
- Falls back to configuration or fails

Override with explicit configuration:

```json
{
  "gpu_vendor": "AMD"
}
```

## Best Practices

1. **Use configuration files** for complex settings
2. **Start with minimal configs** and add as needed
3. **Validate JSON syntax** before running
4. **Use environment variables** for sensitive data
5. **Test locally first** before deploying
6. **Enable verbose logging** when debugging
7. **Document custom configurations** for team use

## Next Steps

- [Usage Guide](usage.md) - Using madengine commands
- [Deployment Guide](deployment.md) - Deploy to clusters
- [Profiling Guide](profiling.md) - Performance analysis
- [Launchers Guide](launchers.md) - Distributed training frameworks

