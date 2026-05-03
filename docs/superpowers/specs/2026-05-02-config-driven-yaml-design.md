# Config-Driven YAML System for madengine

**Date:** 2026-05-02
**Status:** Draft
**Author:** Stephen Shao + Claude

## Overview

Add a `--config` CLI argument to madengine that accepts Hydra-based YAML configuration files with full CLI override support. This replaces the error-prone `--additional-context` JSON string approach with structured, composable YAML configs that can drive the entire workflow from a single file — model selection, deployment target, distributed training, profiling tools, and environment tuning.

## Goals

1. Single `--config` argument drives the full madengine workflow (build + run)
2. Hydra config groups for composable deployment configurations
3. CLI override support via dot-path syntax (`distributed.nnodes=4`)
4. Clean, readable YAML keys with a translator to internal format
5. Backward compatible: `--additional-context` still works and overrides `--config`
6. Extensible for future platforms (bare metal, Singularity, Podman)

## Non-Goals

- Replacing `models.json` or `data.json` with YAML (they remain as-is)
- Adding Hydra's `@hydra.main` decorator (Typer remains the CLI framework)
- Recipe configs (can be added later as a config group)

---

## Config Directory Structure

```
src/madengine/configs/
├── config.yaml                         # Root defaults + top-level settings
│
├── platform/                           # WHERE: execution platform
│   ├── docker.yaml                     #   Docker container (default)
│   ├── bare_metal.yaml                 #   Direct execution, no container (future)
│   ├── singularity.yaml                #   Singularity/Apptainer (future)
│   └── podman.yaml                     #   Podman container (future)
│
├── scheduler/                          # HOW: job scheduling
│   ├── local.yaml                      #   Direct execution on current host (default)
│   ├── slurm.yaml                      #   SLURM HPC cluster
│   └── k8s.yaml                        #   Kubernetes cluster
│
├── hardware/                           # WHAT GPU: vendor + runtime settings
│   ├── amd.yaml                        #   AMD ROCm (default) — vendor, guest_os,
│   │                                   #   device mounts, security opts, renderD
│   ├── nvidia.yaml                     #   NVIDIA CUDA — vendor, --gpus flag
│   └── cpu.yaml                        #   CPU-only — no GPU devices
│
├── launcher/                           # WHAT FRAMEWORK: distributed launcher
│   ├── none.yaml                       #   No distributed launcher (default)
│   ├── torchrun.yaml                   #   PyTorch torchrun
│   ├── deepspeed.yaml                  #   DeepSpeed
│   ├── megatron.yaml                   #   Megatron-LM
│   ├── vllm.yaml                       #   vLLM inference serving
│   ├── sglang.yaml                     #   SGLang inference serving
│   ├── sglang_disagg.yaml              #   SGLang disaggregated prefill/decode
│   ├── torchtitan.yaml                 #   TorchTitan
│   ├── primus.yaml                     #   Primus launcher
│   └── native.yaml                     #   Native distributed (manual setup)
│
├── profile/                            # OPTIONAL: hardware profiles (+profile=)
│   ├── mi300x_8gpu.yaml
│   ├── mi300x_single.yaml
│   ├── mi250x_4gpu.yaml
│   ├── h100_8gpu.yaml
│   └── a100_8gpu.yaml
│
├── env/                                # OPTIONAL: env var bundles (+env=)
│   ├── nccl_debug.yaml
│   ├── nccl_tuned.yaml
│   ├── infiniband.yaml
│   └── miopen_defaults.yaml
│
├── tools/                              # OPTIONAL: profiling tools (+tools=)
│   ├── rocprofv3_lightweight.yaml
│   ├── rocprofv3_comprehensive.yaml
│   ├── power_profiler.yaml
│   ├── vram_profiler.yaml
│   └── rocm_trace_lite.yaml
│
├── data/                               # OPTIONAL: data provider (+data=)
│   ├── local.yaml                      #   Local filesystem data
│   ├── s3.yaml                         #   AWS S3 data source
│   ├── minio.yaml                      #   MinIO object storage
│   └── nas.yaml                        #   NAS/NFS shared storage
│
└── build/                              # OPTIONAL: build settings (+build=)
    ├── default.yaml                    #   Default build settings
    ├── ci.yaml                         #   CI pipeline (no cache, strict)
    └── multi_arch.yaml                 #   Multi-architecture builds
```

**Note:** `platform/` config group stubs (bare_metal, singularity, podman) are created with placeholder content for future extensibility. In Phase 1, only `docker` is functional — the others raise a `ConfigurationError("platform '{name}' is not yet supported")` if selected.

### Config Group Types

| Group | Type | Hydra Syntax | Purpose |
|-------|------|-------------|---------|
| `platform` | Default | `platform=docker` | Execution platform |
| `scheduler` | Default | `scheduler=slurm` | Job scheduler |
| `hardware` | Default | `hardware=amd` | GPU vendor + runtime |
| `launcher` | Default | `launcher=torchrun` | Distributed launcher |
| `profile` | Append-only | `+profile=mi300x_8gpu` | Hardware presets |
| `env` | Append-only | `+env=nccl_tuned` | Env var bundles |
| `tools` | Append-only | `+tools=rocprofv3_lightweight` | Profiling tools |
| `data` | Append-only | `+data=local` | Data provider |
| `build` | Append-only | `+build=ci` | Build settings |

Default groups: exactly one option is selected; changing it replaces the previous selection.
Append-only groups: added on top of existing config via `+` prefix; composable.

---

## YAML Schema

### Root Config (`config.yaml`)

```yaml
defaults:
  - platform: docker
  - scheduler: local
  - hardware: amd
  - launcher: none
  - _self_

# Model selection
model:
  tags: []                         # Model tags to build+run (equivalent to --tags)
  manifest_file: null              # Use existing manifest (equivalent to --manifest-file)
  container_image: null            # Skip build, use image (equivalent to MAD_CONTAINER_IMAGE)
  skip_run: false                  # Build only (equivalent to --skip-model-run)
  timeout: null                    # Run timeout in seconds

# Docker / container settings
docker:
  build_args: {}                   # --build-arg flags
  env_vars: {}                     # --env flags for docker run
  mounts: {}                       # -v host:container volume mounts
  gpus: null                       # GPU device range (auto-detected if null)
  cpus: null                       # CPU affinity (--cpuset-cpus)
  additional_run_options: null     # Extra docker run flags
  keep_alive: false                # Keep containers after run
  clean_cache: false               # Rebuild without cache

# Build settings
build:
  registry: null                   # Docker registry URL
  target_archs: []                 # Target GPU architectures for multi-arch
  manifest_output: build_manifest.json

# Environment variables (passed to container/job — separate from docker.env_vars)
env_vars: {}

# Runtime behavior
debug: false
live_output: false

# Error scanning
log_error:
  pattern_scan: true
  benign_patterns: []
  patterns: []

# Scripts
tools: []
pre_scripts: []
post_scripts: []
encapsulate_script: null

# Data
data_config: data.json

# Output
output: perf.csv
summary_output: null
```

### Scheduler Configs

**`scheduler/local.yaml`:**
```yaml
# @package _global_
# Local execution — no scheduler-specific config needed
```

**`scheduler/slurm.yaml`:**
```yaml
# @package _global_
slurm:
  partition: amd-rccl
  nodes: 1
  gpus_per_node: 8
  time: "24:00:00"
  output_dir: ./slurm_results
  exclusive: true
  modules: []
  account: null
  qos: null
  constraint: null
  nodelist: null
  exclude: null
  results_dir: null
  shared_workspace: null
  network_interface: null

env_vars:
  OMP_NUM_THREADS: "8"
  MIOPEN_FIND_MODE: "1"
```

**`scheduler/k8s.yaml`:**
```yaml
# @package _global_
k8s:
  kubeconfig: ~/.kube/config
  namespace: default
  image_pull_policy: Always
  backoff_limit: 3
  ttl_seconds_after_finished: null
  allow_privileged_profiling: null
  gpu_count: null
  gpu_resource_name: amd.com/gpu
  memory: null
  memory_limit: null
  cpu: null
  cpu_limit: null
  host_ipc: true
  node_selector: {}
  tolerations: []
  nfs_storage_class: nfs-banff
  local_path_storage_class: local-path
  data_storage_class: nfs-banff
  recreate_shared_data_pvc: false
  results_pvc: null
  data_pvc: null
  output_dir: null
  secrets:
    strategy: from_local_credentials
    image_pull_secret_names: []
    runtime_secret_name: null

env_vars:
  OMP_NUM_THREADS: "8"
```

### Hardware Configs

**`hardware/amd.yaml`:**
```yaml
# @package _global_
gpu_vendor: AMD
guest_os: UBUNTU

runtime:
  devices:
    - /dev/kfd
    - /dev/dri
    - /dev/infiniband
  capabilities:
    - SYS_PTRACE
  security_opts:
    - seccomp=unconfined
  network_mode: host
  ipc: host
  groups:
    - video
  use_gpu_flag: false
```

**`hardware/nvidia.yaml`:**
```yaml
# @package _global_
gpu_vendor: NVIDIA
guest_os: UBUNTU

runtime:
  devices: []
  capabilities: []
  security_opts: []
  network_mode: host
  ipc: host
  groups: []
  use_gpu_flag: true
```

**`hardware/cpu.yaml`:**
```yaml
# @package _global_
gpu_vendor: null
guest_os: UBUNTU

runtime:
  devices: []
  capabilities: []
  security_opts: []
  network_mode: null
  ipc: null
  groups: []
  use_gpu_flag: false
```

### Launcher Configs

**`launcher/none.yaml`:**
```yaml
# @package _global_
distributed:
  enabled: false
```

**`launcher/torchrun.yaml`:**
```yaml
# @package _global_
distributed:
  enabled: true
  launcher: torchrun
  backend: nccl
  nnodes: 1
  nproc_per_node: 8
  master_port: 29500
  port: 29500
```

**`launcher/vllm.yaml`:**
```yaml
# @package _global_
distributed:
  enabled: true
  launcher: vllm
  nnodes: 1
  nproc_per_node: 4

vllm:
  kv_cache_size: 0.7
  max_model_len: null
  tensor_parallel_size: null
```

**`launcher/sglang_disagg.yaml`:**
```yaml
# @package _global_
distributed:
  enabled: true
  launcher: sglang-disagg
  backend: nccl
  nnodes: 3
  nproc_per_node: 8
  port: 29500

sglang_disagg:
  prefill_nodes: null
  decode_nodes: null
  transfer_backend: mooncake
```

**`launcher/deepspeed.yaml`:**
```yaml
# @package _global_
distributed:
  enabled: true
  launcher: deepspeed
  backend: nccl
  nnodes: 1
  nproc_per_node: 8
  master_port: 29500
```

**`launcher/megatron.yaml`:**
```yaml
# @package _global_
distributed:
  enabled: true
  launcher: torchrun
  backend: nccl
  nnodes: 1
  nproc_per_node: 8
  master_port: 29500
```

### Profile Configs (append-only)

**`profile/mi300x_8gpu.yaml`:**
```yaml
# @package _global_
# Use: +profile=mi300x_8gpu
# Note: profile keys use gpu_* prefix to avoid collision with hardware/ config group
gpu_type: mi300x
gpu_memory_gb: 192
gpus_per_node: 8

distributed:
  nproc_per_node: 8

env_vars:
  GPU_MAX_HW_QUEUES: "2"
  HSA_ENABLE_SDMA: "0"
  HSA_FORCE_FINE_GRAIN_PCIE: "1"
```

### Env Configs (append-only)

**`env/infiniband.yaml`:**
```yaml
# @package _global_
# Use: +env=infiniband
env_vars:
  NCCL_IB_DISABLE: "0"
  NCCL_IB_HCA: "mlx5_0:1,mlx5_1:1"
  NCCL_SOCKET_IFNAME: ib0
  NCCL_NET_GDR_LEVEL: 3
```

**`env/nccl_debug.yaml`:**
```yaml
# @package _global_
# Use: +env=nccl_debug
env_vars:
  NCCL_DEBUG: INFO
  NCCL_DEBUG_SUBSYS: "INIT,NET,GRAPH"
  TORCH_DISTRIBUTED_DEBUG: DETAIL
```

### Tools Configs (append-only)

**`tools/rocprofv3_comprehensive.yaml`:**
```yaml
# @package _global_
# Use: +tools=rocprofv3_comprehensive
tools:
  - name: rocprofv3_full
    env_vars:
      RCCL_DEBUG: INFO
      HSA_ENABLE_SDMA: "0"
  - name: gpu_info_power_profiler
    env_vars:
      POWER_DEVICE: all
      POWER_SAMPLING_RATE: "0.1"
      POWER_DUAL_GCD: "false"
  - name: gpu_info_vram_profiler
    env_vars:
      VRAM_DEVICE: all
      VRAM_SAMPLING_RATE: "0.1"
  - name: miopen_trace
  - name: rocblas_trace
```

---

## Internal Architecture

### New Module: `src/madengine/config/`

```
src/madengine/config/
├── __init__.py              # Public API: load_config()
├── loader.py                # HydraConfigLoader: Compose API integration
├── translator.py            # Maps clean YAML keys → internal additional_context dict
└── schema.py                # Config validation
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Layer                                 │
│                                                                  │
│  --config file.yaml key=val    → config_args: List[str]         │
│  --tags llama3                 → tags: List[str]                │
│  --timeout 3600                → timeout: int                   │
│  --additional-context '{...}'  → additional_context: str        │
│                                                                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│               HydraConfigLoader.load(config_args)                │
│                                                                  │
│  1. Separate file path from Hydra overrides                     │
│  2. initialize_config_dir("pkg://madengine.configs")            │
│  3. compose(config_name="config", overrides=[...])              │
│  4. If user YAML file: OmegaConf.merge(cfg, user_cfg)          │
│  5. Return DictConfig                                           │
│                                                                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│            ConfigTranslator.to_additional_context(cfg)           │
│                                                                  │
│  Maps clean YAML keys to internal additional_context format:    │
│                                                                  │
│    YAML Key                    → Internal Key                   │
│    ─────────────────────────     ──────────────────────          │
│    docker.build_args           → docker_build_arg               │
│    docker.env_vars             → docker_env_vars                │
│    docker.mounts               → docker_mounts                  │
│    docker.gpus                 → docker_gpus                    │
│    docker.cpus                 → docker_cpus                    │
│    docker.additional_run_options → additional_docker_run_options │
│    model.container_image       → MAD_CONTAINER_IMAGE            │
│    log_error.pattern_scan      → log_error_pattern_scan         │
│    log_error.benign_patterns   → log_error_benign_patterns      │
│    log_error.patterns          → log_error_patterns             │
│    runtime.*                   → (Context runtime settings)     │
│                                                                  │
│  Passthrough keys (no translation):                             │
│    gpu_vendor, guest_os, env_vars, tools, pre_scripts,          │
│    post_scripts, encapsulate_script, debug, slurm, k8s,         │
│    distributed, vllm, sglang_disagg, shared_data                │
│                                                                  │
│  Extracted (not in additional_context):                          │
│    model.tags → returned separately for orchestrator            │
│    model.manifest_file → returned separately                    │
│    model.timeout → returned separately                          │
│    build.registry → returned separately                         │
│    build.target_archs → returned separately                     │
│                                                                  │
│  Returns: (additional_context: dict, metadata: dict)            │
│                                                                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Merge Layer                                    │
│                                                                  │
│  1. Start with translated config dict                           │
│  2. CLI args override equivalent config keys:                   │
│     --tags provided?    → overrides model.tags                  │
│     --timeout provided? → overrides model.timeout               │
│  3. --additional-context merged on top (highest priority)       │
│  4. Result = final additional_context dict                      │
│                                                                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              Existing Pipeline (unchanged)                        │
│                                                                  │
│  BuildOrchestrator(args) → Context(repr(merged_dict))           │
│  RunOrchestrator(args) → ContainerRunner / DeploymentFactory    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Merge Precedence (lowest → highest)

1. **Config group defaults** — `config.yaml` defaults list
2. **Selected config groups** — `scheduler=slurm`, `launcher=torchrun`
3. **Appended config groups** — `+profile=mi300x_8gpu`, `+env=nccl_tuned`
4. **User YAML file** — if `--config /path/to/file.yaml`
5. **Inline Hydra overrides** — `distributed.nnodes=4`
6. **CLI args** — `--tags`, `--timeout` (override equivalent config keys)
7. **`--additional-context`** — highest priority (backward compatibility)

### HydraConfigLoader

```python
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import importlib.resources

class HydraConfigLoader:
    """Loads madengine config using Hydra's Compose API."""

    @staticmethod
    def load(config_args: list[str]) -> DictConfig:
        """Load and compose config from Hydra overrides and/or user YAML.

        Args:
            config_args: Mix of Hydra overrides and optional user YAML path.
                Examples:
                  ["scheduler=slurm", "launcher=torchrun", "distributed.nnodes=4"]
                  ["/path/to/my_job.yaml"]
                  ["/path/to/my_job.yaml", "distributed.nnodes=8"]

        Returns:
            Composed DictConfig with all merges applied.
        """
        user_file, overrides = HydraConfigLoader._parse_args(config_args)

        # Resolve package config directory
        config_dir = str(
            importlib.resources.files("madengine") / "configs"
        )

        # Clear any previous Hydra state
        GlobalHydra.instance().clear()

        with initialize_config_dir(
            config_dir=config_dir, version_base=None
        ):
            cfg = compose(config_name="config", overrides=overrides)

        # Merge user file on top if provided
        if user_file:
            user_cfg = OmegaConf.load(user_file)
            OmegaConf.set_struct(cfg, False)
            cfg = OmegaConf.merge(cfg, user_cfg)

        return cfg

    @staticmethod
    def _parse_args(config_args: list[str]) -> tuple[str | None, list[str]]:
        """Separate user YAML file path from Hydra overrides."""
        user_file = None
        overrides = []
        for arg in config_args:
            if (
                arg.endswith(('.yaml', '.yml'))
                and '=' not in arg
                and not arg.startswith('+')
            ):
                if user_file:
                    raise ConfigurationError(
                        "Only one YAML config file allowed"
                    )
                user_file = arg
            else:
                overrides.append(arg)
        return user_file, overrides
```

### ConfigTranslator

```python
class ConfigTranslator:
    """Translates clean YAML config to internal additional_context format."""

    # YAML key → internal key mapping (only for keys that differ)
    KEY_MAP = {
        "docker.build_args": "docker_build_arg",
        "docker.env_vars": "docker_env_vars",
        "docker.mounts": "docker_mounts",
        "docker.gpus": "docker_gpus",
        "docker.cpus": "docker_cpus",
        "docker.additional_run_options": "additional_docker_run_options",
        "log_error.pattern_scan": "log_error_pattern_scan",
        "log_error.benign_patterns": "log_error_benign_patterns",
        "log_error.patterns": "log_error_patterns",
    }

    # Keys extracted from config (not part of additional_context)
    EXTRACTED_KEYS = {
        "model", "build", "platform", "output",
        "summary_output", "data_config", "live_output",
    }

    @classmethod
    def to_additional_context(
        cls, cfg: DictConfig
    ) -> tuple[dict, dict]:
        """Convert DictConfig to (additional_context, metadata) tuple.

        additional_context: dict in the format expected by existing pipeline.
        metadata: dict with model.tags, build.registry, etc. for the CLI layer.
        """
        raw = OmegaConf.to_container(cfg, resolve=True)

        context = {}
        metadata = {}

        for key, value in raw.items():
            if key in cls.EXTRACTED_KEYS:
                metadata[key] = value
            elif key == "docker":
                # Flatten docker.* to docker_* keys
                for subkey, subval in value.items():
                    internal_key = cls.KEY_MAP.get(
                        f"docker.{subkey}", f"docker_{subkey}"
                    )
                    if subval is not None:
                        context[internal_key] = subval
            elif key == "log_error":
                for subkey, subval in value.items():
                    internal_key = cls.KEY_MAP.get(
                        f"log_error.{subkey}", f"log_error_{subkey}"
                    )
                    context[internal_key] = subval
            elif key == "runtime":
                # Runtime settings stored separately, applied to Context
                metadata["runtime"] = value
            else:
                # Passthrough: gpu_vendor, guest_os, env_vars, slurm,
                # k8s, distributed, tools, pre_scripts, etc.
                if value is not None:
                    context[key] = value

        # Extract MAD_CONTAINER_IMAGE from model metadata
        model = metadata.get("model", {})
        if model and model.get("container_image"):
            context["MAD_CONTAINER_IMAGE"] = model["container_image"]

        return context, metadata
```

### Config Validation (`schema.py`)

```python
class ConfigValidator:
    """Validates composed config for consistency."""

    @staticmethod
    def validate(cfg: DictConfig) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []

        # Cross-field: scheduler=slurm must have slurm section
        scheduler = cfg.get("scheduler", {})
        # (Hydra handles this via config group selection)

        # Conflict: can't have both slurm and k8s
        if cfg.get("slurm") and cfg.get("k8s"):
            errors.append(
                "Cannot specify both 'slurm' and 'k8s' sections"
            )

        # Distributed: if enabled, must have launcher
        dist = cfg.get("distributed", {})
        if dist.get("enabled") and not dist.get("launcher"):
            errors.append(
                "distributed.enabled=true requires distributed.launcher"
            )

        # Type checks
        if dist.get("nnodes") is not None:
            if not isinstance(dist["nnodes"], int) or dist["nnodes"] < 1:
                errors.append("distributed.nnodes must be a positive integer")

        # Warn on unknown top-level keys
        known_keys = {
            "defaults", "platform", "scheduler", "hardware", "launcher",
            "model", "docker", "build", "env_vars", "debug", "live_output",
            "log_error", "tools", "pre_scripts", "post_scripts",
            "encapsulate_script", "data_config", "output", "summary_output",
            "gpu_vendor", "guest_os", "runtime", "slurm", "k8s",
            "kubernetes", "distributed", "vllm", "sglang_disagg",
            "shared_data", "timeout",
        }
        for key in cfg:
            if key not in known_keys:
                errors.append(f"Unknown config key: '{key}'")

        return errors
```

---

## CLI Integration

### Changes to `commands/run.py`

```python
def run(
    tags: Annotated[...] = [],
    # ... existing args ...
    config: Annotated[
        Optional[List[str]],
        typer.Option(
            "--config",
            help=(
                "YAML config file and/or Hydra overrides. "
                "Examples: --config my_job.yaml, "
                "--config scheduler=slurm launcher=torchrun, "
                "--config my_job.yaml distributed.nnodes=4"
            ),
        ),
    ] = None,
    additional_context: Annotated[...] = "{}",
    # ... rest of existing args ...
):
    if config:
        from madengine.config import load_config
        config_ctx, config_meta = load_config(config)

        # Extract model selection from config (CLI args override)
        if not tags and config_meta.get("model", {}).get("tags"):
            tags = config_meta["model"]["tags"]
        if timeout == DEFAULT_TIMEOUT and config_meta.get("model", {}).get("timeout"):
            timeout = config_meta["model"]["timeout"]
        if not manifest_file and config_meta.get("model", {}).get("manifest_file"):
            manifest_file = config_meta["model"]["manifest_file"]
        if not registry and config_meta.get("build", {}).get("registry"):
            registry = config_meta["build"]["registry"]

        # Merge: config_ctx is base, additional_context overrides
        parsed_ac = ast.literal_eval(additional_context) if additional_context != "{}" else {}
        merged = deep_merge(config_ctx, parsed_ac)
        additional_context = repr(merged)

    # ... rest of existing run logic (unchanged) ...
```

### Changes to `commands/build.py`

Same pattern: add `--config` parameter, extract build-relevant metadata, merge with `additional_context`.

---

## Usage Examples

### Single-file workflow (most common)

```yaml
# my_slurm_training.yaml
defaults:
  - /scheduler: slurm
  - /launcher: torchrun
  - /hardware: amd
  - _self_

model:
  tags: [megatron_llama3_70b]

slurm:
  partition: gpu-cluster
  nodes: 4
  gpus_per_node: 8
  time: "48:00:00"
  modules: [rocm/6.2.0]

distributed:
  nnodes: 4
  nproc_per_node: 8

env_vars:
  NCCL_DEBUG: WARN
  GPU_MAX_HW_QUEUES: "2"
  HSA_ENABLE_SDMA: "0"
```

```bash
madengine run --config my_slurm_training.yaml
```

### Config groups + inline overrides (no file)

```bash
# SLURM multi-node with torchrun
madengine run --config scheduler=slurm launcher=torchrun \
  model.tags=[llama3] distributed.nnodes=4 slurm.partition=gpu-high

# K8s vLLM inference with profiling
madengine run --config scheduler=k8s launcher=vllm \
  +tools=rocprofv3_lightweight k8s.namespace=ml-inference \
  model.tags=[vllm_llama]

# Local single-GPU (all defaults, just select model)
madengine run --config model.tags=[dummy]
```

### File + overrides

```bash
# Base config from file, override node count
madengine run --config my_slurm_training.yaml distributed.nnodes=8

# Base config + add profiling tools
madengine run --config my_slurm_training.yaml +tools=power_profiler
```

### Backward compatible

```bash
# --additional-context still works, overrides --config
madengine run --config my_slurm_training.yaml \
  --additional-context '{"slurm": {"partition": "override-partition"}}'

# Pure --additional-context (no --config) still works exactly as before
madengine run --tags dummy -c '{"gpu_vendor": "AMD"}'
```

### Future: bare metal

```bash
# No Docker — direct execution on host
madengine run --config platform=bare_metal scheduler=slurm \
  launcher=torchrun model.tags=[benchmark]
```

---

## Migration Path

### Phase 1: Add --config alongside --additional-context
- Both coexist; `--additional-context` has highest priority
- Existing JSON example configs can be converted to YAML (1:1 mapping via translator)
- No breaking changes

### Phase 2: Convert existing JSON presets to YAML configs
- `deployment/presets/k8s/defaults.json` → `configs/scheduler/k8s.yaml`
- `deployment/presets/slurm/defaults.json` → `configs/scheduler/slurm.yaml`
- `deployment/presets/k8s/profiles/` → `configs/profile/` YAML files
- `examples/profiling-configs/*.json` → `configs/tools/` YAML files
- `examples/k8s-configs/*.json` → example YAML files in `examples/`

### Phase 3: Deprecate --additional-context (future)
- Emit deprecation warning when `--additional-context` is used
- Eventually remove in a major version

---

## Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
    # ... existing ...
    "hydra-core>=1.3",
    "omegaconf>=2.3",
]
```

Both are pure Python with minimal transitive dependencies. `omegaconf` is already a dependency of `hydra-core`.

---

## Testing Strategy

### Unit Tests

- `test_loader.py`: HydraConfigLoader with various override combinations
- `test_translator.py`: ConfigTranslator key mapping, passthrough, extraction
- `test_schema.py`: Validation rules (conflicts, unknown keys, type checks)
- `test_merge.py`: Merge precedence (config < CLI < additional_context)

### Integration Tests

- End-to-end: `--config scheduler=slurm` produces correct `additional_context`
- File + overrides: `--config file.yaml key=value` merges correctly
- Backward compat: `--additional-context` without `--config` unchanged
- Both: `--config` + `--additional-context` merges with correct precedence

### Fixture Configs

- Add YAML equivalents of existing test fixture JSON files
- Test each config group individually and in combination

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/madengine/config/__init__.py` | Public API |
| `src/madengine/config/loader.py` | HydraConfigLoader |
| `src/madengine/config/translator.py` | ConfigTranslator |
| `src/madengine/config/schema.py` | ConfigValidator |
| `src/madengine/configs/config.yaml` | Root config |
| `src/madengine/configs/platform/*.yaml` | Platform configs |
| `src/madengine/configs/scheduler/*.yaml` | Scheduler configs |
| `src/madengine/configs/hardware/*.yaml` | Hardware configs |
| `src/madengine/configs/launcher/*.yaml` | Launcher configs |
| `src/madengine/configs/profile/*.yaml` | Hardware profiles |
| `src/madengine/configs/env/*.yaml` | Env var presets |
| `src/madengine/configs/tools/*.yaml` | Profiling tool configs |
| `src/madengine/configs/data/*.yaml` | Data provider configs |
| `src/madengine/configs/build/*.yaml` | Build setting configs |
| `tests/unit/test_config_loader.py` | Loader tests |
| `tests/unit/test_config_translator.py` | Translator tests |
| `tests/unit/test_config_schema.py` | Validation tests |

## Files to Modify

| File | Change |
|------|--------|
| `pyproject.toml` | Add hydra-core, omegaconf dependencies |
| `src/madengine/cli/commands/run.py` | Add `--config` parameter, integration logic |
| `src/madengine/cli/commands/build.py` | Add `--config` parameter, integration logic |
