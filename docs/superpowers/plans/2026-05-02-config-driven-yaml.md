# Config-Driven YAML System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--config` CLI argument to madengine that loads Hydra-based YAML configs with composable config groups and CLI override support, backward-compatible with `--additional-context`.

**Architecture:** A new `src/madengine/config/` package uses Hydra's Compose API (not `@hydra.main`) to load YAML config groups from `src/madengine/configs/`, then a `ConfigTranslator` maps clean YAML keys to the internal `additional_context` dict format that existing orchestrators expect. The `--config` arg is added to the `run` and `build` Typer commands; `--additional-context` still works and takes highest priority.

**Tech Stack:** hydra-core>=1.3, omegaconf>=2.3 (new deps); Typer (existing CLI); pytest (tests)

---

## File Map

### New Files — Config Package

| File | Responsibility |
|------|---------------|
| `src/madengine/config/__init__.py` | Public API: `load_config(config_args) -> (dict, dict)` |
| `src/madengine/config/loader.py` | `HydraConfigLoader` — Compose API wrapper, separates file path from overrides |
| `src/madengine/config/translator.py` | `ConfigTranslator` — maps YAML keys to `additional_context` format |
| `src/madengine/config/schema.py` | `ConfigValidator` — cross-field checks, unknown key detection |

### New Files — YAML Configs

| Directory | Files |
|-----------|-------|
| `src/madengine/configs/` | `config.yaml` (root) |
| `src/madengine/configs/platform/` | `docker.yaml`, `bare_metal.yaml`, `singularity.yaml`, `podman.yaml` |
| `src/madengine/configs/scheduler/` | `local.yaml`, `slurm.yaml`, `k8s.yaml` |
| `src/madengine/configs/hardware/` | `amd.yaml`, `nvidia.yaml`, `cpu.yaml` |
| `src/madengine/configs/launcher/` | `none.yaml`, `torchrun.yaml`, `deepspeed.yaml`, `megatron.yaml`, `vllm.yaml`, `sglang.yaml`, `sglang_disagg.yaml`, `torchtitan.yaml`, `primus.yaml`, `native.yaml` |
| `src/madengine/configs/profile/` | `mi300x_8gpu.yaml`, `mi300x_single.yaml`, `mi250x_4gpu.yaml`, `h100_8gpu.yaml`, `a100_8gpu.yaml` |
| `src/madengine/configs/env/` | `nccl_debug.yaml`, `nccl_tuned.yaml`, `infiniband.yaml`, `miopen_defaults.yaml` |
| `src/madengine/configs/tools/` | `rocprofv3_lightweight.yaml`, `rocprofv3_comprehensive.yaml`, `power_profiler.yaml`, `vram_profiler.yaml`, `rocm_trace_lite.yaml` |
| `src/madengine/configs/data/` | `local.yaml`, `s3.yaml`, `minio.yaml`, `nas.yaml` |
| `src/madengine/configs/build/` | `default.yaml`, `ci.yaml`, `multi_arch.yaml` |

### New Files — Tests

| File | Responsibility |
|------|---------------|
| `tests/unit/test_hydra_config_loader.py` | HydraConfigLoader unit tests |
| `tests/unit/test_config_translator.py` | ConfigTranslator unit tests |
| `tests/unit/test_config_schema.py` | ConfigValidator unit tests |
| `tests/unit/test_config_integration.py` | End-to-end: `load_config()` → dict |
| `tests/fixtures/configs/` | Test YAML fixtures |

### Modified Files

| File | Change |
|------|--------|
| `pyproject.toml` | Add `hydra-core>=1.3`, `omegaconf>=2.3` to dependencies; add `configs` to wheel force-include |
| `src/madengine/cli/commands/run.py` | Add `--config` parameter, config loading + merge logic |
| `src/madengine/cli/commands/build.py` | Add `--config` parameter, config loading + merge logic |

---

### Task 1: Add Dependencies and Wheel Config

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add hydra-core and omegaconf to dependencies**

In `pyproject.toml`, add to the `dependencies` list after `"pyyaml>=6.0"`:

```toml
dependencies = [
  "pandas",
  "GitPython",
  "jsondiff",
  "sqlalchemy",
  "paramiko",
  "tqdm",
  "typing-extensions",
  "pymongo",
  "toml",
  "typer>=0.9.0",
  "rich>=13.0.0",
  "click>=8.0.0",
  "jinja2>=3.0.0",
  "pyyaml>=6.0",
  "hydra-core>=1.3",
  "omegaconf>=2.3",
]
```

- [ ] **Step 2: Add configs directory to wheel force-include**

In the `[tool.hatch.build.targets.wheel.force-include]` section, add:

```toml
[tool.hatch.build.targets.wheel.force-include]
"src/madengine/scripts" = "madengine/scripts"
"src/madengine/deployment/templates" = "madengine/deployment/templates"
"src/madengine/configs" = "madengine/configs"
```

- [ ] **Step 3: Install updated dependencies**

Run: `pip install -e ".[dev]"`
Expected: Clean install with hydra-core and omegaconf resolved.

- [ ] **Step 4: Verify imports work**

Run: `python -c "from hydra import compose, initialize_config_dir; from omegaconf import OmegaConf, DictConfig; print('OK')"`
Expected: Prints `OK`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "feat(config): add hydra-core and omegaconf dependencies"
```

---

### Task 2: Create YAML Config Files — Root and Default Groups

**Files:**
- Create: `src/madengine/configs/config.yaml`
- Create: `src/madengine/configs/platform/docker.yaml`
- Create: `src/madengine/configs/platform/bare_metal.yaml`
- Create: `src/madengine/configs/platform/singularity.yaml`
- Create: `src/madengine/configs/platform/podman.yaml`
- Create: `src/madengine/configs/scheduler/local.yaml`
- Create: `src/madengine/configs/scheduler/slurm.yaml`
- Create: `src/madengine/configs/scheduler/k8s.yaml`
- Create: `src/madengine/configs/hardware/amd.yaml`
- Create: `src/madengine/configs/hardware/nvidia.yaml`
- Create: `src/madengine/configs/hardware/cpu.yaml`
- Create: `src/madengine/configs/launcher/none.yaml`
- Create: `src/madengine/configs/launcher/torchrun.yaml`
- Create: `src/madengine/configs/launcher/deepspeed.yaml`
- Create: `src/madengine/configs/launcher/megatron.yaml`
- Create: `src/madengine/configs/launcher/vllm.yaml`
- Create: `src/madengine/configs/launcher/sglang.yaml`
- Create: `src/madengine/configs/launcher/sglang_disagg.yaml`
- Create: `src/madengine/configs/launcher/torchtitan.yaml`
- Create: `src/madengine/configs/launcher/primus.yaml`
- Create: `src/madengine/configs/launcher/native.yaml`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src/madengine/configs/{platform,scheduler,hardware,launcher,profile,env,tools,data,build}
```

- [ ] **Step 2: Create root config.yaml**

Write to `src/madengine/configs/config.yaml`:

```yaml
defaults:
  - platform: docker
  - scheduler: local
  - hardware: amd
  - launcher: none
  - _self_

model:
  tags: []
  manifest_file: null
  container_image: null
  skip_run: false
  timeout: null

docker:
  build_args: {}
  env_vars: {}
  mounts: {}
  gpus: null
  cpus: null
  additional_run_options: null
  keep_alive: false
  clean_cache: false

build:
  registry: null
  target_archs: []
  manifest_output: build_manifest.json

env_vars: {}

debug: false
live_output: false

log_error:
  pattern_scan: true
  benign_patterns: []
  patterns: []

tools: []
pre_scripts: []
post_scripts: []
encapsulate_script: null

data_config: data.json

output: perf.csv
summary_output: null
```

- [ ] **Step 3: Create platform configs**

Write to `src/madengine/configs/platform/docker.yaml`:

```yaml
# @package _global_
platform:
  type: docker
```

Write to `src/madengine/configs/platform/bare_metal.yaml`:

```yaml
# @package _global_
platform:
  type: bare_metal
```

Write to `src/madengine/configs/platform/singularity.yaml`:

```yaml
# @package _global_
platform:
  type: singularity
```

Write to `src/madengine/configs/platform/podman.yaml`:

```yaml
# @package _global_
platform:
  type: podman
```

- [ ] **Step 4: Create scheduler configs**

Write to `src/madengine/configs/scheduler/local.yaml`:

```yaml
# @package _global_
```

Write to `src/madengine/configs/scheduler/slurm.yaml`:

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

Write to `src/madengine/configs/scheduler/k8s.yaml`:

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

- [ ] **Step 5: Create hardware configs**

Write to `src/madengine/configs/hardware/amd.yaml`:

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

Write to `src/madengine/configs/hardware/nvidia.yaml`:

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

Write to `src/madengine/configs/hardware/cpu.yaml`:

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

- [ ] **Step 6: Create launcher configs**

Write to `src/madengine/configs/launcher/none.yaml`:

```yaml
# @package _global_
distributed:
  enabled: false
```

Write to `src/madengine/configs/launcher/torchrun.yaml`:

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

Write to `src/madengine/configs/launcher/deepspeed.yaml`:

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

Write to `src/madengine/configs/launcher/megatron.yaml`:

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

Write to `src/madengine/configs/launcher/vllm.yaml`:

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

Write to `src/madengine/configs/launcher/sglang.yaml`:

```yaml
# @package _global_
distributed:
  enabled: true
  launcher: sglang
  backend: nccl
  nnodes: 1
  nproc_per_node: 8
  port: 29500
```

Write to `src/madengine/configs/launcher/sglang_disagg.yaml`:

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

Write to `src/madengine/configs/launcher/torchtitan.yaml`:

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

Write to `src/madengine/configs/launcher/primus.yaml`:

```yaml
# @package _global_
distributed:
  enabled: true
  launcher: primus
  backend: nccl
  nnodes: 1
  nproc_per_node: 8
  master_port: 29500
```

Write to `src/madengine/configs/launcher/native.yaml`:

```yaml
# @package _global_
distributed:
  enabled: true
  launcher: native
  backend: nccl
  nnodes: 1
  nproc_per_node: 8
  master_port: 29500
```

- [ ] **Step 7: Verify Hydra can compose the root config**

Run: `python -c "
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import os
GlobalHydra.instance().clear()
config_dir = os.path.abspath('src/madengine/configs')
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name='config')
print(OmegaConf.to_yaml(cfg))
"`

Expected: Prints the full composed YAML with all default groups merged — `gpu_vendor: AMD`, `distributed.enabled: false`, etc.

- [ ] **Step 8: Commit**

```bash
git add src/madengine/configs/
git commit -m "feat(config): add root config.yaml and default config groups"
```

---

### Task 3: Create YAML Config Files — Append-Only Groups

**Files:**
- Create: `src/madengine/configs/profile/mi300x_8gpu.yaml` (and 4 others)
- Create: `src/madengine/configs/env/nccl_debug.yaml` (and 3 others)
- Create: `src/madengine/configs/tools/rocprofv3_lightweight.yaml` (and 4 others)
- Create: `src/madengine/configs/data/local.yaml` (and 3 others)
- Create: `src/madengine/configs/build/default.yaml` (and 2 others)

- [ ] **Step 1: Create profile configs**

Write to `src/madengine/configs/profile/mi300x_8gpu.yaml`:

```yaml
# @package _global_
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

Write to `src/madengine/configs/profile/mi300x_single.yaml`:

```yaml
# @package _global_
gpu_type: mi300x
gpu_memory_gb: 192
gpus_per_node: 1

distributed:
  nproc_per_node: 1
```

Write to `src/madengine/configs/profile/mi250x_4gpu.yaml`:

```yaml
# @package _global_
gpu_type: mi250x
gpu_memory_gb: 128
gpus_per_node: 4

distributed:
  nproc_per_node: 4

env_vars:
  GPU_MAX_HW_QUEUES: "2"
  HSA_ENABLE_SDMA: "0"
```

Write to `src/madengine/configs/profile/h100_8gpu.yaml`:

```yaml
# @package _global_
gpu_vendor: NVIDIA
guest_os: UBUNTU
gpu_type: h100
gpu_memory_gb: 80
gpus_per_node: 8

runtime:
  devices: []
  capabilities: []
  security_opts: []
  network_mode: host
  ipc: host
  groups: []
  use_gpu_flag: true

distributed:
  nproc_per_node: 8
```

Write to `src/madengine/configs/profile/a100_8gpu.yaml`:

```yaml
# @package _global_
gpu_vendor: NVIDIA
guest_os: UBUNTU
gpu_type: a100
gpu_memory_gb: 80
gpus_per_node: 8

runtime:
  devices: []
  capabilities: []
  security_opts: []
  network_mode: host
  ipc: host
  groups: []
  use_gpu_flag: true

distributed:
  nproc_per_node: 8
```

- [ ] **Step 2: Create env configs**

Write to `src/madengine/configs/env/nccl_debug.yaml`:

```yaml
# @package _global_
env_vars:
  NCCL_DEBUG: INFO
  NCCL_DEBUG_SUBSYS: "INIT,NET,GRAPH"
  TORCH_DISTRIBUTED_DEBUG: DETAIL
```

Write to `src/madengine/configs/env/nccl_tuned.yaml`:

```yaml
# @package _global_
env_vars:
  NCCL_DEBUG: WARN
  TORCH_NCCL_HIGH_PRIORITY: "1"
  GPU_MAX_HW_QUEUES: "2"
  NCCL_TIMEOUT: "600"
  TORCH_NCCL_ASYNC_ERROR_HANDLING: "1"
```

Write to `src/madengine/configs/env/infiniband.yaml`:

```yaml
# @package _global_
env_vars:
  NCCL_IB_DISABLE: "0"
  NCCL_IB_HCA: "mlx5_0:1,mlx5_1:1"
  NCCL_SOCKET_IFNAME: ib0
  NCCL_NET_GDR_LEVEL: "3"
```

Write to `src/madengine/configs/env/miopen_defaults.yaml`:

```yaml
# @package _global_
env_vars:
  MIOPEN_FIND_MODE: "1"
  MIOPEN_USER_DB_PATH: /tmp/.miopen
```

- [ ] **Step 3: Create tools configs**

Write to `src/madengine/configs/tools/rocprofv3_lightweight.yaml`:

```yaml
# @package _global_
tools:
  - name: rocprofv3_lightweight
```

Write to `src/madengine/configs/tools/rocprofv3_comprehensive.yaml`:

```yaml
# @package _global_
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

Write to `src/madengine/configs/tools/power_profiler.yaml`:

```yaml
# @package _global_
tools:
  - name: gpu_info_power_profiler
    env_vars:
      POWER_DEVICE: all
      POWER_SAMPLING_RATE: "0.1"
      POWER_MODE: power
      POWER_DUAL_GCD: "false"
      POWER_OUTPUT_FILE: gpu_info_power_profiler_output.csv
```

Write to `src/madengine/configs/tools/vram_profiler.yaml`:

```yaml
# @package _global_
tools:
  - name: gpu_info_vram_profiler
    env_vars:
      VRAM_DEVICE: all
      VRAM_SAMPLING_RATE: "0.1"
      VRAM_MODE: vram
      VRAM_DUAL_GCD: "false"
      VRAM_OUTPUT_FILE: gpu_info_vram_profiler_output.csv
```

Write to `src/madengine/configs/tools/rocm_trace_lite.yaml`:

```yaml
# @package _global_
tools:
  - name: rocm_trace_lite
    env_vars:
      RTL_MODE: lite
```

- [ ] **Step 4: Create data configs**

Write to `src/madengine/configs/data/local.yaml`:

```yaml
# @package _global_
data:
  provider: local
  path: null
```

Write to `src/madengine/configs/data/s3.yaml`:

```yaml
# @package _global_
data:
  provider: s3
  bucket: null
  prefix: null
  region: null
```

Write to `src/madengine/configs/data/minio.yaml`:

```yaml
# @package _global_
data:
  provider: minio
  endpoint: null
  bucket: null
  access_key: null
  secret_key: null
```

Write to `src/madengine/configs/data/nas.yaml`:

```yaml
# @package _global_
data:
  provider: nas
  mount_path: null
```

- [ ] **Step 5: Create build configs**

Write to `src/madengine/configs/build/default.yaml`:

```yaml
# @package _global_
build:
  registry: null
  target_archs: []
  manifest_output: build_manifest.json
```

Write to `src/madengine/configs/build/ci.yaml`:

```yaml
# @package _global_
docker:
  clean_cache: true

build:
  registry: null
  target_archs: []
  manifest_output: build_manifest.json
```

Write to `src/madengine/configs/build/multi_arch.yaml`:

```yaml
# @package _global_
build:
  registry: null
  target_archs:
    - gfx942
    - gfx90a
    - gfx908
  manifest_output: build_manifest.json
```

- [ ] **Step 6: Verify append-only group composition**

Run: `python -c "
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import os
GlobalHydra.instance().clear()
config_dir = os.path.abspath('src/madengine/configs')
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name='config', overrides=['scheduler=slurm', 'launcher=torchrun', '+profile=mi300x_8gpu', '+env=nccl_debug'])
print(OmegaConf.to_yaml(cfg))
"`

Expected: Prints composed config with SLURM scheduler, torchrun launcher, mi300x profile, and NCCL debug env vars all merged.

- [ ] **Step 7: Commit**

```bash
git add src/madengine/configs/
git commit -m "feat(config): add append-only config groups (profile, env, tools, data, build)"
```

---

### Task 4: Implement HydraConfigLoader

**Files:**
- Create: `src/madengine/config/__init__.py`
- Create: `src/madengine/config/loader.py`
- Test: `tests/unit/test_hydra_config_loader.py`

- [ ] **Step 1: Write failing tests for HydraConfigLoader**

Write to `tests/unit/test_hydra_config_loader.py`:

```python
#!/usr/bin/env python3
"""Tests for HydraConfigLoader."""

import os
import pytest
import tempfile
from pathlib import Path

from omegaconf import DictConfig

from madengine.config.loader import HydraConfigLoader
from madengine.core.errors import ConfigurationError


class TestParseArgs:
    def test_hydra_overrides_only(self):
        user_file, overrides = HydraConfigLoader._parse_args(
            ["scheduler=slurm", "distributed.nnodes=4"]
        )
        assert user_file is None
        assert overrides == ["scheduler=slurm", "distributed.nnodes=4"]

    def test_yaml_file_only(self):
        user_file, overrides = HydraConfigLoader._parse_args(
            ["/path/to/config.yaml"]
        )
        assert user_file == "/path/to/config.yaml"
        assert overrides == []

    def test_yaml_file_with_overrides(self):
        user_file, overrides = HydraConfigLoader._parse_args(
            ["/path/to/config.yaml", "distributed.nnodes=8"]
        )
        assert user_file == "/path/to/config.yaml"
        assert overrides == ["distributed.nnodes=8"]

    def test_yml_extension_recognized(self):
        user_file, overrides = HydraConfigLoader._parse_args(
            ["/path/to/config.yml"]
        )
        assert user_file == "/path/to/config.yml"

    def test_multiple_yaml_files_raises(self):
        with pytest.raises(ConfigurationError, match="Only one YAML"):
            HydraConfigLoader._parse_args(
                ["/path/a.yaml", "/path/b.yaml"]
            )

    def test_append_override_not_treated_as_file(self):
        user_file, overrides = HydraConfigLoader._parse_args(
            ["+profile=mi300x_8gpu"]
        )
        assert user_file is None
        assert overrides == ["+profile=mi300x_8gpu"]

    def test_empty_args(self):
        user_file, overrides = HydraConfigLoader._parse_args([])
        assert user_file is None
        assert overrides == []


class TestLoad:
    def test_defaults_only(self):
        cfg = HydraConfigLoader.load([])
        assert isinstance(cfg, DictConfig)
        assert cfg.gpu_vendor == "AMD"
        assert cfg.guest_os == "UBUNTU"
        assert cfg.distributed.enabled is False

    def test_scheduler_override(self):
        cfg = HydraConfigLoader.load(["scheduler=slurm"])
        assert "slurm" in cfg
        assert cfg.slurm.partition == "amd-rccl"

    def test_launcher_override(self):
        cfg = HydraConfigLoader.load(["launcher=torchrun"])
        assert cfg.distributed.enabled is True
        assert cfg.distributed.launcher == "torchrun"

    def test_inline_value_override(self):
        cfg = HydraConfigLoader.load(
            ["launcher=torchrun", "distributed.nnodes=4"]
        )
        assert cfg.distributed.nnodes == 4

    def test_append_profile(self):
        cfg = HydraConfigLoader.load(["+profile=mi300x_8gpu"])
        assert cfg.gpu_type == "mi300x"
        assert cfg.distributed.nproc_per_node == 8

    def test_user_yaml_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("debug: true\nenv_vars:\n  MY_VAR: hello\n")
            f.flush()
            try:
                cfg = HydraConfigLoader.load([f.name])
                assert cfg.debug is True
                assert cfg.env_vars.MY_VAR == "hello"
            finally:
                os.unlink(f.name)

    def test_user_yaml_with_overrides(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("debug: true\n")
            f.flush()
            try:
                cfg = HydraConfigLoader.load(
                    [f.name, "scheduler=slurm"]
                )
                assert cfg.debug is True
                assert "slurm" in cfg
            finally:
                os.unlink(f.name)

    def test_hardware_nvidia(self):
        cfg = HydraConfigLoader.load(["hardware=nvidia"])
        assert cfg.gpu_vendor == "NVIDIA"
        assert cfg.runtime.use_gpu_flag is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_hydra_config_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'madengine.config'`

- [ ] **Step 3: Implement HydraConfigLoader**

Write to `src/madengine/config/__init__.py`:

```python
"""Config-driven YAML configuration system for madengine."""

from madengine.config.loader import HydraConfigLoader
from madengine.config.translator import ConfigTranslator
from madengine.config.schema import ConfigValidator


def load_config(config_args: list) -> tuple:
    """Load config from Hydra overrides and/or user YAML file.

    Args:
        config_args: List of Hydra overrides and/or a YAML file path.

    Returns:
        Tuple of (additional_context dict, metadata dict).
    """
    cfg = HydraConfigLoader.load(config_args)
    errors = ConfigValidator.validate(cfg)
    if errors:
        from madengine.core.errors import ConfigurationError

        raise ConfigurationError(
            "Config validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    return ConfigTranslator.to_additional_context(cfg)
```

Write to `src/madengine/config/loader.py`:

```python
"""Hydra-based config loader using the Compose API."""

import importlib.resources
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from madengine.core.errors import ConfigurationError


class HydraConfigLoader:
    """Loads madengine config using Hydra's Compose API."""

    @staticmethod
    def load(config_args: list) -> DictConfig:
        """Load and compose config from Hydra overrides and/or user YAML.

        Args:
            config_args: Mix of Hydra overrides and optional user YAML path.

        Returns:
            Composed DictConfig with all merges applied.
        """
        user_file, overrides = HydraConfigLoader._parse_args(config_args)

        config_dir = str(
            Path(importlib.resources.files("madengine")) / "configs"
        )

        if not os.path.isdir(config_dir):
            config_dir = str(
                Path(__file__).parent.parent / "configs"
            )

        GlobalHydra.instance().clear()

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config", overrides=overrides)

        if user_file:
            user_cfg = OmegaConf.load(user_file)
            OmegaConf.set_struct(cfg, False)
            cfg = OmegaConf.merge(cfg, user_cfg)

        return cfg

    @staticmethod
    def _parse_args(config_args: list) -> tuple:
        """Separate user YAML file path from Hydra overrides."""
        user_file = None
        overrides = []
        for arg in config_args:
            if (
                arg.endswith((".yaml", ".yml"))
                and "=" not in arg
                and not arg.startswith("+")
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

- [ ] **Step 4: Create stub translator and schema so imports resolve**

Write to `src/madengine/config/translator.py`:

```python
"""Translates clean YAML config to internal additional_context format."""

from omegaconf import DictConfig, OmegaConf


class ConfigTranslator:
    """Maps YAML config keys to internal additional_context dict format."""

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

    EXTRACTED_KEYS = {
        "model", "build", "platform", "output",
        "summary_output", "data_config", "live_output",
    }

    @classmethod
    def to_additional_context(cls, cfg: DictConfig) -> tuple:
        """Placeholder — implemented in Task 5."""
        return {}, {}
```

Write to `src/madengine/config/schema.py`:

```python
"""Config validation."""

from omegaconf import DictConfig


class ConfigValidator:
    """Validates composed config for consistency."""

    @staticmethod
    def validate(cfg: DictConfig) -> list:
        """Placeholder — implemented in Task 6."""
        return []
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_hydra_config_loader.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/madengine/config/ tests/unit/test_hydra_config_loader.py
git commit -m "feat(config): implement HydraConfigLoader with Compose API"
```

---

### Task 5: Implement ConfigTranslator

**Files:**
- Modify: `src/madengine/config/translator.py`
- Test: `tests/unit/test_config_translator.py`

- [ ] **Step 1: Write failing tests for ConfigTranslator**

Write to `tests/unit/test_config_translator.py`:

```python
#!/usr/bin/env python3
"""Tests for ConfigTranslator."""

import pytest
from omegaconf import OmegaConf

from madengine.config.translator import ConfigTranslator


def make_cfg(overrides: dict) -> "DictConfig":
    """Build a DictConfig from a base + overrides for testing."""
    base = {
        "model": {"tags": [], "manifest_file": None, "container_image": None, "skip_run": False, "timeout": None},
        "docker": {"build_args": {}, "env_vars": {}, "mounts": {}, "gpus": None, "cpus": None, "additional_run_options": None, "keep_alive": False, "clean_cache": False},
        "build": {"registry": None, "target_archs": [], "manifest_output": "build_manifest.json"},
        "env_vars": {},
        "debug": False,
        "live_output": False,
        "log_error": {"pattern_scan": True, "benign_patterns": [], "patterns": []},
        "tools": [],
        "pre_scripts": [],
        "post_scripts": [],
        "encapsulate_script": None,
        "data_config": "data.json",
        "output": "perf.csv",
        "summary_output": None,
        "gpu_vendor": "AMD",
        "guest_os": "UBUNTU",
        "runtime": {"devices": [], "capabilities": [], "security_opts": [], "network_mode": "host", "ipc": "host", "groups": [], "use_gpu_flag": False},
        "platform": {"type": "docker"},
    }
    merged = {**base, **overrides}
    return OmegaConf.create(merged)


class TestDockerKeyMapping:
    def test_build_args_mapped(self):
        cfg = make_cfg({"docker": {"build_args": {"KEY": "val"}, "env_vars": {}, "mounts": {}, "gpus": None, "cpus": None, "additional_run_options": None, "keep_alive": False, "clean_cache": False}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["docker_build_arg"] == {"KEY": "val"}

    def test_env_vars_mapped(self):
        cfg = make_cfg({"docker": {"build_args": {}, "env_vars": {"A": "1"}, "mounts": {}, "gpus": None, "cpus": None, "additional_run_options": None, "keep_alive": False, "clean_cache": False}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["docker_env_vars"] == {"A": "1"}

    def test_null_gpus_excluded(self):
        cfg = make_cfg({})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert "docker_gpus" not in ctx

    def test_non_null_gpus_included(self):
        cfg = make_cfg({"docker": {"build_args": {}, "env_vars": {}, "mounts": {}, "gpus": "0-3", "cpus": None, "additional_run_options": None, "keep_alive": False, "clean_cache": False}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["docker_gpus"] == "0-3"


class TestLogErrorMapping:
    def test_pattern_scan_mapped(self):
        cfg = make_cfg({"log_error": {"pattern_scan": False, "benign_patterns": [], "patterns": []}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["log_error_pattern_scan"] is False

    def test_patterns_mapped(self):
        cfg = make_cfg({"log_error": {"pattern_scan": True, "benign_patterns": ["OK"], "patterns": ["ERR"]}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["log_error_benign_patterns"] == ["OK"]
        assert ctx["log_error_patterns"] == ["ERR"]


class TestPassthroughKeys:
    def test_gpu_vendor_passthrough(self):
        cfg = make_cfg({"gpu_vendor": "NVIDIA"})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["gpu_vendor"] == "NVIDIA"

    def test_env_vars_passthrough(self):
        cfg = make_cfg({"env_vars": {"MY": "VAR"}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["env_vars"] == {"MY": "VAR"}

    def test_slurm_passthrough(self):
        cfg = make_cfg({"slurm": {"partition": "gpu"}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["slurm"] == {"partition": "gpu"}

    def test_distributed_passthrough(self):
        cfg = make_cfg({"distributed": {"enabled": True, "launcher": "torchrun"}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["distributed"]["launcher"] == "torchrun"

    def test_tools_passthrough(self):
        cfg = make_cfg({"tools": [{"name": "rpd"}]})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["tools"] == [{"name": "rpd"}]


class TestExtractedKeys:
    def test_model_extracted(self):
        cfg = make_cfg({"model": {"tags": ["dummy"], "manifest_file": None, "container_image": None, "skip_run": False, "timeout": 300}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert "model" not in ctx
        assert meta["model"]["tags"] == ["dummy"]
        assert meta["model"]["timeout"] == 300

    def test_build_extracted(self):
        cfg = make_cfg({"build": {"registry": "myregistry.io", "target_archs": ["gfx942"], "manifest_output": "build_manifest.json"}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert "build" not in ctx
        assert meta["build"]["registry"] == "myregistry.io"

    def test_platform_extracted(self):
        cfg = make_cfg({})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert "platform" not in ctx
        assert meta["platform"]["type"] == "docker"

    def test_container_image_promoted(self):
        cfg = make_cfg({"model": {"tags": [], "manifest_file": None, "container_image": "myimage:latest", "skip_run": False, "timeout": None}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["MAD_CONTAINER_IMAGE"] == "myimage:latest"

    def test_runtime_extracted(self):
        cfg = make_cfg({})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert "runtime" not in ctx
        assert "runtime" in meta
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_config_translator.py -v`
Expected: FAIL — translator returns empty dicts.

- [ ] **Step 3: Implement ConfigTranslator**

Replace the content of `src/madengine/config/translator.py` with:

```python
"""Translates clean YAML config to internal additional_context format."""

from omegaconf import DictConfig, OmegaConf


class ConfigTranslator:
    """Maps YAML config keys to internal additional_context dict format."""

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

    EXTRACTED_KEYS = {
        "model", "build", "platform", "output",
        "summary_output", "data_config", "live_output",
    }

    @classmethod
    def to_additional_context(cls, cfg: DictConfig) -> tuple:
        """Convert DictConfig to (additional_context, metadata) tuple.

        Returns:
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
                metadata["runtime"] = value
            else:
                if value is not None:
                    context[key] = value

        model = metadata.get("model", {})
        if model and model.get("container_image"):
            context["MAD_CONTAINER_IMAGE"] = model["container_image"]

        return context, metadata
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_config_translator.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/madengine/config/translator.py tests/unit/test_config_translator.py
git commit -m "feat(config): implement ConfigTranslator key mapping"
```

---

### Task 6: Implement ConfigValidator

**Files:**
- Modify: `src/madengine/config/schema.py`
- Test: `tests/unit/test_config_schema.py`

- [ ] **Step 1: Write failing tests for ConfigValidator**

Write to `tests/unit/test_config_schema.py`:

```python
#!/usr/bin/env python3
"""Tests for ConfigValidator."""

import pytest
from omegaconf import OmegaConf

from madengine.config.schema import ConfigValidator


def make_cfg(data: dict) -> "DictConfig":
    return OmegaConf.create(data)


class TestConflictDetection:
    def test_slurm_and_k8s_conflict(self):
        cfg = make_cfg({"slurm": {"partition": "gpu"}, "k8s": {"namespace": "default"}})
        errors = ConfigValidator.validate(cfg)
        assert any("Cannot specify both" in e for e in errors)

    def test_slurm_only_no_conflict(self):
        cfg = make_cfg({"slurm": {"partition": "gpu"}})
        errors = ConfigValidator.validate(cfg)
        assert not any("Cannot specify both" in e for e in errors)

    def test_k8s_only_no_conflict(self):
        cfg = make_cfg({"k8s": {"namespace": "default"}})
        errors = ConfigValidator.validate(cfg)
        assert not any("Cannot specify both" in e for e in errors)


class TestDistributedValidation:
    def test_enabled_without_launcher(self):
        cfg = make_cfg({"distributed": {"enabled": True}})
        errors = ConfigValidator.validate(cfg)
        assert any("requires distributed.launcher" in e for e in errors)

    def test_enabled_with_launcher(self):
        cfg = make_cfg({"distributed": {"enabled": True, "launcher": "torchrun"}})
        errors = ConfigValidator.validate(cfg)
        assert not any("requires distributed.launcher" in e for e in errors)

    def test_invalid_nnodes(self):
        cfg = make_cfg({"distributed": {"enabled": True, "launcher": "torchrun", "nnodes": -1}})
        errors = ConfigValidator.validate(cfg)
        assert any("positive integer" in e for e in errors)

    def test_valid_nnodes(self):
        cfg = make_cfg({"distributed": {"enabled": True, "launcher": "torchrun", "nnodes": 4}})
        errors = ConfigValidator.validate(cfg)
        assert not any("positive integer" in e for e in errors)


class TestUnknownKeys:
    def test_unknown_top_level_key(self):
        cfg = make_cfg({"gpu_vendor": "AMD", "typo_key": "oops"})
        errors = ConfigValidator.validate(cfg)
        assert any("Unknown config key: 'typo_key'" in e for e in errors)

    def test_known_keys_accepted(self):
        cfg = make_cfg({"gpu_vendor": "AMD", "debug": True, "env_vars": {}})
        errors = ConfigValidator.validate(cfg)
        assert not any("Unknown config key" in e for e in errors)


class TestPlatformValidation:
    def test_unsupported_platform(self):
        cfg = make_cfg({"platform": {"type": "bare_metal"}})
        errors = ConfigValidator.validate(cfg)
        assert any("not yet supported" in e for e in errors)

    def test_docker_platform_ok(self):
        cfg = make_cfg({"platform": {"type": "docker"}})
        errors = ConfigValidator.validate(cfg)
        assert not any("not yet supported" in e for e in errors)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_config_schema.py -v`
Expected: FAIL — validator returns empty list.

- [ ] **Step 3: Implement ConfigValidator**

Replace the content of `src/madengine/config/schema.py` with:

```python
"""Config validation for composed Hydra configs."""

from omegaconf import DictConfig


KNOWN_TOP_LEVEL_KEYS = {
    "defaults", "platform", "scheduler", "hardware", "launcher",
    "model", "docker", "build", "env_vars", "debug", "live_output",
    "log_error", "tools", "pre_scripts", "post_scripts",
    "encapsulate_script", "data_config", "output", "summary_output",
    "gpu_vendor", "guest_os", "runtime", "slurm", "k8s",
    "kubernetes", "distributed", "vllm", "sglang_disagg",
    "shared_data", "timeout", "gpu_type", "gpu_memory_gb",
    "gpus_per_node", "data",
}

SUPPORTED_PLATFORMS = {"docker"}


class ConfigValidator:
    """Validates composed config for consistency."""

    @staticmethod
    def validate(cfg: DictConfig) -> list:
        """Return list of validation errors (empty = valid)."""
        errors = []

        raw = dict(cfg) if hasattr(cfg, "keys") else {}

        if raw.get("slurm") and raw.get("k8s"):
            errors.append(
                "Cannot specify both 'slurm' and 'k8s' sections"
            )

        dist = raw.get("distributed")
        if isinstance(dist, dict):
            if dist.get("enabled") and not dist.get("launcher"):
                errors.append(
                    "distributed.enabled=true requires distributed.launcher"
                )
            nnodes = dist.get("nnodes")
            if nnodes is not None:
                if not isinstance(nnodes, int) or nnodes < 1:
                    errors.append(
                        "distributed.nnodes must be a positive integer"
                    )

        platform = raw.get("platform")
        if isinstance(platform, dict):
            ptype = platform.get("type")
            if ptype and ptype not in SUPPORTED_PLATFORMS:
                errors.append(
                    f"Platform '{ptype}' is not yet supported. "
                    f"Supported: {', '.join(sorted(SUPPORTED_PLATFORMS))}"
                )

        for key in raw:
            if key not in KNOWN_TOP_LEVEL_KEYS:
                errors.append(f"Unknown config key: '{key}'")

        return errors
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_config_schema.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/madengine/config/schema.py tests/unit/test_config_schema.py
git commit -m "feat(config): implement ConfigValidator with cross-field checks"
```

---

### Task 7: Integration Test — load_config End-to-End

**Files:**
- Create: `tests/unit/test_config_integration.py`
- Create: `tests/fixtures/configs/test_slurm_job.yaml`

- [ ] **Step 1: Create test fixture YAML**

Write to `tests/fixtures/configs/test_slurm_job.yaml`:

```yaml
model:
  tags: [dummy]

slurm:
  partition: test-partition
  nodes: 2

distributed:
  enabled: true
  launcher: torchrun
  nnodes: 2
  nproc_per_node: 4

env_vars:
  MY_VAR: test_value

debug: true
```

- [ ] **Step 2: Write integration tests**

Write to `tests/unit/test_config_integration.py`:

```python
#!/usr/bin/env python3
"""Integration tests for load_config end-to-end pipeline."""

import os
import pytest
from pathlib import Path

from madengine.config import load_config
from madengine.core.errors import ConfigurationError


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "configs"


class TestLoadConfigEndToEnd:
    def test_defaults_produce_valid_context(self):
        ctx, meta = load_config([])
        assert ctx["gpu_vendor"] == "AMD"
        assert ctx["guest_os"] == "UBUNTU"
        assert meta["model"]["tags"] == []

    def test_scheduler_slurm(self):
        ctx, meta = load_config(["scheduler=slurm"])
        assert "slurm" in ctx
        assert ctx["slurm"]["partition"] == "amd-rccl"

    def test_launcher_torchrun(self):
        ctx, meta = load_config(["launcher=torchrun"])
        assert ctx["distributed"]["enabled"] is True
        assert ctx["distributed"]["launcher"] == "torchrun"

    def test_combined_overrides(self):
        ctx, meta = load_config([
            "scheduler=slurm",
            "launcher=torchrun",
            "distributed.nnodes=4",
            "+env=nccl_debug",
        ])
        assert ctx["distributed"]["nnodes"] == 4
        assert ctx["env_vars"]["NCCL_DEBUG"] == "INFO"
        assert "slurm" in ctx

    def test_user_yaml_file(self):
        yaml_path = str(FIXTURES_DIR / "test_slurm_job.yaml")
        ctx, meta = load_config([yaml_path])
        assert meta["model"]["tags"] == ["dummy"]
        assert ctx["slurm"]["partition"] == "test-partition"
        assert ctx["distributed"]["nnodes"] == 2
        assert ctx["env_vars"]["MY_VAR"] == "test_value"
        assert ctx["debug"] is True

    def test_user_yaml_with_override(self):
        yaml_path = str(FIXTURES_DIR / "test_slurm_job.yaml")
        ctx, meta = load_config([yaml_path, "distributed.nnodes=8"])
        assert ctx["distributed"]["nnodes"] == 8

    def test_docker_keys_translated(self):
        ctx, meta = load_config(["docker.build_args.KEY=val"])
        assert ctx["docker_build_arg"]["KEY"] == "val"

    def test_slurm_and_k8s_conflict_raises(self):
        with pytest.raises(ConfigurationError, match="Cannot specify both"):
            load_config(["scheduler=slurm", "k8s.namespace=test"])

    def test_unsupported_platform_raises(self):
        with pytest.raises(ConfigurationError, match="not yet supported"):
            load_config(["platform=bare_metal"])

    def test_container_image_promoted(self):
        ctx, meta = load_config(
            ["model.container_image=myimage:latest"]
        )
        assert ctx["MAD_CONTAINER_IMAGE"] == "myimage:latest"

    def test_model_tags_in_metadata(self):
        ctx, meta = load_config(["model.tags=[dummy,bert]"])
        assert meta["model"]["tags"] == ["dummy", "bert"]
        assert "model" not in ctx

    def test_profile_append(self):
        ctx, meta = load_config(["+profile=mi300x_8gpu"])
        assert ctx["gpu_type"] == "mi300x"
        assert ctx["env_vars"]["HSA_ENABLE_SDMA"] == "0"

    def test_tools_append(self):
        ctx, meta = load_config(["+tools=rocprofv3_lightweight"])
        assert len(ctx["tools"]) == 1
        assert ctx["tools"][0]["name"] == "rocprofv3_lightweight"
```

- [ ] **Step 3: Run integration tests**

Run: `pytest tests/unit/test_config_integration.py -v`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_config_integration.py tests/fixtures/configs/
git commit -m "test(config): add integration tests for load_config pipeline"
```

---

### Task 8: Integrate --config into CLI run Command

**Files:**
- Modify: `src/madengine/cli/commands/run.py`

- [ ] **Step 1: Add --config parameter and merge logic to run command**

In `src/madengine/cli/commands/run.py`, add the import at the top (after the existing imports, around line 9):

```python
import ast
```

Add the `--config` parameter to the `run` function signature, after the `additional_context_file` parameter (after line 83):

```python
    config: Annotated[
        Optional[List[str]],
        typer.Option(
            "--config",
            help="YAML config file and/or Hydra overrides (e.g., --config my_job.yaml, --config scheduler=slurm launcher=torchrun)",
        ),
    ] = None,
```

After line 165 (`processed_tags = split_comma_separated_tags(tags)`), insert the config loading block:

```python
    # Load --config YAML if provided
    if config:
        from madengine.config import load_config

        config_ctx, config_meta = load_config(config)

        # Config values provide defaults; explicit CLI args override
        if not processed_tags and config_meta.get("model", {}).get("tags"):
            processed_tags = config_meta["model"]["tags"]
        if timeout == DEFAULT_TIMEOUT and config_meta.get("model", {}).get("timeout"):
            timeout = config_meta["model"]["timeout"]
        if not manifest_file and config_meta.get("model", {}).get("manifest_file"):
            manifest_file = config_meta["model"]["manifest_file"]
        if not registry and config_meta.get("build", {}).get("registry"):
            registry = config_meta["build"]["registry"]

        # Merge: config is base, --additional-context overrides
        parsed_ac = {}
        if additional_context and additional_context.strip() != "{}":
            try:
                parsed_ac = json.loads(additional_context)
            except json.JSONDecodeError:
                parsed_ac = ast.literal_eval(additional_context)

        def _deep_merge(base: dict, override: dict) -> dict:
            result = base.copy()
            for k, v in override.items():
                if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                    result[k] = _deep_merge(result[k], v)
                else:
                    result[k] = v
            return result

        merged = _deep_merge(config_ctx, parsed_ac)
        additional_context = repr(merged)
        additional_context_file = None
```

- [ ] **Step 2: Verify the existing test suite still passes**

Run: `pytest tests/unit/test_cli.py -v`
Expected: All existing tests PASS (backward compatibility preserved).

- [ ] **Step 3: Commit**

```bash
git add src/madengine/cli/commands/run.py
git commit -m "feat(config): integrate --config into run command"
```

---

### Task 9: Integrate --config into CLI build Command

**Files:**
- Modify: `src/madengine/cli/commands/build.py`

- [ ] **Step 1: Add --config parameter and merge logic to build command**

In `src/madengine/cli/commands/build.py`, add the import at the top (after existing imports, around line 9):

```python
import ast
```

Add the `--config` parameter to the `build` function signature, after the `additional_context_file` parameter (after line 71):

```python
    config: Annotated[
        Optional[List[str]],
        typer.Option(
            "--config",
            help="YAML config file and/or Hydra overrides (e.g., --config my_job.yaml, --config scheduler=slurm)",
        ),
    ] = None,
```

After line 104 (`processed_tags = split_comma_separated_tags(tags)`), insert the config loading block:

```python
    # Load --config YAML if provided
    if config:
        from madengine.config import load_config

        config_ctx, config_meta = load_config(config)

        # Config values provide defaults; explicit CLI args override
        if not processed_tags and config_meta.get("model", {}).get("tags"):
            processed_tags = config_meta["model"]["tags"]
        if not registry and config_meta.get("build", {}).get("registry"):
            registry = config_meta["build"]["registry"]
        build_meta = config_meta.get("build", {})
        if not target_archs and build_meta.get("target_archs"):
            target_archs = build_meta["target_archs"]

        # Merge: config is base, --additional-context overrides
        parsed_ac = {}
        if additional_context and additional_context.strip() != "{}":
            try:
                parsed_ac = json.loads(additional_context)
            except json.JSONDecodeError:
                parsed_ac = ast.literal_eval(additional_context)

        def _deep_merge(base: dict, override: dict) -> dict:
            result = base.copy()
            for k, v in override.items():
                if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                    result[k] = _deep_merge(result[k], v)
                else:
                    result[k] = v
            return result

        merged = _deep_merge(config_ctx, parsed_ac)
        additional_context = repr(merged)
        additional_context_file = None
```

- [ ] **Step 2: Verify the existing test suite still passes**

Run: `pytest tests/unit/test_cli.py -v`
Expected: All existing tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/madengine/cli/commands/build.py
git commit -m "feat(config): integrate --config into build command"
```

---

### Task 10: Extract _deep_merge to Shared Utility

The `_deep_merge` function is duplicated in both `run.py` and `build.py`. Extract it.

**Files:**
- Modify: `src/madengine/cli/utils.py`
- Modify: `src/madengine/cli/commands/run.py`
- Modify: `src/madengine/cli/commands/build.py`

- [ ] **Step 1: Add deep_merge to cli/utils.py**

At the bottom of `src/madengine/cli/utils.py`, add:

```python
def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflicts."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result
```

- [ ] **Step 2: Update run.py to use shared deep_merge**

In `src/madengine/cli/commands/run.py`, add `deep_merge` to the import from `..utils`:

```python
from ..utils import (
    console,
    setup_logging,
    split_comma_separated_tags,
    create_args_namespace,
    save_summary_with_feedback,
    display_results_table,
    display_performance_table,
    deep_merge,
)
```

Remove the inline `_deep_merge` function definition and replace `_deep_merge(` with `deep_merge(` in the config loading block.

- [ ] **Step 3: Update build.py to use shared deep_merge**

In `src/madengine/cli/commands/build.py`, add `deep_merge` to the import from `..utils`:

```python
from ..utils import (
    console,
    setup_logging,
    split_comma_separated_tags,
    create_args_namespace,
    save_summary_with_feedback,
    display_results_table,
    deep_merge,
)
```

Remove the inline `_deep_merge` function definition and replace `_deep_merge(` with `deep_merge(` in the config loading block.

- [ ] **Step 4: Run all tests**

Run: `pytest tests/unit/ -v --timeout=60`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/madengine/cli/utils.py src/madengine/cli/commands/run.py src/madengine/cli/commands/build.py
git commit -m "refactor(config): extract deep_merge to shared utility"
```

---

### Task 11: Final Verification — Full Test Suite

**Files:**
- No new files — verification only.

- [ ] **Step 1: Run the complete unit test suite**

Run: `pytest tests/unit/ -v --timeout=60`
Expected: All tests PASS including the new config tests.

- [ ] **Step 2: Run the pre-commit hooks**

Run: `pre-commit run --all-files`
Expected: All hooks pass (black, isort, flake8).

- [ ] **Step 3: Verify Hydra config composition end-to-end**

Run: `python -c "
from madengine.config import load_config
ctx, meta = load_config(['scheduler=slurm', 'launcher=torchrun', '+profile=mi300x_8gpu', '+env=nccl_debug', 'model.tags=[dummy]'])
print('Tags:', meta['model']['tags'])
print('Launcher:', ctx['distributed']['launcher'])
print('Partition:', ctx['slurm']['partition'])
print('NCCL_DEBUG:', ctx['env_vars'].get('NCCL_DEBUG'))
print('GPU type:', ctx.get('gpu_type'))
"`

Expected output:
```
Tags: ['dummy']
Launcher: torchrun
Partition: amd-rccl
NCCL_DEBUG: INFO
GPU type: mi300x
```

- [ ] **Step 4: Verify CLI help text includes --config**

Run: `madengine run --help | grep -A2 "config"`
Expected: Shows `--config` option with help text about YAML config files and Hydra overrides.

- [ ] **Step 5: Final commit if any formatting fixes were needed**

```bash
git add -u
git commit -m "style: apply formatting fixes from pre-commit hooks"
```
