# MADEngine CLI Refactoring Plan - Production Ready

> **Version**: 2.0  
> **Last Updated**: November 28, 2025  
> **Status**: Draft for Review

---

## Executive Summary

madengine-cli is a **model automation framework** that works with the [MAD (Model Automation and Dashboarding)](https://github.com/ROCm/MAD) project - a curated AI/ML model hub. This refactoring extends deployment from single-node to multi-node (SLURM/Kubernetes) while maintaining the core automation workflow.

### What madengine-cli Does

```
┌─────────────────────────────────────────────────────────────┐
│ MAD Project (Model Hub)                                     │
│ ├─ models.json: Model definitions with tags                 │
│ ├─ docker/: Dockerfiles for building model environments     │
│ └─ scripts/: Model-specific run scripts                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ madengine-cli Automation Workflow                           │
│                                                              │
│ 1. Discover models from MAD's models.json by tags          │
│ 2. Build Docker image from MAD's Dockerfile                 │
│ 3. Run model workload (Python subprocess automation):       │
│    ├─ Start Docker container                                │
│    ├─ Download data (Minio/AWS/NAS via dataprovider)       │
│    ├─ Run pre-scripts (rocEnvTool, GPU info, profiling)    │
│    ├─ Execute model benchmark (MAD's run.sh)               │
│    ├─ Run post-scripts (collect metrics, end profiling)    │
│    ├─ Parse performance output                              │
│    └─ Remove container                                      │
│ 4. Collect results → perf.csv                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: Pre/post-scripts (rocEnvTool, profiling, data download) are in **madengine** (`src/madengine/scripts/common/`), called via Python subprocess. MAD models only provide the benchmark code.

### Key Objectives

1. **Keep existing workflow intact** - All automation (data download, pre/post-scripts, profiling) works as-is
2. **Extend to multi-node** - SLURM and Kubernetes deployment using existing workflow
3. **Use --additional-context** - No new CLI arguments, deployment config via JSON
4. **Simple templates** - Jinja2 templates for sbatch and K8s Job manifests
5. **Same execution everywhere** - SLURM runs `madengine run` on nodes, K8s runs same flow in containers
6. **vLLM MoE support** - Enable parallelism benchmarking (TP/DP/PP/EP) for inference models

### Critical Design Decisions

✅ **madengine automation is in madengine repo** (`src/madengine/scripts/common/`):
- Pre-scripts: `rocEnvTool`, `gpu_info_pre.sh`, `trace.sh` (start profiling)
- Post-scripts: `gpu_info_post.sh`, `trace.sh` (end profiling), metric collection
- Data download: Python subprocess calling Minio/AWS/NAS providers
- All called via Python subprocess, not separate executable scripts

✅ **MAD models provide only**:
- Dockerfile (dependencies, environment setup)
- run.sh (model benchmark code)
- models.json entry (metadata, tags)

✅ **SLURM deployment**: Each node runs `madengine run` (not docker/singularity)

✅ **Kubernetes deployment**: Pod runs built Docker image, executes same workflow (no docker-in-docker)

✅ **Configuration via --additional-context**: No new CLI arguments, deployment mode in JSON:
```json
{
  "deploy": "slurm",  // or "k8s"
  "slurm": {"partition": "gpu", "nodes": 4},
  "k8s": {"namespace": "ml-bench", "gpu_count": 8}
}
```

---

## Table of Contents

1. [Problem Analysis](#1-problem-analysis)
2. [Architecture Clarification](#2-architecture-clarification)
   - 2.4 [vLLM MoE Parallelism Strategies](#24-vllm-moe-parallelism-strategies)
3. [Proposed Solution](#3-proposed-solution)
   - 3.2 [Enhanced build_manifest.json](#32-enhanced-build_manifestjson)
4. [Implementation Plan](#4-implementation-plan)
5. [Migration Strategy](#5-migration-strategy)
6. [Testing Strategy](#6-testing-strategy)
7. [Timeline & Milestones](#7-timeline--milestones)
8. [Success Criteria](#8-success-criteria)
9. [Risks & Mitigation](#9-risks--mitigation)
- [Appendix A: vLLM MoE Parallelism Benchmarking](#appendix-a-vllm-moe-parallelism-benchmarking)
- [Appendix B: Example Usage](#appendix-b-example-usage)
- [Appendix C: Configuration Examples](#appendix-c-configuration-examples)
- [References](#references)

---

## 1. PROBLEM ANALYSIS

### 1.1 Current Issues

**Terminology Confusion**:
- Current "runners" (SSH/Ansible/K8s/SLURM) distribute **madengine execution itself**
- But users need to distribute **model workload execution** (using torchrun, deepspeed, etc.)
- This creates confusion between "infrastructure" and "execution method"

**Complexity**:
- Four runner types (SSH, Ansible, K8s, SLURM) with different abstractions
- Complex setup process (clone MAD, setup venv, install madengine on each node)
- Not aligned with how K8s and SLURM are actually used in practice

**K8s/SLURM Usage Gap**:
- **K8s Reality**: Users deploy pods with model containers directly, not madengine containers
- **SLURM Reality**: Users submit sbatch scripts that run models, not madengine setup scripts
- Current implementation adds unnecessary indirection

### 1.2 What Works Well (Keep These)

✅ **Build Phase** (`DockerBuilder`):
- Model discovery via tags
- Docker image building with GPU architecture support
- Registry push/pull
- Manifest generation

✅ **Run Phase** (`ContainerRunner`):
- Local Docker container execution
- GPU device mapping
- Performance metric collection
- Timeout management

✅ **Core Components**:
- Context (GPU detection, environment)
- DataProvider (data source management)
- Model discovery system
- Error handling framework

---

## 2. ARCHITECTURE CLARIFICATION

### 2.1 Terminology Alignment

**Infrastructure Layer** (Where workload runs):
```
┌─────────────────────────────────────────────────────┐
│ Infrastructure Targets                              │
├─────────────────────────────────────────────────────┤
│ • Local:      Docker on current node               │
│ • SLURM:      HPC cluster with job scheduler        │
│ • Kubernetes: Container orchestration platform      │
└─────────────────────────────────────────────────────┘
```

**Execution Methods** (How model runs within container):
```
┌─────────────────────────────────────────────────────┐
│ Execution Launchers (Inside Container)             │
├─────────────────────────────────────────────────────┤
│ Training/Fine-tuning:                               │
│ • Single GPU:    python train.py                   │
│ • Multi GPU:     torchrun --nproc_per_node=8       │
│ • Distributed:   torchrun --nnodes=4               │
│ • DeepSpeed:     deepspeed --hostfile=...          │
│ • Megatron:      Megatron-LM launcher              │
│                                                     │
│ Inference Serving (vLLM/SGLang):                   │
│ • vLLM TP:       --tensor-parallel-size 8          │
│ • vLLM DP:       --data-parallel-size 8            │
│ • vLLM PP:       --pipeline-parallel-size 2        │
│ • vLLM EP:       --enable-expert-parallel          │
│ • SGLang:        SGLang server configuration       │
└─────────────────────────────────────────────────────┘
```

**madengine's Scope**:
- ✅ Handles **infrastructure layer** (where to run)
- ✅ Builds Docker images with model code
- ❌ Does NOT implement execution methods (models handle this)

### 2.2 Correct Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        User Commands                           │
│  madengine-cli build    # Build Docker images                  │
│  madengine-cli run      # Run locally OR deploy to infra       │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                   Build Phase (Keep As-Is)                     │
│  • DiscoverModels                                              │
│  • DockerBuilder                                               │
│  • Generate build_manifest.json                                │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────┴────────────────┬───────────────────┐
        │                                 │                   │
        ▼                                 ▼                   ▼
┌──────────────┐              ┌──────────────┐    ┌──────────────┐
│ Local Run    │              │ SLURM Deploy │    │ K8s Deploy   │
│ (Existing)   │              │ (New)        │    │ (New)        │
├──────────────┤              ├──────────────┤    ├──────────────┤
│• Pull image  │              │• Gen sbatch  │    │• Gen pod.yaml│
│• Run container│             │• Submit job  │    │• kubectl apply│
│• Collect perf│              │• Monitor     │    │• Monitor     │
└──────────────┘              └──────────────┘    └──────────────┘
```

### 2.3 Reference Projects Analysis

**K8s Demo (`/home/ysha/amd/k8s-demo`)**:
- Pattern: Generate pod.yaml → `kubectl apply -f pod.yaml`
- Pod runs model container directly (not madengine)
- Simple, straightforward deployment

**SGLang Disagg (`/home/ysha/playground/MAD-private/scripts/sglang_disagg`)**:
- Pattern: Generate sbatch script → `sbatch job.sh`
- Script runs model directly (not madengine setup)
- Uses SLURM for resource allocation

**Primus Project** (https://github.com/AMD-AGI/Primus):
- Supports multiple backends (Megatron-LM, TorchTitan, JAX MaxText)
- Infrastructure-agnostic (can run on SLURM, K8s, etc.)
- madengine should orchestrate infrastructure, Primus handles execution

### 2.4 vLLM MoE Parallelism Strategies

**Reference**: [The vLLM MoE Playbook: A Practical Guide to TP, DP, PP and Expert Parallelism](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)

For inference serving with vLLM (especially MoE models like DeepSeek-R1, Qwen3-235B, Llama-4-Maverick), madengine-cli must support various parallelism strategies for comprehensive benchmarking.

**Parallelism Types**:
```
┌─────────────────────────────────────────────────────────────┐
│ vLLM Parallelism Strategies for MoE Models                 │
├─────────────────────────────────────────────────────────────┤
│ • Tensor Parallelism (TP): Shards layers across GPUs       │
│   └─ Best for: Low latency, interactive workloads          │
│                                                             │
│ • Data Parallelism (DP): Replicates model across GPUs      │
│   └─ Best for: High throughput, batch processing           │
│                                                             │
│ • Pipeline Parallelism (PP): Splits model into stages      │
│   └─ Best for: Very large models, memory constraints       │
│                                                             │
│ • Expert Parallelism (EP): Distributes MoE experts         │
│   └─ Best for: MoE models with many experts                │
│                                                             │
│ • Hybrid: TP+EP, DP+EP (most common for MoE)               │
│   └─ Best for: Balancing latency and throughput            │
└─────────────────────────────────────────────────────────────┘
```

**Key Insights from vLLM MoE Guide**:

1. **TP+EP**: Superior for low-latency interactive workloads
   - Single request processed by all GPUs in parallel
   - Lower latency per request
   - AllReduce communication after each layer

2. **DP+EP**: Better for high-throughput batch processing
   - Multiple requests processed in parallel
   - Higher overall throughput
   - AllToAll communication for expert distribution

3. **Expert Activation Density**: Critical factor
   - Low density (<10%): EP improves performance
   - High density (>20%): EP may add overhead
   - Optimal strategy depends on model architecture

4. **MLA/MQA Attention**: Special handling required
   - Models like DeepSeek-R1 with Multi-Latent Attention
   - Affects KV cache memory requirements
   - Influences DP vs TP choice

**madengine-cli Support**:

madengine-cli enables users to specify vLLM parallelism strategies via `--additional-context`:

```json
{
  "launcher": "vllm",
  "vllm": {
    "tensor_parallel_size": 8,
    "data_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "enable_expert_parallel": true,
    "max_model_len": 32768,
    "distributed_executor_backend": "mp",
    "env_vars": {
      "VLLM_ROCM_USE_AITER": "0",
      "VLLM_ALL2ALL_BACKEND": "allgather_reducescatter"
    }
  }
}
```

This allows benchmarking different parallelism strategies on the same infrastructure (SLURM/K8s) to find optimal configuration for specific models and workloads.

---

## 3. PROPOSED SOLUTION

### 3.1 Clean Command Structure (--additional-context Driven)

**Three Deployment Modes** - All configuration via `--additional-context` (stored in `build_manifest.json`):

```bash
# Mode 1: Local Single Node (Default)
madengine-cli run --tags pyt_bert_training

# Mode 2: SLURM Multi-Node
madengine-cli run --tags pyt_bert_training \
  --additional-context '{
    "deploy": "slurm",
    "slurm": {
      "partition": "gpu",
      "nodes": 4,
      "gpus_per_node": 8,
      "time": "24:00:00",
      "exclusive": true,
      "qos": "normal"
    },
    "distributed": {
      "backend": "torchrun",
      "master_port": 29500,
      "nccl_socket_ifname": "ens14np0"
    },
    "shared_storage": "/nfs/datasets"
  }'

# Mode 3: Kubernetes (with AMD GPU Device Plugin)
madengine-cli run --tags pyt_bert_training \
  --additional-context '{
    "deploy": "k8s",
    "k8s": {
      "namespace": "ml-workloads",
      "gpu_count": 8,
      "gpu_vendor": "amd.com/gpu",
      "memory": "256Gi",
      "cpu": "64",
      "node_selector": {
        "amd.com/gpu.device.id": "0x74a1"
      }
    }
  }'

# vLLM Inference Configuration (SLURM example)
madengine-cli run --tags vllm_deepseek_r1 \
  --additional-context '{
    "deploy": "slurm",
    "slurm": {"partition": "mi300x", "nodes": 1, "gpus_per_node": 8},
    "vllm": {
      "tensor_parallel_size": 8,
      "enable_expert_parallel": true,
      "max_model_len": 32768,
      "port": 8000
    }
  }'

# Or use config file (for CI/CD)
madengine-cli run --tags pyt_bert_training \
  --additional-context-file configs/slurm_4node_training.json
```

**Why --additional-context for Everything**:
- ✅ **Stored in build_manifest.json**: Configuration is versioned and reproducible
- ✅ **CI/CD friendly**: Jenkins can use different config files for 7x24 testing
- ✅ **Production ready**: Same manifest used for build + multiple deployments
- ✅ **No environment pollution**: All config explicit, no hidden env vars
- ✅ **Auditable**: Every deployment has traceable configuration

**Key Design Principles**: 
- ✅ **3 Deployment Types Only**: Local, SLURM, Kubernetes
- ✅ **Configuration in Manifest**: All --additional-context saved to `build_manifest.json`
- ✅ **AMD GPU Device Plugin**: K8s uses standard resource requests (`amd.com/gpu`)
- ✅ **Template-driven**: Jinja2 generates sbatch scripts and K8s Job manifests
- ✅ **Factory Pattern**: Clean abstractions for each deployment type

**Remove These**:
- ❌ SSH/Ansible runners (not needed with SLURM/K8s)
- ❌ `madengine-cli generate/runner` subcommands
- ❌ Environment variable configuration for deployment

### 3.2 Actual madengine Run Workflow

**Understanding what `madengine run` actually does** (same on local, SLURM nodes, K8s containers):

```python
# Simplified view of run_models.py workflow

def run_model(model_info):
    # 1. Build Docker image (or use pre-built from manifest)
    docker_image = build_or_pull_image(model_info)
    
    # 2. Start container
    container = docker.run(
        image=docker_image,
        volumes=[f"{model_scripts}:/workspace"],
        devices=["/dev/kfd", "/dev/dri"],  # GPU devices
        env={
            "MAD_SYSTEM_GPU_ARCHITECTURE": "gfx90a",
            "ROCR_VISIBLE_DEVICES": "0,1,2,3"
        }
    )
    
    # 3. Inside container, madengine automation runs (via subprocess):
    
    # 3a. Download data (if model.data specified in models.json)
    if model_info.get("data"):
        subprocess.run(["python3", "download_data.py", 
                       "--provider", data_provider,  # Minio/AWS/NAS
                       "--dataset", model_info["data"]])
    
    # 3b. Run pre-scripts (from madengine/scripts/common/pre_scripts/)
    subprocess.run(["bash", "src/madengine/scripts/common/pre_scripts/run_rocenv_tool.sh"])
    subprocess.run(["bash", "src/madengine/scripts/common/pre_scripts/gpu_info_pre.sh"])
    subprocess.run(["bash", "src/madengine/scripts/common/pre_scripts/trace.sh"])  # Start profiling
    
    # 3c. Run model benchmark (MAD model's run.sh)
    result = subprocess.run(
        ["bash", "/workspace/run.sh"],  # MAD model script
        capture_output=True
    )
    
    # 3d. Run post-scripts (from madengine/scripts/common/post_scripts/)
    subprocess.run(["bash", "src/madengine/scripts/common/post_scripts/trace.sh"])  # End profiling
    subprocess.run(["bash", "src/madengine/scripts/common/post_scripts/gpu_info_post.sh"])
    
    # 3e. Parse performance from output
    performance = parse_output(result.stdout)  # Look for "performance: X.XX metric"
    
    # 4. Collect metrics and cleanup
    collect_metrics(performance)
    docker.remove(container)
    
    # 5. Write to perf.csv
    write_perf_csv(model_info, performance)
```

**Key Points**:
- ✅ Data download, pre/post-scripts, profiling handled by **madengine** (Python subprocess)
- ✅ MAD models only provide: Dockerfile, run.sh (benchmark code), models.json entry
- ✅ This workflow is **identical** on local, SLURM nodes, and K8s containers
- ✅ No changes needed to MAD repository models

**Deployment Strategy**:
- **Local**: Run `madengine run` directly on current node
- **Manual Multi-Node**: User manually runs `madengine run` on each node with `multi_node_args`
- **SLURM**: Generate sbatch → SLURM allocates nodes → Each node runs `madengine run` with auto-configured `multi_node_args`
- **K8s**: Generate Job → K8s creates pods → Each pod runs same workflow (with built image)

### 3.2b Clean Multi-Node Design (Production-Ready)

**Environment-Based Configuration** (Best Practice):

Instead of manual `NODE_RANK`, `MASTER_ADDR`, let deployment infrastructure provide environment variables:

```python
# MAD model's run.sh reads standard environment variables:
# - SLURM provides: SLURM_PROCID, SLURM_NODEID, SLURM_NODELIST
# - K8s provides: POD_NAME, POD_NAMESPACE, etc.
# - madengine translates these to standard ML vars

# In MAD model's run.sh:
if [ -n "$SLURM_JOB_ID" ]; then
    # SLURM environment
    export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
    export RANK=$SLURM_PROCID
    export WORLD_SIZE=$SLURM_NTASKS
elif [ -n "$KUBERNETES_SERVICE_HOST" ]; then
    # K8s environment
    export MASTER_ADDR="${POD_NAME%%-*}-0.${POD_NAME%%-*}"
    export RANK=$((${POD_NAME##*-}))
fi

# Run with torchrun (auto-detects environment)
torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=${MASTER_PORT:-29500} \
    train.py
```

**madengine's Role**:

```
┌─────────────────────────────────────────────────────────┐
│ User Command                                            │
│ madengine-cli run --tags model                          │
│   --additional-context '{"deploy": "slurm", ...}'       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ madengine Deployment Layer                              │
│                                                          │
│ SlurmDeployment.deploy():                              │
│   1. Render Jinja2 template (job.sh.j2)                │
│   2. Inject: partition, nodes, gpus, time, env vars    │
│   3. Submit: sbatch job.sh                              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ SLURM Scheduler                                         │
│   - Allocates nodes                                     │
│   - Sets SLURM_* environment variables                  │
│   - Runs job.sh on each node                            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Each Node: madengine run                                │
│   - Detects SLURM environment                           │
│   - Runs MAD model automation workflow                  │
│   - Model's run.sh uses SLURM env vars                  │
│   - torchrun auto-discovers nodes/ranks                 │
└─────────────────────────────────────────────────────────┘
```

**Clean Design Benefits**:

| Aspect | Old Manual Approach ❌ | New Clean Design ✅ |
|--------|----------------------|-------------------|
| **Node Discovery** | Manual IP addresses | Auto from SLURM/K8s |
| **Rank Assignment** | Manual NODE_RANK=0,1,2... | Auto from job scheduler |
| **Error Potential** | High (typos, wrong rank) | Low (automated) |
| **Scalability** | Must update for each node | Works for any node count |
| **Configuration** | User must know topology | Job scheduler handles it |
| **Best Practice** | ❌ Manual orchestration | ✅ Let infrastructure handle it |

**Example - 4-Node Training**:

```bash
# Clean approach (production-ready)
madengine-cli run --tags pyt_megatron_lm \
  --additional-context '{
    "deploy": "slurm",
    "slurm": {
      "partition": "gpu",
      "nodes": 4,
      "gpus_per_node": 8
    },
    "distributed": {
      "backend": "torchrun"
    }
  }'

# What happens:
# 1. madengine generates sbatch script with 4 nodes
# 2. SLURM allocates 4 nodes, sets SLURM_NODELIST, SLURM_PROCID, etc.
# 3. Each node's job.sh extracts MASTER_ADDR from SLURM_NODELIST
# 4. torchrun uses SLURM env vars to coordinate across nodes
# 5. No manual configuration needed!
```

### 3.3 build_manifest.json with --additional-context

**Design**: All --additional-context configuration is stored in `build_manifest.json` for reproducibility.

**Current Structure** (from actual manifest):
```json
{
  "built_images": {
    "ci-dummy_dummy.ubuntu.amd": {
      "docker_image": "ci-dummy_dummy.ubuntu.amd",
      "registry_image": "rocm/mad-private:ci-dummy_dummy.ubuntu.amd",
      "docker_sha": "sha256:780ac31518...",
      "build_duration": 358.48
    }
  },
  "built_models": {
    "ci-dummy_dummy.ubuntu.amd": {
      "name": "dummy",
      "dockerfile": "docker/dummy",
      "scripts": "scripts/dummy/run.sh",
      "n_gpus": "1",
      "tags": ["dummies"]
    }
  },
  "context": {
    "gpu_vendor": "AMD",
    "docker_gpus": ""
  },
  "registry": "dockerhub"
}
```

**Enhanced Structure** (with --additional-context stored):

```json
{
  "built_images": { /* ... unchanged ... */ },
  "built_models": { /* ... unchanged ... */ },
  "context": { /* ... unchanged ... */ },
  "registry": "dockerhub",
  
  "deployment_config": {
    "target": "slurm",
    "slurm": {
      "partition": "gpu",
      "nodes": 4,
      "gpus_per_node": 8,
      "time": "24:00:00",
      "exclusive": true,
      "qos": "normal",
      "modules": ["rocm/5.7.0", "python/3.10"]
    },
    "distributed": {
      "backend": "torchrun",
      "master_port": 29500,
      "nccl_socket_ifname": "ens14np0"
    },
    "shared_storage": "/nfs/datasets",
    "vllm": null,
    "k8s": null
  }
}
```

**How It Works**:

```bash
# Step 1: Build with deployment config
madengine-cli build --tags model \
  --additional-context '{
    "deploy": "slurm",
    "slurm": {"partition": "gpu", "nodes": 4},
    "distributed": {"backend": "torchrun"}
  }'

# Result: build_manifest.json contains deployment_config section

# Step 2: Run uses the stored config
madengine-cli run --manifest-file build_manifest.json

# OR override deployment target at runtime
madengine-cli run --manifest-file build_manifest.json \
  --additional-context '{"deploy": "k8s", "k8s": {...}}'
```

**Benefits**:
- ✅ **CI/CD Reproducibility**: Jenkins can rebuild + redeploy with same config
- ✅ **Configuration Versioning**: Manifest files can be committed to git
- ✅ **Audit Trail**: Know exactly what config was used for each deployment
- ✅ **Multi-Target**: Build once, deploy to SLURM or K8s using same manifest
- ✅ **No Hidden State**: All configuration explicit in manifest

### 3.4 Enhanced build_manifest.json (Continued)

Based on the current `build_manifest.json` structure generated by `madengine build`, we'll add deployment configuration fields while maintaining backward compatibility.

**Current Structure** (v1.x):
```json
{
  "built_images": {
    "ci-dummy_dummy.ubuntu.amd": {
      "docker_image": "ci-dummy_dummy.ubuntu.amd",
      "dockerfile": "docker/dummy.ubuntu.amd.Dockerfile",
      "base_docker": "rocm/pytorch",
      "docker_sha": "sha256:780ac31518773c3ae26165584688a6cee3b09f9d1410a175e0a47eece85b1ec7",
      "build_duration": 358.48,
      "build_command": "docker build --no-cache --network=host -t ci-dummy_dummy.ubuntu.amd --pull -f docker/dummy.ubuntu.amd.Dockerfile ./docker",
      "log_file": "dummy_dummy.ubuntu.amd.build.live.log",
      "registry_image": "rocm/mad-private:ci-dummy_dummy.ubuntu.amd"
    }
  },
  "built_models": {
    "ci-dummy_dummy.ubuntu.amd": {
      "name": "dummy",
      "dockerfile": "docker/dummy",
      "scripts": "scripts/dummy/run.sh",
      "n_gpus": "1",
      "owner": "mad.support@amd.com",
      "training_precision": "",
      "tags": ["dummies", "dummy_test_group_1", "dummy_group_1"],
      "args": ""
    }
  },
  "context": {
    "docker_env_vars": {},
    "docker_mounts": {},
    "docker_build_arg": {},
    "gpu_vendor": "AMD",
    "docker_gpus": ""
  },
  "credentials_required": [],
  "registry": "dockerhub"
}
```

**Enhanced Structure** (v2.0) - with deployment support from `--additional-context`:

```json
{
  "built_images": {
    "ci-dummy_dummy.ubuntu.amd": {
      // Existing fields (UNCHANGED - backward compatible)
      "docker_image": "ci-dummy_dummy.ubuntu.amd",
      "dockerfile": "docker/dummy.ubuntu.amd.Dockerfile",
      "base_docker": "rocm/pytorch",
      "docker_sha": "sha256:780ac31518773c3ae26165584688a6cee3b09f9d1410a175e0a47eece85b1ec7",
      "build_duration": 358.48,
      "build_command": "docker build --no-cache --network=host -t ci-dummy_dummy.ubuntu.amd --pull -f docker/dummy.ubuntu.amd.Dockerfile ./docker",
      "log_file": "dummy_dummy.ubuntu.amd.build.live.log",
      "registry_image": "rocm/mad-private:ci-dummy_dummy.ubuntu.amd"
    }
  },
  "built_models": {
    "ci-dummy_dummy.ubuntu.amd": {
      // Existing fields (UNCHANGED)
      "name": "dummy",
      "dockerfile": "docker/dummy",
      "scripts": "scripts/dummy/run.sh",
      "n_gpus": "1",
      "owner": "mad.support@amd.com",
      "training_precision": "",
      "tags": ["dummies", "dummy_test_group_1", "dummy_group_1"],
      "args": "",
      
      // NEW: Execution configuration (populated from --additional-context)
      "execution": {
        "launcher": "python",           // "python", "torchrun", "deepspeed", "vllm", "sglang"
        "nnodes": 1,                    // Number of nodes for distributed execution
        "nproc_per_node": 1,            // Number of processes per node (GPUs)
        "master_port": 29500,           // Master port for distributed communication
        "launcher_args": "",            // Additional launcher-specific arguments
        "env_vars": {}                  // Additional environment variables for execution
      }
    }
  },
  "context": {
    // Existing fields (UNCHANGED)
    "docker_env_vars": {},
    "docker_mounts": {},
    "docker_build_arg": {},
    "gpu_vendor": "AMD",
    "docker_gpus": "",
    
    // NEW: Extended runtime context (from --additional-context)
    "host_os": "UBUNTU",
    "gpu_architecture": "gfx90a",
    "n_gpus": 8
  },
  "credentials_required": [],
  "registry": "dockerhub",
  
  // NEW: Deployment configuration (from --additional-context)
  "deployment": {
    "target": "local",                  // "local", "slurm", "k8s"
    "generated_at": "2025-11-28T10:30:00Z",
    
    // SLURM configuration (when target="slurm")
    "slurm": {
      "partition": "gpu",
      "nodes": 1,
      "ntasks_per_node": 8,
      "gres": "gpu:8",
      "time_limit": "01:00:00",
      "qos": "normal",
      "account": null,
      "modules": ["rocm/5.7.0", "python/3.10"],
      "output_dir": "./slurm_output",
      "work_dir": "/projects/ml",
      "login_node": null
    },
    
    // Kubernetes configuration (when target="k8s")
    "k8s": {
      "namespace": "default",
      "kubeconfig": null,
      "node_selector": {},
      "resources": {
        "requests": {
          "amd.com/gpu": "2",
          "memory": "32Gi",
          "cpu": "8"
        },
        "limits": {
          "amd.com/gpu": "2",
          "memory": "64Gi",
          "cpu": "16"
        }
      },
      "volumes": [],
      "output_dir": "./k8s_manifests"
    }
  },
  
  // NEW: Execution profiles for different launchers (from --additional-context)
  "execution_profiles": {
    // vLLM inference serving configuration
    "vllm": {
      "tensor_parallel_size": 8,
      "data_parallel_size": 1,
      "pipeline_parallel_size": 1,
      "enable_expert_parallel": true,
      "max_model_len": 32768,
      "distributed_executor_backend": "mp",
      "disable_nccl_for_dp": true,
      "swap_space": 16,
      "port": 8000,
      "trust_remote_code": true,
      "env_vars": {
        "VLLM_ROCM_USE_AITER": "0",
        "VLLM_ALL2ALL_BACKEND": "allgather_reducescatter"
      }
    },
    
    // SGLang inference serving configuration
    "sglang": {
      "dp_size": 4,
      "tp_size": 2,
      "port": 30000,
      "mode": "disaggregated"
    },
    
    // Torchrun distributed training configuration
    "torchrun": {
      "nnodes": 4,
      "nproc_per_node": 8,
      "rdzv_backend": "c10d",
      "rdzv_endpoint": "auto"
    },
    
    // DeepSpeed distributed training configuration
    "deepspeed": {
      "num_nodes": 4,
      "num_gpus": 8,
      "hostfile": null,
      "deepspeed_config": null
    }
  }
}
```

**How --additional-context Populates build_manifest.json**:

1. **During Build Phase**:
```bash
madengine-cli build --tags dummy \
  --additional-context '{
    "deploy": "slurm",
    "launcher": "vllm",
    "nnodes": 1,
    "nproc_per_node": 8,
    "vllm": {
      "tensor_parallel_size": 8,
      "enable_expert_parallel": true,
      "max_model_len": 32768
    },
    "slurm": {
      "partition": "gpu",
      "nodes": 1,
      "time_limit": 3600,
      "modules": ["rocm/5.7.0"]
    }
  }'
```

**Results in**:
- `deployment.target` = "slurm"
- `deployment.slurm` = {partition: "gpu", nodes: 1, ...}
- `execution_profiles.vllm` = {tensor_parallel_size: 8, ...}
- `built_models[*].execution.launcher` = "vllm"
- `built_models[*].execution.nnodes` = 1
- `built_models[*].execution.nproc_per_node` = 8

2. **During Run Phase**:
```bash
# Run phase reads build_manifest.json and uses deployment config
madengine-cli run --manifest-file build_manifest.json

# Or override deployment target at runtime
madengine-cli run --manifest-file build_manifest.json \
  --additional-context '{"deploy": "k8s"}'
```

**Backward Compatibility Strategy**:

| Scenario | Behavior |
|----------|----------|
| v1.x manifest + v2.0 CLI | Works - missing fields get defaults (target="local") |
| v2.0 manifest + v1.x CLI | Works - extra fields ignored by v1.x code |
| v2.0 manifest without deployment | Works - defaults to local execution |
| Existing scripts/workflows | Unchanged - all existing fields preserved |

### 3.3 New Directory Structure

```
src/madengine/
├── mad.py                      # Legacy CLI (keep, deprecate gradually)
├── mad_cli.py                  # Modern CLI (refactor)
│
├── core/                       # Keep as-is (stable foundation)
│   ├── context.py
│   ├── docker.py
│   ├── dataprovider.py
│   └── ...
│
├── tools/                      # Keep existing tools
│   ├── discover_models.py     # Keep
│   ├── docker_builder.py      # Keep
│   ├── container_runner.py    # Keep + enhance
│   ├── distributed_orchestrator.py  # Refactor → deployment_orchestrator.py
│   └── ...
│
├── deployment/                 # NEW: Deployment infrastructure
│   ├── __init__.py
│   ├── base.py                # BaseDeployment abstract class
│   ├── local.py               # LocalDeployment (wraps existing)
│   ├── slurm.py               # SlurmDeployment (new)
│   ├── kubernetes.py          # KubernetesDeployment (new)
│   ├── factory.py             # DeploymentFactory
│   └── templates/             # Jinja2 templates
│       ├── slurm/
│       │   ├── job.sh.j2
│       │   └── job_array.sh.j2
│       └── kubernetes/
│           ├── pod.yaml.j2
│           ├── job.yaml.j2
│           └── deployment.yaml.j2
│
└── runners/                    # DEPRECATED (to be removed)
    └── ... (keep for now, mark deprecated)
```

---

## 4. IMPLEMENTATION PLAN

### 4.1 Phase 1: Foundation (Week 1-2)

#### 4.1.1 Create Deployment Abstraction (Production-Ready)

**File**: `src/madengine/deployment/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
from enum import Enum


class DeploymentStatus(Enum):
    """Deployment status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DeploymentConfig:
    """Configuration for deployment"""
    target: str  # "local", "slurm", "k8s"
    manifest_file: str
    additional_context: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 3600
    monitor: bool = True
    cleanup_on_failure: bool = True


@dataclass
class DeploymentResult:
    """Result of deployment operation"""
    status: DeploymentStatus
    deployment_id: str
    message: str
    metrics: Optional[Dict[str, Any]] = None
    logs_path: Optional[str] = None
    artifacts: Optional[List[str]] = None
    
    @property
    def is_success(self) -> bool:
        return self.status == DeploymentStatus.SUCCESS
    
    @property
    def is_failed(self) -> bool:
        return self.status == DeploymentStatus.FAILED


class BaseDeployment(ABC):
    """
    Abstract base class for all deployment targets.
    
    Implements Template Method pattern for deployment workflow.
    Subclasses implement specific deployment logic.
    """
    
    DEPLOYMENT_TYPE: str = "base"
    REQUIRED_TOOLS: List[str] = []  # e.g., ["sbatch"] for SLURM
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.manifest = self._load_manifest(config.manifest_file)
        self.console = self._get_console()
    
    def _load_manifest(self, manifest_file: str) -> Dict:
        """Load and validate build manifest"""
        import json
        from pathlib import Path
        
        manifest_path = Path(manifest_file)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        # Validate required fields
        required = ["built_images", "built_models", "context"]
        missing = [f for f in required if f not in manifest]
        if missing:
            raise ValueError(f"Invalid manifest, missing: {missing}")
        
        return manifest
    
    def _get_console(self):
        """Get Rich console for output"""
        from rich.console import Console
        return Console()
    
    # Template Method - defines workflow
    def execute(self) -> DeploymentResult:
        """
        Execute full deployment workflow (Template Method).
        
        Workflow:
        1. Validate environment and configuration
        2. Prepare deployment artifacts (scripts, manifests)
        3. Deploy to target infrastructure
        4. Monitor until completion (if enabled)
        5. Collect results and metrics
        6. Cleanup (if needed)
        """
        try:
            # Step 1: Validate
            self.console.print(f"[blue]Validating {self.DEPLOYMENT_TYPE} deployment...[/blue]")
            if not self.validate():
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    deployment_id="",
                    message=f"{self.DEPLOYMENT_TYPE} validation failed"
                )
            
            # Step 2: Prepare
            self.console.print(f"[blue]Preparing deployment artifacts...[/blue]")
            if not self.prepare():
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    deployment_id="",
                    message="Preparation failed"
                )
            
            # Step 3: Deploy
            self.console.print(f"[blue]Deploying to {self.DEPLOYMENT_TYPE}...[/blue]")
            result = self.deploy()
            
            if not result.is_success:
                if self.config.cleanup_on_failure:
                    self.cleanup(result.deployment_id)
                return result
            
            # Step 4: Monitor (optional)
            if self.config.monitor:
                result = self._monitor_until_complete(result.deployment_id)
            
            # Step 5: Collect Results
            if result.is_success:
                metrics = self.collect_results(result.deployment_id)
                result.metrics = metrics
            
            return result
            
        except Exception as e:
            self.console.print(f"[red]Deployment error: {e}[/red]")
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id="",
                message=f"Exception: {str(e)}"
            )
    
    def _monitor_until_complete(self, deployment_id: str) -> DeploymentResult:
        """Monitor deployment until completion"""
        import time
        
        self.console.print("[blue]Monitoring deployment...[/blue]")
        
        while True:
            status = self.monitor(deployment_id)
            
            if status.status in [DeploymentStatus.SUCCESS, DeploymentStatus.FAILED]:
                return status
            
            time.sleep(30)  # Check every 30 seconds
    
    # Abstract methods to be implemented by subclasses
    
    @abstractmethod
    def validate(self) -> bool:
        """
        Validate deployment environment and configuration.
        
        Check:
        - Required tools are available
        - Credentials/access are valid
        - Configuration is correct
        
        Returns:
            True if validation passes, False otherwise
        """
        pass
    
    @abstractmethod
    def prepare(self) -> bool:
        """
        Prepare deployment artifacts.
        
        Generate:
        - Deployment scripts (sbatch, Job manifests)
        - Configuration files
        - Environment setup
        
        Returns:
            True if preparation succeeds, False otherwise
        """
        pass
    
    @abstractmethod
    def deploy(self) -> DeploymentResult:
        """
        Execute deployment to target infrastructure.
        
        Submit:
        - SLURM job (sbatch)
        - Kubernetes Job (kubectl apply)
        - etc.
        
        Returns:
            DeploymentResult with status and deployment_id
        """
        pass
    
    @abstractmethod
    def monitor(self, deployment_id: str) -> DeploymentResult:
        """
        Check deployment status.
        
        Query:
        - SLURM job status (squeue)
        - K8s Job status (kubectl get job)
        - etc.
        
        Args:
            deployment_id: ID returned from deploy()
        
        Returns:
            Current status
        """
        pass
    
    @abstractmethod
    def collect_results(self, deployment_id: str) -> Dict[str, Any]:
        """
        Collect execution results and metrics.
        
        Retrieve:
        - Performance metrics (perf.csv)
        - Logs
        - Artifacts
        
        Args:
            deployment_id: ID of completed deployment
        
        Returns:
            Dictionary of metrics and results
        """
        pass
    
    @abstractmethod
    def cleanup(self, deployment_id: str) -> bool:
        """
        Cleanup deployment resources.
        
        Remove:
        - Temporary files
        - Jobs (if cancelled)
        - etc.
        
        Args:
            deployment_id: ID of deployment to clean up
        
        Returns:
            True if cleanup succeeds
        """
        pass
```

**Key Production Features**:
- ✅ **Template Method Pattern**: Clear workflow with hooks
- ✅ **Enum for Status**: Type-safe status handling
- ✅ **Validation**: Check environment before deployment
- ✅ **Error Handling**: Try/catch with cleanup on failure
- ✅ **Monitoring**: Optional progress tracking
- ✅ **Extensibility**: Easy to add new deployment types
- ✅ **Testability**: Each method can be tested independently

#### 4.1.2 Implement LocalDeployment

**File**: `src/madengine/deployment/local.py`

```python
from .base import BaseDeployment, DeploymentConfig, DeploymentResult
from madengine.tools.container_runner import ContainerRunner


class LocalDeployment(BaseDeployment):
    """Local deployment using existing ContainerRunner"""
    
    DEPLOYMENT_TYPE = "local"
    
    def __init__(self, config: DeploymentConfig):
        super().__init__(config)
        self.runner = ContainerRunner(
            context=self._get_context(),
            live_output=config.context.get("live_output", False)
        )
    
    def validate(self) -> bool:
        """Validate local deployment requirements"""
        # Check Docker is available
        # Check GPU if required
        return True
    
    def prepare(self) -> bool:
        """Prepare local deployment"""
        # Existing ContainerRunner handles this
        return True
    
    def deploy(self) -> DeploymentResult:
        """Execute local deployment using ContainerRunner"""
        try:
            # Use existing run_models_from_manifest
            summary = self.runner.run_models_from_manifest(
                manifest_file=self.config.manifest_file,
                timeout=self.config.timeout
            )
            
            return DeploymentResult(
                status="success",
                deployment_id="local",
                message="Local execution completed",
                metrics=summary
            )
        except Exception as e:
            return DeploymentResult(
                status="failed",
                deployment_id="local",
                message=f"Execution failed: {e}"
            )
    
    def monitor(self, deployment_id: str) -> DeploymentResult:
        """Local deployment completes immediately"""
        return DeploymentResult(
            status="success",
            deployment_id=deployment_id,
            message="Complete"
        )
    
    def collect_results(self, deployment_id: str) -> Dict:
        """Results already collected during execution"""
        return {}
    
    def cleanup(self, deployment_id: str) -> bool:
        """No cleanup needed for local"""
        return True
```

#### 4.1.3 Create DeploymentFactory (3 Types Only)

**File**: `src/madengine/deployment/factory.py`

```python
from typing import Dict, Type, Optional
from .base import BaseDeployment, DeploymentConfig


class DeploymentFactory:
    """
    Factory for creating deployment instances.
    
    Supports 3 deployment types:
    - local: Single-node local execution
    - slurm: HPC multi-node via SLURM scheduler
    - k8s: Kubernetes container orchestration
    """
    
    _deployments: Dict[str, Type[BaseDeployment]] = {}
    
    @classmethod
    def register(cls, deployment_type: str, deployment_class: Type[BaseDeployment]):
        """
        Register a deployment type.
        
        Args:
            deployment_type: Unique identifier (e.g., "local", "slurm", "k8s")
            deployment_class: Class implementing BaseDeployment
        """
        cls._deployments[deployment_type] = deployment_class
    
    @classmethod
    def create(cls, target: str, manifest_file: str, additional_context: Dict) -> BaseDeployment:
        """
        Create deployment instance based on target.
        
        Args:
            target: Deployment target ("local", "slurm", "k8s")
            manifest_file: Path to build_manifest.json
            additional_context: Full context from --additional-context
        
        Returns:
            Configured deployment instance
        
        Raises:
            ValueError: If target is not registered
        """
        deployment_class = cls._deployments.get(target)
        
        if not deployment_class:
            available = ", ".join(sorted(cls._deployments.keys()))
            raise ValueError(
                f"Unknown deployment target: '{target}'\n"
                f"Available: {available}\n\n"
                f"Example:\n"
                f'  madengine-cli run --tags model --additional-context \'{{"deploy": "slurm"}}\''
            )
        
        # Create configuration
        config = DeploymentConfig(
            target=target,
            manifest_file=manifest_file,
            additional_context=additional_context
        )
        
        return deployment_class(config)
    
    @classmethod
    def available_deployments(cls) -> list:
        """Get list of registered deployment types"""
        return sorted(cls._deployments.keys())
    
    @classmethod
    def is_available(cls, deployment_type: str) -> bool:
        """Check if deployment type is available"""
        return deployment_type in cls._deployments


# Register the 3 core deployment types
def register_deployments():
    """Register production-ready deployment types"""
    
    # 1. Local (always available)
    from .local import LocalDeployment
    DeploymentFactory.register("local", LocalDeployment)
    
    # 2. SLURM (HPC clusters)
    try:
        from .slurm import SlurmDeployment
        DeploymentFactory.register("slurm", SlurmDeployment)
    except ImportError as e:
        # Optional dependency, fail gracefully
        import warnings
        warnings.warn(f"SLURM deployment not available: {e}")
    
    # 3. Kubernetes (container orchestration)
    try:
        from .kubernetes import KubernetesDeployment
        DeploymentFactory.register("k8s", KubernetesDeployment)
        DeploymentFactory.register("kubernetes", KubernetesDeployment)  # Alias
    except ImportError as e:
        # Optional dependency, fail gracefully
        import warnings
        warnings.warn(f"Kubernetes deployment not available: {e}")


# Auto-register on module import
register_deployments()
```

**Key Features**:
- ✅ **3 Types Only**: Local, SLURM, Kubernetes
- ✅ **Graceful Degradation**: Missing deps don't break import
- ✅ **Clear Error Messages**: Shows available types and example usage
- ✅ **Factory Pattern**: Standard creational pattern
- ✅ **Extensible**: Easy to add new deployment types later

---

### 4.2 Phase 2: SLURM Deployment (Week 3-4)

#### 4.2.1 SLURM Template (Clean, Production-Ready)

**File**: `src/madengine/deployment/templates/slurm/job.sh.j2`

**Key Design**: Clean environment-based approach - SLURM provides env vars, model uses them directly.

```bash
#!/bin/bash
#SBATCH --job-name=madengine-{{ model_name }}
#SBATCH --output={{ output_dir }}/madengine-{{ model_name }}_%j_%t.out
#SBATCH --error={{ output_dir }}/madengine-{{ model_name }}_%j_%t.err
#SBATCH --partition={{ partition }}
#SBATCH --nodes={{ nodes }}
#SBATCH --ntasks={{ nodes }}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node={{ gpus_per_node }}
#SBATCH --time={{ time_limit }}
{% if exclusive %}
#SBATCH --exclusive
{% endif %}
{% if qos %}
#SBATCH --qos={{ qos }}
{% endif %}
{% if account %}
#SBATCH --account={{ account }}
{% endif %}

# =============================================================================
# SLURM Job Configuration Generated by madengine-cli
# Model: {{ model_name }}
# Deployment: {{ nodes }} nodes x {{ gpus_per_node }} GPUs
# =============================================================================

# Load required modules
{% for module in modules %}
module load {{ module }}
{% endfor %}

# =============================================================================
# Environment Setup (Standard ML Environment Variables)
# =============================================================================

# Distributed training environment (auto-configured from SLURM)
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT={{ master_port | default(29500) }}
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export NNODES={{ nodes }}
export GPUS_PER_NODE={{ gpus_per_node }}

# GPU visibility (ROCm/CUDA)
export ROCR_VISIBLE_DEVICES=$(seq -s, 0 $(({{ gpus_per_node }}-1)))
export CUDA_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES

# Network configuration
{% if network_interface %}
export NCCL_SOCKET_IFNAME={{ network_interface }}
export GLOO_SOCKET_IFNAME={{ network_interface }}
{% endif %}

# Distributed backend configuration
{% if distributed_backend %}
export DISTRIBUTED_BACKEND={{ distributed_backend }}
{% endif %}

# Application-specific environment variables
{% for key, value in env_vars.items() %}
export {{ key }}="{{ value }}"
{% endfor %}

# madengine environment
export MAD_SLURM_JOB_ID=$SLURM_JOB_ID
export MAD_NODE_RANK=$SLURM_NODEID
export MAD_TOTAL_NODES={{ nodes }}

# =============================================================================
# Workspace Setup
# =============================================================================

{% if shared_workspace %}
# Use shared workspace (NFS/Lustre)
WORKSPACE={{ shared_workspace }}
{% else %}
# Use node-local scratch
WORKSPACE=$SLURM_TMPDIR
{% endif %}

cd $WORKSPACE

# Copy required files
{% if manifest_file %}
cp {{ manifest_file }} $WORKSPACE/build_manifest.json
{% endif %}
{% if credential_file %}
cp {{ credential_file }} $WORKSPACE/credential.json
{% endif %}
{% if data_file %}
cp {{ data_file }} $WORKSPACE/data.json
{% endif %}

# =============================================================================
# Execute madengine Workflow
# =============================================================================

madengine run \
    {% if manifest_file %}--manifest-file build_manifest.json{% else %}--tags {{ tags }}{% endif %} \
    --timeout {{ timeout | default(3600) }} \
    {% if shared_data %}--force-mirror-local {{ shared_data }}{% endif %} \
    {% if live_output %}--live-output{% endif %}

EXIT_CODE=$?

# =============================================================================
# Collect Results
# =============================================================================

{% if results_dir %}
# Copy performance results to shared location
if [ -f "perf.csv" ]; then
    cp perf.csv {{ results_dir }}/perf_${SLURM_JOB_ID}_node${SLURM_NODEID}.csv
fi

# Copy logs
cp {{ output_dir }}/madengine-{{ model_name }}_${SLURM_JOB_ID}_${SLURM_PROCID}.out \
   {{ results_dir }}/logs/ 2>/dev/null || true
{% endif %}

echo "Node $SLURM_NODEID completed with exit code $EXIT_CODE"
exit $EXIT_CODE
```

**Key Features**:
- ✅ **Standard Environment Variables**: Uses SLURM_*, MASTER_ADDR, RANK, etc.
- ✅ **No Manual Configuration**: SLURM auto-provides node topology
- ✅ **Clean Separation**: Infrastructure (SLURM) vs Application (model)
- ✅ **Flexible Storage**: Shared filesystem or node-local scratch
- ✅ **Production-Ready**: Error handling, logging, result collection
- ✅ **Self-Documenting**: Clear sections with comments

#### 4.2.2 Comparison: Old vs New Multi-Node Design

| Aspect | Old Manual Multi-Node | Old slurm_args | New Unified Design ✅ |
|--------|----------------------|----------------|----------------------|
| **User Experience** | SSH to each node manually | Single command | Single command |
| **Command** | Run on each node with NODE_RANK | `--additional-context '{slurm_args: {...}}'` | `--additional-context '{deploy: slurm, ...}'` |
| **SLURM Submission** | Manual (user manages) | Model script calls sbatch | madengine generates sbatch |
| **Workflow** | Full madengine automation | Bypasses madengine, direct model exec | Full madengine automation |
| **Data Download** | ✅ Yes (dataprovider) | ❌ No (manual in model) | ✅ Yes (dataprovider) |
| **Pre-scripts** | ✅ Yes (rocEnvTool) | ❌ No | ✅ Yes (rocEnvTool) |
| **Profiling** | ✅ Yes | ❌ No | ✅ Yes |
| **Post-scripts** | ✅ Yes | ❌ No | ✅ Yes |
| **Centralized** | N/A | ❌ Model-specific scripts | ✅ Centralized templates |
| **Job Management** | ❌ Manual | ✅ SLURM | ✅ SLURM |
| **Error Handling** | ❌ Manual | ⚠️ Limited | ✅ Full madengine error handling |

**Concrete Example** (Megatron-LM 4-node training):

<details>
<summary>Old Manual Multi-Node (click to expand)</summary>

```bash
# Must SSH to 4 nodes and run separately:
ssh node0 "madengine run --tags pyt_megatron_lm_train_llama2_7b \
  --additional-context '{\"multi_node_args\": {\"RUNNER\": \"torchrun\", \"MASTER_ADDR\": \"10.194.129.113\", \"MASTER_PORT\": \"4000\", \"NNODES\": \"4\", \"NODE_RANK\": \"0\", \"NCCL_SOCKET_IFNAME\": \"ens14np0\"}}' \
  --force-mirror-local /nfs/data"

ssh node1 "madengine run --tags pyt_megatron_lm_train_llama2_7b \
  --additional-context '{\"multi_node_args\": {\"RUNNER\": \"torchrun\", \"MASTER_ADDR\": \"10.194.129.113\", \"MASTER_PORT\": \"4000\", \"NNODES\": \"4\", \"NODE_RANK\": \"1\", \"NCCL_SOCKET_IFNAME\": \"ens14np0\"}}' \
  --force-mirror-local /nfs/data"

# ... node2, node3 ...
# Problem: Manual, error-prone, no job scheduling
```
</details>

<details>
<summary>Old slurm_args (click to expand)</summary>

```bash
# Bypasses madengine automation:
madengine run --tags sglang_disagg \
  --additional-context '{
    "slurm_args": {
      "FRAMEWORK": "sglang_disagg",
      "PREFILL_NODES": "2",
      "DECODE_NODES": "2",
      "PARTITION": "amd-rccl"
    }
  }'

# Problem: 
# - Skips madengine workflow
# - Calls scripts/sglang_disagg/run.sh directly
# - No data download, pre/post-scripts, profiling automation
# - Model-specific SLURM logic
```
</details>

**New Unified Approach** ✅:

```bash
# Single command for Megatron-LM 4-node training
madengine-cli run --tags pyt_megatron_lm_train_llama2_7b \
  --additional-context '{
    "deploy": "slurm",
    "slurm": {
      "partition": "gpu",
      "nodes": 4,
      "gpus_per_node": 8,
      "time": "24:00:00",
      "exclusive": true
    },
    "multi_node_args": {
      "RUNNER": "torchrun",
      "MASTER_PORT": "29500",
      "NCCL_SOCKET_IFNAME": "ens14np0",
      "GLOO_SOCKET_IFNAME": "ens14np0"
    },
    "shared_data": "/nfs/data"
  }'

# What happens:
# 1. madengine generates sbatch script
# 2. Submits to SLURM (sbatch job.sh)
# 3. SLURM allocates 4 nodes
# 4. Each node automatically runs:
#    madengine run --manifest-file build_manifest.json \
#      --additional-context '{
#        "multi_node_args": {
#          "RUNNER": "torchrun",
#          "MASTER_ADDR": "<auto-from-slurm>",
#          "MASTER_PORT": "29500",
#          "NNODES": "4",
#          "NODE_RANK": "<auto-from-slurm>",
#          "NCCL_SOCKET_IFNAME": "ens14np0"
#        }
#      }' \
#      --force-mirror-local /nfs/data
# 5. All madengine automation works on each node
# 6. Results aggregated from all nodes
```

**Benefits**:
- ✅ Single command (vs 4 SSH commands)
- ✅ SLURM job management (queue, priorities, monitoring)
- ✅ Auto-configures MASTER_ADDR and NODE_RANK
- ✅ Full madengine automation on every node
- ✅ Centralized, maintainable

#### 4.2.3 SLURM Deployment Implementation (Production-Ready with Classes)

**File**: `src/madengine/deployment/slurm.py`

```python
import os
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader

from .base import (
    BaseDeployment,
    DeploymentConfig,
    DeploymentResult,
    DeploymentStatus
)


class SlurmDeployment(BaseDeployment):
    """
    SLURM HPC cluster deployment.
    
    Generates sbatch script and submits to SLURM scheduler.
    Each node runs madengine with standard distributed environment variables.
    """
    
    DEPLOYMENT_TYPE = "slurm"
    REQUIRED_TOOLS = ["sbatch", "squeue", "scontrol"]
    
    def __init__(self, config: DeploymentConfig):
        super().__init__(config)
        
        # Parse SLURM configuration
        self.slurm_config = config.additional_context.get("slurm", {})
        self.distributed_config = config.additional_context.get("distributed", {})
        
        # SLURM parameters
        self.partition = self.slurm_config.get("partition", "gpu")
        self.nodes = self.slurm_config.get("nodes", 1)
        self.gpus_per_node = self.slurm_config.get("gpus_per_node", 8)
        self.time_limit = self.slurm_config.get("time", "24:00:00")
        self.output_dir = Path(self.slurm_config.get("output_dir", "./slurm_output"))
        self.login_node = self.slurm_config.get("login_node")
        
        # Setup Jinja2 template engine
        template_dir = Path(__file__).parent / "templates" / "slurm"
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        
        # Generated script path
        self.script_path = None
    
    def validate(self) -> bool:
        """Validate SLURM environment and configuration"""
        # Check required tools
        for tool in self.REQUIRED_TOOLS:
            cmd = ["which", tool]
            if self.login_node:
                cmd = ["ssh", self.login_node] + cmd
            
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                self.console.print(f"[red]✗ Required tool not found: {tool}[/red]")
                return False
        
        # Validate configuration
        if self.nodes < 1:
            self.console.print(f"[red]✗ Invalid nodes: {self.nodes}[/red]")
            return False
        
        if self.gpus_per_node < 1:
            self.console.print(f"[red]✗ Invalid GPUs per node: {self.gpus_per_node}[/red]")
            return False
        
        self.console.print(f"[green]✓ SLURM environment validated[/green]")
        return True
    
    def prepare(self) -> bool:
        """Generate sbatch script from template"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get model info from manifest
            model_keys = list(self.manifest["built_models"].keys())
            if not model_keys:
                raise ValueError("No models in manifest")
            
            model_key = model_keys[0]
            model_info = self.manifest["built_models"][model_key]
            
            # Prepare template context
            context = self._prepare_template_context(model_info)
            
            # Render template
            template = self.jinja_env.get_template("job.sh.j2")
            script_content = template.render(**context)
            
            # Save script
            self.script_path = self.output_dir / f"madengine_{model_info['name']}.sh"
            self.script_path.write_text(script_content)
            self.script_path.chmod(0o755)
            
            self.console.print(f"[green]✓ Generated sbatch script: {self.script_path}[/green]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]✗ Failed to generate script: {e}[/red]")
            return False
    
    def _prepare_template_context(self, model_info: Dict) -> Dict[str, Any]:
        """Prepare context for Jinja2 template rendering"""
        return {
            "model_name": model_info["name"],
            "manifest_file": os.path.abspath(self.config.manifest_file),
            "partition": self.partition,
            "nodes": self.nodes,
            "gpus_per_node": self.gpus_per_node,
            "time_limit": self.time_limit,
            "output_dir": str(self.output_dir),
            "master_port": self.distributed_config.get("port", 29500),
            "distributed_backend": self.distributed_config.get("backend", "nccl"),
            "network_interface": self.slurm_config.get("network_interface"),
            "exclusive": self.slurm_config.get("exclusive", True),
            "qos": self.slurm_config.get("qos"),
            "account": self.slurm_config.get("account"),
            "modules": self.slurm_config.get("modules", []),
            "env_vars": self.config.additional_context.get("env_vars", {}),
            "shared_workspace": self.slurm_config.get("shared_workspace"),
            "shared_data": self.config.additional_context.get("shared_data"),
            "results_dir": self.slurm_config.get("results_dir"),
            "timeout": self.config.timeout,
            "live_output": self.config.additional_context.get("live_output", False),
            "tags": " ".join(model_info.get("tags", [])),
            "credential_file": "credential.json" if Path("credential.json").exists() else None,
            "data_file": "data.json" if Path("data.json").exists() else None,
        }
    
    def deploy(self) -> DeploymentResult:
        """Submit sbatch script to SLURM"""
        if not self.script_path or not self.script_path.exists():
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id="",
                message="Script not generated. Run prepare() first."
            )
        
        try:
            # Submit job
            cmd = ["sbatch", str(self.script_path)]
            if self.login_node:
                cmd = ["ssh", self.login_node] + cmd
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Parse job ID: "Submitted batch job 12345"
                job_id = result.stdout.strip().split()[-1]
                
                self.console.print(f"[green]✓ Submitted SLURM job: {job_id}[/green]")
                self.console.print(f"  Nodes: {self.nodes} x {self.gpus_per_node} GPUs")
                self.console.print(f"  Partition: {self.partition}")
                
                return DeploymentResult(
                    status=DeploymentStatus.SUCCESS,
                    deployment_id=job_id,
                    message=f"SLURM job {job_id} submitted successfully",
                    logs_path=str(self.output_dir)
                )
            else:
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    deployment_id="",
                    message=f"sbatch failed: {result.stderr}"
                )
                
        except subprocess.TimeoutExpired:
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id="",
                message="sbatch submission timed out"
            )
        except Exception as e:
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id="",
                message=f"Deployment error: {str(e)}"
            )
    
    def monitor(self, deployment_id: str) -> DeploymentResult:
        """Check SLURM job status"""
        try:
            # Query job status
            cmd = ["squeue", "-j", deployment_id, "-h", "-o", "%T"]
            if self.login_node:
                cmd = ["ssh", self.login_node] + cmd
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                # Job not found - likely completed or failed
                return self._check_job_completion(deployment_id)
            
            status = result.stdout.strip().upper()
            
            if status in ["RUNNING", "PENDING", "CONFIGURING"]:
                return DeploymentResult(
                    status=DeploymentStatus.RUNNING,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} is {status.lower()}"
                )
            elif status in ["COMPLETED"]:
                return DeploymentResult(
                    status=DeploymentStatus.SUCCESS,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} completed successfully"
                )
            else:  # FAILED, CANCELLED, TIMEOUT, etc.
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} {status.lower()}"
                )
                
        except Exception as e:
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id=deployment_id,
                message=f"Monitor error: {str(e)}"
            )
    
    def _check_job_completion(self, job_id: str) -> DeploymentResult:
        """Check completed job status using sacct"""
        try:
            cmd = ["sacct", "-j", job_id, "-n", "-X", "-o", "State"]
            if self.login_node:
                cmd = ["ssh", self.login_node] + cmd
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                status = result.stdout.strip().upper()
                if "COMPLETED" in status:
                    return DeploymentResult(
                        status=DeploymentStatus.SUCCESS,
                        deployment_id=job_id,
                        message=f"Job {job_id} completed"
                    )
                else:
                    return DeploymentResult(
                        status=DeploymentStatus.FAILED,
                        deployment_id=job_id,
                        message=f"Job {job_id} failed: {status}"
                    )
            
            # Fallback - assume completed
            return DeploymentResult(
                status=DeploymentStatus.SUCCESS,
                deployment_id=job_id,
                message=f"Job {job_id} completed (assumed)"
            )
            
        except Exception:
            return DeploymentResult(
                status=DeploymentStatus.SUCCESS,
                deployment_id=job_id,
                message=f"Job {job_id} completed (status unavailable)"
            )
    
    def collect_results(self, deployment_id: str) -> Dict[str, Any]:
        """Collect performance results from SLURM output files"""
        results = {
            "job_id": deployment_id,
            "nodes": self.nodes,
            "gpus_per_node": self.gpus_per_node,
            "perf_files": [],
            "logs": []
        }
        
        try:
            # Find output files
            output_pattern = f"madengine-*_{deployment_id}_*.out"
            output_files = list(self.output_dir.glob(output_pattern))
            
            results["logs"] = [str(f) for f in output_files]
            
            # Find performance CSV files
            if self.slurm_config.get("results_dir"):
                results_dir = Path(self.slurm_config["results_dir"])
                perf_pattern = f"perf_{deployment_id}_*.csv"
                perf_files = list(results_dir.glob(perf_pattern))
                results["perf_files"] = [str(f) for f in perf_files]
            
            self.console.print(f"[green]✓ Collected results: {len(results['perf_files'])} perf files, "
                             f"{len(results['logs'])} log files[/green]")
            
        except Exception as e:
            self.console.print(f"[yellow]⚠ Results collection incomplete: {e}[/yellow]")
        
        return results
    
    def cleanup(self, deployment_id: str) -> bool:
        """Cancel SLURM job if still running"""
        try:
            cmd = ["scancel", deployment_id]
            if self.login_node:
                cmd = ["ssh", self.login_node] + cmd
            
            subprocess.run(cmd, capture_output=True, timeout=10)
            self.console.print(f"[yellow]Cancelled SLURM job: {deployment_id}[/yellow]")
            return True
            
        except Exception as e:
            self.console.print(f"[yellow]⚠ Cleanup warning: {e}[/yellow]")
            return False
```

**Key Production Features**:
- ✅ **Proper Class Structure**: Inherits from BaseDeployment
- ✅ **Validation**: Checks tools, configuration before deployment
- ✅ **Error Handling**: Try/catch with timeout, proper error messages
- ✅ **Separation of Concerns**: prepare, deploy, monitor, collect are separate
- ✅ **Testability**: Each method can be mocked and tested
- ✅ **Status Tracking**: Uses enum for type-safe status
- ✅ **Result Collection**: Gathers logs and performance files
- ✅ **Cleanup**: Can cancel jobs on failure
- ✅ **Production-Ready**: Timeouts, logging, error recovery
    
    # Extract SLURM parameters
    partition = slurm_config.get("partition", "gpu")
    nodes = slurm_config.get("nodes", 1)
    gpus_per_node = slurm_config.get("gpus_per_node", 8)
    time_limit = slurm_config.get("time", "24:00:00")
    output_dir = slurm_config.get("output_dir", "./slurm_output")
    
    # Setup Jinja2
    template_dir = Path(__file__).parent / "templates" / "slurm"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("job.sh.j2")
    
    # Get model info from manifest
    model_keys = list(manifest["built_models"].keys())
    model_info = manifest["built_models"][model_keys[0]]
    
    # Render sbatch script
    script_content = template.render(
        model_name=model_info["name"],
        manifest_file=os.path.abspath(manifest_file),
        partition=partition,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        time_limit=time_limit,
        output_dir=output_dir,
        master_port=multi_node_args.get("MASTER_PORT", "29500"),
        runner=multi_node_args.get("RUNNER", "torchrun"),
        nccl_socket_ifname=multi_node_args.get("NCCL_SOCKET_IFNAME"),
        exclusive=slurm_config.get("exclusive", True),
        modules=slurm_config.get("modules", []),
        env_vars=additional_context.get("env_vars", {}),
        shared_data=additional_context.get("shared_data"),
        tags=" ".join(model_info.get("tags", [])),
        credential_file="credential.json" if Path("credential.json").exists() else None,
        data_file="data.json" if Path("data.json").exists() else None,
        timeout=additional_context.get("timeout", 3600),
        live_output=additional_context.get("live_output", False)
    )
    
    # Save sbatch script
    os.makedirs(output_dir, exist_ok=True)
    script_file = Path(output_dir) / f"madengine_{model_info['name']}.sh"
    script_file.write_text(script_content)
    script_file.chmod(0o755)
    
    console.print(f"✓ Generated SLURM script: {script_file}")
    
    # Submit to SLURM
    result = subprocess.run(
        ["sbatch", str(script_file)],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        # Parse job ID: "Submitted batch job 12345"
        job_id = result.stdout.strip().split()[-1]
        console.print(f"[green]✓ Submitted SLURM job: {job_id}[/green]")
        
        # Monitor job (optional)
        if additional_context.get("monitor", True):
            monitor_slurm_job(job_id, slurm_config.get("login_node"))
        
        return {"status": "success", "job_id": job_id}
    else:
        console.print(f"[red]✗ Failed to submit SLURM job:[/red]\n{result.stderr}")
        return {"status": "failed", "error": result.stderr}


def monitor_slurm_job(job_id: str, login_node: str = None):
    """Monitor SLURM job until completion"""
    import time
    
    while True:
        # Check job status
        cmd = ["squeue", "-j", job_id, "-h"]
        if login_node:
            cmd = ["ssh", login_node] + cmd
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if not result.stdout.strip():
            # Job completed
            console.print(f"[green]✓ SLURM job {job_id} completed[/green]")
            break
        
        # Still running
        console.print(f"⏳ Job {job_id} running... (checking again in 30s)")
        time.sleep(30)
```

**Key Simplifications**:
- ✅ Simple function (not complex class hierarchy)
- ✅ Generates sbatch script with Jinja2
- ✅ Submits to SLURM with subprocess
- ✅ Optional job monitoring
- ✅ ~100 lines vs ~400 lines in class-based approach
        self.work_dir = slurm_config.get("work_dir", os.getcwd())
        
        # Setup Jinja2 for template rendering
        template_dir = Path(__file__).parent / "templates" / "slurm"
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    def validate(self) -> bool:
        """Validate SLURM deployment requirements"""
        # Check if sbatch is available (or SSH to login node)
        if self.login_node:
            # SSH validation
            result = subprocess.run(
                ["ssh", self.login_node, "which", "sbatch"],
                capture_output=True
            )
            return result.returncode == 0
        else:
            # Local sbatch
            result = subprocess.run(["which", "sbatch"], capture_output=True)
            return result.returncode == 0
    
    def prepare(self) -> bool:
        """Prepare SLURM deployment (generate sbatch scripts)"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate sbatch script for each model
        for model_name, model_info in self.manifest["built_images"].items():
            job_script = self._generate_job_script(model_name, model_info)
            
            script_path = Path(self.output_dir) / f"{model_name}_job.sh"
            with open(script_path, "w") as f:
                f.write(job_script)
            
            # Make executable
            os.chmod(script_path, 0o755)
        
        return True
    
    def _generate_job_script(self, model_name: str, model_info: dict) -> str:
        """Generate sbatch script using Jinja2 template"""
        template = self.jinja_env.get_template("job.sh.j2")
        
        # Prepare template context
        execution = model_info.get("execution", {})
        
        context = {
            "job_name": model_name,
            "output_dir": self.output_dir,
            "partition": self.partition,
            "nnodes": execution.get("nnodes", self.config.nnodes),
            "nproc_per_node": execution.get("nproc_per_node", self.config.nproc_per_node),
            "time_limit": self._format_time(self.config.timeout),
            "master_port": execution.get("master_port", 29500),
            "world_size": execution.get("nnodes", 1) * execution.get("nproc_per_node", 1),
            "modules": self.config.context.get("slurm", {}).get("modules", []),
            "env_vars": self.config.context.get("env_vars", {}),
            "launcher": self.config.launcher,
            "container_image": model_info.get("registry_image"),
            "work_dir": self.work_dir,
            "run_command": self._get_run_command(model_info),
        }
        
        return template.render(**context)
    
    def _get_run_command(self, model_info: dict) -> str:
        """Get the run command from model info"""
        # Default: run.sh from model scripts
        return "./run.sh"
    
    def _format_time(self, seconds: int) -> str:
        """Format timeout in SLURM time format (HH:MM:SS)"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def deploy(self) -> DeploymentResult:
        """Submit SLURM jobs"""
        job_ids = []
        
        for model_name in self.manifest["built_images"].keys():
            script_path = Path(self.output_dir) / f"{model_name}_job.sh"
            
            # Submit job
            if self.login_node:
                cmd = ["ssh", self.login_node, "sbatch", str(script_path)]
            else:
                cmd = ["sbatch", str(script_path)]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse job ID from output: "Submitted batch job 12345"
                job_id = result.stdout.strip().split()[-1]
                job_ids.append(job_id)
            else:
                return DeploymentResult(
                    status="failed",
                    deployment_id="",
                    message=f"Failed to submit {model_name}: {result.stderr}"
                )
        
        return DeploymentResult(
            status="success",
            deployment_id=",".join(job_ids),
            message=f"Submitted {len(job_ids)} SLURM jobs"
        )
    
    def monitor(self, deployment_id: str) -> DeploymentResult:
        """Monitor SLURM job status"""
        job_ids = deployment_id.split(",")
        
        # Check status using squeue
        if self.login_node:
            cmd = ["ssh", self.login_node, "squeue", "-j", deployment_id, "-h"]
        else:
            cmd = ["squeue", "-j", deployment_id, "-h"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if not result.stdout.strip():
            # Job completed or not found
            return DeploymentResult(
                status="success",
                deployment_id=deployment_id,
                message="Jobs completed"
            )
        else:
            # Jobs still running
            return DeploymentResult(
                status="pending",
                deployment_id=deployment_id,
                message=f"{len(job_ids)} jobs running"
            )
    
    def collect_results(self, deployment_id: str) -> Dict:
        """Collect results from SLURM output files"""
        results = {}
        
        for model_name in self.manifest["built_images"].keys():
            # Parse output files
            pattern = f"{self.output_dir}/{model_name}_job_*.out"
            output_files = glob.glob(pattern)
            
            for output_file in output_files:
                # Parse performance metrics from output
                # This depends on model output format
                pass
        
        return results
    
    def cleanup(self, deployment_id: str) -> bool:
        """Cleanup SLURM jobs if needed"""
        # Cancel any remaining jobs
        job_ids = deployment_id.split(",")
        
        if self.login_node:
            cmd = ["ssh", self.login_node, "scancel"] + job_ids
        else:
            cmd = ["scancel"] + job_ids
        
        subprocess.run(cmd, capture_output=True)
        return True
```

---

### 4.3 Phase 3: Kubernetes Deployment (Week 5-6)

#### 4.3.1 Kubernetes Template (Using AMD GPU Device Plugin)

**File**: `src/madengine/deployment/templates/kubernetes/job.yaml.j2`

**Key Design**: 
- Uses built Docker image from build phase
- Requests AMD GPUs via AMD GPU Device Plugin ([k8s-device-plugin](https://github.com/ROCm/k8s-device-plugin))
- Runs same madengine workflow as local execution

**Prerequisites**: AMD GPU Device Plugin must be deployed (DaemonSet):
```bash
kubectl create -f https://raw.githubusercontent.com/ROCm/k8s-device-plugin/master/k8s-ds-amdgpu-dp.yaml
```

**Job Manifest Template**:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: madengine-{{ model_name | lower | replace("_", "-") }}
  namespace: {{ namespace }}
  labels:
    app: madengine
    model: {{ model_name }}
    madengine-job: "true"
spec:
  backoffLimit: {{ backoff_limit | default(3) }}
  completions: 1
  parallelism: 1
  template:
    metadata:
      labels:
        app: madengine
        model: {{ model_name }}
    spec:
      restartPolicy: Never
      
      {% if node_selector %}
      nodeSelector:
        {% for key, value in node_selector.items() %}
        {{ key }}: "{{ value }}"
        {% endfor %}
      {% endif %}
      
      {% if tolerations %}
      tolerations:
      {% for toleration in tolerations %}
      - key: {{ toleration.key }}
        operator: {{ toleration.operator | default("Equal") }}
        value: {{ toleration.value | default("") }}
        effect: {{ toleration.effect | default("NoSchedule") }}
      {% endfor %}
      {% endif %}
      
      containers:
      - name: madengine-{{ model_name | lower }}
        # Use built Docker image from build phase (build_manifest.json)
        image: {{ registry_image }}
        imagePullPolicy: {{ image_pull_policy | default("Always") }}
        
        workingDir: /workspace
        
        command: ["/bin/bash", "-c"]
        args:
          - |
            set -e
            
            echo "==================================================================="
            echo "MADEngine Kubernetes Job"
            echo "Model: {{ model_name }}"
            echo "Namespace: {{ namespace }}"
            echo "Node: $(hostname)"
            echo "==================================================================="
            
            # GPU Information
            if command -v rocminfo &> /dev/null; then
                echo "AMD GPU Information:"
                rocminfo | grep -E "(Name|Device ID|Compute Unit)" || true
            fi
            
            # Set GPU visibility (K8s AMD GPU Device Plugin handles device allocation)
            export ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-0}
            export MAD_SYSTEM_GPU_ARCHITECTURE={{ gpu_architecture | default("gfx90a") }}
            
            # Kubernetes-specific environment
            export MAD_K8S_POD_NAME=${HOSTNAME}
            export MAD_K8S_NAMESPACE={{ namespace }}
            export MAD_K8S_JOB=true
            
            # Additional environment variables from --additional-context
            {% for key, value in env_vars.items() %}
            export {{ key }}="{{ value }}"
            {% endfor %}
            
            # Run MAD model's run.sh (madengine automation workflow)
            # 1. Data download (if dataprovider configured)
            # 2. Pre-scripts (rocEnvTool, GPU info, profiling start)
            # 3. Model benchmark execution
            # 4. Post-scripts (profiling end, metrics collection)
            # 5. Generate perf.csv
            
            cd /workspace
            bash run.sh
            
            EXIT_CODE=$?
            
            # Copy results to shared storage (if configured)
            {% if results_pvc %}
            if [ -f "perf.csv" ]; then
                cp perf.csv /results/perf_{{ model_name }}_${HOSTNAME}.csv
                echo "Results saved to /results/perf_{{ model_name }}_${HOSTNAME}.csv"
            fi
            {% endif %}
            
            echo "Job completed with exit code $EXIT_CODE"
            exit $EXIT_CODE
        
        # AMD GPU Device Plugin resource requests
        # Ref: https://github.com/ROCm/k8s-device-plugin
        resources:
          requests:
            {{ gpu_resource_name }}: "{{ gpu_count }}"
            memory: "{{ memory }}"
            cpu: "{{ cpu }}"
          limits:
            {{ gpu_resource_name }}: "{{ gpu_count }}"
            memory: "{{ memory_limit }}"
            cpu: "{{ cpu_limit }}"
        
        volumeMounts:
        {% if results_pvc %}
        - name: results
          mountPath: /results
        {% endif %}
        {% if data_pvc %}
        - name: data
          mountPath: /data
          readOnly: true
        {% endif %}
        {% if shared_storage_pvc %}
        - name: shared-storage
          mountPath: /shared
        {% endif %}
        {% for volume in custom_volumes %}
        - name: {{ volume.name }}
          mountPath: {{ volume.mount_path }}
          {% if volume.read_only %}readOnly: true{% endif %}
        {% endfor %}
        
        {% if security_context %}
        securityContext:
          {% if security_context.run_as_user %}
          runAsUser: {{ security_context.run_as_user }}
          {% endif %}
          {% if security_context.run_as_group %}
          runAsGroup: {{ security_context.run_as_group }}
          {% endif %}
          capabilities:
            add:
            - SYS_PTRACE  # For rocprof/profiling
        {% endif %}
      
      volumes:
      {% if results_pvc %}
      - name: results
        persistentVolumeClaim:
          claimName: {{ results_pvc }}
      {% endif %}
      {% if data_pvc %}
      - name: data
        persistentVolumeClaim:
          claimName: {{ data_pvc }}
      {% endif %}
      {% if shared_storage_pvc %}
      - name: shared-storage
        persistentVolumeClaim:
          claimName: {{ shared_storage_pvc }}
      {% endif %}
      {% for volume in custom_volumes %}
      - name: {{ volume.name }}
        {% if volume.type == "pvc" %}
        persistentVolumeClaim:
          claimName: {{ volume.claim_name }}
        {% elif volume.type == "configmap" %}
        configMap:
          name: {{ volume.config_name }}
        {% elif volume.type == "secret" %}
        secret:
          secretName: {{ volume.secret_name }}
        {% elif volume.type == "emptydir" %}
        emptyDir: {}
        {% endif %}
      {% endfor %}
```

**Key Features**:
- ✅ **AMD GPU Device Plugin Integration**: Uses `amd.com/gpu` resource name
- ✅ **Node Selection**: Can target specific GPU models via node labels
- ✅ **Built Image**: Uses pre-built Docker image from `build_manifest.json`
- ✅ **Same Workflow**: Runs MAD model's automation (data, pre/post-scripts, profiling)
- ✅ **Result Collection**: Supports PVC for shared results storage
- ✅ **Security**: Optional securityContext for profiling capabilities
- ✅ **Production-Ready**: Error handling, logging, exit codes

**Example --additional-context for K8s**:

```json
{
  "deploy": "k8s",
  "k8s": {
    "namespace": "ml-workloads",
    "gpu_resource_name": "amd.com/gpu",
    "gpu_count": 8,
    "memory": "256Gi",
    "memory_limit": "512Gi",
    "cpu": "64",
    "cpu_limit": "128",
    "node_selector": {
      "amd.com/gpu.device.id": "0x74a1",
      "node-role.kubernetes.io/worker": "true"
    },
    "results_pvc": "madengine-results",
    "data_pvc": "ml-datasets",
    "tolerations": [
      {
        "key": "nvidia.com/gpu",
        "operator": "Exists",
        "effect": "NoSchedule"
      }
    ]
  }
}
```

#### 4.3.2 Kubernetes Deployment Implementation (Simplified)

**File**: `src/madengine/deployment/kubernetes.py`

**Simple function-based approach** (no complex classes):

```python
import os
import json
import yaml
import subprocess
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from rich.console import Console

console = Console()


def deploy_to_k8s(manifest_file: str, additional_context: dict):
    """
    Deploy to Kubernetes cluster - generates and applies Job manifest.
    
    Pod uses built Docker image, runs same workflow as local (no docker-in-docker).
    """
    # Load manifest
    with open(manifest_file) as f:
        manifest = json.load(f)
    
    # Get K8s configuration
    k8s_config = additional_context.get("k8s", {})
    namespace = k8s_config.get("namespace", "default")
    output_dir = k8s_config.get("output_dir", "./k8s_manifests")
    kubeconfig = k8s_config.get("kubeconfig")
    
    # Setup Jinja2
    template_dir = Path(__file__).parent / "templates" / "kubernetes"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("job.yaml.j2")
    
    # Get model and image info from manifest
    model_keys = list(manifest["built_models"].keys())
    model_key = model_keys[0]
    model_info = manifest["built_models"][model_key]
    image_info = manifest["built_images"][model_key]
    
    # Render Job manifest
    job_content = template.render(
        model_name=model_info["name"].lower().replace("_", "-"),
        namespace=namespace,
        registry_image=image_info["registry_image"],  # Built image from build phase
        gpu_count=model_info.get("n_gpus", 1),
        gpu_vendor=manifest["context"].get("gpu_vendor", "AMD"),
        gpu_architecture=manifest["context"].get("gpu_architecture", "gfx90a"),
        memory=k8s_config.get("memory", "128Gi"),
        memory_limit=k8s_config.get("memory_limit", "256Gi"),
        cpu=k8s_config.get("cpu", "32"),
        cpu_limit=k8s_config.get("cpu_limit", "64"),
        node_selector=k8s_config.get("node_selector", {}),
        env_vars=additional_context.get("env_vars", {}),
        model_scripts_path=model_info.get("scripts"),
        data_volume=k8s_config.get("data_volume"),
        data_pvc_name=k8s_config.get("data_pvc_name", "ml-data"),
        custom_volumes=k8s_config.get("volumes", [])
    )
    
    # Save manifest
    os.makedirs(output_dir, exist_ok=True)
    manifest_file = Path(output_dir) / f"madengine_{model_info['name']}.yaml"
    manifest_file.write_text(job_content)
    
    console.print(f"✓ Generated K8s manifest: {manifest_file}")
    
    # Apply to cluster
    cmd = ["kubectl", "apply", "-f", str(manifest_file), "-n", namespace]
    if kubeconfig:
        cmd.extend(["--kubeconfig", kubeconfig])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        job_name = f"madengine-{model_info['name'].lower().replace('_', '-')}"
        console.print(f"[green]✓ Deployed to Kubernetes: {job_name}[/green]")
        
        # Monitor job (optional)
        if additional_context.get("monitor", True):
            monitor_k8s_job(job_name, namespace, kubeconfig)
        
        return {"status": "success", "job_name": job_name}
    else:
        console.print(f"[red]✗ Failed to deploy to K8s:[/red]\n{result.stderr}")
        return {"status": "failed", "error": result.stderr}


def monitor_k8s_job(job_name: str, namespace: str, kubeconfig: str = None):
    """Monitor Kubernetes Job until completion"""
    import time
    
    while True:
        # Check job status
        cmd = ["kubectl", "get", "job", job_name, "-n", namespace, "-o", "json"]
        if kubeconfig:
            cmd.extend(["--kubeconfig", kubeconfig])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            console.print(f"[red]✗ Failed to get job status[/red]")
            break
        
        job_status = json.loads(result.stdout).get("status", {})
        
        if job_status.get("succeeded"):
            console.print(f"[green]✓ K8s job {job_name} completed successfully[/green]")
            break
        elif job_status.get("failed"):
            console.print(f"[red]✗ K8s job {job_name} failed[/red]")
            break
        
        # Still running
        console.print(f"⏳ Job {job_name} running... (checking again in 30s)")
        time.sleep(30)
```

**Key Simplifications**:
- ✅ Simple function (not complex class hierarchy)
- ✅ Uses built Docker image from build phase (no docker-in-docker)
- ✅ Generates Job manifest with Jinja2
- ✅ Applies with kubectl
- ✅ Optional job monitoring
- ✅ ~80 lines vs ~300 lines in class-based approach

---

### 4.4 Phase 4: CLI Integration (Week 3)

#### 4.4.1 Refactor mad_cli.py (Using Factory Pattern)

**Changes to** `src/madengine/mad_cli.py`:

```python
# mad_cli.py updates - Clean integration with DeploymentFactory

from madengine.deployment.factory import DeploymentFactory
from madengine.deployment.base import DeploymentStatus

@app.command(name="run")
def run_command(
    tags: List[str] = typer.Option([], "--tags", "-t"),
    manifest_file: str = typer.Option("", "--manifest-file", "-m"),
    timeout: int = typer.Option(3600, "--timeout"),
    additional_context: str = typer.Option("{}", "--additional-context", "-c"),
    additional_context_file: Optional[str] = typer.Option(None, "--additional-context-file", "-f"),
    live_output: bool = typer.Option(False, "--live-output", "-l"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    Run models locally or deploy to SLURM/K8s.
    
    All configuration via --additional-context (stored in build_manifest.json):
    
    Examples:
      # Local single-node
      madengine-cli run --tags bert
      
      # SLURM multi-node
      madengine-cli run --tags bert --additional-context '{"deploy": "slurm", "slurm": {...}}'
      
      # Kubernetes
      madengine-cli run --tags bert --additional-context '{"deploy": "k8s", "k8s": {...}}'
      
      # Or use config file (for CI/CD)
      madengine-cli run --tags bert --additional-context-file configs/slurm_4node.json
    """
    setup_logging(verbose)
    
    # Parse additional context
    context = _parse_additional_context(additional_context, additional_context_file)
    
    # Add runtime parameters to context
    context["timeout"] = timeout
    context["live_output"] = live_output
    context["verbose"] = verbose
    
    # Get deployment target (default: local)
    deploy_target = context.get("deploy", "local")
    
    # Build phase if tags provided (stores deployment_config in manifest)
    if not manifest_file:
        if not tags:
            console.print("[red]Error:[/red] Either --tags or --manifest-file required")
            raise typer.Exit(1)
        
        console.print("[bold blue]Building Docker images...[/bold blue]")
        manifest_file = _build_phase(tags, context)
        console.print(f"[green]✓ Build complete: {manifest_file}[/green]")
    else:
        # Load existing manifest and merge with current context
        manifest_file = _merge_manifest_context(manifest_file, context)
    
    # Deploy using Factory pattern
    try:
        console.print(f"\n[bold blue]Deploying to {deploy_target}...[/bold blue]")
        
        # Create deployment via Factory
        deployment = DeploymentFactory.create(
            target=deploy_target,
            manifest_file=manifest_file,
            additional_context=context
        )
        
        # Execute deployment (validate → prepare → deploy → monitor → collect)
        result = deployment.execute()
        
        # Display results
        if result.is_success:
            console.print(f"\n[green]✓ Deployment successful![/green]")
            console.print(f"  Deployment ID: {result.deployment_id}")
            console.print(f"  Message: {result.message}")
            
            if result.metrics:
                _display_metrics(result.metrics)
            
            if result.logs_path:
                console.print(f"  Logs: {result.logs_path}")
        else:
            console.print(f"\n[red]✗ Deployment failed[/red]")
            console.print(f"  Status: {result.status.value}")
            console.print(f"  Message: {result.message}")
            raise typer.Exit(1)
            
    except ValueError as e:
        console.print(f"[red]Configuration Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Deployment Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def _build_phase(tags: List[str], additional_context: Dict) -> str:
    """
    Execute build phase and save deployment_config to manifest.
    
    Returns:
        Path to generated build_manifest.json
    """
    from madengine.tools.distributed_orchestrator import DistributedOrchestrator
    
    orchestrator = DistributedOrchestrator(
        build_only_mode=True,
        additional_context=additional_context
    )
    
    manifest_file = orchestrator.build_phase(tags)
    
    # Enhance manifest with deployment_config from --additional-context
    _save_deployment_config_to_manifest(manifest_file, additional_context)
    
    return manifest_file


def _save_deployment_config_to_manifest(manifest_file: str, context: Dict):
    """Add deployment_config section to build_manifest.json"""
    import json
    
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    # Extract deployment configuration
    deployment_config = {
        "target": context.get("deploy", "local"),
        "slurm": context.get("slurm"),
        "k8s": context.get("k8s"),
        "distributed": context.get("distributed"),
        "vllm": context.get("vllm"),
        "sglang": context.get("sglang"),
        "shared_storage": context.get("shared_storage"),
        "env_vars": context.get("env_vars", {})
    }
    
    # Remove None values
    deployment_config = {k: v for k, v in deployment_config.items() if v is not None}
    
    manifest["deployment_config"] = deployment_config
    
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)


def _merge_manifest_context(manifest_file: str, runtime_context: Dict) -> str:
    """
    Merge runtime --additional-context with manifest's deployment_config.
    
    Allows overriding deployment target at runtime:
    - Build with SLURM config
    - Deploy to K8s by overriding at runtime
    """
    import json
    
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    # Merge deployment configs (runtime overrides build-time)
    stored_config = manifest.get("deployment_config", {})
    
    for key in ["deploy", "slurm", "k8s", "distributed", "vllm", "env_vars"]:
        if key in runtime_context:
            stored_config[key] = runtime_context[key]
    
    manifest["deployment_config"] = stored_config
    
    # Write updated manifest
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return manifest_file


def _parse_additional_context(context_str: str, context_file: Optional[str]) -> Dict:
    """Parse --additional-context from string or file"""
    import json
    
    if context_file:
        with open(context_file) as f:
            return json.load(f)
    
    if context_str == "{}":
        return {}
    
    try:
        return json.loads(context_str)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON in --additional-context:[/red] {e}")
        raise typer.Exit(1)


def _display_metrics(metrics: Dict):
    """Display deployment metrics in a table"""
    from rich.table import Table
    
    table = Table(title="Deployment Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in metrics.items():
        table.add_row(str(key), str(value))
    
    console.print(table)
```

**Key Changes**:
- ✅ **Factory Pattern**: Uses `DeploymentFactory.create()`
- ✅ **Manifest Storage**: Saves `deployment_config` from --additional-context
- ✅ **Runtime Override**: Can change deployment target when running existing manifest
- ✅ **Clean Separation**: Build phase, deployment phase clearly separated
- ✅ **Error Handling**: Proper exceptions and user-friendly messages

---

### 4.5 Phase 5: Deprecation & Documentation (Week 8)

#### 4.5.1 Mark Old Runners as Deprecated

```python
# src/madengine/runners/__init__.py

import warnings

warnings.warn(
    "The madengine.runners module is deprecated and will be removed in v2.0. "
    "Please use the new deployment API: madengine.deployment",
    DeprecationWarning,
    stacklevel=2
)
```

#### 4.5.2 Update Documentation

Create `docs/DEPLOYMENT_GUIDE.md` with examples for all three modes.

---

## 5. MIGRATION STRATEGY

### 5.1 Backward Compatibility

**Legacy madengine (mad.py)**:
- ✅ No changes required
- ✅ Continue to use existing core components
- ✅ All existing tests pass
- ⚠️  Mark as deprecated in documentation
- 📅 Remove in v3.0 (12+ months)

**Existing madengine-cli users**:
- ✅ Local execution unchanged
- ✅ `build` command unchanged
- ⚠️  `runner` commands deprecated (print warning)
- ⚠️  `generate` commands deprecated (auto-generated now)
- 📋 Provide migration guide

### 5.2 Migration Path

**For SSH/Ansible users** → Use Local deployment + your own orchestration:
```bash
# Old way (deprecated)
madengine-cli runner ssh --inventory nodes.yml

# New way (v2.0+)
# 1. Build on central node
madengine-cli build --tags models --registry your-registry

# 2. Deploy to each node using your orchestration
ansible-playbook -i inventory.yml deploy_local.yml
  # Playbook runs: madengine-cli run --manifest-file build_manifest.json

# Or use SSH loop
for node in node1 node2 node3; do
  ssh $node "madengine-cli run --manifest-file build_manifest.json"
done
```

**For K8s users** → Use K8s deployment:
```bash
# Old way (complex setup)
madengine-cli generate k8s --manifest-file manifest.json
madengine-cli runner k8s --inventory k8s.yml

# New way (simple)
madengine-cli run --tags models \
  --additional-context '{"deploy": "k8s", "k8s": {"namespace": "prod"}}'
```

**For SLURM users** → Use SLURM deployment:
```bash
# Old way (manual sbatch)
madengine-cli generate slurm --manifest-file manifest.json
# Then manually submit sbatch scripts

# New way (automated)
madengine-cli run --tags models \
  --additional-context '{"deploy": "slurm", "slurm": {"partition": "gpu"}}'
```

---

## 6. TESTING STRATEGY

### 6.1 Unit Tests (Simplified)

```python
# tests/deployment/test_slurm.py
def test_slurm_template_generation():
    """Test SLURM sbatch script generation"""
    from madengine.deployment.slurm import deploy_to_slurm
    
    manifest = {
        "built_models": {"test_model": {"name": "test"}},
        "built_images": {"test_model": {"registry_image": "test:latest"}}
    }
    
    slurm_config = {
        "partition": "gpu",
        "nodes": 2,
        "gpus_per_node": 8
    }
    
    # Generate script
    deploy_to_slurm(manifest, slurm_config)
    
    # Verify script created
    assert Path("madengine_slurm.sh").exists()
    
    # Verify content
    content = Path("madengine_slurm.sh").read_text()
    assert "madengine run" in content
    assert "#SBATCH --partition=gpu" in content

# tests/deployment/test_kubernetes.py
def test_k8s_manifest_generation():
    """Test Kubernetes Job manifest generation"""
    from madengine.deployment.kubernetes import deploy_to_k8s
    
    manifest = {
        "built_models": {"test_model": {"name": "test"}},
        "built_images": {"test_model": {"registry_image": "test:latest"}}
    }
    
    k8s_config = {
        "namespace": "test-ns",
        "gpu_count": 8,
        "memory": "128Gi"
    }
    
    # Generate manifest
    deploy_to_k8s(manifest, k8s_config)
    
    # Verify manifest created
    assert Path("madengine_job.yaml").exists()
    
    # Verify content
    content = Path("madengine_job.yaml").read_text()
    assert "image: test:latest" in content
    assert "namespace: test-ns" in content
    assert "amd.com/gpu:" in content
```

### 6.2 Integration Tests

```python
# tests/integration/test_end_to_end.py
@pytest.mark.integration
def test_local_end_to_end():
    """Test full workflow: build + local run"""
    # Build phase
    result = subprocess.run([
        "madengine-cli", "build",
        "--tags", "dummy",
        "--registry", "localhost:5000"
    ])
    assert result.returncode == 0
    
    # Run phase (local)
    result = subprocess.run([
        "madengine-cli", "run",
        "--manifest-file", "build_manifest.json"
    ])
    assert result.returncode == 0

@pytest.mark.slurm
def test_slurm_deployment():
    """Test SLURM deployment (requires SLURM cluster)"""
    result = subprocess.run([
        "madengine-cli", "run",
        "--manifest-file", "build_manifest.json",
        "--additional-context", '{"deploy": "slurm"}'
    ])
    assert result.returncode == 0
```

---

## 7. TIMELINE & MILESTONES (Simplified)

### Week 1: SLURM Templates & Integration
- [x] Design review (this document)
- [ ] Create SLURM Jinja2 template (job.sh.j2)
- [ ] Implement `deploy_to_slurm()` function
- [ ] Add routing in `mad_cli.py` based on `--additional-context`
- [ ] Test sbatch script generation

**Deliverable**: SLURM deployment working (generate + submit sbatch)

### Week 2: Kubernetes Templates & Integration  
- [ ] Create Kubernetes Jinja2 template (job.yaml.j2)
- [ ] Implement `deploy_to_k8s()` function
- [ ] Test K8s Job manifest generation
- [ ] Test kubectl apply

**Deliverable**: K8s deployment working (generate + apply manifest)

### Week 3: Testing & Examples
- [ ] Unit tests for template generation
- [ ] Integration tests with actual SLURM/K8s clusters
- [ ] Test with MAD training models (PyTorch BERT, etc.)
- [ ] Test with MAD inference models (vLLM, SGLang)
- [ ] Verify data download, pre/post-scripts work on distributed nodes

**Deliverable**: All workflows tested end-to-end

### Week 4: Documentation & Polish
- [ ] Mark old `runner` commands as deprecated
- [ ] Update README.md with deployment examples
- [ ] Create configuration file examples (slurm_config.json, k8s_config.json)
- [ ] Add vLLM MoE parallelism examples
- [ ] Migration guide for existing users
- [ ] Final testing

**Deliverable**: Production-ready v2.0 release

---

**Total Time**: 4 weeks (vs 8 weeks in complex approach)

**Key Simplifications**:
- ✅ No complex class hierarchies → Simple functions + Jinja2
- ✅ No deployment factories → Direct routing in CLI
- ✅ Reuse existing ContainerRunner for local → No LocalDeployment class
- ✅ Focus on template quality → Easy to customize

---

## 8. SUCCESS CRITERIA

### Technical
- [ ] All existing tests pass (backward compatibility)
- [ ] New deployment tests pass (local, SLURM, K8s)
- [ ] Template generation works correctly
- [ ] Performance equivalent or better than v1.x

### Usability
- [ ] Simpler CLI (fewer commands)
- [ ] Clear deployment model (3 modes)
- [ ] Better error messages
- [ ] Comprehensive documentation

### Maintainability
- [ ] Reduced code complexity
- [ ] Better separation of concerns
- [ ] Easier to add new deployment targets
- [ ] Clear deprecation path

---

## 9. RISKS & MITIGATION

### Risk 1: Breaking Changes
**Mitigation**: Extensive testing, deprecation warnings, migration guide

### Risk 2: Template Complexity
**Mitigation**: Start with simple templates, iterate based on real usage

### Risk 3: Cluster Access for Testing
**Mitigation**: Mock-based unit tests + optional integration tests

### Risk 4: User Adoption
**Mitigation**: Clear documentation, migration examples, both APIs work during transition

---

## APPENDIX A: vLLM MoE Parallelism Benchmarking

### A.1 Parallelism Strategy Decision Framework

Based on the [vLLM MoE Playbook](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html), use this table to select optimal parallelism strategy:

| Workload Type | Concurrency | Expert Density | Recommended Strategy | Configuration |
|---------------|-------------|----------------|---------------------|---------------|
| Interactive (chatbot) | Low | Any | TP + EP | `tensor_parallel_size=8, enable_expert_parallel=true` |
| Batch processing | High | <10% | DP + EP | `data_parallel_size=8, enable_expert_parallel=true` |
| Batch processing | High | >20% | DP only | `data_parallel_size=8, enable_expert_parallel=false` |
| Very large model | Any | Any | TP + PP | `tensor_parallel_size=4, pipeline_parallel_size=2` |
| MLA/MQA models | Low | Any | TP + EP | Optimized for KV cache |

### A.2 DeepSeek-R1 Benchmarking Examples

**Model**: DeepSeek-R1 (671B parameters, 256 routed + 1 shared experts, 8 experts/token, MLA)

#### Strategy 1: TP+EP (Low Latency - Interactive)

```bash
# Local single-node benchmark
madengine-cli run --tags deepseek_r1 \
  --additional-context '{
    "launcher": "vllm",
    "vllm": {
      "tensor_parallel_size": 8,
      "enable_expert_parallel": true,
      "max_model_len": 32768,
      "distributed_executor_backend": "mp",
      "disable_nccl_for_dp": true,
      "swap_space": 16,
      "env_vars": {
        "VLLM_ROCM_USE_AITER": "0"
      }
    }
  }'
```

#### Strategy 2: DP+EP (High Throughput - Batch)

```bash
# SLURM deployment for throughput benchmark
madengine-cli run --tags deepseek_r1 \
  --additional-context '{
    "deploy": "slurm",
    "launcher": "vllm",
    "vllm": {
      "tensor_parallel_size": 1,
      "data_parallel_size": 8,
      "enable_expert_parallel": true,
      "max_model_len": 32768,
      "distributed_executor_backend": "mp",
      "disable_nccl_for_dp": true,
      "swap_space": 16,
      "env_vars": {
        "VLLM_ROCM_USE_AITER": "0",
        "VLLM_ALL2ALL_BACKEND": "allgather_reducescatter"
      }
    },
    "slurm": {
      "partition": "gpu",
      "nodes": 1,
      "ntasks_per_node": 8,
      "gres": "gpu:8",
      "time_limit": 3600
    }
  }'
```

### A.3 Qwen3-235B Parallelism Comparison

**Model**: Qwen3-235B-A22B-Instruct (128 routed experts, 8 experts/token, 6.25% activation density)

```bash
# Kubernetes deployment for multi-strategy comparison

# Strategy 1: TP=8 (baseline)
madengine-cli run --tags qwen3_235b \
  --additional-context-file configs/qwen3_tp8.json

# Strategy 2: TP=8 + EP (optimized for low density MoE)
madengine-cli run --tags qwen3_235b \
  --additional-context-file configs/qwen3_tp8_ep.json

# Strategy 3: DP=8 + EP (high throughput)
madengine-cli run --tags qwen3_235b \
  --additional-context-file configs/qwen3_dp8_ep.json
```

**Config files**:

`configs/qwen3_tp8.json`:
```json
{
  "deploy": "k8s",
  "launcher": "vllm",
  "vllm": {
    "tensor_parallel_size": 8,
    "max_model_len": 32768,
    "env_vars": {"VLLM_ROCM_USE_AITER": "1"}
  },
  "k8s": {
    "namespace": "vllm-benchmark",
    "gpu_vendor": "AMD"
  }
}
```

`configs/qwen3_tp8_ep.json`:
```json
{
  "deploy": "k8s",
  "launcher": "vllm",
  "vllm": {
    "tensor_parallel_size": 8,
    "enable_expert_parallel": true,
    "max_model_len": 32768,
    "env_vars": {"VLLM_ROCM_USE_AITER": "1"}
  },
  "k8s": {
    "namespace": "vllm-benchmark",
    "gpu_vendor": "AMD"
  }
}
```

### A.4 Llama-4-Maverick (128 Experts) Benchmark

```bash
# SLURM deployment for MoE model with high expert count
madengine-cli run --tags llama4_maverick \
  --additional-context '{
    "deploy": "slurm",
    "launcher": "vllm",
    "vllm": {
      "tensor_parallel_size": 8,
      "enable_expert_parallel": true,
      "max_model_len": 32768,
      "swap_space": 16,
      "env_vars": {"VLLM_ROCM_USE_AITER": "1"}
    },
    "slurm": {
      "partition": "mi300x",
      "nodes": 1,
      "ntasks_per_node": 8
    }
  }'
```

### A.5 SGLang Disaggregated Inference (Multi-Node SLURM)

**From [existing docs](https://github.com/ROCm/madengine/blob/coketaste/slurm-integrate/docs/how-to-run-multi-node.md)**: SGLang disaggregated prefill/decode architecture.

**Old Approach** (bypassed madengine):
```bash
# OLD: Model-specific SLURM script handles everything
madengine run --tags sglang_disagg \
  --additional-context '{
    "slurm_args": {
      "FRAMEWORK": "sglang_disagg",
      "PREFILL_NODES": "2",
      "DECODE_NODES": "2",
      "PARTITION": "amd-rccl",
      "TIME": "12:00:00"
    }
  }'
# Problem: Skips madengine workflow, calls scripts/sglang_disagg/run.sh directly
```

**New Approach** (unified with madengine automation):
```bash
# NEW: Centralized deployment + madengine automation
madengine-cli run --tags sglang_disagg_qwen3_32b \
  --additional-context '{
    "deploy": "slurm",
    "slurm": {
      "partition": "amd-rccl",
      "nodes": 4,
      "gpus_per_node": 8,
      "time": "12:00:00"
    },
    "sglang": {
      "mode": "disaggregated",
      "prefill_nodes": 2,
      "decode_nodes": 2,
      "dp_size": 2,
      "tp_size": 8
    }
  }'

# Generates sbatch → Each node runs madengine with:
# - Data download (if needed)
# - Pre-scripts (system info, profiling)
# - SGLang server startup (prefill or decode based on node)
# - Post-scripts (metrics collection)
```

**Benefits of New Approach**:
- ✅ Centralized SLURM template (not model-specific scripts)
- ✅ All madengine automation works (data, profiling, metrics)
- ✅ Easier to customize and maintain
- ✅ Consistent with other workloads

### A.6 Multi-Node Training Examples

#### Megatron-LM Llama2 Training (4-Node SLURM)

**Old Approach** (manual multi-node):
```bash
# OLD: Must SSH to each node manually
ssh node0 "madengine run --tags pyt_megatron_lm_train_llama2_7b --additional-context '{multi_node_args: {RUNNER: torchrun, MASTER_ADDR: 10.194.129.113, NODE_RANK: 0, NNODES: 4}}' --force-mirror-local /nfs/data"
ssh node1 "madengine run --tags pyt_megatron_lm_train_llama2_7b --additional-context '{multi_node_args: {RUNNER: torchrun, MASTER_ADDR: 10.194.129.113, NODE_RANK: 1, NNODES: 4}}' --force-mirror-local /nfs/data"
ssh node2 "madengine run --tags pyt_megatron_lm_train_llama2_7b --additional-context '{multi_node_args: {RUNNER: torchrun, MASTER_ADDR: 10.194.129.113, NODE_RANK: 2, NNODES: 4}}' --force-mirror-local /nfs/data"
ssh node3 "madengine run --tags pyt_megatron_lm_train_llama2_7b --additional-context '{multi_node_args: {RUNNER: torchrun, MASTER_ADDR: 10.194.129.113, NODE_RANK: 3, NNODES: 4}}' --force-mirror-local /nfs/data"
# Problem: Manual, error-prone, no job management
```

**New Approach** (automated SLURM):
```bash
# NEW: Single command, automated deployment
madengine-cli run --tags pyt_megatron_lm_train_llama2_7b \
  --additional-context '{
    "deploy": "slurm",
    "slurm": {
      "partition": "gpu",
      "nodes": 4,
      "gpus_per_node": 8,
      "time": "24:00:00",
      "exclusive": true
    },
    "multi_node_args": {
      "RUNNER": "torchrun",
      "MASTER_PORT": "29500",
      "NCCL_SOCKET_IFNAME": "ens14np0",
      "GLOO_SOCKET_IFNAME": "ens14np0"
    },
    "shared_data": "/nfs/data"
  }'

# What happens:
# 1. Generates sbatch script with 4 nodes
# 2. SLURM allocates 4 nodes
# 3. Each node runs madengine with auto-configured NODE_RANK and MASTER_ADDR
# 4. Shared filesystem /nfs/data used for data and results
# 5. torchrun coordinates across nodes
# 6. All nodes collect metrics, aggregate results
```

### A.7 Multi-Configuration Automated Benchmarking

```bash
# Automated benchmarking across multiple parallelism strategies
#!/bin/bash

STRATEGIES=("tp8" "tp8_ep" "dp8" "dp8_ep")
MODEL="deepseek_r1"

for strategy in "${STRATEGIES[@]}"; do
  echo "Running ${strategy} strategy..."
  
  madengine-cli run --tags ${MODEL} \
    --additional-context-file "configs/${MODEL}_${strategy}.json" \
    --summary-output "results/${MODEL}_${strategy}_results.json"
  
  sleep 60  # Cool down between runs
done

# Generate comparison report
madengine report compare \
  --input results/${MODEL}_*_results.json \
  --output ${MODEL}_parallelism_comparison.html
```

---

## APPENDIX B: Example Usage

### B.1 Local Execution

```bash
# Simple local run (unchanged)
madengine-cli run --tags dummy

# With explicit context
madengine-cli run --tags dummy \
  --additional-context '{"deploy": "local"}'
```

### B.2 SLURM Multi-Node Deployment

#### Training Model (Megatron-LM)

```bash
# 4-node Megatron-LM training with automated SLURM submission
madengine-cli run --tags pyt_megatron_lm_train_llama2_7b \
  --additional-context '{
    "deploy": "slurm",
    "slurm": {
      "partition": "gpu",
      "nodes": 4,
      "gpus_per_node": 8,
      "time": "24:00:00",
      "exclusive": true,
      "modules": ["rocm/5.7.0", "python/3.10"]
    },
    "multi_node_args": {
      "RUNNER": "torchrun",
      "MASTER_PORT": "29500",
      "NCCL_SOCKET_IFNAME": "ens14np0",
      "GLOO_SOCKET_IFNAME": "ens14np0"
    },
    "shared_data": "/nfs/data"
  }'

# What this does:
# 1. Generates sbatch script
# 2. Submits to SLURM
# 3. Each of 4 nodes runs: madengine run with proper multi_node_args
# 4. Full automation on each node (data, pre/post-scripts, profiling)
# 5. Aggregates results
```

#### Inference Model (vLLM)

```bash
# vLLM inference on SLURM with TP+EP
madengine-cli run --tags vllm_deepseek_r1_tp8_ep \
  --additional-context '{
    "deploy": "slurm",
    "slurm": {
      "partition": "mi300x",
      "nodes": 1,
      "gpus_per_node": 8,
      "time": "04:00:00"
    },
    "vllm": {
      "tensor_parallel_size": 8,
      "enable_expert_parallel": true,
      "max_model_len": 32768
    }
  }'
```

#### Using Config File

```bash
# With config file for easier management
madengine-cli run --tags pyt_bert_training \
  --additional-context-file configs/slurm_4node.json
```

`configs/slurm_4node.json`:
```json
{
  "deploy": "slurm",
  "slurm": {
    "partition": "gpu",
    "nodes": 4,
    "gpus_per_node": 8,
    "time": "12:00:00",
    "modules": ["rocm/5.7.0"]
  },
  "multi_node_args": {
    "RUNNER": "torchrun"
  },
  "shared_data": "/nfs/datasets"
}
```

### B.3 Kubernetes Deployment

```bash
# Basic K8s deployment
madengine-cli run --tags llama_inference \
  --additional-context '{
    "deploy": "k8s",
    "launcher": "python",
    "nnodes": 2,
    "nproc_per_node": 4,
    "k8s": {
      "namespace": "ml-workloads",
      "gpu_vendor": "AMD",
      "memory": "64Gi",
      "node_selector": {"gpu-type": "mi250x"}
    }
  }'
```

---

## APPENDIX C: Configuration Examples

### C.1 SLURM Configuration

```json
{
  "deploy": "slurm",
  "launcher": "torchrun",
  "nnodes": 4,
  "nproc_per_node": 8,
  "slurm": {
    "login_node": "hpc-login.example.com",
    "partition": "gpu",
    "qos": "high",
    "account": "ml-research",
    "time_limit": 14400,
    "modules": [
      "rocm/5.7.0",
      "python/3.10",
      "git/2.40"
    ],
    "output_dir": "./slurm_jobs",
    "work_dir": "/projects/ml/experiments"
  },
  "env_vars": {
    "NCCL_DEBUG": "INFO",
    "NCCL_IB_HCA": "mlx5_0"
  }
}
```

### C.2 Kubernetes Configuration

```json
{
  "deploy": "k8s",
  "launcher": "torchrun",
  "nnodes": 2,
  "nproc_per_node": 4,
  "k8s": {
    "namespace": "ml-prod",
    "kubeconfig": "~/.kube/config",
    "gpu_vendor": "AMD",
    "memory": "64Gi",
    "memory_limit": "128Gi",
    "cpu": "16",
    "cpu_limit": "32",
    "node_selector": {
      "gpu-type": "mi250x",
      "zone": "us-west1-a"
    },
    "volumes": [
      {
        "name": "data",
        "type": "pvc",
        "claim_name": "ml-data",
        "mount_path": "/data"
      }
    ],
    "output_dir": "./k8s_manifests"
  },
  "env_vars": {
    "NCCL_DEBUG": "INFO"
  }
}
```

### C.3 vLLM MoE Parallelism Configurations

#### C.3.1 DeepSeek-R1 TP+EP (Low Latency)

```json
{
  "deploy": "slurm",
  "launcher": "vllm",
  "vllm": {
    "tensor_parallel_size": 8,
    "enable_expert_parallel": true,
    "max_model_len": 32768,
    "distributed_executor_backend": "mp",
    "disable_nccl_for_dp": true,
    "swap_space": 16,
    "port": 8000,
    "env_vars": {
      "VLLM_ROCM_USE_AITER": "0"
    }
  },
  "slurm": {
    "partition": "gpu",
    "nodes": 1,
    "ntasks_per_node": 8,
    "gres": "gpu:8",
    "time_limit": 3600
  }
}
```

#### C.3.2 DeepSeek-R1 DP+EP (High Throughput)

```json
{
  "deploy": "k8s",
  "launcher": "vllm",
  "vllm": {
    "tensor_parallel_size": 1,
    "data_parallel_size": 8,
    "enable_expert_parallel": true,
    "max_model_len": 32768,
    "distributed_executor_backend": "mp",
    "disable_nccl_for_dp": true,
    "swap_space": 16,
    "port": 8000,
    "env_vars": {
      "VLLM_ROCM_USE_AITER": "0",
      "VLLM_ALL2ALL_BACKEND": "allgather_reducescatter"
    }
  },
  "k8s": {
    "namespace": "vllm-prod",
    "gpu_vendor": "AMD",
    "memory": "256Gi",
    "cpu": "64"
  }
}
```

#### C.3.3 Qwen3-235B TP Only (Baseline)

```json
{
  "deploy": "local",
  "launcher": "vllm",
  "vllm": {
    "tensor_parallel_size": 8,
    "max_model_len": 32768,
    "distributed_executor_backend": "mp",
    "swap_space": 16,
    "env_vars": {
      "VLLM_ROCM_USE_AITER": "1"
    }
  }
}
```

#### C.3.4 Llama-4-Maverick TP+EP (128 Experts)

```json
{
  "deploy": "slurm",
  "launcher": "vllm",
  "vllm": {
    "tensor_parallel_size": 8,
    "enable_expert_parallel": true,
    "max_model_len": 32768,
    "distributed_executor_backend": "mp",
    "swap_space": 16,
    "env_vars": {
      "VLLM_ROCM_USE_AITER": "1"
    }
  },
  "slurm": {
    "partition": "mi300x",
    "nodes": 1,
    "ntasks": 8
  }
}
```

---

## REFERENCES

### Industry Best Practices & Documentation

1. **vLLM MoE Parallelism Guide** (AMD ROCm)  
   **[The vLLM MoE Playbook: A Practical Guide to TP, DP, PP and Expert Parallelism](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)**  
   - Comprehensive guide on parallelism strategies for MoE models
   - Benchmark results on AMD Instinct™ MI300X GPUs
   - Decision framework for strategy selection based on workload type
   - Critical insights on TP+EP vs DP+EP tradeoffs
   - Expert activation density analysis
   - MLA/MQA attention considerations

2. **Primus Training Framework** (AMD-AGI)  
   https://github.com/AMD-AGI/Primus  
   - Flexible training framework for large-scale models on AMD GPUs
   - Multiple backend support (Megatron-LM, TorchTitan, JAX MaxText)
   - Infrastructure-agnostic design (SLURM, K8s compatible)
   - ROCm-optimized components

3. **MAD Model Hub** (ROCm)  
   https://github.com/ROCm/MAD  
   - Centralized AI model repository for AMD GPU ecosystem
   - Standardized model interfaces and Docker configurations
   - Script templates for training and inference

### Key Parallelism Concepts

**Tensor Parallelism (TP)**:
- Shards model layers across GPUs
- All GPUs collaborate on same computation
- Requires AllReduce communication after each layer
- Best for: Low latency, single request processing, interactive workloads

**Data Parallelism (DP)**:
- Replicates entire model across GPUs
- Each replica processes different requests independently
- No communication between replicas during inference
- Best for: High throughput, batch processing, concurrent requests

**Expert Parallelism (EP)**:
- Distributes MoE experts across GPUs (modifier for TP or DP)
- Only activated experts participate in computation
- Requires AllToAll communication in DP+EP mode
- Best for: MoE models with low expert activation density (<10%)
- May add overhead for high density models (>20%)

**Pipeline Parallelism (PP)**:
- Splits model into sequential stages across GPUs
- Different GPUs process different layers
- Enables deployment of models too large for TP alone
- Best for: Very large models, memory-constrained scenarios

### vLLM Parallelism Strategies for Production

| Strategy | Communication | Use Case | Latency | Throughput |
|----------|---------------|----------|---------|------------|
| TP only | AllReduce | Small models, low latency | Low | Medium |
| TP + EP | AllReduce | MoE interactive, low density | Low | Medium |
| DP only | None | High throughput, dense models | Medium | High |
| DP + EP | AllToAll | MoE batch processing | Medium | High |
| TP + PP | AllReduce + P2P | Very large models | Medium | Medium |

---

**Document Status**: Ready for Review  
**Next Steps**: Approve plan → Begin Phase 1 implementation


