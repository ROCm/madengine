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

## 2. PRODUCTION-READY ARCHITECTURE

### 2.1 Layered Architecture (Best Practices)

madengine-cli follows a **clean layered architecture** with separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: PRESENTATION                        │
│                   (CLI Entry Points)                            │
│                                                                 │
│  mad_cli.py                                                     │
│  ├─ build_command()  → BuildOrchestrator                        │
│  └─ run_command()    → RunOrchestrator                          │
│                                                                 │
│  Responsibilities:                                              │
│  • Parse CLI arguments                                          │
│  • Validate input                                               │
│  • Delegate to orchestration layer                              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LAYER 2: ORCHESTRATION                         │
│              (Workflow Management)                              │
│                                                                 │
│  orchestration/                                                 │
│  ├─ build_orchestrator.py                                       │
│  │   └─ Orchestrates: Discover → Build → Generate manifest     │
│  │                                                               │
│  └─ run_orchestrator.py                                         │
│      └─ Orchestrates: Load manifest → Route to execution       │
│                                                                 │
│  Responsibilities:                                              │
│  • Workflow coordination                                        │
│  • Decision making (local vs distributed)                       │
│  • Phase separation (build-only, run-only, full workflow)       │
│  • Delegate to execution/deployment layers                      │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
┌───────────────────────────┐  ┌───────────────────────────┐
│   LAYER 3a: EXECUTION     │  │   LAYER 3b: DEPLOYMENT    │
│   (Local Single-Node)     │  │   (Distributed Multi-Node)│
│                           │  │                           │
│  execution/               │  │  deployment/              │
│  └─ container_runner.py   │  │  ├─ base.py               │
│                           │  │  ├─ factory.py            │
│  Responsibilities:        │  │  ├─ slurm.py   (CLI)      │
│  • Docker container exec  │  │  └─ kubernetes.py (Lib)   │
│  • Local GPU management   │  │                           │
│  • Performance collection │  │  Responsibilities:        │
│                           │  │  • Generate deployment    │
│                           │  │    scripts/manifests      │
│                           │  │  • Submit to scheduler    │
│                           │  │  • Monitor execution      │
│                           │  │  • Collect results        │
└───────────────────────────┘  └───────────────────────────┘
```

### 2.2 Key Architectural Principles

1. **Separation of Concerns**: Each layer has one clear responsibility
2. **Dependency Inversion**: High-level orchestration depends on abstractions
3. **Open/Closed Principle**: Easy to extend (new deployment types) without modifying existing code
4. **Single Responsibility**: Each class/module does one thing well
5. **Interface Segregation**: Clean interfaces between layers

### 2.3 Workflow Support

The architecture supports **both separate and combined phases**:

```bash
# Separate Phases (distributed build/run)
madengine-cli build --tags model --registry docker.io
madengine-cli run --manifest-file build_manifest.json

# Full Workflow (single command - current behavior preserved)
madengine-cli run --tags model  # Builds + Runs locally

# Full Workflow with Distributed Deployment (new)
madengine-cli run --tags model --additional-context '{"deploy": "slurm", ...}'
```

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
- ❌ **Entire `runners/` folder** (replaced by `deployment/`)
- ❌ SSH/Ansible runners (not needed with SLURM/K8s)
- ❌ `madengine-cli generate/runner` subcommands (unified via `run`)
- ❌ Environment variable configuration for deployment

---

### 3.2 What's Being Removed (Detailed)

#### ❌ DELETE: `src/madengine/runners/` (Entire Folder)

The old `runners/` module is **completely replaced** by the new `deployment/` architecture.

**Files being deleted**:
```
src/madengine/runners/
├── __init__.py                 # ❌ DELETE
├── base.py                     # ❌ DELETE → Replaced by deployment/base.py
├── factory.py                  # ❌ DELETE → Replaced by deployment/factory.py
├── ssh_runner.py               # ❌ DELETE (SSH out of scope)
├── ansible_runner.py           # ❌ DELETE (Ansible out of scope)
├── k8s_runner.py              # ❌ DELETE → Replaced by deployment/kubernetes.py
├── slurm_runner.py            # ❌ DELETE → Replaced by deployment/slurm.py
├── orchestrator_generation.py  # ❌ DELETE (Jinja2 used directly)
├── template_generator.py       # ❌ DELETE (Jinja2 used directly)
└── templates/                  # ❌ DELETE → Replaced by deployment/templates/
    ├── ansible/
    ├── k8s/
    └── slurm/
```

**Why complete removal**:
1. **Replaced by better design**: New `deployment/` uses production-ready patterns
2. **Different approach**: Old runners used complex wrapper classes, new uses direct libraries/CLI
3. **Scope reduction**: No SSH/Ansible support in new architecture
4. **Cleaner separation**: New layered architecture (orchestration vs deployment)

**Migration mapping**:
```python
# OLD (being deleted)
from madengine.runners.factory import RunnerFactory
runner = RunnerFactory.create_runner("slurm", inventory="slurm.yml")
runner.execute_workload(...)

# NEW (replacement)
from madengine.deployment.factory import DeploymentFactory
deployment = DeploymentFactory.create(
    target="slurm",
    manifest_file="build_manifest.json",
    additional_context={...}
)
deployment.execute()
```

---

#### ❌ REMOVE: CLI Sub-Commands

**Old CLI commands being removed**:
```bash
# These NO LONGER EXIST in new architecture:
madengine-cli generate ansible --manifest-file manifest.json  # ❌ REMOVED
madengine-cli generate k8s --manifest-file manifest.json      # ❌ REMOVED  
madengine-cli generate slurm --manifest-file manifest.json    # ❌ REMOVED
madengine-cli runner ssh --inventory nodes.yml                # ❌ REMOVED
madengine-cli runner ansible --inventory cluster.yml          # ❌ REMOVED
madengine-cli runner k8s --inventory k8s.yml                  # ❌ REMOVED
madengine-cli runner slurm --inventory slurm.yml              # ❌ REMOVED
```

**Replaced by unified command**:
```bash
# NEW: Single command with --additional-context
madengine-cli run --tags model --additional-context '{"deploy": "slurm", ...}'
madengine-cli run --tags model --additional-context '{"deploy": "k8s", ...}'

# Auto-generation during deployment (no manual generate step needed)
# Templates generated and applied automatically
```

**Why removed**:
- **Simpler UX**: One command instead of 7+ commands
- **Automatic generation**: Templates auto-generated during deployment
- **Unified config**: Everything via `--additional-context`
- **Less maintenance**: Fewer commands = less code to maintain

---

#### ❌ REMOVE: SSH and Ansible Support

**Decision**: New architecture supports **3 targets only**:
1. ✅ **Local**: Single-node execution
2. ✅ **SLURM**: HPC cluster deployment
3. ✅ **Kubernetes**: Cloud/on-prem orchestration

**Not supported** (users manage themselves):
- ❌ SSH runner
- ❌ Ansible runner

**Rationale**:
- SLURM + K8s cover 95% of production use cases
- SSH/Ansible are generic tools (users can orchestrate themselves)
- Reduces scope → Better focus → Production-ready faster
- Simpler codebase → Easier to maintain

**For users who need custom orchestration**:
```bash
# Use Ansible playbook to call madengine on each node
ansible-playbook -i inventory.yml run_madengine.yml

# Playbook content:
# - hosts: gpu_nodes
#   tasks:
#     - name: Run madengine
#       command: madengine-cli run --manifest-file build_manifest.json
```

---

### 3.3 Actual madengine Run Workflow

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
      "work_dir": "/projects/ml"
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

### 3.3 Production-Ready Directory Structure

```
src/madengine/
├── mad.py                      # Layer 1: Legacy CLI (keep for compatibility)
├── mad_cli.py                  # Layer 1: Modern CLI (REFACTOR - simplified routing)
│
├── orchestration/              # Layer 2: NEW - Workflow Orchestration
│   ├── __init__.py
│   ├── build_orchestrator.py  # Orchestrates build workflow
│   └── run_orchestrator.py    # Orchestrates run workflow (build+run or run-only)
│
├── execution/                  # Layer 3a: NEW - Local Execution
│   ├── __init__.py
│   └── container_runner.py    # Moved from tools/ (handles Docker locally)
│
├── deployment/                 # Layer 3b: NEW - Distributed Deployment
│   ├── __init__.py
│   ├── base.py                # BaseDeployment abstract class
│   ├── factory.py             # DeploymentFactory (2 types: slurm, k8s)
│   ├── slurm.py               # SlurmDeployment (uses CLI: sbatch/squeue)
│   ├── kubernetes.py          # KubernetesDeployment (uses library: kubernetes)
│   └── templates/             # Jinja2 templates
│       ├── slurm/
│       │   └── job.sh.j2      # SLURM sbatch script template
│       └── kubernetes/
│           └── job.yaml.j2    # K8s Job manifest template (optional)
│
├── tools/                      # Supporting Tools (used by orchestrators)
│   ├── discover_models.py     # Model discovery (used by build_orchestrator)
│   ├── docker_builder.py      # Docker image building (used by build_orchestrator)
│   ├── distributed_orchestrator.py  # DEPRECATED - to be removed
│   └── ...
│
├── core/                       # Foundation Layer (unchanged)
│   ├── context.py             # GPU/OS detection, environment
│   ├── docker.py              # Docker client wrapper
│   ├── dataprovider.py        # Data source management
│   ├── console.py             # Output formatting
│   └── errors.py              # Error handling
│
└── runners/                    # ❌ REMOVED - Replaced by deployment/
    └── (DELETE ENTIRE FOLDER)
    # Old files being removed:
    # - base.py
    # - factory.py
    # - ssh_runner.py         → Removed (out of scope)
    # - ansible_runner.py     → Removed (out of scope)
    # - k8s_runner.py         → Replaced by deployment/kubernetes.py
    # - slurm_runner.py       → Replaced by deployment/slurm.py
    # - orchestrator_generation.py → Removed (templates used instead)
    # - template_generator.py → Removed (Jinja2 used directly)

Dependencies in pyproject.toml:
  - kubernetes (for K8s deployment layer)
  - jinja2 (for template rendering)
  - No SLURM library needed (uses CLI commands)
```

**Migration Path**:
1. Create new `orchestration/`, `execution/`, `deployment/` directories
2. Refactor `distributed_orchestrator.py` → `build_orchestrator.py` + `run_orchestrator.py`
3. Move `tools/container_runner.py` → `execution/container_runner.py`
4. **DELETE** entire `runners/` folder (replaced by `deployment/`)
5. Update `mad_cli.py` to use new orchestrators
6. Remove `generate` and `runner` CLI sub-commands (no longer needed)

---

## 4. IMPLEMENTATION PLAN

### 4.0 Implementation Strategy

**Approach**: Incremental refactoring with zero breaking changes

1. **Create new architecture** alongside existing code
2. **Gradually migrate** functionality from old to new
3. **Maintain backward compatibility** throughout
4. **Deprecate old code** only after new code is proven
5. **Test continuously** at each step

### 4.1 Phase 1: Orchestration Layer (Week 1)

**Goal**: Create the orchestration layer that coordinates build and run workflows.

#### 4.1.1 Create Orchestration Layer

**Step 1**: Create `orchestration/` directory structure

**Step 2**: Extract build workflow from `distributed_orchestrator.py`

**File**: `src/madengine/orchestration/build_orchestrator.py`

This orchestrator coordinates the build workflow:
1. Discover models by tags
2. Build Docker images
3. Generate build_manifest.json
4. Save deployment_config from --additional-context

(See implementation in detailed code section)

**Step 3**: Create run workflow orchestrator

**File**: `src/madengine/orchestration/run_orchestrator.py`

This orchestrator coordinates the run workflow:
1. Load manifest or trigger build if needed
2. Determine target (local vs distributed)
3. Delegate to execution or deployment layer
4. Collect results

Supports both:
- **Run-only** mode: `madengine-cli run --manifest-file build_manifest.json`
- **Full workflow** mode: `madengine-cli run --tags model` (builds + runs)

(See implementation in detailed code section)

**Step 4**: Update `mad_cli.py` to use orchestrators

```python
# mad_cli.py - simplified routing

@app.command()
def build(...):
    from madengine.orchestration.build_orchestrator import BuildOrchestrator
    
    orchestrator = BuildOrchestrator(args, additional_context)
    manifest_file = orchestrator.execute(registry, clean_cache)
    console.print(f"[green]✓ Build complete: {manifest_file}[/green]")


@app.command()
def run(...):
    from madengine.orchestration.run_orchestrator import RunOrchestrator
    
    orchestrator = RunOrchestrator(args, additional_context)
    results = orchestrator.execute(manifest_file, tags, timeout)
    console.print(f"[green]✓ Execution complete[/green]")
```

#### 4.1.2 Create Deployment Abstraction (Production-Ready)

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
    """Configuration for distributed deployment"""
    target: str  # "slurm", "k8s" (NOT "local" - that uses container_runner)
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

#### 4.1.2 Local Execution (No LocalDeployment Needed)

**Important**: Local execution is NOT a "deployment" - it uses existing `container_runner.py` directly.

**Why No LocalDeployment?**
- ❌ Would be an unnecessary wrapper around container_runner
- ❌ Adds abstraction with zero benefit
- ❌ "Deploy locally" doesn't make semantic sense
- ✅ container_runner.py already works perfectly

**Implementation** (in `mad_cli.py`):

```python
def run_command(...):
    deploy_target = context.get("deploy", "local")
    
    if deploy_target == "local":
        # Use existing container_runner directly (no wrapper)
        _run_local(manifest_file, timeout, live_output)
    else:
        # Use Factory for distributed deployments
        deployment = DeploymentFactory.create(
            target=deploy_target,
            manifest_file=manifest_file,
            additional_context=context
        )
        result = deployment.execute()


def _run_local(manifest_file: str, timeout: int, live_output: bool):
    """
    Run locally using existing container_runner.
    
    This is the proven, existing implementation - no changes needed.
    """
    from madengine.tools.container_runner import ContainerRunner
    
    runner = ContainerRunner(
        live_output=live_output,
        timeout=timeout
    )
    
    # Existing, proven implementation
    runner.run_models_from_manifest(manifest_file)
```

**Benefits**:
- ✅ Reuses existing, proven code
- ✅ No unnecessary abstraction
- ✅ Clear semantics: "run" vs "deploy"
- ✅ Simpler codebase

#### 4.1.3 Create DeploymentFactory (2 Types - Distributed Only)

**File**: `src/madengine/deployment/factory.py`

```python
from typing import Dict, Type, Optional
from .base import BaseDeployment, DeploymentConfig


class DeploymentFactory:
    """
    Factory for creating DISTRIBUTED deployment instances.
    
    Supports 2 deployment types:
    - slurm: HPC multi-node via SLURM scheduler
    - k8s: Kubernetes container orchestration
    
    Note: Local execution uses container_runner.py directly (not a "deployment").
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


# Register the 2 distributed deployment types
def register_deployments():
    """Register production-ready distributed deployment types"""
    
    # 1. SLURM (HPC clusters)
    try:
        from .slurm import SlurmDeployment
        DeploymentFactory.register("slurm", SlurmDeployment)
    except ImportError as e:
        # Optional dependency, fail gracefully
        import warnings
        warnings.warn(f"SLURM deployment not available: {e}")
    
    # 2. Kubernetes (container orchestration)
    try:
        from .kubernetes import KubernetesDeployment
        DeploymentFactory.register("k8s", KubernetesDeployment)
        DeploymentFactory.register("kubernetes", KubernetesDeployment)  # Alias
    except ImportError as e:
        # Optional dependency, fail gracefully
        import warnings
        warnings.warn(f"Kubernetes deployment not available: {e}")
    
    # Note: Local execution uses container_runner.py directly (no registration needed)


# Auto-register on module import
register_deployments()
```

**Key Features**:
- ✅ **2 Types Only**: SLURM, Kubernetes (distributed deployments)
- ✅ **Graceful Degradation**: Missing deps don't break import
- ✅ **Clear Error Messages**: Shows available types and example usage
- ✅ **Factory Pattern**: Standard creational pattern for distributed deployments
- ✅ **Extensible**: Easy to add new deployment types later
- ✅ **Local Execution**: Uses container_runner.py directly (no factory overhead)

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

#### 4.2.3 SLURM Deployment Implementation (Using CLI Commands)

**File**: `src/madengine/deployment/slurm.py`

**Implementation Strategy**: Uses SLURM CLI commands (`sbatch`, `squeue`, `scancel`) via subprocess

**Why CLI Instead of Python Library**:
- ✅ **Zero dependencies**: No `pyslurm` installation needed
- ✅ **Portability**: Works with any SLURM version
- ✅ **Industry standard**: Used by Airflow, Prefect, Ray
- ✅ **Simplicity**: Direct, no C extension compilation
- ✅ **Reliability**: SLURM CLI is always available on clusters

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
    SLURM HPC cluster deployment using CLI commands.
    
    **Assumption**: User has already SSH'd to SLURM login node manually.
    madengine-cli is executed ON the login node, not remotely.
    
    Uses subprocess to call SLURM CLI commands locally:
    - sbatch: Submit jobs to SLURM scheduler
    - squeue: Monitor job status
    - scancel: Cancel jobs
    - scontrol: Get cluster info
    
    **Workflow**:
    1. User: ssh login_node@hpc.example.com
    2. User: madengine-cli run --tags model --additional-context '{"deploy": "slurm", ...}'
    3. madengine-cli: Runs sbatch locally (no SSH needed)
    
    No Python SLURM library required (zero dependencies).
    No SSH handling needed (user is already on login node).
    """
    
    DEPLOYMENT_TYPE = "slurm"
    REQUIRED_TOOLS = ["sbatch", "squeue", "scontrol"]  # Must be available locally
    
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
        
        # Setup Jinja2 template engine
        template_dir = Path(__file__).parent / "templates" / "slurm"
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        
        # Generated script path
        self.script_path = None
    
    def validate(self) -> bool:
        """Validate SLURM commands are available locally"""
        # Check required SLURM CLI tools
        for tool in self.REQUIRED_TOOLS:
            result = subprocess.run(
                ["which", tool],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                self.console.print(
                    f"[red]✗ Required tool not found: {tool}[/red]\n"
                    f"[yellow]Make sure you are on a SLURM login node[/yellow]"
                )
                return False
        
        # Verify we can query SLURM cluster
        result = subprocess.run(
            ["sinfo", "-h"],
            capture_output=True,
            timeout=10
        )
        if result.returncode != 0:
            self.console.print("[red]✗ Cannot query SLURM (sinfo failed)[/red]")
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
        """Submit sbatch script to SLURM scheduler (locally)"""
        if not self.script_path or not self.script_path.exists():
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id="",
                message="Script not generated. Run prepare() first."
            )
        
        try:
            # Submit job to SLURM (runs locally on login node)
            result = subprocess.run(
                ["sbatch", str(self.script_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
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
        """Check SLURM job status (locally)"""
        try:
            # Query job status using squeue (runs locally)
            result = subprocess.run(
                ["squeue", "-j", deployment_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
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
        """Check completed job status using sacct (locally)"""
        try:
            result = subprocess.run(
                ["sacct", "-j", job_id, "-n", "-X", "-o", "State"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
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
        """Cancel SLURM job if still running (locally)"""
        try:
            subprocess.run(
                ["scancel", deployment_id],
                capture_output=True,
                timeout=10
            )
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
            monitor_slurm_job(job_id)
        
        return {"status": "success", "job_id": job_id}
    else:
        console.print(f"[red]✗ Failed to submit SLURM job:[/red]\n{result.stderr}")
        return {"status": "failed", "error": result.stderr}


def monitor_slurm_job(job_id: str):
    """Monitor SLURM job until completion (locally)"""
    import time
    
    while True:
        # Check job status using squeue (runs locally)
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h"],
            capture_output=True,
            text=True
        )
        
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
        """Validate SLURM deployment requirements (locally)"""
        # Check if sbatch is available on this login node
        result = subprocess.run(["which", "sbatch"], capture_output=True)
        if result.returncode != 0:
            console.print("[red]✗ sbatch not found. Make sure you are on a SLURM login node.[/red]")
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
        """Submit SLURM jobs (locally)"""
        job_ids = []
        
        for model_name in self.manifest["built_images"].keys():
            script_path = Path(self.output_dir) / f"{model_name}_job.sh"
            
            # Submit job using sbatch (runs locally on login node)
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True
            )
            
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
        """Monitor SLURM job status (locally)"""
        job_ids = deployment_id.split(",")
        
        # Check status using squeue (runs locally)
        result = subprocess.run(
            ["squeue", "-j", deployment_id, "-h"],
            capture_output=True,
            text=True
        )
        
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
        """Cleanup SLURM jobs if needed (locally)"""
        # Cancel any remaining jobs using scancel (runs locally)
        job_ids = deployment_id.split(",")
        
        subprocess.run(
            ["scancel"] + job_ids,
            capture_output=True
        )
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

#### 4.3.2 Kubernetes Deployment Implementation (Using Python Library)

**File**: `src/madengine/deployment/kubernetes.py`

**Implementation Strategy**: Uses Kubernetes Python client library (NOT kubectl CLI)

**Why Python Library Instead of kubectl**:
- ✅ **Type safety**: Typed API, no string parsing
- ✅ **Better error handling**: Python exceptions, not stderr parsing
- ✅ **Production standard**: Used by Kubeflow, Argo, Ray
- ✅ **Programmatic control**: Direct API access
- ✅ **Retry logic**: Built-in retry mechanisms
- ✅ **No kubectl required**: Works in Python-only environments

**Dependencies**: Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
kubernetes = ["kubernetes>=28.0.0"]
```

**Implementation**:

```python
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

from .base import (
    BaseDeployment,
    DeploymentConfig,
    DeploymentResult,
    DeploymentStatus
)


class KubernetesDeployment(BaseDeployment):
    """
    Kubernetes cluster deployment using Python client library.
    
    Uses kubernetes Python API for type-safe, production-ready deployment:
    - client.BatchV1Api(): Job creation and management
    - client.CoreV1Api(): Pod logs and status
    
    Requires AMD GPU Device Plugin: https://github.com/ROCm/k8s-device-plugin
    """
    
    DEPLOYMENT_TYPE = "k8s"
    REQUIRED_TOOLS = []  # No CLI tools needed, uses Python library
    
    def __init__(self, config: DeploymentConfig):
        if not KUBERNETES_AVAILABLE:
            raise ImportError(
                "Kubernetes Python library not installed.\n"
                "Install with: pip install madengine[kubernetes]\n"
                "Or: pip install kubernetes"
            )
        
        super().__init__(config)
        
        # Parse K8s configuration
        self.k8s_config = config.additional_context.get("k8s", {})
        self.namespace = self.k8s_config.get("namespace", "default")
        self.gpu_resource_name = self.k8s_config.get("gpu_resource_name", "amd.com/gpu")
        
        # Load Kubernetes configuration
        kubeconfig_path = self.k8s_config.get("kubeconfig")
        try:
            if kubeconfig_path:
                config.load_kube_config(config_file=kubeconfig_path)
            else:
                # Try in-cluster first, then default kubeconfig
                try:
                    config.load_incluster_config()
                except:
                    config.load_kube_config()
        except Exception as e:
            raise RuntimeError(f"Failed to load Kubernetes config: {e}")
        
        # Initialize API clients
        self.batch_v1 = client.BatchV1Api()
        self.core_v1 = client.CoreV1Api()
        
        # Generated Job name
        self.job_name = None
    
    def validate(self) -> bool:
        """Validate Kubernetes cluster access and configuration"""
        try:
            # Test cluster connectivity
            version = client.VersionApi().get_code()
            self.console.print(f"[green]✓ Connected to K8s cluster (v{version.major}.{version.minor})[/green]")
            
            # Check if namespace exists
            try:
                self.core_v1.read_namespace(self.namespace)
                self.console.print(f"[green]✓ Namespace '{self.namespace}' exists[/green]")
            except ApiException as e:
                if e.status == 404:
                    self.console.print(f"[yellow]⚠ Namespace '{self.namespace}' not found[/yellow]")
                    # Could create it here, or fail
                    return False
                raise
            
            # Validate AMD GPU Device Plugin is deployed (check for amd.com/gpu resource)
            nodes = self.core_v1.list_node()
            amd_gpu_nodes = [n for n in nodes.items 
                           if self.gpu_resource_name in n.status.allocatable]
            
            if not amd_gpu_nodes:
                self.console.print(
                    f"[yellow]⚠ No nodes with {self.gpu_resource_name} found[/yellow]\n"
                    f"[yellow]  Ensure AMD GPU Device Plugin is deployed:[/yellow]\n"
                    f"[yellow]  kubectl create -f https://raw.githubusercontent.com/ROCm/k8s-device-plugin/master/k8s-ds-amdgpu-dp.yaml[/yellow]"
                )
                return False
            
            self.console.print(f"[green]✓ Found {len(amd_gpu_nodes)} AMD GPU nodes[/green]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]✗ Validation failed: {e}[/red]")
            return False
    
    def prepare(self) -> bool:
        """Prepare K8s Job manifest"""
        try:
            # Get model info
            model_keys = list(self.manifest["built_models"].keys())
            if not model_keys:
                raise ValueError("No models in manifest")
            
            model_key = model_keys[0]
            model_info = self.manifest["built_models"][model_key]
            image_info = self.manifest["built_images"][model_key]
            
            # Generate job name (K8s compatible: lowercase, hyphens)
            self.job_name = f"madengine-{model_info['name'].lower().replace('_', '-')}"
            
            # Build Job manifest using Python objects (not YAML template)
            self.job_manifest = self._build_job_manifest(model_info, image_info)
            
            self.console.print(f"[green]✓ Prepared Job manifest: {self.job_name}[/green]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]✗ Failed to prepare manifest: {e}[/red]")
            return False
    
    def _build_job_manifest(self, model_info: Dict, image_info: Dict) -> client.V1Job:
        """Build K8s Job manifest using Python objects"""
        gpu_count = int(model_info.get("n_gpus", 1))
        
        # Container specification
        container = client.V1Container(
            name=self.job_name,
            image=image_info["registry_image"],
            image_pull_policy=self.k8s_config.get("image_pull_policy", "Always"),
            working_dir="/workspace",
            command=["/bin/bash", "-c"],
            args=[self._get_container_script(model_info)],
            resources=client.V1ResourceRequirements(
                requests={
                    self.gpu_resource_name: str(gpu_count),
                    "memory": self.k8s_config.get("memory", "128Gi"),
                    "cpu": self.k8s_config.get("cpu", "32")
                },
                limits={
                    self.gpu_resource_name: str(gpu_count),
                    "memory": self.k8s_config.get("memory_limit", "256Gi"),
                    "cpu": self.k8s_config.get("cpu_limit", "64")
                }
            ),
            volume_mounts=self._build_volume_mounts()
        )
        
        # Pod specification
        pod_spec = client.V1PodSpec(
            restart_policy="Never",
            containers=[container],
            node_selector=self.k8s_config.get("node_selector", {}),
            tolerations=self._build_tolerations(),
            volumes=self._build_volumes()
        )
        
        # Job specification
        job_spec = client.V1JobSpec(
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={
                        "app": "madengine",
                        "model": model_info["name"]
                    }
                ),
                spec=pod_spec
            ),
            backoff_limit=self.k8s_config.get("backoff_limit", 3),
            completions=1,
            parallelism=1
        )
        
        # Complete Job object
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=self.job_name,
                namespace=self.namespace,
                labels={
                    "app": "madengine",
                    "model": model_info["name"],
                    "madengine-job": "true"
                }
            ),
            spec=job_spec
        )
        
        return job
    
    def _get_container_script(self, model_info: Dict) -> str:
        """Generate container startup script"""
        return """
        set -e
        echo "MADEngine Kubernetes Job Starting..."
        
        # GPU visibility (AMD GPU Device Plugin handles allocation)
        export ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-0}
        
        # Run MAD model automation workflow
        cd /workspace
        bash run.sh
        
        # Copy results if configured
        if [ -f "perf.csv" ] && [ -d "/results" ]; then
            cp perf.csv /results/perf_${HOSTNAME}.csv
        fi
        
        echo "Job completed with exit code $?"
        """
    
    def _build_volume_mounts(self) -> list:
        """Build volume mounts from configuration"""
        mounts = []
        
        if self.k8s_config.get("results_pvc"):
            mounts.append(client.V1VolumeMount(
                name="results",
                mount_path="/results"
            ))
        
        if self.k8s_config.get("data_pvc"):
            mounts.append(client.V1VolumeMount(
                name="data",
                mount_path="/data",
                read_only=True
            ))
        
        return mounts
    
    def _build_volumes(self) -> list:
        """Build volumes from configuration"""
        volumes = []
        
        if self.k8s_config.get("results_pvc"):
            volumes.append(client.V1Volume(
                name="results",
                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                    claim_name=self.k8s_config["results_pvc"]
                )
            ))
        
        if self.k8s_config.get("data_pvc"):
            volumes.append(client.V1Volume(
                name="data",
                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                    claim_name=self.k8s_config["data_pvc"]
                )
            ))
        
        return volumes
    
    def _build_tolerations(self) -> list:
        """Build tolerations from configuration"""
        tolerations_config = self.k8s_config.get("tolerations", [])
        tolerations = []
        
        for tol in tolerations_config:
            tolerations.append(client.V1Toleration(
                key=tol.get("key"),
                operator=tol.get("operator", "Equal"),
                value=tol.get("value", ""),
                effect=tol.get("effect", "NoSchedule")
            ))
        
        return tolerations
    
    def deploy(self) -> DeploymentResult:
        """Submit Job to Kubernetes cluster"""
        try:
            # Create Job using Python API
            job = self.batch_v1.create_namespaced_job(
                namespace=self.namespace,
                body=self.job_manifest
            )
            
            self.console.print(f"[green]✓ Submitted K8s Job: {self.job_name}[/green]")
            self.console.print(f"  Namespace: {self.namespace}")
            self.console.print(f"  Image: {self.job_manifest.spec.template.spec.containers[0].image}")
            
            return DeploymentResult(
                status=DeploymentStatus.SUCCESS,
                deployment_id=self.job_name,
                message=f"Job {self.job_name} created successfully"
            )
            
        except ApiException as e:
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id="",
                message=f"K8s API error: {e.reason} - {e.body}"
            )
        except Exception as e:
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id="",
                message=f"Deployment error: {str(e)}"
            )
    
    def monitor(self, deployment_id: str) -> DeploymentResult:
        """Monitor Job status using Python API"""
        try:
            job = self.batch_v1.read_namespaced_job_status(
                name=deployment_id,
                namespace=self.namespace
            )
            
            # Check job conditions
            if job.status.succeeded:
                return DeploymentResult(
                    status=DeploymentStatus.SUCCESS,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} completed successfully"
                )
            
            if job.status.failed:
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} failed"
                )
            
            if job.status.active:
                return DeploymentResult(
                    status=DeploymentStatus.RUNNING,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} running ({job.status.active} active pods)"
                )
            
            return DeploymentResult(
                status=DeploymentStatus.PENDING,
                deployment_id=deployment_id,
                message=f"Job {deployment_id} pending"
            )
            
        except ApiException as e:
            if e.status == 404:
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} not found"
                )
            raise
    
    def collect_results(self, deployment_id: str) -> Dict[str, Any]:
        """Collect Job results and logs"""
        results = {
            "job_name": deployment_id,
            "namespace": self.namespace,
            "logs": []
        }
        
        try:
            # Get pods for this job
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"job-name={deployment_id}"
            )
            
            # Collect logs from each pod
            for pod in pods.items:
                pod_name = pod.metadata.name
                try:
                    log = self.core_v1.read_namespaced_pod_log(
                        name=pod_name,
                        namespace=self.namespace
                    )
                    results["logs"].append({
                        "pod": pod_name,
                        "log": log
                    })
                except ApiException:
                    pass
            
            self.console.print(f"[green]✓ Collected logs from {len(results['logs'])} pods[/green]")
            
        except Exception as e:
            self.console.print(f"[yellow]⚠ Results collection incomplete: {e}[/yellow]")
        
        return results
    
    def cleanup(self, deployment_id: str) -> bool:
        """Delete Job and associated pods"""
        try:
            # Delete Job (propagates to pods)
            self.batch_v1.delete_namespaced_job(
                name=deployment_id,
                namespace=self.namespace,
                propagation_policy="Background"
            )
            
            self.console.print(f"[yellow]Deleted K8s Job: {deployment_id}[/yellow]")
            return True
            
        except ApiException as e:
            if e.status == 404:
                return True  # Already deleted
            self.console.print(f"[yellow]⚠ Cleanup warning: {e.reason}[/yellow]")
            return False
        except Exception as e:
            self.console.print(f"[yellow]⚠ Cleanup error: {e}[/yellow]")
            return False
```

**Key Production Features**:
- ✅ **Python API**: Type-safe, no string parsing
- ✅ **Native Kubernetes objects**: `client.V1Job`, `client.V1Pod`
- ✅ **Better error handling**: ApiException with status codes
- ✅ **No kubectl dependency**: Pure Python
- ✅ **In-cluster support**: Can run inside K8s pod
- ✅ **Comprehensive**: Job creation, monitoring, log collection, cleanup
- ✅ **AMD GPU Integration**: Uses `amd.com/gpu` resource from Device Plugin

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

### 4.5 Phase 5: Cleanup & Documentation (Week 8)

#### 4.5.1 Delete Old `runners/` Folder

**Action**: Complete removal of deprecated code

```bash
# Delete entire runners/ directory
rm -rf src/madengine/runners/

# Files being deleted:
# - src/madengine/runners/__init__.py
# - src/madengine/runners/base.py
# - src/madengine/runners/factory.py
# - src/madengine/runners/ssh_runner.py
# - src/madengine/runners/ansible_runner.py
# - src/madengine/runners/k8s_runner.py
# - src/madengine/runners/slurm_runner.py
# - src/madengine/runners/orchestrator_generation.py
# - src/madengine/runners/template_generator.py
# - src/madengine/runners/templates/

# Also delete old distributed_orchestrator.py
rm src/madengine/tools/distributed_orchestrator.py
```

**Verify no imports remain**:
```bash
# Search for any remaining imports
grep -r "from madengine.runners" src/
grep -r "import madengine.runners" src/
grep -r "distributed_orchestrator" src/

# All should return empty (no matches)
```

#### 4.5.2 Remove CLI Sub-Commands

Update `src/madengine/mad_cli.py`:

```python
# REMOVE these sub-applications:
# generate_app = typer.Typer(...)  # ❌ DELETE
# runner_app = typer.Typer(...)    # ❌ DELETE

# KEEP only:
app = typer.Typer(...)  # Main app with build, run, discover commands
```

**Commands removed**:
- `madengine-cli generate` (entire sub-command)
- `madengine-cli runner` (entire sub-command)

**Commands kept**:
- ✅ `madengine-cli build`
- ✅ `madengine-cli run`
- ✅ `madengine-cli discover`

#### 4.5.3 Update Documentation

Create `docs/DEPLOYMENT_GUIDE.md` with examples for all three modes:
- Local single-node execution
- SLURM multi-node deployment
- Kubernetes cluster deployment

Update `README.md` to reflect new architecture and removed features.

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

### Week 1: Base Classes & SLURM
- [x] Design review (this document)
- [ ] Create `deployment/base.py` (BaseDeployment, DeploymentConfig, DeploymentResult)
- [ ] Create `deployment/factory.py` (DeploymentFactory - 2 types)
- [ ] Create SLURM Jinja2 template (job.sh.j2)
- [ ] Implement `deployment/slurm.py` (SlurmDeployment class)
- [ ] Update `mad_cli.py` routing (local vs distributed)
- [ ] Test sbatch script generation

**Deliverable**: SLURM deployment working (generate + submit sbatch)

### Week 2: Kubernetes Integration  
- [ ] Verify AMD GPU Device Plugin is deployed on K8s cluster
- [ ] Create Kubernetes Jinja2 template (job.yaml.j2)
- [ ] Implement `deployment/kubernetes.py` (KubernetesDeployment class)
- [ ] Test K8s Job manifest generation with `amd.com/gpu` resources
- [ ] Test kubectl apply and pod scheduling
- [ ] Test with AMD GPU node selectors

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
- [ ] Clear execution model (local run + 2 distributed deployments)
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

## 10. REMOVAL VS REPLACEMENT SUMMARY

### Complete Mapping: Old → New

| Old (Being Removed) | New (Replacement) | Status |
|---------------------|-------------------|--------|
| **`runners/` folder** | `deployment/` folder | ✅ Complete replacement |
| `runners/base.py` | `deployment/base.py` | ✅ Redesigned with better abstractions |
| `runners/factory.py` | `deployment/factory.py` | ✅ Simplified factory pattern |
| `runners/slurm_runner.py` | `deployment/slurm.py` | ✅ Uses CLI commands (subprocess) |
| `runners/k8s_runner.py` | `deployment/kubernetes.py` | ✅ Uses Python library (kubernetes) |
| `runners/ssh_runner.py` | ❌ None | ⚠️ Removed (out of scope) |
| `runners/ansible_runner.py` | ❌ None | ⚠️ Removed (out of scope) |
| `runners/orchestrator_generation.py` | Jinja2 direct usage | ✅ Simpler, no wrapper |
| `runners/template_generator.py` | Jinja2 direct usage | ✅ Simpler, no wrapper |
| `runners/templates/` | `deployment/templates/` | ✅ Moved and simplified |
| `distributed_orchestrator.py` | `orchestration/build_orchestrator.py` + `orchestration/run_orchestrator.py` | ✅ Split for clarity |
| `generate` CLI sub-command | Auto-generation in deployment | ✅ No manual step needed |
| `runner` CLI sub-command | `run` with `--additional-context` | ✅ Unified command |

### What Users Need to Know

#### ❌ These Commands NO LONGER EXIST:
```bash
madengine-cli generate ansible  # Removed
madengine-cli generate k8s      # Removed
madengine-cli generate slurm    # Removed
madengine-cli runner ssh        # Removed
madengine-cli runner ansible    # Removed
madengine-cli runner k8s        # Removed
madengine-cli runner slurm      # Removed
```

#### ✅ Use These Instead:
```bash
# Local execution (unchanged)
madengine-cli run --tags model

# SLURM deployment (NEW unified approach)
madengine-cli run --tags model --additional-context '{"deploy": "slurm", ...}'

# Kubernetes deployment (NEW unified approach)
madengine-cli run --tags model --additional-context '{"deploy": "k8s", ...}'
```

### Code Deletion Checklist

When implementing Phase 5, ensure these are **completely deleted**:

- [ ] Delete `src/madengine/runners/` directory (ALL files)
- [ ] Delete `src/madengine/tools/distributed_orchestrator.py`
- [ ] Remove `generate_app` from `mad_cli.py`
- [ ] Remove `runner_app` from `mad_cli.py`
- [ ] Remove all `from madengine.runners` imports across codebase
- [ ] Remove references in `pyproject.toml` (if any)
- [ ] Remove references in tests (update to use new `deployment/`)
- [ ] Update documentation to reflect removal

---

## 11. PRODUCTION-READY ARCHITECTURE SUMMARY

### 11.1 Checklist Verification ✅

Based on the comprehensive analysis and architectural decisions:

#### ✅ 1. Support Separate Build/Run Phases
**Status**: FULLY SUPPORTED

```bash
# Separate phases (distributed build/run)
madengine-cli build --tags model --registry docker.io
madengine-cli run --manifest-file build_manifest.json
```

**Implementation**: 
- `BuildOrchestrator`: Handles build workflow independently
- `RunOrchestrator`: Loads manifest and executes (checks for existing manifest first)

---

#### ✅ 2. Support Full Workflow (Build+Run in One Command)
**Status**: FULLY SUPPORTED (Backward Compatible)

```bash
# Full workflow - current behavior PRESERVED
madengine-cli run --tags model

# Detection logic in RunOrchestrator:
if not manifest_file or not os.path.exists(manifest_file):
    if tags:
        self._build_phase(tags)  # Build first, then run
```

**Backward Compatibility**: Existing users can continue using `madengine-cli run --tags` for combined workflow.

---

#### ✅ 3. SLURM Uses CLI Commands (subprocess)
**Status**: IMPLEMENTED

**Approach**: `subprocess.run(['sbatch', ...])` - NO Python library

**Rationale**:
- ✅ Zero dependencies (`pyslurm` not needed)
- ✅ Works with any SLURM version
- ✅ Industry standard (Airflow, Prefect, Ray use CLI)
- ✅ Simple, reliable, portable

**Implementation**: `src/madengine/deployment/slurm.py`
```python
class SlurmDeployment(BaseDeployment):
    REQUIRED_TOOLS = ["sbatch", "squeue", "scontrol"]  # CLI tools
    
    def deploy(self):
        result = subprocess.run(
            ['sbatch', str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
```

---

#### ✅ 4. Kubernetes Uses Python Library
**Status**: IMPLEMENTED

**Approach**: `from kubernetes import client, config` - Official Python client

**Rationale**:
- ✅ Type-safe API (no string parsing)
- ✅ Better error handling (Python exceptions)
- ✅ Production standard (Kubeflow, Argo use it)
- ✅ No kubectl installation required
- ✅ Works in-cluster and out-of-cluster

**Implementation**: `src/madengine/deployment/kubernetes.py`
```python
class KubernetesDeployment(BaseDeployment):
    def __init__(self, config):
        from kubernetes import client, config as k8s_config
        k8s_config.load_kube_config()
        self.batch_v1 = client.BatchV1Api()
    
    def deploy(self):
        job = self.batch_v1.create_namespaced_job(
            namespace=self.namespace,
            body=self.job_manifest
        )
```

**Dependency**: `pip install kubernetes` (added to `pyproject.toml` optional dependencies)

---

#### ✅ 5. Proper Layered Architecture
**Status**: IMPLEMENTED

```
┌─────────────────────────────────────┐
│  LAYER 1: Presentation (mad_cli.py) │  ← CLI argument parsing
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  LAYER 2: Orchestration             │  ← Workflow coordination
│  ├─ BuildOrchestrator               │
│  └─ RunOrchestrator                 │
└─────────┬──────────────┬────────────┘
          │              │
          ▼              ▼
┌──────────────┐  ┌─────────────────┐
│ LAYER 3a:    │  │ LAYER 3b:       │
│ Execution    │  │ Deployment      │
│ (Local)      │  │ (Distributed)   │
│              │  │                 │
│ container_   │  │ ├─ slurm.py     │
│ runner.py    │  │ └─ kubernetes.py│
└──────────────┘  └─────────────────┘
```

**Benefits**:
- Clear separation of concerns
- Easy to test (mock each layer)
- Extensible (add new deployment types)
- Maintainable (changes isolated to layers)

---

#### ✅ 6. Best Practices & Code Quality
**Status**: PRODUCTION-READY

**Design Patterns Applied**:
- ✅ **Factory Pattern**: `DeploymentFactory` for dynamic deployment selection
- ✅ **Strategy Pattern**: `BaseDeployment` with SLURM/K8s implementations
- ✅ **Template Method**: Common workflow in base, specifics in subclasses
- ✅ **Dependency Injection**: Context and config passed to orchestrators

**Industry Standards**:
- ✅ SLURM CLI approach (matches Airflow, Prefect, Ray)
- ✅ Kubernetes Python client (matches Kubeflow, Argo Workflows)
- ✅ Jinja2 templates (industry standard for config generation)
- ✅ Type hints throughout (Python 3.8+ standards)

**Testing Strategy**:
- ✅ Mock subprocess for SLURM testing
- ✅ Mock kubernetes.client for K8s testing
- ✅ Layer isolation enables unit testing
- ✅ Integration tests with real clusters (optional)

---

### 11.2 Workflow Examples

#### Example 1: Local Single-Node (Current Behavior)
```bash
madengine-cli run --tags dummy
# → BuildOrchestrator builds image
# → RunOrchestrator detects local
# → container_runner.py executes
```

#### Example 2: Separate Build/Run for SLURM

**User Workflow** (manual SSH to login node):
```bash
# Step 1: On local/build machine
madengine-cli build --tags llama2 --registry docker.io
# Generates: build_manifest.json

# Step 2: Copy manifest to SLURM cluster
scp build_manifest.json user@hpc-login.example.com:~/

# Step 3: SSH to SLURM login node (MANUAL)
ssh user@hpc-login.example.com

# Step 4: On SLURM login node, run madengine-cli
madengine-cli run --manifest-file build_manifest.json \
  --additional-context '{
    "deploy": "slurm",
    "slurm": {"partition": "gpu", "nodes": 4, "gpus_per_node": 8}
  }'

# What happens:
# → User is already ON login node (no SSH needed by madengine-cli)
# → RunOrchestrator loads manifest
# → SlurmDeployment generates sbatch script
# → subprocess.run(['sbatch', 'job.sh'])  ← Runs locally
# → SLURM scheduler allocates nodes and runs job
```

**Key Point**: madengine-cli does NOT handle SSH. User manually SSHs to login node first.

#### Example 3: Full Workflow to Kubernetes
```bash
madengine-cli run --tags vllm-mixtral \
  --additional-context '{
    "deploy": "k8s",
    "k8s": {"namespace": "ml-prod", "gpus": 8}
  }'
# → BuildOrchestrator builds (no manifest provided)
# → RunOrchestrator routes to K8s
# → KubernetesDeployment.batch_v1.create_namespaced_job(...)
```

---

### 11.3 Migration Path

**Phase 1** (Weeks 1-2): Create orchestration layer
- ✅ No breaking changes
- ✅ Existing code continues working
- ✅ New orchestrators coexist with `distributed_orchestrator.py`

**Phase 2** (Weeks 3-4): Implement SLURM deployment
- ✅ SLURM CLI commands (subprocess)
- ✅ Jinja2 templates
- ✅ Full madengine workflow on each node

**Phase 3** (Weeks 5-6): Implement K8s deployment
- ✅ Kubernetes Python library
- ✅ AMD GPU Device Plugin integration
- ✅ Type-safe Job creation and monitoring

**Phase 4** (Week 7): Integration & Testing
- ✅ Update `mad_cli.py` to use orchestrators
- ✅ Mark `distributed_orchestrator.py` deprecated
- ✅ Comprehensive testing

**Phase 5** (Week 8): Cleanup & Removal
- ✅ **DELETE** entire `runners/` directory (replaced by `deployment/`)
- ✅ **DELETE** `distributed_orchestrator.py` (replaced by orchestrators)
- ✅ **REMOVE** `generate` and `runner` CLI sub-commands
- ✅ Verify no remaining imports of old modules
- ✅ Update documentation with migration guide

---

### 11.4 Dependencies Summary

**Core Dependencies** (already in project):
- `jinja2`: Template rendering (SLURM scripts, K8s manifests)
- `typer`: CLI framework
- `rich`: Terminal UI

**Optional Dependencies** (add to `pyproject.toml`):
```toml
[project.optional-dependencies]
kubernetes = ["kubernetes>=28.0.0"]
all = ["kubernetes>=28.0.0"]
```

**NO Dependencies Needed**:
- ❌ `pyslurm`: NOT used (SLURM uses CLI commands)
- ❌ `kubectl`: NOT required (K8s uses Python library)

**Installation**:
```bash
# Base install (local + SLURM)
pip install madengine

# With Kubernetes support
pip install madengine[kubernetes]

# Everything
pip install madengine[all]
```

---

### 11.5 Success Criteria

✅ **Backward Compatibility**: Existing `madengine-cli run --tags` continues working  
✅ **Separate Phases**: Build and run can be executed independently  
✅ **Full Workflow**: Single command can build+run (local or distributed)  
✅ **Best Practices**: Industry-standard approaches (CLI for SLURM, library for K8s)  
✅ **Production-Ready**: Proper error handling, logging, monitoring  
✅ **Extensible**: Easy to add new deployment targets  
✅ **Testable**: Layer isolation enables comprehensive testing  
✅ **Maintainable**: Clear architecture, good documentation  

---

**Document Status**: ✅ Ready for Implementation  
**Architecture**: ✅ Production-Ready with Best Practices  
**Next Steps**: Begin Phase 1 - Create Orchestration Layer


