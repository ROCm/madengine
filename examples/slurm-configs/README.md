# SLURM Configuration Examples

This directory contains example configurations for deploying madengine workloads on SLURM HPC clusters.

## 📋 Convention Over Configuration

**No explicit `deploy` field needed!** The presence of a `slurm` field automatically indicates SLURM deployment:

```json
{
  "slurm": {
    "partition": "amd-rccl",
    "nodes": 2,
    "gpus_per_node": 8
  }
}
```

**⚠️ Important:** The default partition is `amd-rccl` (for AMD RCCL clusters). If your cluster uses a different partition name (e.g., `gpu`, `compute`, `batch`), override it in your configuration:

```json
{
  "slurm": {
    "partition": "your-partition-name"  // Override default
  }
}
```

**Check your cluster's partitions:**
```bash
sinfo -o "%P"  # List all available partitions
```

The deployment type is **inferred** from the configuration structure:
- ✅ Deployment type (k8s/slurm/local) inferred from config structure
- ✅ Layered defaults: base → profile → user configuration
- ✅ Intelligent profile selection based on node count

## 📁 Example Configurations

### Training Configurations (`basic/`)

| File | Description | Nodes | GPUs | Use Case |
|------|-------------|-------|------|----------|
| `01-torchrun-single-node-single-gpu.json` | Single GPU training | 1 | 1 | Quick tests, small models |
| `02-single-node-multi-gpu.json` | Single node, 8 GPUs | 1 | 8 | Single-node distributed workload |
| `03-multi-node-basic.json` | 2 nodes, 8 GPUs each | 2 | 16 | Multi-node distributed workload |
| `04-multi-node-advanced.json` | 4 nodes, advanced features | 4 | 32 | Production-scale training |

### vLLM Inference Configurations (`basic/`)

| File | Description | Nodes | GPUs | Use Case |
|------|-------------|-------|------|----------|
| `05-vllm-single-node.json` | Single node vLLM | 1 | 4 | Single-node LLM inference |
| `06-vllm-multi-node.json` | Multi-node vLLM | 2 | 8 | Multi-node LLM inference with Ray |

### Minimal Examples (`minimal/`)

Stripped-down configurations showing only essential fields:
- `single-gpu-minimal.json` - Minimal single GPU config
- `multi-gpu-minimal.json` - Minimal 8 GPU config
- `multi-node-minimal.json` - Minimal 2-node config
- `vllm-single-node-minimal.json` - Minimal vLLM single-node
- `vllm-multi-node-minimal.json` - Minimal vLLM multi-node

## 🔄 Configuration Workflow

Understanding how configurations flow through madengine:

```
┌──────────────────────────────────────────────────┐
│ 1. Config File (*.json)                          │
│    - Contains: slurm, distributed, env_vars      │
└──────────────────┬───────────────────────────────┘
                   │ --additional-context-file
                   ↓
┌──────────────────────────────────────────────────┐
│ 2. madengine-cli build                           │
│    - BuildOrchestrator._save_deployment_config() │
│    - Extracts env_vars, slurm, distributed       │
└──────────────────┬───────────────────────────────┘
                   │
                   ↓
┌──────────────────────────────────────────────────┐
│ 3. build_manifest.json                           │
│    - deployment_config.env_vars (saved)          │
│    - deployment_config.slurm (saved)             │
└──────────────────┬───────────────────────────────┘
                   │ --manifest-file
                   ↓
┌──────────────────────────────────────────────────┐
│ 4. madengine-cli run                             │
│    - RunOrchestrator._execute_*()                │
│    - Loads deployment_config from manifest       │
└──────────────────┬───────────────────────────────┘
                   │
                   ↓
┌──────────────────────────────────────────────────┐
│ 5. Docker Container Environment                  │
│    - env_vars passed to container                │
│    - SLURM job submitted with configuration      │
└──────────────────────────────────────────────────┘
```

**Key Points:**
- ✅ **Config files are the source of truth** - Don't edit `build_manifest.json` manually
- ✅ **Build phase embeds configuration** - Configuration is saved during build for use at runtime
- ✅ **Run phase uses manifest** - All settings come from the generated manifest
- ✅ **Environment variables flow automatically** - From config → manifest → Docker

## 🚀 Quick Start

### 1. Build-and-Run Workflow (Recommended)

When using configuration files with `env_vars`, use the two-phase workflow:

```bash
# SSH to SLURM login node first
ssh user@hpc-cluster.example.com

# Phase 1: Build with configuration
MODEL_DIR=models/my-model madengine-cli build \
  --tags model_tag \
  --additional-context-file examples/slurm-configs/03-multi-node-basic.json \
  --manifest-output build_manifest.json

# Phase 2: Run from manifest
MODEL_DIR=models/my-model madengine-cli run \
  --manifest-file build_manifest.json
```

**Why two phases?**
- Build phase embeds your `env_vars` and deployment config into the manifest
- Run phase uses the pre-configured manifest
- Ensures consistency across builds and deployments

### 2. Direct Run (For Simple Cases)

For quick tests without custom `env_vars`:

```bash
madengine-cli run --tags model_tag \
  --additional-context-file examples/slurm-configs/minimal/single-gpu-minimal.json
```

### 3. CLI Override

```bash
madengine-cli run --tags model_tag \
  --additional-context '{
    "slurm": {
      "partition": "gpu",
      "nodes": 2,
      "gpus_per_node": 8,
      "time": "24:00:00"
    }
  }'
```

### 4. Hybrid Approach (File + CLI Override)

```bash
# Use base config, override specific fields
madengine-cli run --tags model_tag \
  --additional-context-file examples/slurm-configs/03-multi-node-basic.json \
  --additional-context '{"slurm": {"nodes": 4, "time": "48:00:00"}}'
```

## 🔄 Distributed Workload Support

The SLURM deployment **automatically configures distributed execution** for multi-node and multi-GPU setups (training with torchrun/deepspeed or inference with vLLM/SGLang):

### How It Works

1. **Environment Variables**: SLURM sets distributed execution environment (MASTER_ADDR, MASTER_PORT, RANK, etc.)
2. **MAD_MULTI_NODE_RUNNER**: Automatically configured with the appropriate `torchrun` command
3. **Docker Containers**: Environment variables are passed into containers via `docker_env_vars`
4. **Model Scripts**: Use `$MAD_MULTI_NODE_RUNNER` to launch training (see below)

### Model Script Pattern

Your model's run script should use the `MAD_MULTI_NODE_RUNNER` environment variable:

```bash
#!/bin/bash
# Example: scripts/my_model/run.sh

# MAD_MULTI_NODE_RUNNER is automatically set by madengine for distributed workloads
if [ -z "$MAD_MULTI_NODE_RUNNER" ]; then
    # Fallback for standalone execution
    N_GPUS="${MAD_RUNTIME_NGPUS:-1}"
    MAD_MULTI_NODE_RUNNER="torchrun --standalone --nproc_per_node=$N_GPUS"
fi

# Launch your Python training script with torchrun
$MAD_MULTI_NODE_RUNNER train.py --your-args
```

### Distributed Environment Variables

The following variables are automatically available in your containers:

| Variable | Description | Example |
|----------|-------------|---------|
| `MASTER_ADDR` | Master node address | `node001` |
| `MASTER_PORT` | Master communication port | `29500` |
| `WORLD_SIZE` | Total number of processes | `16` (2 nodes × 8 GPUs) |
| `RANK` | Global process rank | `0`, `1`, ... |
| `LOCAL_RANK` | Local GPU rank on node | `0-7` |
| `NNODES` | Number of nodes | `2` |
| `NPROC_PER_NODE` | GPUs per node | `8` |
| `MAD_MULTI_NODE_RUNNER` | Complete torchrun command | `torchrun --nnodes=2 ...` |

### Example Configurations

**Single-Node Multi-GPU (Data Parallel)**:
```json
{
  "slurm": {
    "nodes": 1,
    "gpus_per_node": 8
  }
}
```
→ `MAD_MULTI_NODE_RUNNER="torchrun --standalone --nproc_per_node=8"`

**Multi-Node Distributed Workload**:
```json
{
  "slurm": {
    "nodes": 4,
    "gpus_per_node": 8
  }
}
```
→ `MAD_MULTI_NODE_RUNNER="torchrun --nnodes=4 --nproc_per_node=8 --node_rank=$SLURM_PROCID --master_addr=$MASTER_ADDR --master_port=29500"`

### Verification

Check that distributed execution is configured correctly:

```bash
# In your SLURM output logs, you should see:
Distributed Execution Configuration:
  NNODES: 2
  GPUS_PER_NODE: 8
  TOTAL_GPUS: 16
  MASTER_ADDR: node001
  MASTER_PORT: 29500
  NODE_RANK: 0
  Launcher: torchrun (distributed)
  MAD_MULTI_NODE_RUNNER: torchrun --nnodes=2 --nproc_per_node=8 ...
```

## 🚀 vLLM Inference Configurations

vLLM is a high-throughput LLM inference engine. madengine provides pre-configured setups for both single-node and multi-node deployments.

### Memory Management

vLLM configurations include critical memory management environment variables to prevent OOM (Out of Memory) errors, especially in multi-node deployments with pipeline parallelism.

#### Key Environment Variables

**1. `VLLM_KV_CACHE_SIZE`**
- **Purpose**: Limits the percentage of GPU memory allocated for KV cache
- **Default in configs**: `0.8` (80% of available GPU memory)
- **Why needed**: Prevents vLLM from aggressively allocating all available memory, which can cause fragmentation and OOM errors
- **Tuning**: 
  - Increase (e.g., `0.9`) if you have large memory headroom
  - Decrease (e.g., `0.6`, `0.7`) if experiencing OOM errors

**2. `PYTORCH_CUDA_ALLOC_CONF`**
- **Purpose**: Configures PyTorch's CUDA/HIP memory allocator
- **Value**: `expandable_segments:True`
- **Why needed**: Reduces memory fragmentation by allowing the allocator to expand memory segments dynamically
- **Reference**: [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

### vLLM Configuration Files

**Single-Node Configurations:**
- `05-vllm-single-node.json` - Full single-node config with NCCL settings
- `vllm-single-node-minimal.json` - Minimal single-node config (in `minimal/` directory)

**Multi-Node Configurations:**
- `06-vllm-multi-node.json` - Full multi-node config with NCCL and Ray settings
- `vllm-multi-node-minimal.json` - Minimal multi-node config (in `minimal/` directory)

### vLLM Workflow Example

```bash
# 1. Build with vLLM configuration
MODEL_DIR=models/llama2-70b madengine-cli build \
  --tags vllm \
  --additional-context-file examples/slurm-configs/basic/06-vllm-multi-node.json \
  --manifest-output build_manifest.json

# 2. Verify memory management env_vars were embedded
grep -A 10 "env_vars" build_manifest.json
# Should show:
#   "VLLM_KV_CACHE_SIZE": "0.8"
#   "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"

# 3. Run the inference job
MODEL_DIR=models/llama2-70b madengine-cli run \
  --manifest-file build_manifest.json
```

### vLLM Parallelism Strategies

vLLM automatically selects parallelism based on your configuration:

**Single-Node (TP only)**:
```json
{
  "slurm": {
    "nodes": 1,
    "gpus_per_node": 4
  },
  "distributed": {
    "launcher": "vllm",
    "nnodes": 1,
    "nproc_per_node": 4
  }
}
```
→ **Tensor Parallelism (TP) = 4** across GPUs

**Multi-Node (TP + PP)**:
```json
{
  "slurm": {
    "nodes": 2,
    "gpus_per_node": 4
  },
  "distributed": {
    "launcher": "vllm",
    "nnodes": 2,
    "nproc_per_node": 4
  }
}
```
→ **Tensor Parallelism (TP) = 4** within each node  
→ **Pipeline Parallelism (PP) = 2** across nodes  
→ **Requires Ray cluster** for multi-node coordination

## ⚙️ Configuration Layers

madengine uses intelligent multi-layer configuration merging:

```
┌─────────────────────────────────┐
│  1. Base Defaults               │ ← slurm/defaults.json
│     (partition, time, etc.)     │
├─────────────────────────────────┤
│  2. Profile Selection           │ ← single-node.json or multi-node.json
│     (auto-selected by nodes)    │   (based on nodes count)
├─────────────────────────────────┤
│  3. User Configuration          │ ← Your config file + CLI args
│     (highest priority)          │
└─────────────────────────────────┘
```

### Profile Auto-Selection

- **Single-node profile**: Applied when `nodes == 1`
- **Multi-node profile**: Applied when `nodes > 1`

## 📝 Configuration Reference

### SLURM Section

```json
{
  "slurm": {
    "partition": "amd-rccl",         // SLURM partition name (default: amd-rccl)
    "nodes": 2,                      // Number of nodes
    "gpus_per_node": 8,             // GPUs per node
    "time": "24:00:00",             // Wall time (HH:MM:SS)
    "output_dir": "./slurm_output", // Local output directory
    "results_dir": "/shared/results", // Shared results collection
    "shared_workspace": "/shared/workspace", // Shared workspace (NFS/Lustre)
    "exclusive": true,               // Exclusive node access
    "qos": "high",                   // Quality of Service
    "account": "project-name",       // SLURM account
    "network_interface": "ib0",      // Network interface (ib0/eth0)
    "modules": ["rocm/5.7.0"]       // Environment modules to load
  }
}
```

### Distributed Execution Section

```json
{
  "distributed": {
    "launcher": "torchrun",    // Launcher type: torchrun, vllm, sglang, deepspeed, megatron
    "backend": "nccl",         // Communication backend (nccl/gloo)
    "port": 29500,             // Master node port
    "nnodes": 2,               // Number of nodes (overrides slurm.nodes if set)
    "nproc_per_node": 8        // GPUs per node (overrides slurm.gpus_per_node if set)
  }
}
```

**Supported Launchers:**
- `torchrun`: PyTorch distributed training (default)
- `vllm`: vLLM inference engine (TP/PP parallelism)
- `sglang`: SGLang inference engine
- `deepspeed`: DeepSpeed training framework
- `megatron`: Megatron-LM large model training
- Custom: Set environment variables, model script handles launcher

**Note**: For vLLM and SGLang, the model script handles process spawning directly.
For torchrun/deepspeed/megatron, use `$MAD_MULTI_NODE_RUNNER` in your model script.

### Environment Variables

```json
{
  "env_vars": {
    "NCCL_DEBUG": "WARN",
    "NCCL_SOCKET_IFNAME": "ib0",
    "OMP_NUM_THREADS": "8",
    "MIOPEN_FIND_MODE": "1",
    "VLLM_KV_CACHE_SIZE": "0.8",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
  }
}
```

**Note**: Environment variables set in config files are:
1. Saved to `deployment_config.env_vars` during `build` phase
2. Automatically passed to Docker containers during `run` phase
3. Available to your model scripts inside containers

## 🔍 Common Use Cases

### Testing on Single GPU

```bash
madengine-cli run --tags my_model \
  --additional-context-file examples/slurm-configs/minimal/single-gpu-minimal.json
```

### Multi-Node Training

```bash
# Build with config
MODEL_DIR=models/my-model madengine-cli build \
  --tags training \
  --additional-context-file examples/slurm-configs/03-multi-node-basic.json

# Run from manifest
MODEL_DIR=models/my-model madengine-cli run \
  --manifest-file build_manifest.json
```

### vLLM Single-Node Inference

```bash
# Build with vLLM config
MODEL_DIR=models/llama2-13b madengine-cli build \
  --tags vllm \
  --additional-context-file examples/slurm-configs/basic/05-vllm-single-node.json

# Run inference
MODEL_DIR=models/llama2-13b madengine-cli run \
  --manifest-file build_manifest.json
```

### vLLM Multi-Node Inference

```bash
# Build with multi-node vLLM config
MODEL_DIR=models/llama2-70b madengine-cli build \
  --tags vllm \
  --additional-context-file examples/slurm-configs/basic/06-vllm-multi-node.json

# Run multi-node inference
MODEL_DIR=models/llama2-70b madengine-cli run \
  --manifest-file build_manifest.json
```

### Production Deployment with Shared Storage

```bash
madengine-cli build --tags my_model \
  --additional-context-file examples/slurm-configs/04-multi-node-advanced.json

madengine-cli run --manifest-file build_manifest.json
```

### Custom vLLM Memory Settings

For custom memory configurations, create a new config file:

```json
{
  "slurm": {
    "partition": "amd-rccl",
    "nodes": 2,
    "gpus_per_node": 8,
    "time": "04:00:00"
  },
  
  "distributed": {
    "launcher": "vllm",
    "nnodes": 2,
    "nproc_per_node": 8
  },
  
  "env_vars": {
    "VLLM_KV_CACHE_SIZE": "0.7",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "HSA_FORCE_FINE_GRAIN_PCIE": "1"
  }
}
```

## 🛠️ Advanced Features

### Custom Environment Modules

Load specific software versions:

```json
{
  "slurm": {
    "modules": [
      "rocm/5.7.0",
      "gcc/11.2.0",
      "openmpi/4.1.4"
    ]
  }
}
```

### Shared Filesystem

Configure shared workspace and data:

```json
{
  "slurm": {
    "shared_workspace": "/lustre/workspace",
    "results_dir": "/lustre/results"
  },
  "shared_data": "/lustre/datasets"
}
```

### Network Configuration

For InfiniBand clusters:

```json
{
  "slurm": {
    "network_interface": "ib0"
  },
  "env_vars": {
    "NCCL_SOCKET_IFNAME": "ib0",
    "NCCL_IB_DISABLE": "0",
    "NCCL_IB_HCA": "mlx5_0:1,mlx5_1:1"
  }
}
```

## 📊 Monitoring Jobs

After submission, monitor your SLURM job:

```bash
# Check job status
squeue -u $USER

# View job details
scontrol show job <job_id>

# View output logs (real-time)
tail -f slurm_output/madengine-*_<job_id>_*.out

# View error logs
tail -f slurm_output/madengine-*_<job_id>_*.err

# Cancel job if needed
scancel <job_id>
```

## 🐛 Troubleshooting

### Job Fails Immediately

- Check SLURM partition exists: `sinfo`
- Verify GPU resources available: `sinfo -o "%P %.5a %.10l %.6D %.6t %N %G"`
- Check SLURM account/QoS settings
- Review job script: `slurm_output/madengine_*.sh`

### Out of Memory Errors

**General OOM**:
- Reduce batch size or model size
- Use gradient accumulation
- Enable CPU offloading
- Check available GPU memory: `rocm-smi` or `amd-smi`

**vLLM-Specific OOM** (`torch.OutOfMemoryError: HIP out of memory`):

**Symptom**: Error during vLLM initialization or KV cache allocation:
```
torch.OutOfMemoryError: HIP out of memory. Tried to allocate 22.14 GiB. 
GPU has a total capacity of 191.98 GiB of which 145.02 GiB is free.
```

**Root Cause**: Memory fragmentation or aggressive KV cache allocation

**Solutions**:
1. **Reduce KV cache size**: 
   ```json
   "env_vars": {
     "VLLM_KV_CACHE_SIZE": "0.6"  // Try 0.6 or 0.7
   }
   ```
2. **Enable expandable segments** (should already be in configs):
   ```json
   "env_vars": {
     "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
   }
   ```
3. **Reduce parallelism**: Use fewer GPUs or nodes for smaller models
4. **Check GPU memory**: `rocm-smi` or `amd-smi` to verify available memory
5. **Rebuild with updated config**: Don't edit `build_manifest.json` - update the source config file and rebuild

### NCCL/Communication Errors

- Verify network interface name: `ifconfig` or `ip addr`
- Check InfiniBand status: `ibstat` (if using IB)
- Test connectivity between nodes
- Set correct `NCCL_SOCKET_IFNAME` in `env_vars`

### vLLM Ray Connection Failures

**Symptom**: `Failed to connect to GCS at address <node>:6379`

**Solutions**:
1. Check network connectivity between nodes
2. Ensure Ray port (6379) is accessible
3. Verify NCCL/RCCL environment variables are set correctly
4. For smaller models, consider using tensor parallelism only (single node)

### Module Load Failures

- List available modules: `module avail`
- Check module syntax: `module load rocm/5.7.0` (manual test)
- Verify module names match cluster configuration

## 💡 Best Practices

### General

1. **Start Small**: Test on single GPU first, then scale up
2. **Use Configuration Files**: Prefer config files over CLI arguments for reproducibility
3. **Build-Then-Run**: Use two-phase workflow when configs include `env_vars`
4. **Use Shared Storage**: Configure shared workspace for multi-node jobs
5. **Network Configuration**: Properly configure NCCL for your network fabric
6. **Resource Requests**: Request exclusive node access for large jobs
7. **Time Limits**: Set realistic wall times (add buffer for checkpointing)
8. **Output Collection**: Use `results_dir` to collect outputs from all nodes

### vLLM-Specific

1. **Memory Management**: Always include `VLLM_KV_CACHE_SIZE` and `PYTORCH_CUDA_ALLOC_CONF`
2. **Start Conservative**: Use `VLLM_KV_CACHE_SIZE: "0.8"` initially, tune if needed
3. **Test Locally First**: Validate vLLM configs on single-node before scaling to multi-node
4. **Monitor Memory**: Check GPU memory usage during initialization
5. **Don't Edit Manifests**: Always modify source config files, not generated `build_manifest.json`
6. **Rebuild After Changes**: Re-run `build` phase when changing `env_vars`

### Configuration Management

1. **Version Control**: Keep your config files in git
2. **Naming Convention**: Use descriptive names (e.g., `my-project-vllm-8gpu.json`)
3. **Documentation**: Add `_comment` and `_description` fields to configs
4. **Reusability**: Create base configs and override specific fields
5. **Validation**: Test configs on small scale before production runs

## 🎯 Example Workflow

### Standard Training Workflow

```bash
# 1. SSH to SLURM login node
ssh user@hpc-cluster.example.com

# 2. Load any required modules (if needed before madengine)
module load python/3.9

# 3. Build with configuration
MODEL_DIR=models/my-model madengine-cli build \
  --tags llama2_training \
  --additional-context-file examples/slurm-configs/03-multi-node-basic.json \
  --manifest-output build_manifest.json

# 4. Run from manifest
MODEL_DIR=models/my-model madengine-cli run \
  --manifest-file build_manifest.json

# 5. Monitor job
watch squeue -u $USER

# 6. Check logs when complete
ls -lh slurm_output/
tail -f slurm_output/madengine-*_<job_id>_*.out
```

### vLLM Inference Workflow

```bash
# 1. SSH to SLURM login node
ssh user@hpc-cluster.example.com

# 2. Build vLLM image with memory management config
MODEL_DIR=models/llama2-70b madengine-cli build \
  --tags vllm \
  --additional-context-file examples/slurm-configs/basic/06-vllm-multi-node.json \
  --manifest-output build_manifest.json

# 3. Verify configuration was embedded
grep -A 5 "VLLM_KV_CACHE_SIZE" build_manifest.json

# 4. Submit inference job
MODEL_DIR=models/llama2-70b madengine-cli run \
  --manifest-file build_manifest.json

# 5. Monitor for OOM errors
tail -f slurm_output/madengine-*_<job_id>_*.err | grep -i "memory"

# 6. If OOM occurs, adjust config and rebuild
# Edit your config file to set VLLM_KV_CACHE_SIZE to 0.6 or 0.7
# Then repeat steps 2-4
```

## 📚 Related Documentation

- [How to Run Multi-Node](../../docs/how-to-run-multi-node.md)
- [K8s Configuration Examples](../k8s-configs/)
- [SLURM Official Documentation](https://slurm.schedmd.com/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [vLLM Distributed Inference](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
- [SGLang Distributed Serving](https://sgl-project.github.io/)

---

**Note**: All configurations assume you've already SSH'd to the SLURM login node. madengine runs `sbatch` locally on the login node - no remote SSH handling needed.
