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

### Basic Examples

| File | Description | Nodes | GPUs | Use Case |
|------|-------------|-------|------|----------|
| `01-single-node-single-gpu.json` | Single GPU testing | 1 | 1 | Quick tests, small models |
| `02-single-node-multi-gpu.json` | Single node, 8 GPUs | 1 | 8 | Single-node distributed training |
| `03-multi-node-basic.json` | 2 nodes, 8 GPUs each | 2 | 16 | Multi-node distributed training |
| `04-multi-node-advanced.json` | 4 nodes, advanced features | 4 | 32 | Production-scale training |

### Minimal Examples (`minimal/`)

Stripped-down configurations showing only essential fields:
- `single-gpu-minimal.json` - Minimal single GPU config
- `multi-gpu-minimal.json` - Minimal 8 GPU config
- `multi-node-minimal.json` - Minimal 2-node config

## 🚀 Quick Start

### 1. Using Configuration File

```bash
# SSH to SLURM login node first
ssh user@hpc-cluster.example.com

# Run with configuration file
madengine-cli run --tags model_tag \
  --additional-context-file examples/slurm-configs/03-multi-node-basic.json
```

### 2. Using CLI Arguments

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

### 3. Hybrid Approach (File + CLI Override)

```bash
# Use base config, override specific fields
madengine-cli run --tags model_tag \
  --additional-context-file examples/slurm-configs/03-multi-node-basic.json \
  --additional-context '{"slurm": {"nodes": 4, "time": "48:00:00"}}'
```

## 🔄 Distributed Training Support

The SLURM deployment **automatically configures distributed training** for multi-node and multi-GPU setups:

### How It Works

1. **Environment Variables**: SLURM sets distributed training environment (MASTER_ADDR, MASTER_PORT, RANK, etc.)
2. **MAD_MULTI_NODE_RUNNER**: Automatically configured with the appropriate `torchrun` command
3. **Docker Containers**: Environment variables are passed into containers via `docker_env_vars`
4. **Model Scripts**: Use `$MAD_MULTI_NODE_RUNNER` to launch training (see below)

### Model Script Pattern

Your model's run script should use the `MAD_MULTI_NODE_RUNNER` environment variable:

```bash
#!/bin/bash
# Example: scripts/my_model/run.sh

# MAD_MULTI_NODE_RUNNER is automatically set by madengine for distributed training
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

**Multi-Node Distributed Training**:
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

Check that distributed training is configured correctly:

```bash
# In your SLURM output logs, you should see:
Distributed Training Configuration:
  NNODES: 2
  GPUS_PER_NODE: 8
  TOTAL_GPUS: 16
  MASTER_ADDR: node001
  MASTER_PORT: 29500
  NODE_RANK: 0
  Launcher: torchrun (distributed)
  MAD_MULTI_NODE_RUNNER: torchrun --nnodes=2 --nproc_per_node=8 ...
```

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

### Distributed Training Section

```json
{
  "distributed": {
    "backend": "nccl",    // Communication backend (nccl/gloo)
    "port": 29500         // Master node port
  }
}
```

### Environment Variables

```json
{
  "env_vars": {
    "NCCL_DEBUG": "WARN",
    "NCCL_SOCKET_IFNAME": "ib0",
    "OMP_NUM_THREADS": "8",
    "MIOPEN_FIND_MODE": "1"
  }
}
```

## 🔍 Common Use Cases

### Testing on Single GPU

```bash
madengine-cli run --tags my_model \
  --additional-context-file examples/slurm-configs/minimal/single-gpu-minimal.json
```

### Multi-Node Training

```bash
madengine-cli run --tags my_model \
  --additional-context-file examples/slurm-configs/03-multi-node-basic.json
```

### Production Deployment with Shared Storage

```bash
madengine-cli run --tags my_model \
  --additional-context-file examples/slurm-configs/04-multi-node-advanced.json
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

# View output logs
tail -f slurm_output/madengine-*_<job_id>_*.out

# Cancel job if needed
scancel <job_id>
```

## 🐛 Troubleshooting

### Job Fails Immediately

- Check SLURM partition exists: `sinfo`
- Verify GPU resources available: `sinfo -o "%P %.5a %.10l %.6D %.6t %N %G"`
- Check SLURM account/QoS settings

### Out of Memory Errors

- Reduce batch size or model size
- Use gradient accumulation
- Enable CPU offloading

### NCCL/Communication Errors

- Verify network interface name: `ifconfig` or `ip addr`
- Check InfiniBand status: `ibstat` (if using IB)
- Test connectivity between nodes

### Module Load Failures

- List available modules: `module avail`
- Check module syntax: `module load rocm/5.7.0` (manual test)

## 📚 Related Documentation

- [How to Run Multi-Node](../../docs/how-to-run-multi-node.md)
- [K8s Configuration Examples](../k8s-configs/)
- [SLURM Official Documentation](https://slurm.schedmd.com/)

## 💡 Best Practices

1. **Start Small**: Test on single GPU first, then scale up
2. **Use Shared Storage**: Configure shared workspace for multi-node jobs
3. **Network Configuration**: Properly configure NCCL for your network fabric
4. **Resource Requests**: Request exclusive node access for large jobs
5. **Time Limits**: Set realistic wall times (add buffer for checkpointing)
6. **Output Collection**: Use `results_dir` to collect outputs from all nodes

## 🎯 Example Workflow

```bash
# 1. SSH to SLURM login node
ssh user@hpc-cluster.example.com

# 2. Load any required modules (if needed before madengine)
module load python/3.9

# 3. Run madengine with SLURM config
madengine-cli run --tags llama2_training \
  --additional-context-file examples/slurm-configs/03-multi-node-basic.json

# 4. Monitor job
watch squeue -u $USER

# 5. Check logs when complete
ls -lh slurm_output/
```

---

**Note**: All configurations assume you've already SSH'd to the SLURM login node. madengine runs `sbatch` locally on the login node - no remote SSH handling needed.

