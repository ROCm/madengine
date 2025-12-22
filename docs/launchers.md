# Distributed Launchers Guide

Complete reference for all distributed execution launchers supported by madengine.

---

## Overview

madengine provides unified support for multiple distributed frameworks, enabling seamless execution across training and inference workloads on both Kubernetes and SLURM clusters.

### Supported Launchers

| Launcher | Type | Use Case | K8s | SLURM | Multi-Node |
|----------|------|----------|-----|-------|------------|
| **torchrun** | Training | PyTorch DDP/FSDP training | ✅ | ✅ | ✅ |
| **DeepSpeed** | Training | ZeRO optimization training | ✅ | ✅ | ✅ |
| **Megatron-LM** | Training | Large-scale transformer training | ✅ | ✅ | ✅ |
| **TorchTitan** | Training | LLM pre-training (FSDP2+TP+PP) | ✅ | ✅ | ✅ |
| **vLLM** | Inference | High-throughput LLM serving | ✅ | ✅ | ✅ |
| **SGLang** | Inference | Fast LLM inference | ✅ | ✅ | ✅ |

---

## Quick Start

### Basic Configuration

```json
{
  "distributed": {
    "launcher": "torchrun",
    "nnodes": 2,
    "nproc_per_node": 8
  }
}
```

### Deployment

```bash
# Build with configuration
madengine-cli build --tags my_model \
  --additional-context-file config.json

# Deploy to K8s or SLURM
madengine-cli run --manifest-file build_manifest.json
```

---

## Launcher Details

### 1. torchrun (PyTorch Distributed)

**Purpose**: Standard PyTorch distributed training with DDP/FSDP

**When to Use**:
- ✅ Multi-GPU/multi-node PyTorch training
- ✅ Data Parallel or Fully Sharded Data Parallel
- ✅ Standard distributed training patterns

**Configuration**:
```json
{
  "distributed": {
    "launcher": "torchrun",
    "nnodes": 2,
    "nproc_per_node": 8,
    "master_port": 29500
  }
}
```

**Features**:
- Automatic rank assignment
- NCCL backend for GPU communication
- Elastic training support
- Compatible with all PyTorch models

**Examples**:
- K8s: `examples/k8s-configs/minimal/torchrun-multi-gpu-minimal.json`
- SLURM: `examples/slurm-configs/minimal/torchrun-multi-node-minimal.json`

---

### 2. DeepSpeed

**Purpose**: Memory-efficient training with ZeRO optimization

**When to Use**:
- ✅ Large models that don't fit in GPU memory
- ✅ ZeRO optimization stages (ZeRO-1, ZeRO-2, ZeRO-3)
- ✅ Gradient accumulation and mixed precision

**Configuration**:
```json
{
  "distributed": {
    "launcher": "deepspeed",
    "nnodes": 4,
    "nproc_per_node": 8
  }
}
```

**Features**:
- ZeRO memory optimization
- Pipeline parallelism
- Gradient accumulation
- Mixed precision training
- Automatic hostfile generation (K8s)

**Architecture**:
- Uses its own launcher (not torchrun)
- Manages process spawning internally
- Requires DeepSpeed config file in model script

**Examples**:
- SLURM: `examples/slurm-configs/basic/04-multi-node-advanced.json`

---

### 3. Megatron-LM

**Purpose**: Large-scale transformer model training

**When to Use**:
- ✅ GPT, BERT, T5 style transformers
- ✅ Tensor and pipeline parallelism
- ✅ Very large models (70B+ parameters)

**Configuration**:
```json
{
  "distributed": {
    "launcher": "megatron",
    "nnodes": 4,
    "nproc_per_node": 8
  }
}
```

**Features**:
- Tensor parallelism across GPUs
- Pipeline parallelism across nodes
- Optimized for transformer architectures
- Built on top of torchrun
- Automatic TP/PP size configuration

**Availability**:
- ✅ K8s: Fully supported (dedicated launcher)
- ✅ SLURM: Fully supported

**Examples**:
- K8s: `examples/k8s-configs/minimal/megatron-lm-minimal.json`
- K8s Multi-node: `examples/k8s-configs/basic/megatron-lm-multi-node-basic.json`
- SLURM: `examples/slurm-configs/minimal/megatron-lm-minimal.json`
- SLURM Multi-node: `examples/slurm-configs/basic/09-megatron-lm-multi-node.json`

**Environment Variables** (automatically set by launcher):
```bash
# Megatron-Core standard variables
TENSOR_MODEL_PARALLEL_SIZE    # Tensor parallelism (GPUs per node)
PIPELINE_MODEL_PARALLEL_SIZE  # Pipeline parallelism (typically = nnodes)
CONTEXT_PARALLEL_SIZE         # Context parallelism (default: 1)
```

**Note**: The launcher automatically configures:
- Single-node: TP only (PP=1)
- Multi-node: TP across GPUs + PP across nodes

---

### 4. TorchTitan

**Purpose**: Production LLM pre-training with multi-dimensional parallelism

**Reference**: [pytorch/torchtitan](https://github.com/pytorch/torchtitan)

**When to Use**:
- ✅ Llama 3.1 (8B to 405B) pre-training
- ✅ Multi-dimensional parallelism (FSDP2 + TP + PP + CP)
- ✅ Production-scale LLM training

**Configuration**:
```json
{
  "distributed": {
    "launcher": "torchtitan",
    "nnodes": 4,
    "nproc_per_node": 8
  }
}
```

**Parallelism Strategies**:
- **FSDP2**: Fully Sharded Data Parallel v2 for parameter sharding
- **TP**: Tensor Parallel - split model layers across GPUs
- **PP**: Pipeline Parallel - split model stages across nodes
- **CP**: Context Parallel - distributed context processing

**Features**:
- Uses torchrun as underlying launcher
- Configured via TOML files
- Automatic parallelism detection
- Float8 and MXFP8 support
- Gradient accumulation
- Distributed checkpointing

**Environment Variables**:
```bash
TORCHTITAN_TENSOR_PARALLEL_SIZE=8
TORCHTITAN_PIPELINE_PARALLEL_SIZE=4
TORCHTITAN_FSDP_ENABLED=1
TORCHTITAN_CONTEXT_PARALLEL_SIZE=1
```

**Single vs Multi-Node**:
- Single-node: TP only across GPUs
- Multi-node: TP + PP + FSDP2 combined

**Examples**:
- K8s: `examples/k8s-configs/minimal/torchtitan-single-node-minimal.json`
- SLURM: `examples/slurm-configs/minimal/torchtitan-single-node-minimal.json`

**Model Configuration** (TOML):
```toml
[model]
name = "llama3"
flavor = "8B"

[training]
tensor_parallel_degree = 8
pipeline_parallel_degree = 1
batch_size = 1
seq_len = 8192
```

---

### 5. vLLM

**Purpose**: High-throughput LLM inference serving

**Reference**: [vllm-project/vllm](https://github.com/vllm-project/vllm)

**When to Use**:
- ✅ LLM inference with high throughput
- ✅ Continuous batching
- ✅ PagedAttention for memory efficiency

**Configuration**:
```json
{
  "distributed": {
    "launcher": "vllm",
    "nnodes": 2,
    "nproc_per_node": 4
  }
}
```

**Features**:
- Continuous batching for high throughput
- PagedAttention memory optimization
- Tensor Parallelism support
- Ray for distributed coordination
- No torchrun needed (manages own processes)

**Architecture**:
- Single-node: TP across GPUs, no Ray
- Multi-node (K8s): Data Parallelism with independent replicas per pod
- Multi-node (SLURM): TP + PP with Ray cluster

**Environment Variables**:
```bash
VLLM_TENSOR_PARALLEL_SIZE=4
VLLM_PIPELINE_PARALLEL_SIZE=1
VLLM_DISTRIBUTED_BACKEND="auto"  # or "ray" for multi-node
```

**Examples**:
- K8s: `examples/k8s-configs/minimal/vllm-single-node-minimal.json`
- SLURM: `examples/slurm-configs/minimal/vllm-single-node-minimal.json`

---

### 6. SGLang

**Purpose**: Fast LLM inference with structured generation

**Reference**: [sgl-project/sglang](https://github.com/sgl-project/sglang)

**When to Use**:
- ✅ Structured LLM generation
- ✅ Fast inference with caching
- ✅ OpenAI-compatible API

**Configuration**:
```json
{
  "distributed": {
    "launcher": "sglang",
    "nnodes": 2,
    "nproc_per_node": 4
  }
}
```

**Features**:
- Native launcher (sglang.launch_server)
- RadixAttention for prefix caching
- Tensor Parallelism
- Ray for distributed execution
- No torchrun needed

**Architecture**:
- Single-node: TP across GPUs
- Multi-node: Native multi-node support with Ray

**Environment Variables**:
```bash
SGLANG_TENSOR_PARALLEL_SIZE=4
SGLANG_PIPELINE_PARALLEL_SIZE=1
```

**Examples**:
- K8s: `examples/k8s-configs/minimal/sglang-single-node-minimal.json`
- SLURM: `examples/slurm-configs/basic/05-vllm-single-node.json` (similar pattern)

---

## Comparison Matrix

### Training Launchers

| Feature | torchrun | DeepSpeed | Megatron-LM | TorchTitan |
|---------|----------|-----------|-------------|------------|
| **Data Parallel** | ✅ DDP | ✅ ZeRO | ✅ | ✅ FSDP2 |
| **Tensor Parallel** | ❌ | ❌ | ✅ | ✅ |
| **Pipeline Parallel** | ❌ | ✅ | ✅ | ✅ |
| **Memory Efficiency** | Medium | High (ZeRO) | High | Very High |
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Model Size** | Small-Medium | Medium-Large | Very Large | Very Large |
| **K8s Support** | ✅ | ✅ | ❌ | ✅ |
| **SLURM Support** | ✅ | ✅ | ✅ | ✅ |

### Inference Launchers

| Feature | vLLM | SGLang |
|---------|------|--------|
| **Throughput** | Very High | High |
| **Memory Efficiency** | PagedAttention | RadixAttention |
| **Batching** | Continuous | Continuous |
| **API** | OpenAI-compatible | OpenAI-compatible |
| **Structured Gen** | Limited | ✅ Native |
| **Multi-Node** | ✅ Ray | ✅ Ray |
| **K8s Support** | ✅ | ✅ |
| **SLURM Support** | ✅ | ✅ |

---

## Configuration Best Practices

### 1. Launcher Selection

**Training Workloads**:
```
Small models (< 1B)        → torchrun
Medium models (1B-10B)     → DeepSpeed or torchrun
Large models (10B-70B)     → TorchTitan or Megatron-LM
Very large (70B+)          → TorchTitan with full parallelism
```

**Inference Workloads**:
```
High throughput           → vLLM
Structured generation     → SGLang
Memory constrained        → vLLM (PagedAttention)
```

### 2. Resource Allocation

**GPU Count Guidelines**:
```json
{
  "k8s": {
    "gpu_count": 8  // Matches nproc_per_node
  },
  "distributed": {
    "nnodes": 4,
    "nproc_per_node": 8  // Total: 32 GPUs
  }
}
```

**Memory Recommendations**:
- torchrun: 16GB per GPU minimum
- DeepSpeed: 32GB per GPU (ZeRO-3)
- TorchTitan: 64GB+ per GPU (large models)
- vLLM: 32GB per GPU (depends on model size)

### 3. Multi-Node Setup

**Kubernetes**:
- Automatic headless service creation
- Pod discovery via DNS
- Uses `JOB_COMPLETION_INDEX` for rank

**SLURM**:
- Uses SLURM environment variables
- Automatic node discovery
- Network interface configuration

---

## Environment Variables

### Common Variables (All Launchers)

```bash
NNODES=4                    # Number of nodes
NPROC_PER_NODE=8           # GPUs per node
NODE_RANK=0                 # Current node rank (0-based)
MASTER_ADDR=master.local    # Master node address
MASTER_PORT=29500           # Master communication port
```

### Launcher-Specific

**torchrun**:
```bash
MAD_MULTI_NODE_RUNNER="torchrun --nnodes=4 --nproc_per_node=8 ..."
```

**DeepSpeed**:
```bash
MAD_MULTI_NODE_RUNNER="deepspeed --num_gpus=8 --hostfile=/tmp/hostfile ..."
```

**Megatron-LM**:
```bash
# Megatron-Core standard environment variables
TENSOR_MODEL_PARALLEL_SIZE=8         # Tensor parallelism size
PIPELINE_MODEL_PARALLEL_SIZE=4       # Pipeline parallelism size
CONTEXT_PARALLEL_SIZE=1              # Context parallelism size
MAD_MULTI_NODE_RUNNER="torchrun ..."  # Uses torchrun (SLURM only)
```

**TorchTitan**:
```bash
TORCHTITAN_TENSOR_PARALLEL_SIZE=8
TORCHTITAN_PIPELINE_PARALLEL_SIZE=4
TORCHTITAN_FSDP_ENABLED=1
MAD_MULTI_NODE_RUNNER="torchrun ..."
```

**vLLM**:
```bash
VLLM_TENSOR_PARALLEL_SIZE=4
VLLM_DISTRIBUTED_BACKEND="ray"
# No MAD_MULTI_NODE_RUNNER (vLLM manages processes)
```

**SGLang**:
```bash
SGLANG_TENSOR_PARALLEL_SIZE=4
NCCL_INIT_ADDR="master:29500"
# No MAD_MULTI_NODE_RUNNER (SGLang manages processes)
```

---

## Troubleshooting

### Common Issues

**1. Launcher Not Found**
```bash
Error: Unknown launcher type 'xyz'
```
Solution: Use one of: `torchrun`, `deepspeed`, `megatron`, `torchtitan`, `vllm`, `sglang`

**2. Multi-Node Communication Fails**
```bash
Error: Connection timeout to master node
```
Solutions:
- Check network connectivity between nodes
- Verify `MASTER_ADDR` is correct
- Ensure firewall allows `MASTER_PORT`
- For K8s: Check headless service created

**3. GPU Visibility Issues**
```bash
Error: Expected 8 GPUs but found 0
```
Solutions:
- Verify `gpu_count` matches `nproc_per_node`
- Check GPU resource name (`amd.com/gpu` vs `nvidia.com/gpu`)
- Ensure ROCm/CUDA drivers installed

**4. Ray Cluster Issues (vLLM/SGLang)**
```bash
Error: Ray cluster failed to start
```
Solutions:
- Clean existing Ray processes: `ray stop --force`
- Check port 6379 is available
- Verify network interface configuration
- For multi-node: ensure pods can communicate

---

## Advanced Topics

### Custom Launcher Scripts

madengine provides `$MAD_MULTI_NODE_RUNNER` for frameworks that use torchrun:

```bash
#!/bin/bash
# Your model script

# For torchrun/deepspeed/megatron/torchtitan
$MAD_MULTI_NODE_RUNNER your_training_script.py --args

# For vLLM/sglang (no MAD_MULTI_NODE_RUNNER)
python your_inference_script.py --args
```

### Launcher Detection

madengine automatically:
1. Detects launcher from `distributed.launcher` field
2. Sets up appropriate environment variables
3. Generates launcher-specific commands
4. Creates multi-node infrastructure (K8s services, SLURM env)

### Performance Optimization

**AMD MI300X**:
```json
{
  "context": {
    "env_vars": {
      "PYTORCH_TUNABLEOP_ENABLED": "1",
      "NCCL_IB_DISABLE": "0",
      "NCCL_NET_GDR_LEVEL": "5"
    }
  }
}
```

**NVIDIA H100/A100**:
```json
{
  "context": {
    "env_vars": {
      "NCCL_ALGO": "Ring",
      "NCCL_PROTO": "Simple",
      "CUDA_DEVICE_MAX_CONNECTIONS": "1"
    }
  }
}
```

---

## References

### Official Documentation
- [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [DeepSpeed](https://www.deepspeed.ai/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [TorchTitan](https://github.com/pytorch/torchtitan)
- [vLLM](https://docs.vllm.ai/)
- [SGLang](https://github.com/sgl-project/sglang)

### madengine Documentation
- [K8s Configuration Guide](../examples/k8s-configs/README.md)
- [SLURM Configuration Guide](../examples/slurm-configs/README.md)
- [How to Run Multi-Node](how-to-run-multi-node.md)

### Example Configurations
- [K8s Examples](../examples/k8s-configs/)
- [SLURM Examples](../examples/slurm-configs/)
- [Test Fixtures](../tests/fixtures/dummy/)

