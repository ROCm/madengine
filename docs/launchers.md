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
| **SGLang Disaggregated** | Inference | Large-scale disaggregated inference | ✅ | ✅ | ✅ (min 3) |

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
madengine build --tags my_model \
  --additional-context-file config.json

# Deploy to K8s or SLURM
madengine run --manifest-file build_manifest.json
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
- SLURM: `examples/slurm-configs/basic/07-sglang-single-node.json`

---

### 7. SGLang Disaggregated (NEW!)

**Purpose**: Large-scale disaggregated LLM inference with specialized prefill/decode clusters

**Reference**: [sgl-project/sglang](https://github.com/sgl-project/sglang) | [Mooncake Framework](https://github.com/kvcache-ai/Mooncake)

**When to Use**:
- ✅ Large-scale LLM inference (multi-node clusters)
- ✅ Optimized resource allocation (separate prefill/decode)
- ✅ High-throughput production deployments
- ✅ Workload-specific optimization (tune prefill/decode ratio)

**Architecture**:

SGLang Disaggregated separates inference into specialized node pools:

```
┌─────────────────────────────────────────────────┐
│         SGLang Disaggregated Cluster            │
├─────────────────────────────────────────────────┤
│  Node 0:        Proxy (Load Balancer)           │
│  Nodes 1-P:     Prefill Servers (~40%)          │
│  Nodes P+1-N:   Decode Servers (~60%)           │
│                                                  │
│  Communication: Mooncake (KV cache transfer)    │
└─────────────────────────────────────────────────┘
```

**Configuration**:

```json
{
  "distributed": {
    "launcher": "sglang-disagg",
    "nnodes": 5,
    "nproc_per_node": 8,
    "sglang_disagg": {
      "prefill_nodes": 2,
      "decode_nodes": 2
    }
  }
}
```

**Minimum Requirements**:
- **Nodes**: Minimum 3 nodes (1 proxy + 1 prefill + 1 decode)
- **GPUs**: Minimum 1 GPU per node (for tensor parallelism)
- **Network**: High-speed interconnect (InfiniBand recommended for production)

**Node Roles**:
1. **Proxy Node (Rank 0)**: Load balancer, request router (mini_lb)
2. **Prefill Nodes**: Process input prompts, generate KV cache
3. **Decode Nodes**: Receive KV cache, generate output tokens

**Automatic Split (Default)**:
- Uses 40/60 golden ratio for prefill/decode
- Formula: `prefill = max(1, (nnodes - 1) * 2 // 5)`

| Total Nodes | Proxy | Prefill | Decode |
|-------------|-------|---------|--------|
| 3 | 1 | 1 (33%) | 1 (33%) |
| 5 | 1 | 2 (40%) | 2 (40%) |
| 7 | 1 | 2 (29%) | 4 (57%) |
| 11 | 1 | 4 (40%) | 6 (60%) |

**Custom Split (NEW Feature!)**:

Override automatic split based on workload characteristics:

```json
{
  "distributed": {
    "launcher": "sglang-disagg",
    "nnodes": 7,
    "nproc_per_node": 8,
    "sglang_disagg": {
      "prefill_nodes": 4,
      "decode_nodes": 2
    }
  }
}
```

**Custom Split Use Cases**:

| Workload Type | Recommended Split | Example (7 nodes) |
|---------------|------------------|-------------------|
| Long prompts (code gen) | 60% prefill | `prefill: 4, decode: 2` |
| Long outputs (creative) | 30% prefill | `prefill: 2, decode: 4` |
| Balanced (default) | 40% prefill | Omit sglang_disagg |
| Document processing | 50% prefill | `prefill: 3, decode: 3` |

**Validation Rules**:
- `prefill_nodes >= 1`
- `decode_nodes >= 1`
- `prefill_nodes + decode_nodes + 1 == nnodes`

**Features**:
- Disaggregated prefill/decode architecture
- Mooncake framework for KV cache transfer
- Automatic or custom node role assignment
- RadixAttention for KV cache efficiency
- Ray cluster coordination
- No torchrun needed (manages own processes)

**Environment Variables (K8s)**:
```bash
POD_INDEX=${JOB_COMPLETION_INDEX}  # Pod index for role assignment
TOTAL_PODS=5                        # Total number of pods
PREFILL_COUNT=2                     # Number of prefill nodes
DECODE_COUNT=2                      # Number of decode nodes
TP_SIZE=8                           # Tensor parallel size
```

**Environment Variables (SLURM)**:
```bash
SGLANG_DISAGG_MODE="enabled"
SGLANG_DISAGG_PREFILL_NODES=2
SGLANG_DISAGG_DECODE_NODES=2
SGLANG_DISAGG_TOTAL_NODES=5
SGLANG_TP_SIZE=8
SGLANG_NODE_RANK=${SLURM_PROCID}
SGLANG_NODE_IPS="10.0.0.1,10.0.0.2,..."
```

**Examples**:
- K8s Minimal: `examples/k8s-configs/minimal/sglang-disagg-minimal.json`
- K8s Basic: `examples/k8s-configs/basic/sglang-disagg-multi-node-basic.json`
- K8s Custom: `examples/k8s-configs/basic/sglang-disagg-custom-split.json`
- SLURM Minimal: `examples/slurm-configs/minimal/sglang-disagg-minimal.json`
- SLURM Basic: `examples/slurm-configs/basic/sglang-disagg-multi-node.json`
- SLURM Custom: `examples/slurm-configs/basic/sglang-disagg-custom-split.json`

**Comparison: SGLang vs SGLang Disaggregated**:

| Feature | SGLang | SGLang Disaggregated |
|---------|--------|---------------------|
| **Architecture** | Unified | Separated prefill/decode |
| **Min Nodes** | 1 | 3 |
| **Node Types** | Same for all | Specialized (proxy/prefill/decode) |
| **KV Transfer** | In-memory | Mooncake framework |
| **Load Balancer** | Ray | mini_lb (dedicated) |
| **Best For** | General inference | Large-scale clusters |
| **Optimization** | General | Workload-specific tuning |

**Production Considerations**:
1. **Install Mooncake**: Full framework with RDMA support
2. **Configure Network**: InfiniBand/RoCE for high-speed KV transfer
3. **Setup etcd**: For distributed coordination
4. **Monitor Metrics**: Track prefill latency, decode throughput, queue depths
5. **Tune Split**: Adjust prefill/decode ratio based on workload

**Performance Tuning**:
```bash
# Start with automatic split
madengine run --tags model --config minimal-config.json

# Monitor bottleneck (prefill latency vs decode throughput)
# If prefill is bottleneck → increase prefill nodes
# If decode is bottleneck → increase decode nodes

# Apply custom split
madengine run --tags model --config custom-split-config.json
```

**Troubleshooting**:

1. **"requires minimum 3 nodes"**
   - Solution: Set `nnodes >= 3`

2. **"prefill_nodes + decode_nodes + 1 must equal nnodes"**
   - Solution: Verify math in custom split configuration

3. **Pod/Node stuck in Init**
   - K8s: Check headless service creation
   - SLURM: Verify node IP discovery

4. **High KV cache transfer latency**
   - Enable RDMA/InfiniBand
   - Configure Mooncake transfer backend
   - Check network connectivity

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

| Feature | vLLM | SGLang | SGLang Disaggregated |
|---------|------|--------|----------------------|
| **Throughput** | Very High | High | Very High |
| **Memory Efficiency** | PagedAttention | RadixAttention | RadixAttention + Mooncake |
| **Batching** | Continuous | Continuous | Continuous |
| **API** | OpenAI-compatible | OpenAI-compatible | OpenAI-compatible |
| **Structured Gen** | Limited | ✅ Native | ✅ Native |
| **Multi-Node** | ✅ Ray | ✅ Ray | ✅ Ray + mini_lb |
| **Architecture** | Unified | Unified | Disaggregated |
| **Min Nodes** | 1 | 1 | 3 |
| **Specialization** | ❌ | ❌ | ✅ Prefill/Decode |
| **Custom Split** | ❌ | ❌ | ✅ |
| **K8s Support** | ✅ | ✅ | ✅ |
| **SLURM Support** | ✅ | ✅ | ✅ |

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
High throughput            → vLLM or SGLang Disaggregated
Structured generation      → SGLang or SGLang Disaggregated
Memory constrained         → vLLM (PagedAttention)
Large-scale clusters (5+)  → SGLang Disaggregated
Workload-specific tuning   → SGLang Disaggregated
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

**SGLang Disaggregated**:
```bash
SGLANG_DISAGG_MODE="enabled"
SGLANG_DISAGG_PREFILL_NODES=2
SGLANG_DISAGG_DECODE_NODES=2
SGLANG_DISAGG_TOTAL_NODES=5
SGLANG_TP_SIZE=8
SGLANG_NODE_RANK=${SLURM_PROCID}
# No MAD_MULTI_NODE_RUNNER (SGLang disagg manages processes)
```

---

## Troubleshooting

### Common Issues

**1. Launcher Not Found**
```bash
Error: Unknown launcher type 'xyz'
```
Solution: Use one of: `torchrun`, `deepspeed`, `megatron`, `torchtitan`, `vllm`, `sglang`, `sglang-disagg`

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

