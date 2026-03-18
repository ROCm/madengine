# SGLang Distributed Inference - madengine Integration

This directory contains scripts for running SGLang distributed inference on SLURM clusters through madengine.

## Overview

**SGLang** is a fast serving framework for large language models and vision-language models, featuring:
- **RadixAttention**: Efficient KV cache with automatic prefix caching
- **Native Distributed Launcher**: Uses `python3 -m sglang.launch_server` (NO torchrun needed!)
- **Tensor Parallelism (TP)**: Split model across GPUs within a node
- **Ray-based coordination**: Automatic distributed inference across nodes
- **High throughput**: Optimized for both single and multi-node deployments

## Key Difference from vLLM

**SGLang does NOT use torchrun!** It has its own native launcher:
- **SGLang**: `python3 -m sglang.launch_server` (Ray-based)
- **vLLM**: Can use `torchrun` or direct Python launch

## Files

- `run.sh` - Wrapper script that uses SGLang's native launcher
- `run_sglang_inference.py` - Python benchmark using SGLang Runtime API
- `README.md` - This documentation file

## Architecture

### Single-Node Multi-GPU (Tensor Parallelism)

```
┌─────────────────────────────────────────┐
│  Node 1 (4 GPUs with TP)                │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │
│  │ GPU0 │─│ GPU1 │─│ GPU2 │─│ GPU3 │  │
│  │Shard │ │Shard │ │Shard │ │Shard │  │
│  │  1/4 │ │  2/4 │ │  3/4 │ │  4/4 │  │
│  └──────┘ └──────┘ └──────┘ └──────┘  │
└─────────────────────────────────────────┘
```

**Command**: `python3 -m sglang.launch_server --model-path MODEL --tp 4`

### Multi-Node Multi-GPU (TP + Load Balancing)

```
┌─────────────────────────────────────────┐
│  Node 1 (TP Group 1)                    │
│  ┌──────────────────────────────────┐  │
│  │  GPUs 0-3 (Full Model Copy)     │  │
│  └──────────────────────────────────┘  │
└──────────────┬──────────────────────────┘
               │ Ray Coordination
┌──────────────┴──────────────────────────┐
│  Node 2 (TP Group 2)                    │
│  ┌──────────────────────────────────┐  │
│  │  GPUs 0-3 (Full Model Copy)     │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

**Commands**:
```bash
# Node 1 (rank 0)
python3 -m sglang.launch_server --model-path MODEL --tp 4 \
    --nnodes 2 --node-rank 0 --nccl-init-addr MASTER_IP:PORT

# Node 2 (rank 1)
python3 -m sglang.launch_server --model-path MODEL --tp 4 \
    --nnodes 2 --node-rank 1 --nccl-init-addr MASTER_IP:PORT
```

## Usage

### Quick Start with madengine

#### Single-Node Inference (4 GPUs)

```bash
madengine run \
  --model-name dummy_sglang \
  --additional-context-file examples/slurm-configs/minimal/sglang-single-node-minimal.json
```

#### Multi-Node Inference (2 nodes × 4 GPUs)

```bash
madengine run \
  --model-name dummy_sglang \
  --additional-context-file examples/slurm-configs/minimal/sglang-multi-node-minimal.json
```

### Execution Modes

The script supports two execution modes:

#### 1. Server Mode (OpenAI-compatible API)

Launches SGLang as a server that exposes an OpenAI-compatible API:

```bash
export SGLANG_EXECUTION_MODE=server
./run.sh
```

The server will be accessible at `http://localhost:30000` and supports:
- `/v1/completions` - Text completion endpoint
- `/v1/chat/completions` - Chat completion endpoint
- `/v1/models` - List available models

#### 2. Offline Mode (Batch Inference - Default)

Runs batch inference directly for benchmarking:

```bash
export SGLANG_EXECUTION_MODE=offline  # or leave unset
./run.sh
```

This mode is better for:
- Performance benchmarking
- Batch processing
- Integration testing

### Manual Execution

If you want to run the scripts directly without madengine:

#### Single-Node (4 GPUs with TP)

```bash
export NNODES=1
export NPROC_PER_NODE=4
export MASTER_ADDR=localhost
export MASTER_PORT=29500
./run.sh
```

#### Multi-Node (2 nodes × 4 GPUs with TP)

On master node (rank 0):
```bash
export NNODES=2
export NPROC_PER_NODE=4
export NODE_RANK=0
export MASTER_ADDR=master-node-hostname
export MASTER_PORT=29500
./run.sh
```

On worker node (rank 1):
```bash
export NNODES=2
export NPROC_PER_NODE=4
export NODE_RANK=1
export MASTER_ADDR=master-node-hostname
export MASTER_PORT=29500
./run.sh
```

## Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `NNODES` | Number of nodes | `2` |
| `NPROC_PER_NODE` | GPUs per node | `4` |
| `NODE_RANK` | Current node rank (0-indexed) | `0` |
| `MASTER_ADDR` | Master node address | `node001` |
| `MASTER_PORT` | Communication port | `29500` |
| `SGLANG_EXECUTION_MODE` | `server` or `offline` | `offline` |

**Note**: Unlike vLLM, SGLang does NOT use `MAD_MULTI_NODE_RUNNER` (torchrun). It has its own launcher!

### SGLang-Specific Settings

Environment variables in your Slurm config:

```json
{
  "env_vars": {
    "SGLANG_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "SGLANG_ENABLE_FLASHINFER": "1",
    "SGLANG_ENABLE_RADIX_CACHE": "1",
    "SGLANG_RADIX_CACHE_SIZE": "0.9",
    "SGLANG_EXECUTION_MODE": "offline"
  }
}
```

### Custom Models

To use a different model, modify `run.sh`:

For server mode:
```bash
python3 -m sglang.launch_server \
    --model-path "meta-llama/Llama-2-7b-hf" \
    --tp $TP_SIZE \
    --nnodes $NNODES \
    --node-rank $NODE_RANK \
    --nccl-init-addr "${MASTER_ADDR}:${MASTER_PORT}"
```

For offline mode:
```bash
python3 run_sglang_inference.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --tp-size $TP_SIZE \
    --nnodes $NNODES
```

## SGLang Native Launcher Examples

### Server Mode

```bash
# Single-node server (4 GPUs)
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-hf \
    --tp 4 \
    --host 0.0.0.0 \
    --port 30000

# Multi-node server (2 nodes, 4 GPUs each)
# Node 0:
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-hf \
    --tp 4 \
    --nnodes 2 \
    --node-rank 0 \
    --nccl-init-addr 192.168.1.100:29500

# Node 1:
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-hf \
    --tp 4 \
    --nnodes 2 \
    --node-rank 1 \
    --nccl-init-addr 192.168.1.100:29500
```

### Offline Mode (Python API)

```python
import sglang as sgl

# Single-node
runtime = sgl.Runtime(
    model_path="meta-llama/Llama-2-7b-hf",
    tp_size=4,
)

# Multi-node
runtime = sgl.Runtime(
    model_path="meta-llama/Llama-2-7b-hf",
    tp_size=4,
    nnodes=2,
    node_rank=0,  # Set appropriately per node
    nccl_init_addr="192.168.1.100:29500",
)

# Generate
outputs = runtime.generate(
    ["The future of AI is"],
    sampling_params={"max_new_tokens": 128}
)
```

## Performance Tuning

### ROCm Optimizations

For AMD GPUs (included in Dockerfile):

```bash
# HSA optimizations
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HSA_ENABLE_SDMA=0
export GPU_MAX_HW_QUEUES=2

# NCCL optimizations
export NCCL_DEBUG=WARN
export NCCL_MIN_NCHANNELS=16

# Network interface
export NCCL_SOCKET_IFNAME=eth0  # or ib0 for InfiniBand
```

### Memory Management

Adjust GPU memory utilization:

```python
runtime = sgl.Runtime(
    model_path=args.model,
    tp_size=args.tp_size,
    mem_fraction_static=0.90,  # Use 90% of GPU memory
)
```

### Batch Size

For higher throughput, increase concurrent requests:

```python
NUM_PROMPTS = 200  # Increase from default 100
```

## Comparison: SGLang vs vLLM Launchers

| Feature | vLLM | SGLang |
|---------|------|--------|
| **Launcher** | `torchrun` or `vllm serve` | `python3 -m sglang.launch_server` |
| **Coordination** | Ray (optional) | Ray (built-in, required) |
| **Multi-node Setup** | torchrun handles ranks | SGLang launcher handles ranks |
| **Attention** | PagedAttention | RadixAttention (prefix caching) |
| **Prefix Caching** | Manual | Automatic |
| **Best For** | General inference | Complex workflows with shared prefixes |

**Key Insight**: SGLang does NOT need torchrun because it has its own native distributed launcher!

## Troubleshooting

### Issue: "No module named 'sglang'"

**Solution**: Ensure you're using the official SGLang Docker image:
```dockerfile
FROM lmsysorg/sglang:latest
```

Or install SGLang:
```bash
pip install "sglang[all]"
```

### Issue: Multi-node initialization hangs

**Solutions**:
1. Verify `MASTER_ADDR` is accessible from all nodes
2. Check firewall rules for Ray ports (6379, 8265, 10001-10100)
3. Ensure `NCCL_SOCKET_IFNAME` is set correctly
4. Verify NCCL init address is reachable: `telnet $MASTER_ADDR $MASTER_PORT`

### Issue: Out of memory errors

**Solutions**:
1. Reduce `mem_fraction_static` (e.g., from 0.90 to 0.80)
2. Use more GPUs (increase TP size)
3. Use a smaller model
4. Enable FlashInfer if not already: `SGLANG_ENABLE_FLASHINFER=1`

### Issue: Ray initialization failures

**Solutions**:
1. Check Ray is installed: `python3 -c "import ray; print(ray.__version__)"`
2. Clear Ray temp files: `rm -rf /tmp/ray/*`
3. Verify network connectivity between nodes
4. Check Ray logs: `cat /tmp/ray/session_*/logs/*`

## Output Format

The benchmark script outputs performance metrics in madengine format:

```
performance: 45.23 requests_per_second
tokens_per_second: 5789.12
model: facebook/opt-125m
tp_size: 4
nnodes: 2
```

madengine automatically parses these metrics and stores them in `perf.csv`.

## References

- **SGLang GitHub**: https://github.com/sgl-project/sglang
- **SGLang Documentation**: https://docs.sglang.ai/
- **SGLang Native Launcher**: https://github.com/sgl-project/sglang#distributed-serving
- **madengine Documentation**: See `examples/slurm-configs/README.md`
- **ROCm Documentation**: https://rocm.docs.amd.com/

## Support

For issues specific to:
- **madengine integration**: Contact mad.support@amd.com
- **SGLang itself**: Open issue at https://github.com/sgl-project/sglang/issues
- **ROCm compatibility**: Check ROCm documentation or AMD support
