# vLLM Distributed Inference for madengine

This directory contains vLLM inference benchmarking scripts for AMD ROCm GPUs.

## ⚠️ IMPORTANT: ROCm Build Instructions

**The current Dockerfile uses a mock vLLM module for testing infrastructure.**

For **production deployments**, you must build vLLM from source with ROCm support:

1. Uncomment the vLLM build section in `docker/dummy_vllm.ubuntu.amd.Dockerfile`
2. Or install manually: `pip install git+https://github.com/vllm-project/vllm.git`

Note: vLLM's PyPI package (`pip install vllm`) is CUDA-only and will fail with ROCm.

## Overview

vLLM is a high-throughput and memory-efficient inference and serving engine for Large Language Models (LLMs). It features:

- **PagedAttention**: Efficient KV cache management inspired by OS virtual memory paging
- **Continuous Batching**: Dynamic request batching for maximum throughput
- **Tensor Parallelism (TP)**: Split model weights across GPUs within a node
- **Pipeline Parallelism (PP)**: Split model layers across multiple nodes
- **ROCm Support**: Optimized for AMD Instinct GPUs (MI200/MI300 series)

## Files

- `run.sh`: Wrapper script that launches vLLM inference with proper environment setup
- `run_vllm_inference.py`: Main Python script that runs the vLLM benchmark
- `README.md`: This file

## Architecture

### Single-Node Multi-GPU (Tensor Parallelism)
```
Node 1: [GPU0] [GPU1] [GPU2] [GPU3]
        └──────── Model Split ────────┘
```
- Model weights split across all GPUs
- Each GPU holds a portion of the model
- Forward pass requires communication between GPUs

### Multi-Node Multi-GPU (Tensor + Pipeline Parallelism)
```
Node 1: [GPU0] [GPU1] [GPU2] [GPU3] <- Layers 1-N/2
Node 2: [GPU0] [GPU1] [GPU2] [GPU3] <- Layers N/2+1-N
```
- Pipeline parallelism splits layers across nodes
- Tensor parallelism splits weights within each node
- Optimized for very large models

## Configuration

### Environment Variables

**vLLM Core Settings:**
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`: Allow longer sequence lengths
- `VLLM_USE_MODELSCOPE=False`: Disable ModelScope
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`: Use spawn for multiprocessing
- `VLLM_LOGGING_LEVEL=INFO`: Set logging level

**ROCm 7.x Optimizations:**
- `HSA_FORCE_FINE_GRAIN_PCIE=1`: Enable fine-grained PCIe access
- `HSA_ENABLE_SDMA=0`: Disable SDMA for stability
- `GPU_MAX_HW_QUEUES=2`: Optimize hardware queue configuration
- `NCCL_DEBUG=WARN`: NCCL debugging level
- `PYTORCH_ROCM_ARCH=gfx90a;gfx940;gfx941;gfx942`: Target AMD GPU architectures

### Command Line Arguments

The `run_vllm_inference.py` script accepts:

- `--model`: Model name or path (default: `facebook/opt-125m`)
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism
- `--pipeline-parallel-size`: Number of nodes for pipeline parallelism
- `--enforce-eager`: Disable CUDA graph for compatibility

## Usage

### Local Testing (Single GPU)
```bash
cd /path/to/scripts/dummy_vllm
python3 run_vllm_inference.py --model facebook/opt-125m
```

### Single-Node Multi-GPU (via madengine)
```bash
madengine run \
  --model-name dummy_vllm \
  --additional-config examples/slurm-configs/minimal/vllm-single-node-minimal.json
```

### Multi-Node Multi-GPU (via madengine)
```bash
madengine run \
  --model-name dummy_vllm \
  --additional-config examples/slurm-configs/minimal/vllm-multi-node-minimal.json
```

## Slurm Configuration Examples

### Single-Node (4 GPUs with Tensor Parallelism)
```json
{
  "slurm": {
    "partition": "amd-rccl",
    "nodes": 1,
    "gpus_per_node": 4,
    "time": "02:00:00"
  },
  "distributed": {
    "launcher": "vllm",
    "nnodes": 1,
    "nproc_per_node": 4
  }
}
```

### Multi-Node (2 Nodes × 4 GPUs with TP + PP)
```json
{
  "slurm": {
    "partition": "amd-rccl",
    "nodes": 2,
    "gpus_per_node": 4,
    "time": "04:00:00"
  },
  "distributed": {
    "launcher": "vllm",
    "nnodes": 2,
    "nproc_per_node": 4
  }
}
```

## Model Selection

### Small Models (Testing)
- `facebook/opt-125m` (125M parameters, ~250MB)
- `facebook/opt-350m` (350M parameters, ~700MB)

### Medium Models (Production)
- `facebook/opt-6.7b` (6.7B parameters, ~13GB)
- `meta-llama/Llama-2-7b-hf` (7B parameters, ~14GB)
- `mistralai/Mistral-7B-v0.1` (7B parameters, ~14GB)

### Large Models (Multi-GPU Required)
- `meta-llama/Llama-2-13b-hf` (13B parameters, ~26GB)
- `meta-llama/Llama-2-70b-hf` (70B parameters, ~140GB)

**Note**: Ensure you have access to gated models (e.g., Llama-2) via Hugging Face authentication.

## Performance Metrics

The script outputs the following metrics:
- **Throughput**: Requests per second
- **Token Generation Rate**: Tokens per second
- **Average Latency**: Milliseconds per request
- **Total Prompts**: Number of prompts processed
- **Total Time**: End-to-end execution time

## Troubleshooting

### Out of Memory (OOM) Errors
- GPU memory utilization is set to 0.70 (70%) by default for stability
- If you still encounter OOM errors:
  - Use a smaller model or reduce `max_model_len` in the script
  - Increase tensor parallelism size to split the model across more GPUs
  - Check for other processes using GPU memory before running

### Slow Performance
- Enable CUDA graphs (remove `--enforce-eager`)
- Verify NCCL settings for multi-GPU
- Check GPU memory utilization

### Model Download Issues
- Set `HF_HOME` for Hugging Face cache directory
- Use `huggingface-cli login` for gated models
- Pre-download models to shared storage

## References

- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM Documentation](https://docs.vllm.ai/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [madengine Documentation](../../../../../../README.md)

## Support

For issues or questions:
- vLLM: [GitHub Issues](https://github.com/vllm-project/vllm/issues)
- madengine: Contact mad.support@amd.com

