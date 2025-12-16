#!/usr/bin/env python3
"""
vLLM V1 Engine Distributed Inference Benchmark

vLLM V1 Engine Architecture:
- Tensor Parallelism (TP): Split model across GPUs within a node
- Data Parallelism (DP): Run multiple replicas for higher throughput
- Pipeline Parallelism (PP): Split model layers across nodes (experimental)

Launch modes:
  Single-node/single-GPU: TP=1, DP=1
  Single-node/multi-GPU (TP): TP=N, DP=1 (model split across GPUs)
  Single-node/multi-GPU (DP): TP=1, DP=N (multiple replicas)
  Multi-node: Use Ray backend with proper configuration
"""

import os
import sys
import time
import argparse
import socket
from typing import List, Optional

# Configure environment before importing vLLM
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
os.environ.setdefault("VLLM_USE_MODELSCOPE", "False")

# V1 Engine specific settings
os.environ.setdefault("VLLM_USE_V1", "1")  # Explicitly use V1 engine
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

try:
    from vllm import LLM, SamplingParams
    import torch
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please ensure vLLM and PyTorch are installed")
    sys.exit(1)

# Configuration
DEFAULT_MODEL = "facebook/opt-125m"  # Small model for testing
NUM_PROMPTS = 100
MAX_TOKENS = 128
TEMPERATURE = 0.8
TOP_P = 0.95

# Sample prompts for inference
SAMPLE_PROMPTS = [
    "The future of artificial intelligence is",
    "Machine learning has revolutionized",
    "Deep learning models are capable of",
    "Natural language processing enables",
    "Computer vision systems can",
]


def print_header(args):
    """Print benchmark header with configuration."""
    print("=" * 70)
    print("vLLM V1 Engine Distributed Inference Benchmark")
    print("=" * 70)
    print(f"Hostname: {socket.gethostname()}")
    print(f"Model: {args.model}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"Pipeline Parallel Size: {args.pipeline_parallel_size}")
    
    # Calculate total parallelism
    total_gpus = args.tensor_parallel_size * args.pipeline_parallel_size
    print(f"Total GPUs (TP × PP): {total_gpus}")
    
    # Data parallelism is automatic in V1 if more GPUs are available
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if available_gpus > total_gpus:
        data_parallel_size = available_gpus // total_gpus
        print(f"Data Parallel Size (auto): {data_parallel_size}")
    
    print(f"Number of prompts: {NUM_PROMPTS}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Distributed backend: {args.distributed_backend}")
    print("=" * 70)


def generate_prompts(num_prompts: int) -> List[str]:
    """Generate list of prompts for inference."""
    prompts = []
    for i in range(num_prompts):
        # Cycle through sample prompts
        base_prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
        prompts.append(f"{base_prompt} (request {i+1})")
    return prompts


def run_inference(args):
    """Run vLLM V1 inference benchmark."""
    print("\n" + "=" * 70)
    print("Initializing vLLM V1 Engine")
    print("=" * 70)
    
    # Determine distributed backend
    # For single-node: use 'mp' (multiprocessing) or None
    # For multi-node: use 'ray'
    if args.distributed_backend == "auto":
        nnodes = int(os.environ.get("NNODES", "1"))
        distributed_backend = "ray" if nnodes > 1 else None
    else:
        distributed_backend = args.distributed_backend if args.distributed_backend != "none" else None
    
    print(f"Using distributed backend: {distributed_backend or 'default'}")
    
    # Initialize vLLM LLM engine with V1-specific settings
    try:
        llm_kwargs = {
            "model": args.model,
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "trust_remote_code": True,
            "dtype": "auto",
            "gpu_memory_utilization": 0.70,  # Reduced to 70% to avoid OOM errors
            "max_model_len": 2048,
            "disable_log_stats": True,  # Reduce logging noise
        }
        
        # Add distributed backend if specified
        if distributed_backend:
            llm_kwargs["distributed_executor_backend"] = distributed_backend
        
        # V1 engine specific: enforce_eager mode for compatibility
        if args.enforce_eager:
            llm_kwargs["enforce_eager"] = True
        
        llm = LLM(**llm_kwargs)
        print("✓ vLLM V1 engine initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize vLLM engine: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )
    
    print(f"\n{'=' * 70}")
    print("Running Inference")
    print("=" * 70)
    
    # Generate prompts
    prompts = generate_prompts(NUM_PROMPTS)
    
    # Warmup run (not timed)
    print("\nWarmup: Running 10 prompts...")
    warmup_prompts = prompts[:10]
    _ = llm.generate(warmup_prompts, sampling_params)
    print("✓ Warmup complete")
    
    # Benchmark run (timed)
    print(f"\nBenchmark: Running {NUM_PROMPTS} prompts...")
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate metrics
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    throughput = NUM_PROMPTS / elapsed_time
    tokens_per_second = total_tokens / elapsed_time
    
    # Print results
    print(f"\n{'=' * 70}")
    print("Benchmark Results")
    print("=" * 70)
    print(f"Total prompts: {NUM_PROMPTS}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Throughput: {throughput:.2f} requests/second")
    print(f"Token generation: {tokens_per_second:.2f} tokens/second")
    print(f"Average latency: {(elapsed_time / NUM_PROMPTS) * 1000:.2f} ms/request")
    print("=" * 70)
    
    # Print sample outputs
    print("\n" + "=" * 70)
    print("Sample Outputs (first 3)")
    print("=" * 70)
    for i, output in enumerate(outputs[:3]):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\n[Prompt {i+1}]: {prompt}")
        print(f"[Output {i+1}]: {generated_text[:200]}...")  # First 200 chars
    
    # MADEngine output format
    print(f"\nperformance: {throughput:.2f} requests_per_second")
    print(f"tokens_per_second: {tokens_per_second:.2f}")
    print(f"model: {args.model}")
    print(f"tensor_parallel_size: {args.tensor_parallel_size}")
    print(f"pipeline_parallel_size: {args.pipeline_parallel_size}")
    
    # Determine what backend was actually used
    if args.distributed_backend == "auto":
        nnodes = int(os.environ.get("NNODES", "1"))
        actual_backend = "ray" if nnodes > 1 else "default"
    else:
        actual_backend = args.distributed_backend if args.distributed_backend != "none" else "default"
    print(f"distributed_backend: {actual_backend}")
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="vLLM V1 Engine Distributed Inference Benchmark"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name or path (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=1,
        help="Number of nodes for pipeline parallelism (default: 1)"
    )
    parser.add_argument(
        "--distributed-backend",
        type=str,
        choices=["auto", "ray", "mp", "none"],
        default="auto",
        help="Distributed backend: auto (default), ray (multi-node), mp (multiprocessing), none"
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph for compatibility"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.tensor_parallel_size < 1:
        print("Error: tensor-parallel-size must be >= 1")
        return 1
    
    if args.pipeline_parallel_size < 1:
        print("Error: pipeline-parallel-size must be >= 1")
        return 1
    
    # Print configuration
    print_header(args)
    
    # Run inference benchmark
    return run_inference(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

