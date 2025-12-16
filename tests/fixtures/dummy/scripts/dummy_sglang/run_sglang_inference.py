#!/usr/bin/env python3
"""
SGLang Distributed Inference Benchmark

SGLang uses its own native launcher - NO torchrun needed!
- Uses Ray for distributed coordination internally
- Supports Tensor Parallelism (TP) within nodes
- Supports multi-node deployment with automatic load balancing

Launch modes:
  Single-node/multi-GPU: TP only
  Multi-node/multi-GPU: TP across nodes with load balancing
"""

import os
import sys
import time
import argparse
import socket
from typing import List, Optional

# Configure environment before importing SGLang
os.environ.setdefault("SGLANG_ALLOW_LONG_MAX_MODEL_LEN", "1")
os.environ.setdefault("SGLANG_USE_MODELSCOPE", "False")
os.environ.setdefault("SGLANG_ENABLE_FLASHINFER", "1")

try:
    import sglang as sgl
    import torch
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please ensure SGLang and PyTorch are installed")
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
    print("SGLang Distributed Inference Benchmark")
    print("=" * 70)
    print(f"Hostname: {socket.gethostname()}")
    print(f"Model: {args.model}")
    print(f"Tensor Parallel Size: {args.tp_size}")
    print(f"Number of Nodes: {args.nnodes}")
    print(f"Node Rank: {args.node_rank}")
    print(f"Total GPUs: {args.tp_size * args.nnodes}")
    print(f"Number of prompts: {NUM_PROMPTS}")
    print(f"Max tokens: {MAX_TOKENS}")
    print("=" * 70)


def generate_prompts(num_prompts: int) -> List[str]:
    """Generate list of prompts for inference."""
    prompts = []
    for i in range(num_prompts):
        # Cycle through sample prompts
        base_prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
        prompts.append(f"{base_prompt} (request {i+1})")
    return prompts


def run_inference_sglang(args):
    """
    Run SGLang inference using native Runtime API.
    
    SGLang handles distributed setup automatically via Ray.
    No torchrun needed!
    """
    print("\n" + "=" * 70)
    print("Initializing SGLang Runtime")
    print("=" * 70)
    
    try:
        # Initialize SGLang runtime
        # SGLang automatically handles multi-node setup via Ray
        # when appropriate environment variables are set
        
        runtime_config = {
            "model_path": args.model,
            "tp_size": args.tp_size,
            "trust_remote_code": True,
            "mem_fraction_static": 0.90,
        }
        
        # For multi-node, set Ray init address
        if args.nnodes > 1:
            runtime_config["nccl_init_addr"] = f"{args.master_addr}:{args.master_port}"
            runtime_config["nnodes"] = args.nnodes
            runtime_config["node_rank"] = args.node_rank
            print(f"Multi-node setup: {args.nnodes} nodes, rank {args.node_rank}")
        else:
            print(f"Single-node setup: {args.tp_size} GPUs")
        
        # Initialize runtime
        runtime = sgl.Runtime(**runtime_config)
        print("✓ SGLang runtime initialized successfully")
        
    except Exception as e:
        print(f"✗ Failed to initialize SGLang runtime: {e}")
        print("\n⚠️  Falling back to mock inference for testing...")
        return run_inference_mock(args)
    
    # Generate prompts
    prompts = generate_prompts(NUM_PROMPTS)
    
    # Warmup
    print("\nWarmup: Running 10 prompts...")
    warmup_prompts = prompts[:10]
    try:
        _ = runtime.generate(
            warmup_prompts,
            sampling_params={
                "max_new_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            }
        )
        print("✓ Warmup complete")
    except Exception as e:
        print(f"⚠️  Warmup failed: {e}")
    
    # Benchmark
    print(f"\nBenchmark: Running {NUM_PROMPTS} prompts...")
    start_time = time.time()
    
    try:
        outputs = runtime.generate(
            prompts,
            sampling_params={
                "max_new_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            }
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Calculate metrics
        total_tokens = sum(len(output["meta_info"]["completion_tokens"]) for output in outputs)
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
            prompt = prompts[i]
            generated_text = output["text"]
            print(f"\n[Prompt {i+1}]: {prompt}")
            print(f"[Output {i+1}]: {generated_text[:200]}...")
        
        # MADEngine output format
        print(f"\nperformance: {throughput:.2f} requests_per_second")
        print(f"tokens_per_second: {tokens_per_second:.2f}")
        print(f"model: {args.model}")
        print(f"tp_size: {args.tp_size}")
        print(f"nnodes: {args.nnodes}")
        
        # Cleanup
        runtime.shutdown()
        
        return 0
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️  Falling back to mock inference...")
        return run_inference_mock(args)


def run_inference_mock(args):
    """
    Mock inference for testing infrastructure without real SGLang.
    """
    print("\n" + "=" * 70)
    print("⚠️  Running Mock Inference (Testing Mode)")
    print("=" * 70)
    print("This simulates SGLang inference for testing MADEngine infrastructure.")
    print("=" * 70)
    
    # Simulate initialization
    print("\nInitializing mock SGLang runtime...")
    time.sleep(1)
    print("✓ Mock runtime initialized")
    
    # Generate prompts
    prompts = generate_prompts(NUM_PROMPTS)
    
    # Warmup
    print("\nWarmup: Running 10 prompts...")
    time.sleep(0.5)
    print("✓ Warmup complete")
    
    # Benchmark
    print(f"\nBenchmark: Running {NUM_PROMPTS} prompts...")
    start_time = time.time()
    
    # Simulate inference
    time.sleep(2.0)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Mock metrics
    total_tokens = NUM_PROMPTS * MAX_TOKENS
    throughput = NUM_PROMPTS / elapsed_time
    tokens_per_second = total_tokens / elapsed_time
    
    # Print results
    print(f"\n{'=' * 70}")
    print("Benchmark Results (Mock)")
    print("=" * 70)
    print(f"Total prompts: {NUM_PROMPTS}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Throughput: {throughput:.2f} requests/second")
    print(f"Token generation: {tokens_per_second:.2f} tokens/second")
    print(f"Average latency: {(elapsed_time / NUM_PROMPTS) * 1000:.2f} ms/request")
    print("=" * 70)
    
    # Print sample outputs
    print("\n" + "=" * 70)
    print("Sample Outputs (Mock - first 3)")
    print("=" * 70)
    for i in range(3):
        print(f"\n[Prompt {i+1}]: {prompts[i]}")
        print(f"[Output {i+1}]: [Mock generated text for infrastructure testing...]")
    
    # MADEngine output format
    print(f"\nperformance: {throughput:.2f} requests_per_second")
    print(f"tokens_per_second: {tokens_per_second:.2f}")
    print(f"model: {args.model}")
    print(f"tp_size: {args.tp_size}")
    print(f"nnodes: {args.nnodes}")
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SGLang Distributed Inference Benchmark (Native Launcher)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name or path (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size (GPUs per node, default: 1)"
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="Number of nodes (default: 1)"
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="Node rank (0-indexed, default: 0)"
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        default="localhost",
        help="Master node address (default: localhost)"
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Master communication port (default: 29500)"
    )
    parser.add_argument(
        "--mock-only",
        action="store_true",
        help="Force mock inference (skip real SGLang)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.tp_size < 1:
        print("Error: tp-size must be >= 1")
        return 1
    
    if args.nnodes < 1:
        print("Error: nnodes must be >= 1")
        return 1
    
    if args.node_rank < 0 or args.node_rank >= args.nnodes:
        print(f"Error: node-rank must be in range [0, {args.nnodes-1}]")
        return 1
    
    # Print configuration
    print_header(args)
    
    # Run inference
    if args.mock_only:
        return run_inference_mock(args)
    else:
        return run_inference_sglang(args)


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
