#!/usr/bin/env python3
"""
SGLang Disaggregated Dummy Inference Script

Simulates the disaggregated prefill/decode architecture for testing.
This is a lightweight test that validates the launcher setup without
requiring actual models or Mooncake infrastructure.
"""

import os
import sys
import time
import socket
from typing import Optional


def get_node_info() -> dict:
    """Extract node information from environment variables."""
    return {
        "node_rank": int(os.getenv("SGLANG_NODE_RANK", "0")),
        "total_nodes": int(os.getenv("SGLANG_DISAGG_TOTAL_NODES", "3")),
        "prefill_nodes": int(os.getenv("SGLANG_DISAGG_PREFILL_NODES", "1")),
        "decode_nodes": int(os.getenv("SGLANG_DISAGG_DECODE_NODES", "1")),
        "tp_size": int(os.getenv("SGLANG_TP_SIZE", "1")),
        "master_port": int(os.getenv("MASTER_PORT", "29500")),
        "hostname": socket.gethostname(),
    }


def determine_node_role(node_rank: int, prefill_nodes: int) -> str:
    """Determine if this node is proxy, prefill, or decode."""
    if node_rank == 0:
        return "proxy"
    elif node_rank <= prefill_nodes:
        return "prefill"
    else:
        return "decode"


def simulate_proxy_node(info: dict):
    """Simulate proxy/load balancer node."""
    print("=" * 60)
    print("🔀 PROXY NODE (Load Balancer)")
    print("=" * 60)
    print(f"Hostname: {info['hostname']}")
    print(f"Node Rank: {info['node_rank']}")
    print(f"Master Port: {info['master_port']}")
    print(f"Prefill Nodes: {info['prefill_nodes']}")
    print(f"Decode Nodes: {info['decode_nodes']}")
    print("-" * 60)
    
    print("\n[Proxy] Initializing load balancer...")
    time.sleep(1)
    
    print("[Proxy] Waiting for prefill nodes to be ready...")
    for i in range(1, info['prefill_nodes'] + 1):
        print(f"  ✓ Prefill node {i} connected")
        time.sleep(0.5)
    
    print("[Proxy] Waiting for decode nodes to be ready...")
    for i in range(info['prefill_nodes'] + 1, info['total_nodes']):
        print(f"  ✓ Decode node {i} connected")
        time.sleep(0.5)
    
    print("\n[Proxy] All nodes connected. Load balancer ready!")
    print("[Proxy] Simulating request routing...")
    
    # Simulate some requests
    for req_id in range(1, 4):
        print(f"\n[Proxy] Request {req_id}:")
        print(f"  → Routing to prefill node {(req_id % info['prefill_nodes']) + 1}")
        time.sleep(0.3)
        print(f"  → KV cache transferred via Mooncake")
        time.sleep(0.3)
        print(f"  → Routing to decode node {info['prefill_nodes'] + ((req_id % info['decode_nodes']) + 1)}")
        time.sleep(0.3)
        print(f"  ✓ Request {req_id} completed")
    
    print("\n[Proxy] Test complete. Shutting down...")


def simulate_prefill_node(info: dict):
    """Simulate prefill node."""
    print("=" * 60)
    print("⚡ PREFILL NODE")
    print("=" * 60)
    print(f"Hostname: {info['hostname']}")
    print(f"Node Rank: {info['node_rank']}")
    print(f"Tensor Parallel Size: {info['tp_size']}")
    print(f"Role: Prompt Processing")
    print("-" * 60)
    
    print("\n[Prefill] Initializing prefill server...")
    time.sleep(1)
    
    print("[Prefill] Loading model shards...")
    for shard in range(info['tp_size']):
        print(f"  ✓ Shard {shard + 1}/{info['tp_size']} loaded")
        time.sleep(0.3)
    
    print("\n[Prefill] Server ready. Listening for requests...")
    time.sleep(1)
    
    print("[Prefill] Processing prompts...")
    for batch in range(1, 4):
        print(f"\n[Prefill] Batch {batch}:")
        print(f"  → Processing prompt tokens...")
        time.sleep(0.5)
        print(f"  → Generating KV cache...")
        time.sleep(0.5)
        print(f"  → Transferring KV cache via Mooncake...")
        time.sleep(0.3)
        print(f"  ✓ Batch {batch} complete")
    
    print("\n[Prefill] Test complete. Shutting down...")


def simulate_decode_node(info: dict):
    """Simulate decode node."""
    print("=" * 60)
    print("🔤 DECODE NODE")
    print("=" * 60)
    print(f"Hostname: {info['hostname']}")
    print(f"Node Rank: {info['node_rank']}")
    print(f"Tensor Parallel Size: {info['tp_size']}")
    print(f"Role: Token Generation")
    print("-" * 60)
    
    print("\n[Decode] Initializing decode server...")
    time.sleep(1)
    
    print("[Decode] Loading model shards...")
    for shard in range(info['tp_size']):
        print(f"  ✓ Shard {shard + 1}/{info['tp_size']} loaded")
        time.sleep(0.3)
    
    print("\n[Decode] Server ready. Listening for KV caches...")
    time.sleep(1)
    
    print("[Decode] Generating tokens...")
    for batch in range(1, 4):
        print(f"\n[Decode] Batch {batch}:")
        print(f"  → Receiving KV cache via Mooncake...")
        time.sleep(0.5)
        print(f"  → Generating tokens...")
        for token in range(1, 6):
            print(f"    Token {token}/5", end="\r")
            time.sleep(0.2)
        print(f"    ✓ Generated 5 tokens")
        print(f"  ✓ Batch {batch} complete")
    
    print("\n[Decode] Test complete. Shutting down...")


def main():
    """Main entry point for disaggregated inference simulation."""
    print("\n" + "=" * 60)
    print("SGLang Disaggregated Inference Simulation")
    print("=" * 60 + "\n")
    
    # Get node information
    info = get_node_info()
    role = determine_node_role(info["node_rank"], info["prefill_nodes"])
    
    print(f"Cluster Configuration:")
    print(f"  Total Nodes: {info['total_nodes']}")
    print(f"  Prefill Nodes: {info['prefill_nodes']} (ranks 1-{info['prefill_nodes']})")
    print(f"  Decode Nodes: {info['decode_nodes']} (ranks {info['prefill_nodes']+1}-{info['total_nodes']-1})")
    print(f"  Proxy Node: 1 (rank 0)")
    print(f"\nThis Node:")
    print(f"  Rank: {info['node_rank']}")
    print(f"  Role: {role.upper()}")
    print(f"  Hostname: {info['hostname']}")
    print()
    
    # Simulate based on role
    try:
        if role == "proxy":
            simulate_proxy_node(info)
        elif role == "prefill":
            simulate_prefill_node(info)
        elif role == "decode":
            simulate_decode_node(info)
        else:
            print(f"❌ ERROR: Unknown role '{role}'")
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("✅ Simulation Complete")
        print("=" * 60)
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

