#!/usr/bin/env python3
"""
PyTorch Distributed Training Benchmark for MADEngine

This benchmark demonstrates typical PyTorch distributed training patterns:
- DistributedDataParallel (DDP) for multi-GPU/multi-node training
- Synthetic data generation for reproducible benchmarks
- Proper GPU device assignment using LOCAL_RANK
- Gradient synchronization across processes
- Throughput measurement (samples/sec, images/sec)
- Compatible with torchrun launcher

Usage:
  # Single GPU
  torchrun --standalone --nproc_per_node=1 run_torchrun.py
  
  # Multi-GPU (single node)
  torchrun --standalone --nproc_per_node=8 run_torchrun.py
  
  # Multi-node (via K8s with torchrun)
  torchrun --nnodes=4 --nproc_per_node=8 --master_addr=... run_torchrun.py
"""

import os
import sys
import time
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Configuration
BATCH_SIZE = 128  # Per-GPU batch size
NUM_EPOCHS = 5
NUM_BATCHES = 100  # Number of synthetic batches per epoch
IMAGE_SIZE = 224
NUM_CLASSES = 1000

# Get distributed environment variables (set by torchrun)
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
master_addr = os.environ.get("MASTER_ADDR", "localhost")
master_port = os.environ.get("MASTER_PORT", "29500")


def print_header():
    """Print benchmark header"""
    print("=" * 70)
    print("MADEngine PyTorch Distributed Training Benchmark")
    print("=" * 70)
    print(f"Hostname: {socket.gethostname()}")
    print(f"Rank: {rank}/{world_size}")
    print(f"Local Rank (GPU): {local_rank}")
    if world_size > 1:
        print(f"Master: {master_addr}:{master_port}")
    print(f"\nConfiguration:")
    print(f"  Batch Size (per GPU): {BATCH_SIZE}")
    print(f"  Global Batch Size: {BATCH_SIZE * world_size}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batches per Epoch: {NUM_BATCHES}")
    print(f"  Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Num Classes: {NUM_CLASSES}")
    print("=" * 70)


class SimpleCNN(nn.Module):
    """Simple CNN model for benchmarking"""
    def __init__(self, num_classes=1000):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def generate_synthetic_batch(batch_size, device):
    """Generate synthetic data for benchmarking"""
    images = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
    return images, labels


def train_epoch(model, optimizer, criterion, epoch, device):
    """Train for one epoch with accurate distributed throughput measurement"""
    model.train()
    epoch_start = time.time()
    total_samples = 0
    total_loss = 0.0
    
    for batch_idx in range(NUM_BATCHES):
        batch_start = time.time()
        
        # Generate synthetic data
        images, labels = generate_synthetic_batch(BATCH_SIZE, device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass (gradients are automatically synchronized across GPUs)
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        batch_time = time.time() - batch_start
        total_samples += BATCH_SIZE
        total_loss += loss.item()
        
        # Print progress from rank 0
        if rank == 0 and (batch_idx + 1) % 20 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            throughput = BATCH_SIZE * world_size / batch_time
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                  f"Batch [{batch_idx+1}/{NUM_BATCHES}] "
                  f"Loss: {loss.item():.4f} "
                  f"Throughput: {throughput:.2f} samples/sec")
    
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / NUM_BATCHES
    
    # ========================================================================
    # Accurate Distributed Throughput Measurement (Best Practice)
    # ========================================================================
    # Calculate local throughput for this rank
    local_samples = NUM_BATCHES * BATCH_SIZE
    local_throughput = local_samples / epoch_time
    
    # Aggregate metrics across all ranks using all_reduce
    if world_size > 1:
        # Convert to tensors for all_reduce
        local_throughput_tensor = torch.tensor([local_throughput], device=device)
        epoch_time_tensor = torch.tensor([epoch_time], device=device)
        
        # Sum all local throughputs to get true global throughput
        global_throughput_tensor = local_throughput_tensor.clone()
        dist.all_reduce(global_throughput_tensor, op=dist.ReduceOp.SUM)
        
        # Get max epoch time (slowest node determines overall speed)
        max_epoch_time_tensor = epoch_time_tensor.clone()
        dist.all_reduce(max_epoch_time_tensor, op=dist.ReduceOp.MAX)
        
        # Get min epoch time (fastest node)
        min_epoch_time_tensor = epoch_time_tensor.clone()
        dist.all_reduce(min_epoch_time_tensor, op=dist.ReduceOp.MIN)
        
        global_throughput = global_throughput_tensor.item()
        max_epoch_time = max_epoch_time_tensor.item()
        min_epoch_time = min_epoch_time_tensor.item()
        
        # Calculate load imbalance
        time_imbalance = ((max_epoch_time - min_epoch_time) / max_epoch_time) * 100 if max_epoch_time > 0 else 0.0
        
    else:
        # Single GPU
        global_throughput = local_throughput
        max_epoch_time = epoch_time
        min_epoch_time = epoch_time
        time_imbalance = 0.0
    
    # Return metrics dictionary
    metrics = {
        'avg_loss': avg_loss,
        'local_throughput': local_throughput,
        'global_throughput': global_throughput,
        'epoch_time': epoch_time,
        'max_epoch_time': max_epoch_time,
        'min_epoch_time': min_epoch_time,
        'time_imbalance': time_imbalance
    }
    
    return metrics


def main():
    """Main training function"""
    print_header()
    
    # Create per-process MIOpen cache directory to avoid database conflicts
    # This must be done AFTER torchrun sets LOCAL_RANK environment variable
    # This prevents "Duplicate ID" errors and database corruption in multi-GPU training
    if "MIOPEN_USER_DB_PATH" in os.environ:
        # Construct the per-process MIOpen path using actual local_rank value
        # Cannot use expandvars() because the template uses ${LOCAL_RANK} syntax
        miopen_template = os.environ["MIOPEN_USER_DB_PATH"]
        # Replace ${LOCAL_RANK} or $LOCAL_RANK with actual value
        miopen_path = miopen_template.replace("${LOCAL_RANK:-0}", str(local_rank)).replace("$LOCAL_RANK", str(local_rank))
        os.makedirs(miopen_path, exist_ok=True)
        print(f"[Rank {rank}] ✓ Created MIOpen cache directory: {miopen_path}")
    
    # Initialize distributed training
    if world_size > 1:
        print(f"\n[Rank {rank}] Initializing distributed process group...")
        # Best practice: Specify device_ids to avoid PyTorch warnings
        dist.init_process_group(
            backend="nccl",
            init_method=f"env://",  # Use environment variables (set by torchrun)
            world_size=world_size,
            rank=rank
        )
        print(f"[Rank {rank}] ✓ Process group initialized")
        print(f"[Rank {rank}]   Backend: {dist.get_backend()}")
        print(f"[Rank {rank}]   World Size: {dist.get_world_size()}")
    else:
        print(f"\n=== Running in Standalone Mode (Single GPU) ===")
    
    # Set device
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"[Rank {rank}] PyTorch sees {num_gpus} GPU(s)")
        print(f"[Rank {rank}] LOCAL_RANK={local_rank}, attempting to use cuda:{local_rank}")
        
        if local_rank >= num_gpus:
            print(f"[Rank {rank}] ERROR: LOCAL_RANK {local_rank} >= available GPUs {num_gpus}")
            print(f"[Rank {rank}] Using cuda:0 instead")
            device = torch.device("cuda:0")
        else:
            device = torch.device(f"cuda:{local_rank}")
        
        torch.cuda.set_device(device)
        print(f"[Rank {rank}] Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print(f"[Rank {rank}] Warning: CUDA not available, using CPU")
    
    # Create model
    print(f"\n[Rank {rank}] Creating model...")
    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    
    # Wrap model with DDP for distributed training
    if world_size > 1:
        # Best practice: Explicitly specify device_ids for DDP
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,  # Ensure buffers (like BatchNorm stats) are synced
            find_unused_parameters=False  # Set True only if needed (performance impact)
        )
        print(f"[Rank {rank}] ✓ Model wrapped with DistributedDataParallel")
    
    # Create optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Synchronize before training
    if world_size > 1:
        # Best practice: Specify device to avoid warnings
        dist.barrier(device_ids=[local_rank])
    
    if rank == 0:
        print(f"\n{'='*70}")
        print("Starting Training")
        print(f"{'='*70}")
    
    # Training loop
    all_metrics = []
    for epoch in range(NUM_EPOCHS):
        metrics = train_epoch(
            model, optimizer, criterion, epoch, device
        )
        all_metrics.append(metrics)
        
        if rank == 0:
            print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] Complete:")
            print(f"  Average Loss: {metrics['avg_loss']:.4f}")
            print(f"  Global Throughput: {metrics['global_throughput']:.2f} samples/sec")
            print(f"  Images/sec: {metrics['global_throughput']:.2f}")
    
            # Show load imbalance warning if significant
            if metrics['time_imbalance'] > 5.0:
                print(f"  ⚠️  Load Imbalance: {metrics['time_imbalance']:.1f}%")
    
    # Calculate average metrics across all epochs
    avg_global_throughput = sum(m['global_throughput'] for m in all_metrics) / len(all_metrics)
    avg_local_throughput = sum(m['local_throughput'] for m in all_metrics) / len(all_metrics)
    avg_time_imbalance = sum(m['time_imbalance'] for m in all_metrics) / len(all_metrics)
    
    # Get topology information
    nproc_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    num_nodes = (world_size + nproc_per_node - 1) // nproc_per_node if nproc_per_node > 0 else 1
    node_rank = rank // nproc_per_node if nproc_per_node > 0 else 0
    
    # Synchronize before final output
    if world_size > 1:
        dist.barrier(device_ids=[local_rank])
    
    # Each node's rank 0 reports local performance
    if local_rank == 0:
        print(f"\n[Node {node_rank}] Local Performance Summary:")
        print(f"  Node Throughput: {avg_local_throughput * nproc_per_node:.2f} samples/sec")
        print(f"  GPUs on Node: {nproc_per_node}")
        print(f"  Avg Time per Epoch: {all_metrics[-1]['epoch_time']:.2f}s")
    
    # Synchronize again before global rank 0 output
    if world_size > 1:
        dist.barrier(device_ids=[local_rank])
    
    # Global rank 0 reports aggregated performance
    if rank == 0:
        print(f"\n{'='*70}")
        print("Training Complete - GLOBAL METRICS")
        print(f"{'='*70}")
        print(f"Topology: {num_nodes} nodes × {nproc_per_node} GPUs/node = {world_size} total GPUs")
        print(f"Global Throughput: {avg_global_throughput:.2f} samples/sec")
        print(f"Per-GPU Throughput: {avg_global_throughput/world_size:.2f} samples/sec")
        print(f"Global Batch Size: {BATCH_SIZE * world_size}")
        
        # Calculate scaling efficiency
        # Ideal throughput = single GPU throughput * number of GPUs
        ideal_single_gpu_throughput = avg_global_throughput / world_size
        ideal_throughput = ideal_single_gpu_throughput * world_size
        scaling_efficiency = (avg_global_throughput / ideal_throughput) * 100 if ideal_throughput > 0 else 100.0
        print(f"Scaling Efficiency: {scaling_efficiency:.1f}%")
        
        if avg_time_imbalance > 5.0:
            print(f"Average Load Imbalance: {avg_time_imbalance:.1f}%")
        
        print(f"{'='*70}")
        
        # Save results with topology information
        with open("training_results.txt", "w") as f:
            f.write(f"Training Results\n")
            f.write(f"================\n")
            f.write(f"Hostname: {socket.gethostname()}\n")
            f.write(f"Topology: {num_nodes} nodes × {nproc_per_node} GPUs/node\n")
            f.write(f"World Size: {world_size}\n")
            f.write(f"Global Batch Size: {BATCH_SIZE * world_size}\n")
            f.write(f"Epochs: {NUM_EPOCHS}\n")
            f.write(f"Global Throughput: {avg_global_throughput:.2f} samples/sec\n")
            f.write(f"Scaling Efficiency: {scaling_efficiency:.1f}%\n")
        
        # Output performance metric for MADEngine (REQUIRED FORMAT)
        # Use GLOBAL throughput (sum of all nodes - accurate measurement)
        print(f"\nperformance: {avg_global_throughput:.2f} samples_per_second")
        
        # Output topology metadata for parsing
        print(f"topology: {num_nodes} nodes {nproc_per_node} gpus_per_node {world_size} total_gpus")
        print(f"scaling_efficiency: {scaling_efficiency:.2f}")

    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()
        if rank == 0:
            print(f"✓ Process group destroyed")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[Rank {rank}] ✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
