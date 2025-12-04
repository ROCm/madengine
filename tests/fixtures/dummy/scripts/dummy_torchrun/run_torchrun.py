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
    """Train for one epoch"""
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
    epoch_throughput = (NUM_BATCHES * BATCH_SIZE * world_size) / epoch_time
    
    return avg_loss, epoch_throughput


def main():
    """Main training function"""
    print_header()
    
    # Initialize distributed training
    if world_size > 1:
        print(f"\n[Rank {rank}] Initializing distributed process group...")
        dist.init_process_group(backend="nccl")
        print(f"[Rank {rank}] ✓ Process group initialized")
        print(f"[Rank {rank}]   Backend: {dist.get_backend()}")
        print(f"[Rank {rank}]   World Size: {dist.get_world_size()}")
    else:
        print(f"\n=== Running in Standalone Mode (Single GPU) ===")
    
    # Set device
    if torch.cuda.is_available():
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
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print(f"[Rank {rank}] ✓ Model wrapped with DistributedDataParallel")
    
    # Create optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Synchronize before training
    if world_size > 1:
        dist.barrier()
    
    if rank == 0:
        print(f"\n{'='*70}")
        print("Starting Training")
        print(f"{'='*70}")
    
    # Training loop
    all_throughputs = []
    for epoch in range(NUM_EPOCHS):
        avg_loss, epoch_throughput = train_epoch(
            model, optimizer, criterion, epoch, device
        )
        all_throughputs.append(epoch_throughput)
        
        if rank == 0:
            print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] Complete:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Throughput: {epoch_throughput:.2f} samples/sec")
            print(f"  Images/sec: {epoch_throughput:.2f}")
    
    # Calculate final metrics
    avg_throughput = sum(all_throughputs) / len(all_throughputs)
    
    # Synchronize before final output
    if world_size > 1:
        dist.barrier()
    
    if rank == 0:
        print(f"\n{'='*70}")
        print("Training Complete")
        print(f"{'='*70}")
        print(f"Average Throughput: {avg_throughput:.2f} samples/sec")
        print(f"Global Batch Size: {BATCH_SIZE * world_size}")
        print(f"Number of GPUs: {world_size}")
        print(f"{'='*70}")
        
        # Save results
        with open("training_results.txt", "w") as f:
            f.write(f"Training Results\n")
            f.write(f"================\n")
            f.write(f"Hostname: {socket.gethostname()}\n")
            f.write(f"World Size: {world_size}\n")
            f.write(f"Global Batch Size: {BATCH_SIZE * world_size}\n")
            f.write(f"Epochs: {NUM_EPOCHS}\n")
            f.write(f"Average Throughput: {avg_throughput:.2f} samples/sec\n")
        
        # Output performance metric for MADEngine (REQUIRED FORMAT)
        print(f"\nperformance: {avg_throughput:.2f} samples_per_second")
    
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
