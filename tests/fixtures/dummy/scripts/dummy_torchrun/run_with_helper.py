#!/usr/bin/env python3
"""
PyTorch Distributed Training Benchmark with Helper Modules

This script demonstrates:
- Multi-file Python project structure
- Importing model architecture from helper module
- Separating concerns (config, model, training)
- Best practices for distributed training
"""

import os
import sys
import time
import socket
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Import from helper module
from helper import (
    ResNetModel,
    SyntheticDataset,
    BenchmarkConfig,
    print_distributed_info,
    print_gpu_info,
    calculate_model_size
)

# Get distributed environment variables (set by torchrun)
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))


def print_header(config):
    """Print benchmark header"""
    print("=" * 70)
    print("madengine PyTorch Benchmark (with Helper Modules)")
    print("=" * 70)
    print(f"Hostname: {socket.gethostname()}")
    print(f"Rank: {rank}/{world_size}")
    print(f"Local Rank (GPU): {local_rank}")
    print(f"\n{config}")
    print("=" * 70)


def train_epoch(model, dataset, optimizer, criterion, epoch, device, config):
    """Train for one epoch"""
    model.train()
    epoch_start = time.time()
    total_loss = 0.0
    
    for batch_idx in range(dataset.num_batches):
        batch_start = time.time()
        
        # Generate synthetic data
        images, labels = dataset.generate_batch(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass (gradients automatically synchronized)
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        batch_time = time.time() - batch_start
        total_loss += loss.item()
        
        # Print progress from rank 0
        if rank == 0 and (batch_idx + 1) % 20 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            throughput = config.batch_size * world_size / batch_time
            print(f"Epoch [{epoch+1}/{config.num_epochs}] "
                  f"Batch [{batch_idx+1}/{dataset.num_batches}] "
                  f"Loss: {loss.item():.4f} "
                  f"Throughput: {throughput:.2f} samples/sec")
    
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / dataset.num_batches
    epoch_throughput = (dataset.num_batches * config.batch_size * world_size) / epoch_time
    
    return avg_loss, epoch_throughput


def main():
    """Main training function"""
    # Load configuration
    config = BenchmarkConfig()
    
    print_header(config)
    
    # Print distributed info
    print_distributed_info(rank, local_rank, world_size)
    
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
        print_gpu_info(rank, device)
    else:
        device = torch.device("cpu")
        print(f"[Rank {rank}] Warning: CUDA not available, using CPU")
    
    # Create model from helper module
    print(f"\n[Rank {rank}] Creating ResNet model from helper module...")
    model = ResNetModel(
        num_classes=config.num_classes,
        num_blocks=config.resnet_blocks
    ).to(device)
    
    # Print model info
    if rank == 0:
        total_params, trainable_params = calculate_model_size(model)
        print(f"\nModel Statistics:")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        print(f"  Model Size: {total_params * 4 / 1e6:.2f} MB (FP32)")
    
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
    
    # Create dataset
    dataset = SyntheticDataset(
        num_samples=config.num_batches * config.batch_size,
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_classes=config.num_classes
    )
    
    # Create optimizer and loss function
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
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
    all_throughputs = []
    for epoch in range(config.num_epochs):
        avg_loss, epoch_throughput = train_epoch(
            model, dataset, optimizer, criterion, epoch, device, config
        )
        all_throughputs.append(epoch_throughput)
        
        if rank == 0:
            print(f"\nEpoch [{epoch+1}/{config.num_epochs}] Complete:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Throughput: {epoch_throughput:.2f} samples/sec")
    
    # Calculate final metrics
    avg_throughput = sum(all_throughputs) / len(all_throughputs)
    
    # Synchronize before final output
    if world_size > 1:
        dist.barrier(device_ids=[local_rank])
    
    if rank == 0:
        print(f"\n{'='*70}")
        print("Training Complete")
        print(f"{'='*70}")
        print(f"Average Throughput: {avg_throughput:.2f} samples/sec")
        print(f"Global Batch Size: {config.batch_size * world_size}")
        print(f"Number of GPUs: {world_size}")
        print(f"Model: ResNet with {sum(config.resnet_blocks)} blocks")
        print(f"{'='*70}")
        
        # Save results
        with open("training_results_helper.txt", "w") as f:
            f.write(f"Training Results (with Helper Modules)\n")
            f.write(f"======================================\n")
            f.write(f"Hostname: {socket.gethostname()}\n")
            f.write(f"World Size: {world_size}\n")
            f.write(f"Global Batch Size: {config.batch_size * world_size}\n")
            f.write(f"Epochs: {config.num_epochs}\n")
            f.write(f"Model: ResNet-{sum(config.resnet_blocks)*2+2}\n")
            f.write(f"Average Throughput: {avg_throughput:.2f} samples/sec\n")
        
        # Output performance metric for madengine (REQUIRED FORMAT)
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
