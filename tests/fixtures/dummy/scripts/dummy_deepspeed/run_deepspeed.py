#!/usr/bin/env python3
"""
DeepSpeed Training Benchmark - Uses deepspeed launcher

Demonstrates DeepSpeed features:
- ZeRO optimizer stages
- Gradient accumulation
- Mixed precision training
- Uses deepspeed launcher (NOT torchrun)

Launch with deepspeed launcher:
  deepspeed --num_gpus=2 run_deepspeed.py
"""

import os
import sys
import time
import socket
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import deepspeed

# Configuration
NUM_EPOCHS = 3
NUM_BATCHES = 50
IMAGE_SIZE = 224
NUM_CLASSES = 1000

class SimpleModel(nn.Module):
    """Simple model for DeepSpeed testing"""
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def print_header(args):
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if rank == 0:
        print("=" * 70)
        print("DeepSpeed Distributed Training Benchmark")
        print("=" * 70)
        print(f"Hostname: {socket.gethostname()}")
        print(f"World Size: {world_size}")
        print(f"DeepSpeed Config: {args.deepspeed_config}")
        print(f"Training: {NUM_EPOCHS} epochs, {NUM_BATCHES} batches/epoch")
        print("=" * 70)

def train_epoch(model_engine, criterion, epoch):
    model_engine.train()
    start_time = time.time()
    total_loss = 0
    
    rank = model_engine.local_rank
    micro_batch_size = model_engine.train_micro_batch_size_per_gpu()
    
    for batch_idx in range(NUM_BATCHES):
        # Synthetic data
        inputs = torch.randn(
            micro_batch_size, 3, IMAGE_SIZE, IMAGE_SIZE,
            device=model_engine.device
        )
        labels = torch.randint(
            0, NUM_CLASSES, (micro_batch_size,),
            device=model_engine.device
        )
        
        # Forward pass
        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass (DeepSpeed handles gradients, optimization)
        model_engine.backward(loss)
        model_engine.step()
        
        total_loss += loss.item()
        
        if rank == 0 and (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}] Batch [{batch_idx+1}/{NUM_BATCHES}] Loss: {loss.item():.4f}")
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / NUM_BATCHES
    
    # Calculate throughput
    world_size = model_engine.world_size
    throughput = (NUM_BATCHES * micro_batch_size * world_size) / epoch_time
    
    return avg_loss, throughput

def main():
    # Parse DeepSpeed args
    parser = argparse.ArgumentParser()
    # local_rank default should come from environment (set by torchrun)
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    parser.add_argument('--deepspeed_config', type=str, default='ds_config.json')
    args = parser.parse_args()
    
    # Handle config file path - supports multiple locations for K8s/local execution
    config_found = False
    original_config_path = args.deepspeed_config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try 1: Check as-is (current directory or absolute path)
    if os.path.exists(args.deepspeed_config):
        config_found = True
        print(f"[Config] Found DeepSpeed config: {args.deepspeed_config}")
    
    # Try 2: Check relative to script directory (for K8s execution)
    if not config_found:
        config_path = os.path.join(script_dir, args.deepspeed_config)
        if os.path.exists(config_path):
            args.deepspeed_config = config_path
            config_found = True
            print(f"[Config] Found DeepSpeed config in script directory: {config_path}")
    
    # Try 3: Check in scripts/dummy_deepspeed/ directory (for local execution)
    if not config_found:
        local_config_path = os.path.join('scripts/dummy_deepspeed', args.deepspeed_config)
        if os.path.exists(local_config_path):
            args.deepspeed_config = local_config_path
            config_found = True
            print(f"[Config] Found DeepSpeed config in scripts directory: {local_config_path}")
    
    # Error if not found
    if not config_found:
        print(f"\n❌ Error: DeepSpeed config not found!")
        print(f"Searched for: {original_config_path}")
        print(f"Locations tried:")
        print(f"  1. Current directory: {os.getcwd()}/{original_config_path}")
        print(f"  2. Script directory: {os.path.join(script_dir, original_config_path)}")
        print(f"  3. Scripts directory: scripts/dummy_deepspeed/{original_config_path}")
        print(f"\nCurrent directory: {os.getcwd()}")
        print(f"Files in current directory:")
        try:
            for f in os.listdir('.'):
                print(f"  - {f}")
        except Exception as e:
            print(f"  (Cannot list: {e})")
        print(f"\nScript location: {os.path.abspath(__file__)}")
        sys.exit(1)
    
    print_header(args)
    
    # Initialize PyTorch distributed backend BEFORE DeepSpeed
    # This prevents DeepSpeed from trying to use MPI
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        print(f"✓ PyTorch distributed initialized (backend: nccl)")
    
    # Create model
    model = SimpleModel(NUM_CLASSES)
    
    # Initialize DeepSpeed
    # Note: When using deepspeed launcher with --deepspeed_config arg,
    # do NOT pass config parameter to initialize() - it causes a conflict
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
    )
    
    criterion = nn.CrossEntropyLoss()
    
    rank = model_engine.local_rank
    
    if rank == 0:
        print(f"\n✓ DeepSpeed initialized")
        print(f"  ZeRO Stage: {model_engine.zero_optimization_stage()}")
        print(f"  Micro Batch Size: {model_engine.train_micro_batch_size_per_gpu()}")
        print(f"  Gradient Accumulation: {model_engine.gradient_accumulation_steps()}")
        print(f"\nStarting training...\n")
    
    # Training loop
    all_throughputs = []
    for epoch in range(NUM_EPOCHS):
        avg_loss, throughput = train_epoch(model_engine, criterion, epoch)
        all_throughputs.append(throughput)
        
        if rank == 0:
            print(f"\nEpoch {epoch+1} Complete: Loss={avg_loss:.4f}, Throughput={throughput:.2f} samples/sec\n")
    
    if rank == 0:
        avg_throughput = sum(all_throughputs) / len(all_throughputs)
        print(f"{'='*70}")
        print(f"DeepSpeed Training Complete")
        print(f"  Average Throughput: {avg_throughput:.2f} samples/sec")
        print(f"  ZeRO Stage: {model_engine.zero_optimization_stage()}")
        print(f"  World Size: {model_engine.world_size}")
        print(f"{'='*70}")
        
        # madengine output format
        print(f"\nperformance: {avg_throughput:.2f} samples_per_second")
        print(f"deepspeed_config: ZeRO_stage={model_engine.zero_optimization_stage()}")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
