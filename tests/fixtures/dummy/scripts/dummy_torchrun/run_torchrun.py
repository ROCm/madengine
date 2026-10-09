#!/usr/bin/env python3
"""
PyTorch Distributed Training Benchmark for madengine

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
BATCH_SIZE = int(os.environ.get("DUMMY_BATCH_SIZE", "128"))
NUM_EPOCHS = int(os.environ.get("DUMMY_NUM_EPOCHS", "5"))
NUM_BATCHES = int(os.environ.get("DUMMY_NUM_BATCHES", "100"))
IMAGE_SIZE = int(os.environ.get("DUMMY_IMAGE_SIZE", "224"))
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
    print("madengine PyTorch Distributed Training Benchmark")
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


def run_one_batch(model, optimizer, criterion, device):
    """One forward/backward/step. Dynolog's iteration capture counts the step."""
    images, labels = generate_synthetic_batch(BATCH_SIZE, device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def _kineto_request_accepted():
    """True once dynolog_trigger.sh has had a gputrace request matched."""
    result_path = os.environ.get(
        "DYNOLOG_TRIGGER_RESULT", "/tmp/madengine_dynolog_trigger.result"
    )
    try:
        with open(result_path, "r", encoding="utf-8") as handle:
            return handle.read().strip() == "accepted"
    except OSError:
        return False


def _kineto_trace_written():
    """True once Kineto has renamed a finished trace into its output directory."""
    kineto_dir = os.environ.get("TORCH_PROFILE_KINETO_DIR", "/tmp/madengine_kineto")
    if not os.path.isdir(kineto_dir):
        return False
    for name in os.listdir(kineto_dir):
        if name.endswith(".json") or name.endswith(".json.gz"):
            return True
    return False


def train_until_kineto(model, optimizer, criterion, device):
    """Keep stepping until an on-demand Kineto trace can finish.

    A fixed batch count is not long enough on a fast GPU: the trigger sleeps
    before its first request, and the stop script kills that request if the
    process has already exited. Stay inside optimizer.step() until the request
    has been accepted and enough further steps have run for the capture window
    to close, or until DUMMY_KINETO_TIMEOUT_S.
    """
    model.train()
    epoch_start = time.time()
    timeout_s = float(os.environ.get("DUMMY_KINETO_TIMEOUT_S", "90"))
    deadline = epoch_start + timeout_s
    total_loss = 0.0
    batch_idx = 0
    # dyno can report the request accepted before this process's Kineto thread
    # has the config. Iteration capture only counts optimizer.step() calls made
    # after that, so keep stepping until the trace file exists.
    while time.time() < deadline:
        batch_start = time.time()
        loss_value = run_one_batch(model, optimizer, criterion, device)
        batch_time = time.time() - batch_start
        total_loss += loss_value
        batch_idx += 1

        if local_rank == 0 and batch_idx % 20 == 0:
            print(
                f"Kineto hold batch [{batch_idx}] "
                f"Loss: {loss_value:.4f} "
                f"Throughput: {BATCH_SIZE / batch_time:.2f} samples/sec (local)"
            )

        if _kineto_trace_written():
            if local_rank == 0:
                print(f"Kineto trace file is ready after {batch_idx} batches")
            break
    else:
        if local_rank == 0:
            print(
                f"Kineto hold reached {timeout_s:.0f}s "
                f"without a trace file "
                f"(dyno accepted={_kineto_request_accepted()}, batches={batch_idx})"
            )

    epoch_time = max(time.time() - epoch_start, 1e-6)
    completed = max(batch_idx, 1)
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    local_gpu_throughput = (completed * BATCH_SIZE) / epoch_time
    return {
        "avg_loss": total_loss / completed,
        "node_throughput": local_gpu_throughput * local_world_size,
        "epoch_time": epoch_time,
        "local_world_size": local_world_size,
    }


def train_epoch(model, optimizer, criterion, epoch, device):
    """Train for one epoch with node-local throughput measurement"""
    model.train()
    epoch_start = time.time()
    total_samples = 0
    total_loss = 0.0
    
    for batch_idx in range(NUM_BATCHES):
        batch_start = time.time()
        loss_value = run_one_batch(model, optimizer, criterion, device)
        
        batch_time = time.time() - batch_start
        total_samples += BATCH_SIZE
        total_loss += loss_value
        
        # Print progress from local rank 0 on each node
        if local_rank == 0 and (batch_idx + 1) % 20 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            throughput = BATCH_SIZE / batch_time  # Local throughput
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                  f"Batch [{batch_idx+1}/{NUM_BATCHES}] "
                  f"Loss: {loss.item():.4f} "
                  f"Throughput: {throughput:.2f} samples/sec (local)")
    
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / NUM_BATCHES
    
    # ========================================================================
    # Node-Local Throughput Measurement
    # ========================================================================
    # Calculate throughput for ALL GPUs on THIS NODE
    local_samples = NUM_BATCHES * BATCH_SIZE
    local_gpu_throughput = local_samples / epoch_time
    
    # Get local world size (GPUs per node)
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    
    # Node throughput = sum of all local GPUs on this node
    # In data parallel, each GPU processes the same throughput
    node_throughput = local_gpu_throughput * local_world_size
    
    # Return metrics dictionary
    metrics = {
        'avg_loss': avg_loss,
        'node_throughput': node_throughput,
        'epoch_time': epoch_time,
        'local_world_size': local_world_size
    }
    
    return metrics


def main():
    """Main training function"""
    # Start timer for total test duration
    test_start_time = time.time()
    
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
    
    # Get topology information early (needed for logging)
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    node_rank = rank // local_world_size if local_world_size > 0 else 0
    
    if local_rank == 0:
        print(f"\n{'='*70}")
        print(f"[Node {node_rank}] Starting Training")
        print(f"{'='*70}")
    
    # Training loop. The dynolog e2e holds the process in optimizer.step()
    # until a trace can be requested; a fixed batch count finishes first on
    # a fast GPU and the trigger is killed during its warmup sleep.
    all_metrics = []
    if os.environ.get("DUMMY_WAIT_FOR_KINETO") == "1" and world_size == 1:
        all_metrics.append(
            train_until_kineto(model, optimizer, criterion, device)
        )
    else:
        for epoch in range(NUM_EPOCHS):
            metrics = train_epoch(
                model, optimizer, criterion, epoch, device
            )
            all_metrics.append(metrics)

            if local_rank == 0:
                print(f"\n[Node {node_rank}] Epoch [{epoch+1}/{NUM_EPOCHS}] Complete:")
                print(f"  Average Loss: {metrics['avg_loss']:.4f}")
                print(f"  Node Throughput: {metrics['node_throughput']:.2f} samples/sec")
                print(f"  Local GPUs: {metrics['local_world_size']}")
    
    # Calculate average node throughput across all epochs
    avg_node_throughput = sum(m['node_throughput'] for m in all_metrics) / len(all_metrics)
    avg_epoch_time = sum(m['epoch_time'] for m in all_metrics) / len(all_metrics)
    
    # Calculate num_nodes for reference
    num_nodes = (world_size + local_world_size - 1) // local_world_size if local_world_size > 0 else 1
    
    # Synchronize before final output
    if world_size > 1:
        dist.barrier(device_ids=[local_rank])
    
    # ========================================================================
    # Node-Local Performance Reporting (NEW - Best Practice)
    # Each node reports its OWN performance
    # Madengine will collect from ALL nodes and aggregate
    # ========================================================================
    if local_rank == 0:
        print(f"\n{'='*70}")
        print("Node Performance Summary")
        print(f"{'='*70}")
        print(f"Node ID: {node_rank}")
        print(f"Node Hostname: {socket.gethostname()}")
        print(f"Local GPUs: {local_world_size}")
        print(f"Node Throughput: {avg_node_throughput:.2f} samples_per_second")
        print(f"Avg Time per Epoch: {avg_epoch_time:.2f}s")
        print(f"{'='*70}")
        
        # CRITICAL: Standard output format for madengine parsing
        print(f"performance: {avg_node_throughput:.2f} samples_per_second", flush=True)
        print(f"node_id: {node_rank}", flush=True)
        print(f"local_gpus: {local_world_size}", flush=True)
        
        # Calculate and print test duration
        test_duration = time.time() - test_start_time
        print(f"test_duration: {test_duration:.2f}s", flush=True)
        sys.stdout.flush()

    
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
