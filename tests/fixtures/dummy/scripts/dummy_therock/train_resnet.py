#!/usr/bin/env python3
"""
Simple ResNet50 Training Benchmark for TheRock

This script benchmarks ResNet50 training performance using PyTorch
on TheRock's ROCm distribution.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import time
import sys

# Configuration
BATCH_SIZE = 64
NUM_ITERATIONS = 100
IMAGE_SIZE = 224


def main():
    print("=" * 70)
    print("ResNet50 Training Benchmark (TheRock)")
    print("=" * 70)
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")
    
    # Create model
    print("\nCreating ResNet50 model...")
    model = models.resnet50(pretrained=False, num_classes=1000).to(device)
    model.train()
    
    # Setup optimizer and loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Iterations: {NUM_ITERATIONS}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    
    # Warmup
    print("\nWarming up (10 iterations)...")
    for _ in range(10):
        images = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        labels = torch.randint(0, 1000, (BATCH_SIZE,), device=device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running benchmark ({NUM_ITERATIONS} iterations)...")
    start_time = time.time()
    
    for i in range(NUM_ITERATIONS):
        images = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        labels = torch.randint(0, 1000, (BATCH_SIZE,), device=device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{NUM_ITERATIONS}")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calculate metrics
    duration = end_time - start_time
    total_images = BATCH_SIZE * NUM_ITERATIONS
    images_per_sec = total_images / duration
    
    print("\n" + "=" * 70)
    print("Benchmark Results:")
    print(f"  Total Images Processed: {total_images}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Throughput: {images_per_sec:.2f} images/sec")
    print("=" * 70)
    
    # madengine performance output (required format)
    print(f"\nperformance: {images_per_sec:.2f} images_per_second")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

