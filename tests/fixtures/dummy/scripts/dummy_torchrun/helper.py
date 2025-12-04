#!/usr/bin/env python3
"""
Helper modules for PyTorch distributed training benchmark.

This module demonstrates:
- Separating model architecture into a dedicated module
- Reusable data loading utilities
- Configuration management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        out = F.relu(out)
        return out


class ResNetModel(nn.Module):
    """
    ResNet-style model for distributed training benchmark.
    
    This is a more realistic model architecture compared to SimpleCNN,
    demonstrating residual connections and deeper networks.
    """
    def __init__(self, num_classes=1000, num_blocks=[2, 2, 2, 2]):
        super(ResNetModel, self).__init__()
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        """Create a layer with multiple residual blocks"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.pool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class SyntheticDataset:
    """
    Synthetic dataset generator for benchmarking.
    
    Generates random data on-the-fly to avoid I/O bottlenecks
    and provide consistent benchmarking results.
    """
    def __init__(self, num_samples, batch_size, image_size=224, num_classes=1000):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_batches = num_samples // batch_size
    
    def generate_batch(self, device):
        """Generate a synthetic batch of images and labels"""
        images = torch.randn(self.batch_size, 3, self.image_size, 
                            self.image_size, device=device)
        labels = torch.randint(0, self.num_classes, (self.batch_size,), 
                              device=device)
        return images, labels
    
    def __len__(self):
        return self.num_batches


class BenchmarkConfig:
    """Configuration for distributed training benchmark"""
    def __init__(self):
        # Training hyperparameters
        self.batch_size = 128
        self.num_epochs = 5
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.weight_decay = 1e-4
        
        # Data configuration
        self.image_size = 224
        self.num_classes = 1000
        self.num_batches = 100
        
        # Model configuration
        self.model_type = "resnet"  # or "simple_cnn"
        self.resnet_blocks = [2, 2, 2, 2]  # ResNet-18 style
    
    def __str__(self):
        return (
            f"BenchmarkConfig(\n"
            f"  batch_size={self.batch_size},\n"
            f"  num_epochs={self.num_epochs},\n"
            f"  learning_rate={self.learning_rate},\n"
            f"  image_size={self.image_size},\n"
            f"  num_classes={self.num_classes},\n"
            f"  model_type={self.model_type}\n"
            f")"
        )


def print_distributed_info(rank, local_rank, world_size):
    """Print distributed training information"""
    import socket
    import os
    
    print(f"\n[Rank {rank}] Distributed Training Info:")
    print(f"  Hostname: {socket.gethostname()}")
    print(f"  Global Rank: {rank}")
    print(f"  Local Rank: {local_rank}")
    print(f"  World Size: {world_size}")
    print(f"  Master Addr: {os.environ.get('MASTER_ADDR', 'N/A')}")
    print(f"  Master Port: {os.environ.get('MASTER_PORT', 'N/A')}")


def print_gpu_info(rank, device):
    """Print GPU information"""
    if torch.cuda.is_available():
        print(f"\n[Rank {rank}] GPU Info:")
        print(f"  Device: {device}")
        print(f"  GPU Name: {torch.cuda.get_device_name(device)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    else:
        print(f"\n[Rank {rank}] Warning: CUDA not available, using CPU")


def calculate_model_size(model):
    """Calculate total number of parameters in model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def greet(name):
    """Simple greeting function (for backward compatibility)"""
    print(f"Hello from helper module! Greeting: {name}")
