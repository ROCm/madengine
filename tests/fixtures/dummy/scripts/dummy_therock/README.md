# dummy_therock - PyTorch Benchmark with TheRock

## Overview

This model benchmarks PyTorch ResNet50 training performance using [TheRock](https://github.com/ROCm/TheRock), AMD's lightweight open source build system for HIP and ROCm.

## What is TheRock?

TheRock (The HIP Environment and ROCm Kit) is AMD's modern distribution system for ROCm, released as version 7.10 in December 2025. Unlike traditional ROCm installations via apt packages, TheRock distributes ROCm components as Python pip packages, making it lightweight and easy to integrate.

## Benchmark Details

- **Model**: ResNet50 (image classification)
- **Task**: Training with synthetic data
- **Batch Size**: 64 images
- **Iterations**: 100 training steps
- **Image Size**: 224x224
- **Metric**: Images per second (throughput)

## Files

```
dummy_therock/
├── docker/dummy_therock.ubuntu.amd.Dockerfile  # Docker image with rocm/pytorch
├── scripts/dummy_therock/
│   ├── run.sh                                   # Main entry point
│   ├── train_resnet.py                          # ResNet50 training benchmark
│   └── README.md                                # This file
```

## Usage

### With madengine

```bash
# Build and run the model
cd /path/to/madengine
python3 -m madengine.cli.run_models \
    --models-json tests/fixtures/dummy/models.json \
    --tags dummy_therock

# Or run with specific GPU count
python3 -m madengine.cli.run_models \
    --models-json tests/fixtures/dummy/models.json \
    --model-name dummy_therock \
    --n-gpus 1
```

### Standalone

```bash
# Build Docker image
docker build -f tests/fixtures/dummy/docker/dummy_therock.ubuntu.amd.Dockerfile \
    -t dummy_therock .

# Run benchmark
docker run --rm --device=/dev/kfd --device=/dev/dri \
    --network host --ipc=host --group-add video \
    -v $(pwd)/tests/fixtures/dummy/scripts/dummy_therock:/workspace/scripts \
    dummy_therock \
    bash /workspace/scripts/run.sh
```

## Expected Output

The benchmark will output:

```
========================================================================
ResNet50 Training Benchmark with TheRock
========================================================================

=== PyTorch Configuration ===
PyTorch: 2.x.x
CUDA Available: True
HIP: 6.x.xxxxx

========================================================================
======================================================================
ResNet50 Training Benchmark (TheRock)
======================================================================
Device: cuda:0
GPU: AMD Instinct MI300X
GPU Count: 1

Creating ResNet50 model...
Batch Size: 64
Iterations: 100
Image Size: 224x224

Warming up (10 iterations)...
Running benchmark (100 iterations)...
  Progress: 20/100
  Progress: 40/100
  Progress: 60/100
  Progress: 80/100
  Progress: 100/100

======================================================================
Benchmark Results:
  Total Images Processed: 6400
  Duration: 45.23 seconds
  Throughput: 141.52 images/sec
======================================================================

performance: 141.52 images_per_second
```

## Performance Metrics

The model reports performance in the madengine standard format:

```
performance: <value> images_per_second
```

This metric is automatically captured by madengine and written to `perf.csv`.

## Tags

- `dummies` - Test/dummy model
- `therock` - Uses TheRock ROCm distribution
- `pytorch` - PyTorch framework
- `rocm` - AMD ROCm platform

## Notes

- Based on `rocm/pytorch:latest` which uses TheRock's ROCm distribution
- Runs a real ResNet50 training workload (not just dummy output)
- Suitable for validating PyTorch + ROCm functionality
- Performance varies by GPU architecture (MI300X, MI250X, etc.)

