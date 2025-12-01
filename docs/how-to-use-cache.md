# How to Use Run Caching

madengine includes a powerful caching system that saves Docker images and run configurations, allowing you to speed up repeated model runs significantly.

## Overview

The caching system provides **automatic cache matching** - just add `--enable-cache` to your existing command and it handles everything automatically!

Key features:
- **Automatic**: No IDs to remember, no manual steps
- **Fast**: 5-50x speedup on repeated runs
- **Smart**: Exact matching based on full configuration
- **Simple**: Just add one flag to your existing commands
- Uses LRU (Least Recently Used) eviction policy
- Works with multi-node configurations

## Basic Usage - Automatic Cache Matching

### The Simple Way (Recommended)

**Just add `--enable-cache` to your existing command!**

```bash
# First run: builds and caches
madengine run --tags pyt_huggingface_bert --enable-cache --live-output
# Takes: 3-15 minutes (normal build time)
# Output: "Docker image saved to cache: run_1701234567_abc123"

# Second run: AUTOMATICALLY uses cache - SAME COMMAND!
madengine run --tags pyt_huggingface_bert --enable-cache --live-output
# Takes: 10-60 seconds (loads from cache)
# Output: "Cache hit! Loading Docker image from cache..."
```

**That's it!** No cache IDs to remember, no extra steps. The cache system automatically:
1. ✅ Checks if a matching cache entry exists
2. ✅ Loads the cached Docker image if found
3. ✅ Builds and caches if not found

### How Automatic Matching Works

The cache system generates a unique key based on:
- Model tags
- Docker image name
- Additional context parameters
- Dockerfile path
- All configuration settings

**Same configuration = Cache hit!**

```bash
# These will use the same cache:
madengine run --tags bert --enable-cache
madengine run --tags bert --enable-cache  # Cache hit!

# These will use DIFFERENT caches:
madengine run --tags bert --enable-cache
madengine run --tags bert --enable-cache --additional-context "{'guest_os': 'CENTOS'}"  # Different config, different cache
```

### What You'll See

**First run (cache miss):**
```
Building Docker image...
Build Duration: 245.3 seconds
Saving Docker image to cache...
Docker image saved to cache: run_1701234567_abc123
```

**Second run (cache hit):**
```
Cache hit! Loading Docker image from cache: run_1701234567_abc123
Loading Docker image from: ~/.madengine/cache/run_1701234567_abc123/docker_image.tar
Docker image loaded from cache successfully
```

## Real-World Examples

### Development Workflow

```bash
# Developing a model - run multiple times
madengine run --tags my_model --enable-cache --live-output
# First run: 10 minutes (build)

# Make code changes, test again
madengine run --tags my_model --enable-cache --live-output
# Subsequent runs: 30 seconds (cache hit!)

# Test with different context
madengine run --tags my_model --enable-cache --additional-context "{'guest_os': 'CENTOS'}" --live-output
# Different config: 10 minutes (new cache entry)

# Back to original config
madengine run --tags my_model --enable-cache --live-output
# Same config as first run: 30 seconds (cache hit!)
```

### CI/CD Pipeline

```yaml
# .gitlab-ci.yml or similar
test_model:
  script:
    # Just add --enable-cache to your existing command!
    - madengine run --tags $MODEL_NAME --enable-cache --output results.csv
  # First pipeline run: builds and caches
  # Subsequent runs: loads from cache automatically
```

### Benchmarking

```bash
# Run same model with different parameters
madengine run --tags bert --enable-cache --additional-context "{'batch_size': '32'}"
madengine run --tags bert --enable-cache --additional-context "{'batch_size': '64'}"
madengine run --tags bert --enable-cache --additional-context "{'batch_size': '128'}"
# Each configuration gets its own cache entry
# Re-running same config uses cache automatically
```

## Advanced: Run From Cache by ID (Optional)

If you need to explicitly run from a specific cache ID, you can use the `--from-cache` flag:

```bash
# Step 1: List cache entries to get cache ID
madengine cache list

# Step 2: Run directly from cache (loads image and runs model)
madengine run --from-cache run_1701234567_abc123 --live-output
```

This single command:
- Loads the Docker image from cache (if not already loaded)
- Automatically uses the cached model tags
- Runs the model immediately
- No need to specify tags or rebuild anything!

**Example workflow:**
```bash
# Initial run with caching
madengine run --tags pyt_huggingface_bert --enable-cache --live-output
# Output: Docker image saved to cache: run_1701234567_abc123

# Later, run from cache with one command
madengine run --from-cache run_1701234567_abc123 --live-output
# Loads cached image and runs immediately!
```

### Cache Hit vs Cache Miss

**Cache Hit** (configuration matches existing cache):
```
Cache hit! Loading Docker image from cache: run_1234567890_abc123
Loading Docker image from: ~/.madengine/cache/run_1234567890_abc123/docker_image.tar
Docker image loaded from cache successfully
```

**Cache Miss** (new or different configuration):
```
Cache miss. Will build Docker image and save to cache.
Building Docker image...
Build Duration: 245.3 seconds
Saving Docker image to cache...
Docker image saved to cache: run_1234567890_abc123
```

## Run From Cache - Detailed Guide

### Why Use --from-cache?

The `--from-cache` flag provides a seamless way to run models from cache:

**Benefits:**
- ✅ **One Command**: Load image and run model in single command
- ✅ **No Tags Needed**: Automatically uses cached model tags
- ✅ **Fast**: Skips discovery and uses cached Docker image
- ✅ **Simple**: Just specify cache ID, nothing else

### How It Works

When you use `--from-cache <cache_id>`:

1. **Loads Cache Entry**: Retrieves configuration from cache
2. **Checks Docker Image**: If image exists locally, uses it immediately
3. **Loads If Needed**: If image not found, loads from tar file
4. **Sets Configuration**: Applies cached tags and settings
5. **Runs Model**: Executes the model with cached setup

### Getting Cache IDs

First, find available cache entries:

```bash
madengine cache list
```

Output:
```
5 cache entries:
====================================================================================================
ID: run_1701234567_abc123
  Image: ci-pyt_huggingface_bert_ubuntu
  Tags: pyt_huggingface_bert
  Created: 2024-12-01T10:30:00
  Last Used: 2024-12-01T15:45:00
  Size: 3.24 GB
  Nodes: 1
----------------------------------------------------------------------------------------------------
```

Copy the cache ID (e.g., `run_1701234567_abc123`) and use it with `--from-cache`.

### Running From Cache

```bash
# Basic usage
madengine run --from-cache run_1701234567_abc123

# With live output (recommended)
madengine run --from-cache run_1701234567_abc123 --live-output

# With custom output file
madengine run --from-cache run_1701234567_abc123 --output my_results.csv
```

### What You'll See

When running from cache:

```
Running from cache: run_1701234567_abc123
================================================================================
Cache Entry: run_1701234567_abc123
Docker Image: ci-pyt_huggingface_bert_ubuntu
Model Tags: pyt_huggingface_bert
Nodes: 1
Created: 2024-12-01T10:30:00
Last Used: 2024-12-01T15:45:00
================================================================================
Docker image 'ci-pyt_huggingface_bert_ubuntu' already exists locally

Using cached tags: pyt_huggingface_bert

================================================================================
Running model with cached Docker image...
================================================================================

[Model execution output...]

================================================================================
Model ran successfully from cache!
================================================================================
```

### Error Handling

If cache ID not found:
```bash
$ madengine run --from-cache invalid_id

Error: Cache entry 'invalid_id' not found

Available cache entries:
  - run_1701234567_abc123: ci-pyt_huggingface_bert_ubuntu (tags: pyt_huggingface_bert)
  - run_1701234568_def456: ci-pyt_torchvision_alexnet_centos (tags: pyt_torchvision_alexnet)

Tip: Use 'madengine cache list' to see all available cache entries
```

### Combining With Other Flags

You can combine `--from-cache` with other flags:

```bash
# With live output
madengine run --from-cache <cache_id> --live-output

# With custom output file
madengine run --from-cache <cache_id> --output custom.csv

# With keep-alive (keep container running)
madengine run --from-cache <cache_id> --keep-alive

# With custom cache directory
madengine run --from-cache <cache_id> --cache-dir /custom/cache

# Multiple flags
madengine run --from-cache <cache_id> --live-output --keep-alive --output results.csv
```

### Comparison: Three Ways to Use Cache

**1. Automatic Cache Matching (--enable-cache) ⭐ RECOMMENDED:**
```bash
madengine run --tags bert --enable-cache
# On first run: builds and caches
# On second run with SAME command: automatic cache hit!
# SIMPLEST: Just add --enable-cache to your existing commands
```

**2. Run From Cache by ID (--from-cache) - Advanced:**
```bash
madengine run --from-cache run_1701234567_abc123
# Explicitly runs from specific cache ID
# Useful for: running old configs, sharing cache IDs
```

**3. Standalone Executable - For Distribution:**
```bash
./dist/run_models run_1701234567_abc123
# Runs on systems without Python/madengine
# Useful for: portable execution, deployments
```

**Recommendation:** Use **automatic matching** (`--enable-cache`) for regular work. It's the simplest and requires no manual tracking of cache IDs!

## Configuration Options

### Cache Settings

Customize cache behavior with command-line flags:

```bash
# Custom cache directory
madengine run --tags <model> --enable-cache --cache-dir /path/to/cache

# Increase entry limit
madengine run --tags <model> --enable-cache --cache-max-entries 10

# Increase size limit (in GB)
madengine run --tags <model> --enable-cache --cache-max-size-gb 200

# Combine options
madengine run --tags <model> --enable-cache \
  --cache-max-entries 20 \
  --cache-max-size-gb 500 \
  --cache-dir /mnt/fast-storage/madengine-cache
```

### Force Rebuild

To rebuild and update cache (ignore existing cache):

```bash
madengine run --tags <model> --enable-cache --force-rebuild
```

## Cache Management

### List Cache Entries

View all cached runs:

```bash
madengine cache list
```

Output:
```
5 cache entries:
====================================================================================================
ID: run_1701234567_abc123
  Image: ci-pyt_huggingface_bert_ubuntu
  Tags: pyt_huggingface_bert
  Created: 2024-12-01T10:30:00
  Last Used: 2024-12-01T15:45:00
  Size: 3.24 GB
  Nodes: 1
----------------------------------------------------------------------------------------------------
...
```

### Cache Statistics

View cache usage statistics:

```bash
madengine cache stats
```

Output:
```
Cache Statistics:
============================================================
Cache Directory: /home/user/.madengine/cache
Total Entries: 5 / 5
Total Size: 15.67 GB / 100.00 GB
Usage: 15.7%
============================================================
```

### Show Cache Entry Details

View detailed information about a specific cache entry:

```bash
madengine cache info run_1701234567_abc123
```

Output:
```
Cache Entry: run_1701234567_abc123
================================================================================
Cache Key: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
Docker Image: ci-pyt_huggingface_bert_ubuntu
Docker Tar: /home/user/.madengine/cache/run_1701234567_abc123/docker_image.tar
Model Tags: pyt_huggingface_bert
Created: 2024-12-01T10:30:00
Last Used: 2024-12-01T15:45:00
Size: 3.24 GB
Nodes: 1
Cache Directory: /home/user/.madengine/cache/run_1701234567_abc123

Run Configuration:
{
  "model_name": "pyt_huggingface_bert",
  "dockerfile": "/path/to/Dockerfile",
  ...
}
```

### Clear Cache

Remove all cache entries:

```bash
# With confirmation prompt
madengine cache clear

# Skip confirmation
madengine cache clear -y
```

### Evict Specific Entry

Remove a specific cache entry:

```bash
madengine cache evict run_1701234567_abc123
```

## Cache Matching

The cache uses **exact matching** based on:
- Docker image name
- Model tags
- Additional context (all parameters)
- Dockerfile path

Any change in these parameters will result in a cache miss and create a new cache entry.

### Examples

**Same run (cache hit):**
```bash
# First run
madengine run --tags bert --enable-cache --additional-context "{'guest_os': 'UBUNTU'}"

# Second run (cache hit)
madengine run --tags bert --enable-cache --additional-context "{'guest_os': 'UBUNTU'}"
```

**Different context (cache miss):**
```bash
# First run
madengine run --tags bert --enable-cache --additional-context "{'guest_os': 'UBUNTU'}"

# Different OS (cache miss, creates new entry)
madengine run --tags bert --enable-cache --additional-context "{'guest_os': 'CENTOS'}"
```

## Automatic Eviction

When cache limits are exceeded, the system automatically evicts the **least recently used** entries:

### By Entry Count
Default: 5 entries maximum

When you add the 6th entry:
1. Oldest (least recently used) entry is evicted
2. New entry is added

### By Total Size
Default: 100GB maximum

When total cache size exceeds limit:
1. Oldest entries are evicted until size is below limit
2. New entry is added

## Multi-Node Support

The cache system works with multi-node configurations:

```bash
# Cache multi-node run
madengine run --tags multi_node_model --enable-cache

# Cache entry stores node_count
madengine cache info <cache_id>
# Shows: Nodes: 4
```

## Standalone Executable

Build a portable executable that can run cached models on systems without Python or madengine installed.

### Building the Executable

```bash
# Install PyInstaller
pip install pyinstaller

# Build executable
./build_executable.sh
```

This creates:
- `dist/run_models` (Linux/Mac)
- `dist/run_models.exe` (Windows)

### Using the Executable

```bash
# List cache entries
./dist/run_models --list

# Show cache info
./dist/run_models --info run_1701234567_abc123

# Load Docker image from cache
./dist/run_models run_1701234567_abc123
```

The executable:
- Loads the Docker image from cached tar file
- Updates last used timestamp
- Provides instructions for running the model

### Distribution

The executable can be distributed to other systems:
1. Copy the executable
2. Copy the cache directory (optional, or create new cache)
3. Run without needing Python/madengine installation

## Performance Benefits

### Time Savings

Typical Docker build times:
- Initial build: 3-15 minutes (depending on model)
- Cached load: 10-60 seconds (depending on image size)

Speedup: **5-50x faster**

### Disk Space

Cached tar files:
- Small images: 500MB - 2GB
- Medium images: 2GB - 5GB
- Large images: 5GB - 15GB

With default 100GB limit and 5 entry limit, you can cache:
- ~20 small images, or
- ~10 medium images, or
- ~6-7 large images

## Best Practices

### When to Use Caching

✅ **Good use cases:**
- Development: Testing different models/configs repeatedly
- CI/CD: Running same models across multiple pipelines
- Benchmarking: Comparing performance with consistent environments
- Training: Iterating on model parameters

❌ **Avoid caching when:**
- Running one-off experiments
- Disk space is extremely limited
- Images change frequently (always cache miss)

### Cache Directory Location

Choose fast storage:
```bash
# Use SSD/NVMe for best performance
madengine run --tags <model> --enable-cache --cache-dir /mnt/nvme/madengine-cache
```

### Monitoring Cache

Regularly check cache statistics:
```bash
# Add to your workflow
madengine cache stats

# Clean up when needed
madengine cache clear
```

### Multi-User Environments

For shared systems, use per-user cache directories:
```bash
# User-specific cache
export MADENGINE_CACHE_DIR=/shared/cache/$USER
madengine run --tags <model> --enable-cache
```

## Troubleshooting

### Cache Miss When Expected Hit

Check that all parameters match:
```bash
# Show what's in cache
madengine cache list

# Compare with your run parameters
madengine cache info <cache_id>
```

### Failed to Load from Cache

If Docker image fails to load:
1. Cache will automatically fall back to building
2. Old cache entry is kept but not used
3. New cache entry is created after successful build

### Disk Space Issues

If cache fills up:
```bash
# Check current usage
madengine cache stats

# Clear specific entries
madengine cache evict <cache_id>

# Or clear all
madengine cache clear
```

### Corrupted Cache

If cache index is corrupted:
```bash
# Backup and remove cache
mv ~/.madengine/cache ~/.madengine/cache.backup
madengine cache stats  # Creates new cache
```

## Environment Variables

Configure cache behavior via environment variables:

```bash
# Override default cache directory
export MADENGINE_CACHE_DIR=/path/to/cache

# Enable caching by default
export MADENGINE_CACHE_ENABLED=1

# Set maximum cache size
export MADENGINE_CACHE_MAX_SIZE_GB=200
```

## Advanced Usage

### Programmatic Cache Access

Access cache from Python:

```python
from madengine.core.cache import CacheManager

# Initialize cache manager
cache_mgr = CacheManager(cache_dir="/path/to/cache")

# List entries
entries = cache_mgr.list_cache_entries()

# Get entry
entry = cache_mgr.get_cache_entry_by_id("run_1234567890_abc123")

# Load docker image
from madengine.core.docker import Docker
Docker.load_image_from_tar(entry.docker_tar_path)
```

### Custom Cache Keys

Generate cache keys programmatically:

```python
cache_key = cache_mgr.generate_cache_key(
    docker_image="ci-model",
    model_tags=["tag1", "tag2"],
    additional_context={"key": "value"},
    dockerfile="/path/to/Dockerfile"
)
```

## Summary

The madengine caching system provides:
- ⚡ **Speed**: 5-50x faster repeated runs
- 💾 **Smart Storage**: LRU eviction with size limits
- 🔧 **Flexibility**: Exact matching with full configuration
- 🚀 **Portability**: Standalone executable for distribution
- 📊 **Management**: CLI tools for monitoring and cleanup

Enable caching to accelerate your model development and testing workflows!
