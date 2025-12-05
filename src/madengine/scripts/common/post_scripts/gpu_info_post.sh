#!/usr/bin/env bash
# 
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
# 

set -e
set -x

tool=$1

# Output filename is tool_output.csv (e.g., gpu_info_power_profiler_output.csv)
OUTPUT=${tool}_output.csv

# In Docker local execution, prof.csv is in current directory (run_directory)
# In K8s execution, prof.csv is also in current directory (/workspace)
echo "Current directory: $(pwd)"
echo "Looking for profiler output..."

# Check if the profiler already wrote to the final output file
# (This happens when OUTPUT_FILE env var is set in tools.json)
if [ -f "$OUTPUT" ]; then
    echo "✓ Profiler output already exists: $OUTPUT"
    chmod a+rw "${OUTPUT}"
    echo "Profiler output saved to: $(pwd)/${OUTPUT}"
    exit 0
fi

# Otherwise, look for prof.csv (default output name) and rename it
echo "Looking for prof.csv..."
ls -la prof.csv 2>/dev/null || echo "prof.csv not found in current directory"

if [ ! -f "prof.csv" ]; then
    echo "Error: Neither $OUTPUT nor prof.csv found in $(pwd)"
    echo "Directory contents:"
    ls -la
    exit 1
fi

# Move the profiler output to the final location
mv prof.csv "$OUTPUT"

chmod a+rw "${OUTPUT}"

echo "Profiler output saved to: $(pwd)/${OUTPUT}"
