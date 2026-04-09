#!/usr/bin/env bash
# 
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
# 

set -x

tool=$1

# Output filename is tool_output.csv (e.g., gpu_info_power_profiler_output.csv)
OUTPUT=${tool}_output.csv

# In Docker local execution, prof.csv is in current directory (run_directory)
# In K8s execution, prof.csv is also in current directory (/workspace)
echo "Current directory: $(pwd)"
echo "Looking for profiler output for tool: $tool..."

# Check if the profiler already wrote to the final output file
# (This happens when OUTPUT_FILE env var is set in tools.json)
if [ -f "$OUTPUT" ]; then
    echo "✓ Profiler output already exists: $OUTPUT"
    chmod a+rw "${OUTPUT}"
    echo "Profiler output saved to: $(pwd)/${OUTPUT}"
    exit 0
fi

# When multiple gpu_info tools are stacked together, they may create their outputs
# with different filenames. Look for the specific output file by checking common locations.

# Check if any profiler output files exist
echo "Looking for any *_profiler_output.csv files..."
ls -la *_profiler_output.csv 2>/dev/null || echo "No *_profiler_output.csv files found"

# When tools are stacked, one tool might have created its output file while another didn't
# This is expected behavior - don't fail the entire run
if [ ! -f "$OUTPUT" ]; then
    echo "⚠️  Warning: $OUTPUT not found in $(pwd)"
    echo "⚠️  This may be expected if multiple gpu_info tools are stacked together"
    echo "⚠️  and only one ran successfully. Checking for any profiler outputs..."
    
    # Check if prof.csv exists (default output name)
    if [ -f "prof.csv" ]; then
        echo "Found prof.csv - renaming to $OUTPUT"
        mv prof.csv "$OUTPUT"
        chmod a+rw "${OUTPUT}"
        echo "Profiler output saved to: $(pwd)/${OUTPUT}"
        exit 0
    fi
    
    # List all CSV files for debugging
    echo "Available CSV files in directory:"
    ls -la *.csv 2>/dev/null || echo "No CSV files found"
    
    # Don't fail - just warn and exit successfully
    # This allows other stacked tools to complete their post-scripts
    echo "⚠️  Profiler output $OUTPUT not found - skipping (non-fatal)"
    exit 0
fi

# If we get here, OUTPUT exists but wasn't caught by the first check
chmod a+rw "${OUTPUT}"
echo "Profiler output saved to: $(pwd)/${OUTPUT}"
