#!/usr/bin/env bash
# 
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
# 
# Start gpu_info_power_profiler in background mode

set -x

echo "Starting GPU power profiler in background..."

# Get environment variables from tools.json (with POWER_ prefix to avoid conflicts)
DEVICE=${POWER_DEVICE:-"all"}
SAMPLING_RATE=${POWER_SAMPLING_RATE:-"0.1"}
MODE=${POWER_MODE:-"power"}
OUTPUT_FILE=${POWER_OUTPUT_FILE:-"gpu_info_power_profiler_output.csv"}
DUAL_GCD=${POWER_DUAL_GCD:-"false"}

# Export environment variables for the profiler (without prefix for the profiler script)
export DEVICE
export SAMPLING_RATE
export MODE
export OUTPUT_FILE
export DUAL_GCD

# Create a marker file to track profiler status
PROFILER_PID_FILE="/tmp/gpu_info_power_profiler.pid"
PROFILER_START_FILE="/tmp/gpu_info_power_profiler.started"

# Start profiler in background using a wrapper approach
# The profiler will run "tail -f /dev/null" as a dummy command that runs forever
# We'll kill it in the post-script after the actual workload completes
echo "Launching power profiler..."
nohup python3 ../scripts/common/tools/gpu_info_profiler.py tail -f /dev/null > /tmp/gpu_info_power_profiler.log 2>&1 &
PROFILER_PID=$!

# Save PID for later termination
echo "$PROFILER_PID" > "$PROFILER_PID_FILE"
echo "✓ GPU power profiler started (PID: $PROFILER_PID)"

# Give profiler time to initialize
sleep 2

# Touch start marker
touch "$PROFILER_START_FILE"

echo "✓ GPU power profiler initialization complete"

