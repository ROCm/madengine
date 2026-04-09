#!/usr/bin/env bash
# 
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
# 
# Stop gpu_info_vram_profiler and collect output

set -x

echo "Stopping GPU VRAM profiler..."

PROFILER_PID_FILE="/tmp/gpu_info_vram_profiler.pid"
PROFILER_START_FILE="/tmp/gpu_info_vram_profiler.started"

# Check if profiler was started
if [ ! -f "$PROFILER_START_FILE" ]; then
    echo "⚠️  Warning: VRAM profiler was not started - skipping"
    exit 0
fi

# Check if PID file exists
if [ ! -f "$PROFILER_PID_FILE" ]; then
    echo "⚠️  Warning: VRAM profiler PID file not found - profiler may not be running"
    exit 0
fi

# Read PID
PROFILER_PID=$(cat "$PROFILER_PID_FILE")

# Check if process is still running
if ! kill -0 "$PROFILER_PID" 2>/dev/null; then
    echo "⚠️  Warning: VRAM profiler process (PID: $PROFILER_PID) is not running"
else
    echo "Sending termination signal to VRAM profiler (PID: $PROFILER_PID)..."
    
    # Send SIGTERM to gracefully stop the profiler
    kill -TERM "$PROFILER_PID" 2>/dev/null || true
    
    # Wait for profiler to finish writing output (max 10 seconds)
    WAIT_COUNT=0
    while kill -0 "$PROFILER_PID" 2>/dev/null && [ $WAIT_COUNT -lt 20 ]; do
        sleep 0.5
        WAIT_COUNT=$((WAIT_COUNT + 1))
    done
    
    # Force kill if still running
    if kill -0 "$PROFILER_PID" 2>/dev/null; then
        echo "⚠️  Profiler did not stop gracefully, force killing..."
        kill -9 "$PROFILER_PID" 2>/dev/null || true
    fi
    
    echo "✓ GPU VRAM profiler stopped"
fi

# Clean up temporary files
rm -f "$PROFILER_PID_FILE" "$PROFILER_START_FILE"

echo "✓ VRAM profiler cleanup complete"

# Show profiler log if it exists
if [ -f "/tmp/gpu_info_vram_profiler.log" ]; then
    echo "=== VRAM Profiler Log ==="
    tail -20 /tmp/gpu_info_vram_profiler.log || true
    echo "=========================="
fi

