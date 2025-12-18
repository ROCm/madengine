#!/bin/bash
#
# SLURM Epilog Script for GPU Cleanup
# 
# This script should be installed on SLURM compute nodes to ensure
# GPU processes are properly cleaned up after each job.
#
# Installation:
#   1. Copy this script to /etc/slurm/epilog.sh on all compute nodes
#   2. Make it executable: chmod +x /etc/slurm/epilog.sh
#   3. Add to /etc/slurm/slurm.conf:
#      Epilog=/etc/slurm/epilog.sh
#   4. Restart SLURM: sudo systemctl restart slurmd
#
# This script runs as root after each job completes/fails
#

LOG_FILE="/var/log/slurm/epilog.log"
mkdir -p "$(dirname "$LOG_FILE")"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Job ${SLURM_JOB_ID:-unknown}] $1" >> "$LOG_FILE"
}

log_message "=== Epilog script starting ==="

# Function to kill GPU processes
cleanup_gpu_processes() {
    log_message "Checking for GPU processes..."
    
    # Try AMD GPUs first
    if [ -x /opt/rocm/bin/amd-smi ]; then
        log_message "Detected AMD ROCm installation, checking for processes..."
        
        # Get PIDs using amd-smi
        PIDS=$(amd-smi process 2>/dev/null | grep -v PID | awk '{print $1}' | grep -E '^[0-9]+$' | sort -u)
        
        if [ ! -z "$PIDS" ]; then
            log_message "Found GPU processes to clean: $PIDS"
            for pid in $PIDS; do
                if ps -p $pid > /dev/null 2>&1; then
                    log_message "Killing GPU process: $pid"
                    kill -9 $pid 2>/dev/null || true
                    sleep 0.5
                fi
            done
        else
            log_message "No GPU processes found via amd-smi"
        fi
        
        # Try fuser on GPU devices as backup
        for device in /dev/kfd /dev/dri/renderD*; do
            if [ -e "$device" ]; then
                DEVICE_PIDS=$(fuser "$device" 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$')
                if [ ! -z "$DEVICE_PIDS" ]; then
                    log_message "Found processes using $device: $DEVICE_PIDS"
                    for pid in $DEVICE_PIDS; do
                        if ps -p $pid > /dev/null 2>&1; then
                            log_message "Killing process using $device: $pid"
                            kill -9 $pid 2>/dev/null || true
                        fi
                    done
                fi
            fi
        done
    fi
    
    # Try NVIDIA GPUs
    if [ -x /usr/bin/nvidia-smi ]; then
        log_message "Detected NVIDIA GPU installation, checking for processes..."
        
        PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+$')
        
        if [ ! -z "$PIDS" ]; then
            log_message "Found NVIDIA GPU processes to clean: $PIDS"
            for pid in $PIDS; do
                if ps -p $pid > /dev/null 2>&1; then
                    log_message "Killing NVIDIA GPU process: $pid"
                    kill -9 $pid 2>/dev/null || true
                    sleep 0.5
                fi
            done
        else
            log_message "No NVIDIA GPU processes found"
        fi
    fi
}

# Function to kill Ray processes
cleanup_ray_processes() {
    log_message "Cleaning up Ray processes..."
    
    # Kill Ray worker processes
    RAY_PIDS=$(pgrep -f "ray::" 2>/dev/null || true)
    if [ ! -z "$RAY_PIDS" ]; then
        log_message "Found Ray processes: $RAY_PIDS"
        pkill -9 -f "ray::" 2>/dev/null || true
        sleep 1
    else
        log_message "No Ray processes found"
    fi
    
    # Kill vLLM worker processes
    VLLM_PIDS=$(pgrep -f "RayWorkerWrapper" 2>/dev/null || true)
    if [ ! -z "$VLLM_PIDS" ]; then
        log_message "Found vLLM worker processes: $VLLM_PIDS"
        pkill -9 -f "RayWorkerWrapper" 2>/dev/null || true
        sleep 1
    else
        log_message "No vLLM worker processes found"
    fi
    
    # Kill any vllm processes
    VLLM_MAIN_PIDS=$(pgrep -f "vllm" 2>/dev/null || true)
    if [ ! -z "$VLLM_MAIN_PIDS" ]; then
        log_message "Found vLLM main processes: $VLLM_MAIN_PIDS"
        pkill -9 -f "vllm" 2>/dev/null || true
        sleep 1
    fi
}

# Function to clean Docker containers (if any are still running)
cleanup_docker_containers() {
    if command -v docker &> /dev/null; then
        log_message "Checking for stale Docker containers..."
        
        # Find containers that might be from madengine
        CONTAINERS=$(docker ps -q --filter "name=container_rocm" 2>/dev/null || true)
        if [ ! -z "$CONTAINERS" ]; then
            log_message "Found stale containers: $CONTAINERS"
            for container in $CONTAINERS; do
                log_message "Stopping container: $container"
                docker stop --time=5 "$container" 2>/dev/null || true
                docker rm -f "$container" 2>/dev/null || true
            done
        else
            log_message "No stale Docker containers found"
        fi
    fi
}

# Function to reset GPU state
reset_gpu_state() {
    log_message "Resetting GPU state..."
    
    # AMD GPU reset
    if [ -x /opt/rocm/bin/rocm-smi ]; then
        log_message "Resetting AMD GPUs..."
        /opt/rocm/bin/rocm-smi --gpureset 2>/dev/null || log_message "GPU reset failed (may require reboot)"
    fi
    
    # NVIDIA GPU reset (requires nvidia-smi)
    if [ -x /usr/bin/nvidia-smi ]; then
        log_message "Resetting NVIDIA GPUs..."
        nvidia-smi --gpu-reset -i 0 2>/dev/null || log_message "GPU reset failed (may require reboot)"
    fi
}

# Main cleanup sequence
log_message "Starting cleanup sequence for job ${SLURM_JOB_ID:-unknown}"

# Step 1: Kill Ray and vLLM processes first
cleanup_ray_processes

# Step 2: Clean Docker containers
cleanup_docker_containers

# Step 3: Kill any remaining GPU processes
cleanup_gpu_processes

# Step 4: Reset GPU state (optional, may cause brief GPU unavailability)
# Uncomment if needed:
# reset_gpu_state

log_message "=== Epilog script completed ==="

exit 0

