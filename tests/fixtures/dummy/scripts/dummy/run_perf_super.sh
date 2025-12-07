#!/bin/bash
# 
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
# 
# Script to generate dummy results for perf_entry_super testing

# Parse config argument
CONFIG_FILE=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) CONFIG_FILE="$2"; shift ;;
        *) echo "Unknown parameter: $1" ;;
    esac
    shift
done

# Generate comprehensive results with best-practice performance metrics
# Includes: latency percentiles, resource utilization, reliability metrics, and throughput
cat > perf_dummy_super.csv << 'EOF'
model,performance,metric,status,throughput,latency_mean_ms,latency_p50_ms,latency_p90_ms,latency_p95_ms,latency_p99_ms,gpu_memory_used_mb,gpu_memory_total_mb,gpu_utilization_percent,cpu_utilization_percent,total_time_seconds,warmup_iterations,measured_iterations,error_count,success_rate_percent,samples_processed
dummy/model-1,1234.56,tokens/s,SUCCESS,1234.56,8.1,7.9,12.3,15.2,22.8,12288,32768,85.3,42.1,120.5,10,100,0,100.0,123456
dummy/model-2,2345.67,requests/s,SUCCESS,2345.67,4.3,4.1,6.8,8.5,12.3,16384,32768,78.2,38.5,180.3,10,150,2,99.87,352350
dummy/model-3,345.78,ms,SUCCESS,28.92,345.78,340.5,425.3,512.7,678.9,8192,32768,92.1,55.3,240.8,5,50,0,100.0,1447
EOF

cp perf_dummy_super.csv ../

