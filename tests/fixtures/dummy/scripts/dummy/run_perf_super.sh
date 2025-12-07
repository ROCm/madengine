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

# Generate results matching config rows
echo "model,performance,metric,status
dummy/model-1,1234.56,tokens/s,SUCCESS
dummy/model-2,2345.67,requests/s,SUCCESS
dummy/model-3,345.78,ms,SUCCESS" > perf_dummy_super.csv

cp perf_dummy_super.csv ../

