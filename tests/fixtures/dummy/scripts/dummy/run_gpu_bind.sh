#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#

node_ids=()


if [[ "$MAD_GPU_VENDOR" == *"AMD"* ]]; then
    if [[ $MAD_SYSTEM_HIP_VERSION < 6.1 ]]; then
        output=$(rocm-smi --showuniqueid)
        while IFS= read -r line; do
            if [[ $line == *"Unique ID"* ]]; then
                unique_id=$(echo $line | awk -F ':' '{print $3}' | awk '{$1=$1};1')
                # use unique id as node id is available from ROCm 6.1
                node_ids+=($unique_id)
            fi
        done <<< "$output"
    else
        output=$(grep -r drm_render_minor /sys/devices/virtual/kfd/kfd/topology/nodes 2>/dev/null)
        while IFS= read -r line; do
            if [[ $line != *"Operation not permitted"* ]] && [[ ! $line =~ \ 0$ ]]; then
                node_id=$(echo $line | sed -n 's|.*/\([0-9]\+\)/.*|\1|p')
                node_ids+=($node_id)
            fi
        done <<< "$output"
    fi
else
    output=$(nvidia-smi --list-gpus)
    while IFS= read -r line; do
        if [[ $line =~ UUID:\ (.*)\) ]]; then
            node_ids+=("${BASH_REMATCH[1]}")
        fi
    done <<< "$output"
fi


echo 'model,performance,metric' > results_dummy_gpubind.csv
for i in "${!node_ids[@]}"
do
    echo 'gpu'${i}','${node_ids[$i]}',gpu_bound' >> results_dummy_gpubind.csv
done
cp results_dummy_gpubind.csv ../
