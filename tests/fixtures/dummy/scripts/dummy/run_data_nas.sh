#!/bin/bash
# 
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
# 

if [ -z ${MAD_DATAHOME+x} ]; then
    echo "MAD_DATAHOME is NOT set"
    exit 1
else
    echo "MAD_DATAHOME is set"
fi

# Check if data location exists (either mounted or downloaded)
if [ ! -d "${MAD_DATAHOME}" ]; then
    echo "${MAD_DATAHOME} directory does not exist"
    exit 1
fi

# Check if it's a mounted filesystem (for traditional NAS)
mountCode=`mount | grep "${MAD_DATAHOME}"`

if [ -z "$mountCode" ]; then
    echo "${MAD_DATAHOME} is NOT mounted (data downloaded to directory)"
    # For K8s/downloaded data, check if directory has content
    if [ -n "$(ls -A ${MAD_DATAHOME} 2>/dev/null)" ]; then
        echo "${MAD_DATAHOME} has data (downloaded)"
        echo "performance: $RANDOM samples_per_second"
    else
        echo "${MAD_DATAHOME} is empty (test environment - data provider works but source is empty)"
        echo "performance: $RANDOM samples_per_second (simulated)"
    fi
else
    echo "${MAD_DATAHOME} is mounted"
    echo "performance: $RANDOM samples_per_second"
fi


