#!/bin/bash
# madengine K8s Data Provider - Local
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# Usage: download_local.sh <dataname> <datapath> <datahome>

set -e

DATANAME=$1
DATAPATH=$2
DATAHOME=${3:-/data_dlm_0}

echo "=== Local Data Provider ==="
echo "Data: $DATANAME"
echo "Path: $DATAPATH"
echo "Target: $DATAHOME"

# For local data, the path should already be mounted as a volume
# Just verify it exists and calculate size

if [ ! -e "$DATAPATH" ]; then
    echo "Error: Local data path does not exist: $DATAPATH"
    exit 1
fi

# If DATAHOME is different from DATAPATH, we might need to symlink or the data is already mounted
if [ "$DATAPATH" != "$DATAHOME" ]; then
    echo "Note: Data is at $DATAPATH, expected at $DATAHOME"
    echo "Assuming data is pre-mounted by K8s volume"
fi

# Calculate size
SIZE=$(du -sh $DATAHOME 2>/dev/null | cut -f1 || du -sh $DATAPATH 2>/dev/null | cut -f1 || echo "0")

echo "✓ Local data verified"
echo "Size: $SIZE"

# Export metrics
mkdir -p /tmp
echo "MAD_DATA_DOWNLOAD_DURATION=0" >> /tmp/mad_metrics.env
echo "MAD_DATA_SIZE=$SIZE" >> /tmp/mad_metrics.env
echo "MAD_DATA_PROVIDER_TYPE=local" >> /tmp/mad_metrics.env
echo "MAD_DATANAME=$DATANAME" >> /tmp/mad_metrics.env

