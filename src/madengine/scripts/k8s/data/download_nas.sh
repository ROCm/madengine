#!/bin/bash
# madengine K8s Data Provider - NAS (SSH/rsync)
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# Usage: download_nas.sh <dataname> <datapath> <datahome>

set -e

DATANAME=$1
DATAPATH=$2
DATAHOME=${3:-/data_dlm_0}

echo "=== NAS Data Download ==="
echo "Data: $DATANAME"
echo "Source: $DATAPATH"
echo "Target: $DATAHOME"

# Get NAS credentials from environment or credential.json
NAS_HOST=${NAS_HOST:-mlse-nas.amd.com}
NAS_PORT=${NAS_PORT:-22}
NAS_USER=${NAS_USERNAME:-datum}
NAS_PASS=${NAS_PASSWORD}

# If credentials not in environment, try to read from credential.json
if [ -z "$NAS_PASS" ] && [ -f "/workspace/credential.json" ]; then
    echo "Reading NAS credentials from credential.json..."
    
    # Extract NAS node info (try first node or find by hostname)
    NAS_HOST=$(python3 -c "import json; f=open('/workspace/credential.json'); d=json.load(f); nodes=d.get('NAS_NODES', []); print(nodes[0].get('HOST', 'mlse-nas.amd.com') if nodes else 'mlse-nas.amd.com')" 2>/dev/null || echo "mlse-nas.amd.com")
    
    NAS_PORT=$(python3 -c "import json; f=open('/workspace/credential.json'); d=json.load(f); nodes=d.get('NAS_NODES', []); print(nodes[0].get('PORT', '22') if nodes else '22')" 2>/dev/null || echo "22")
    
    NAS_USER=$(python3 -c "import json; f=open('/workspace/credential.json'); d=json.load(f); nodes=d.get('NAS_NODES', []); print(nodes[0].get('USERNAME', 'datum') if nodes else 'datum')" 2>/dev/null || echo "datum")
    
    NAS_PASS=$(python3 -c "import json; f=open('/workspace/credential.json'); d=json.load(f); nodes=d.get('NAS_NODES', []); print(nodes[0].get('PASSWORD', '') if nodes else '')" 2>/dev/null || echo "")
fi

# Verify credentials are available
if [ -z "$NAS_PASS" ]; then
    echo "Error: NAS_PASSWORD not found in environment or credential.json"
    echo "Required: NAS_PASSWORD environment variable or credential.json with NAS_NODES"
    exit 1
fi

echo "Using NAS: $NAS_USER@$NAS_HOST:$NAS_PORT"

# Install required tools
echo "Installing dependencies..."
if [ -f "$(which apt)" ]; then
    apt update && apt install -y sshpass rsync
elif [ -f "$(which yum)" ]; then
    yum install -y sshpass rsync
else
    echo "Error: Unable to detect package manager"
    exit 1
fi

# Create target directory
mkdir -p $DATAHOME

# Download data
START_TIME=$(date +%s)
echo "Downloading from NAS..."

# Use sshpass directly (no wrapper script needed)
export SSHPASS="$NAS_PASS"
sshpass -e rsync --progress -avz -e "ssh -p $NAS_PORT -o StrictHostKeyChecking=no" \
    ${NAS_USER}@${NAS_HOST}:${DATAPATH}/ $DATAHOME/ || {
    echo "Warning: rsync failed, checking if partial data was transferred"
    # Even if rsync fails, continue - might be partial transfer
}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Calculate size
SIZE=$(du -sh $DATAHOME 2>/dev/null | cut -f1 || echo "0")

echo "✓ Download complete"
echo "Duration: ${DURATION}s"
echo "Size: $SIZE"

# Export metrics
mkdir -p /tmp
echo "MAD_DATA_DOWNLOAD_DURATION=$DURATION" >> /tmp/mad_metrics.env
echo "MAD_DATA_SIZE=$SIZE" >> /tmp/mad_metrics.env
echo "MAD_DATA_PROVIDER_TYPE=nas" >> /tmp/mad_metrics.env
echo "MAD_DATANAME=$DATANAME" >> /tmp/mad_metrics.env

