#!/bin/bash
# MADEngine K8s Data Provider - MinIO
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# Usage: download_minio.sh <dataname> <datapath> <datahome>

set -e

DATANAME=$1
DATAPATH=$2
DATAHOME=${3:-/data_dlm_0}

echo "=== MinIO Data Download ==="
echo "Data: $DATANAME"
echo "Source: $DATAPATH"
echo "Target: $DATAHOME"

# Get credentials from environment or credential.json
MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY:-${MAD_MINIO_USERNAME}}
MINIO_SECRET_KEY=${MINIO_SECRET_KEY:-${MAD_MINIO_PASSWORD}}
MINIO_ENDPOINT=${MINIO_ENDPOINT:-https://minio-frameworks.amd.com}

# If credentials not in environment, try to read from credential.json
if [ -z "$MINIO_ACCESS_KEY" ] && [ -f "/workspace/credential.json" ]; then
    echo "Reading MinIO credentials from credential.json..."
    MINIO_ACCESS_KEY=$(python3 -c "import json; f=open('/workspace/credential.json'); d=json.load(f); print(d.get('MAD_MINIO', {}).get('USERNAME', ''))" 2>/dev/null || echo "")
    MINIO_SECRET_KEY=$(python3 -c "import json; f=open('/workspace/credential.json'); d=json.load(f); print(d.get('MAD_MINIO', {}).get('PASSWORD', ''))" 2>/dev/null || echo "")
    MINIO_ENDPOINT=$(python3 -c "import json; f=open('/workspace/credential.json'); d=json.load(f); print(d.get('MAD_MINIO', {}).get('ENDPOINT_URL', 'https://minio-frameworks.amd.com'))" 2>/dev/null || echo "https://minio-frameworks.amd.com")
fi

# Verify credentials are available
if [ -z "$MINIO_ACCESS_KEY" ] || [ -z "$MINIO_SECRET_KEY" ]; then
    echo "Error: MinIO credentials not found in environment or credential.json"
    echo "Required: MINIO_ACCESS_KEY, MINIO_SECRET_KEY"
    exit 1
fi

# Install AWS CLI if not present
if ! command -v aws &> /dev/null; then
    echo "Installing AWS CLI..."
    pip3 --no-cache-dir install --upgrade awscli
fi

# Configure AWS CLI for MinIO
export AWS_ACCESS_KEY_ID=$MINIO_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=$MINIO_SECRET_KEY
export AWS_ENDPOINT_URL_S3=$MINIO_ENDPOINT

# Create target directory
mkdir -p $DATAHOME

# Download data
START_TIME=$(date +%s)
echo "Downloading..."

if aws --endpoint-url $MINIO_ENDPOINT s3 ls $DATAPATH 2>/dev/null | grep PRE; then
    # Directory download
    aws --endpoint-url $MINIO_ENDPOINT s3 sync $DATAPATH $DATAHOME
else
    # Single file download
    aws --endpoint-url $MINIO_ENDPOINT s3 sync \
        $(dirname $DATAPATH) $DATAHOME \
        --exclude="*" --include="$(basename $DATAPATH)"
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Calculate size
SIZE=$(du -sh $DATAHOME 2>/dev/null | cut -f1 || echo "0")

echo "✓ Download complete"
echo "Duration: ${DURATION}s"
echo "Size: $SIZE"

# Export metrics for collection
mkdir -p /tmp
echo "MAD_DATA_DOWNLOAD_DURATION=$DURATION" >> /tmp/mad_metrics.env
echo "MAD_DATA_SIZE=$SIZE" >> /tmp/mad_metrics.env
echo "MAD_DATA_PROVIDER_TYPE=minio" >> /tmp/mad_metrics.env
echo "MAD_DATANAME=$DATANAME" >> /tmp/mad_metrics.env

