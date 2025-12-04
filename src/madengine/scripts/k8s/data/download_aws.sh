#!/bin/bash
# MADEngine K8s Data Provider - AWS S3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# Usage: download_aws.sh <dataname> <datapath> <datahome>

set -e

DATANAME=$1
DATAPATH=$2
DATAHOME=${3:-/data_dlm_0}
AWS_REGION=${AWS_REGION:-us-east-2}

echo "=== AWS S3 Data Download ==="
echo "Data: $DATANAME"
echo "Source: $DATAPATH"
echo "Target: $DATAHOME"
echo "Region: $AWS_REGION"

# Get credentials from environment
export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-${MAD_AWS_ACCESS_KEY}}
export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-${MAD_AWS_SECRET_KEY}}

# Install AWS CLI if not present
if ! command -v aws &> /dev/null; then
    echo "Installing AWS CLI..."
    pip3 --no-cache-dir install --upgrade awscli
fi

# Create target directory
mkdir -p $DATAHOME

# Download data
START_TIME=$(date +%s)
echo "Downloading..."

if aws --region=$AWS_REGION s3 ls $DATAPATH 2>/dev/null | grep "PRE"; then
    # Directory download
    aws --region=$AWS_REGION s3 sync $DATAPATH $DATAHOME
else
    # Single file download
    aws --region=$AWS_REGION s3 sync \
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

# Export metrics
mkdir -p /tmp
echo "MAD_DATA_DOWNLOAD_DURATION=$DURATION" >> /tmp/mad_metrics.env
echo "MAD_DATA_SIZE=$SIZE" >> /tmp/mad_metrics.env
echo "MAD_DATA_PROVIDER_TYPE=aws" >> /tmp/mad_metrics.env
echo "MAD_DATANAME=$DATANAME" >> /tmp/mad_metrics.env

