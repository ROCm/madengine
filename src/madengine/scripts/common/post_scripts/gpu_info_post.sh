#!/usr/bin/env bash
# 
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
# 

set -e
set -x

tool=$1
model_name=$2

# Use model name if provided, otherwise fallback to tool name
if [ -n "$model_name" ]; then
    OUTPUT=${tool}_${model_name}_output.csv
else
    OUTPUT=${tool}_output.csv
fi

SAVESPACE=/myworkspace/

cd $SAVESPACE
if [ -d "$OUTPUT" ]; then
	mkdir "$OUTPUT"
fi

mv prof.csv "$OUTPUT"

chmod -R a+rw "${SAVESPACE}/${OUTPUT}"
