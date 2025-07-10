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
    OUTPUT=${tool}_${model_name}_output
else
    OUTPUT=${tool}_output
fi

SAVESPACE=/myworkspace/

mkdir "$OUTPUT"

case "$tool" in

rpd)
	python3 ./rocmProfileData/tools/rpd2tracing.py trace.rpd trace.json
	
	# Rename trace files with model name if provided
	if [ -n "$model_name" ]; then
		if [ -f "trace.rpd" ]; then
			mv trace.rpd "trace_${model_name}.rpd"
		fi
		if [ -f "trace.json" ]; then
			mv trace.json "trace_${model_name}.json"
		fi
		mv "trace_${model_name}.rpd" "trace_${model_name}.json" "$OUTPUT" 2>/dev/null || true
	else
		mv trace.rpd trace.json "$OUTPUT" 2>/dev/null || true
	fi
	
	cp -vLR --preserve=all "$OUTPUT" "$SAVESPACE"
	;;

rocprof)
	mv results* "$OUTPUT"
	cp -vLR --preserve=all "$OUTPUT" "$SAVESPACE"
	;;

esac

chmod -R a+rw "${SAVESPACE}/${OUTPUT}"
