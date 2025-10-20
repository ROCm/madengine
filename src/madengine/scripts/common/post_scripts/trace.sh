#!/usr/bin/env bash
# 
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
# 

set -e
set -x

tool=$1

OUTPUT=${tool}_output
SAVESPACE=/myworkspace/

mkdir "$OUTPUT"

case "$tool" in

rpd)
	echo "RPD post-script: Checking for trace.rpd file..."
	if [ -f "trace.rpd" ]; then
		echo "RPD post-script: trace.rpd found, size: $(stat -c%s trace.rpd) bytes"
		ls -la trace.rpd
	else
		echo "RPD post-script: ERROR - trace.rpd file not found!"
		echo "Current directory contents:"
		ls -la
		# Still create output directory and copy what we can
		touch "$OUTPUT/trace.rpd"  # Create empty file so test can find directory structure
	fi
	
	echo "RPD post-script: Checking for rpd2tracing.py script..."
	if [ -f "./rocmProfileData/tools/rpd2tracing.py" ]; then
		echo "RPD post-script: rpd2tracing.py found"
		if [ -f "trace.rpd" ] && [ -s "trace.rpd" ]; then
			python3 ./rocmProfileData/tools/rpd2tracing.py trace.rpd trace.json
			mv trace.rpd trace.json "$OUTPUT"
		else
			echo "RPD post-script: Skipping rpd2tracing.py because trace.rpd is missing or empty"
			# Create empty files so the directory structure exists
			touch "$OUTPUT/trace.rpd"  
			touch "$OUTPUT/trace.json"
		fi
	else
		echo "RPD post-script: ERROR - rpd2tracing.py not found!"
		echo "Contents of ./rocmProfileData/:"
		ls -la ./rocmProfileData/ || echo "rocmProfileData directory not found"
		# Create empty files so the directory structure exists
		touch "$OUTPUT/trace.rpd"
		touch "$OUTPUT/trace.json"
	fi
	
	cp -vLR --preserve=all "$OUTPUT" "$SAVESPACE"
	;;

rocprof)
	mv results* "$OUTPUT"
	cp -vLR --preserve=all "$OUTPUT" "$SAVESPACE"
	;;

esac

chmod -R a+rw "${SAVESPACE}/${OUTPUT}"
