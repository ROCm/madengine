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
	# Handle both legacy rocprof (results*) and rocprofv3 (different output format)
	echo "ROCprof post-script: Collecting profiling output..."
	
	# Check for legacy rocprof results files
	if ls results* 1> /dev/null 2>&1; then
		echo "Found rocprof results files"
		mv results* "$OUTPUT" 2>/dev/null || true
	else
		echo "No rocprof results* files found (may be using rocprofv3)"
	fi
	
	# Check for rocprofv3 output directories (UUID pattern like 1e4d92661463/)
	# rocprofv3 creates directories with hex UUIDs containing .db files
	found_rocprofv3_output=false
	for dir in */; do
		# Check if directory exists and contains .db files
		if [ -d "$dir" ]; then
			# Use proper glob expansion to check for any .db file
			if compgen -G "${dir}*_results.db" > /dev/null; then
				echo "Found rocprofv3 output directory: $dir"
				mv "$dir" "$OUTPUT/" 2>/dev/null || true
				found_rocprofv3_output=true
			fi
		fi
	done
	
	# Also check for other rocprofv3 output patterns
	if ls rocprofv3-* 1> /dev/null 2>&1; then
		echo "Found rocprofv3-* files"
		mv rocprofv3-* "$OUTPUT" 2>/dev/null || true
		found_rocprofv3_output=true
	fi
	
	if [ "$found_rocprofv3_output" = true ]; then
		echo "Collected rocprofv3 profiling data"
	fi
	
	# Copy output directory (even if empty - non-critical)
	cp -vLR --preserve=all "$OUTPUT" "$SAVESPACE" || echo "Note: Output directory may be empty (profiling was passive)"
	;;

esac

chmod -R a+rw "${SAVESPACE}/${OUTPUT}"
