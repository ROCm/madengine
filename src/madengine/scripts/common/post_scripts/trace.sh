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

mkdir -p "$OUTPUT"

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
	
	# Check for CSV trace files in subdirectories (rocprof can create hostname subdirectories)
	# Look for patterns like: hostname/pid_kernel_trace.csv, hostname/pid_hip_api_trace.csv, etc.
	csv_found=false
	for dir in */; do
		if [ -d "$dir" ]; then
			# Check for CSV files matching rocprof patterns
			if compgen -G "${dir}*_trace.csv" > /dev/null || compgen -G "${dir}*_api_trace.csv" > /dev/null; then
				echo "Found rocprof CSV files in directory: $dir"
				# Copy CSV files to output directory, preserving subdirectory structure
				mkdir -p "$OUTPUT/$dir"
				cp -v "${dir}"*.csv "$OUTPUT/$dir/" 2>/dev/null || true
				csv_found=true
			fi
		fi
	done
	
	if [ "$csv_found" = true ]; then
		echo "Collected rocprof CSV trace files from subdirectories"
	fi
	
	# Consolidate rocprofv3 CSV files so MAD-agent finds rocprofv3_output_* names.
	# rocprofv3 may write agent_info in -o prefix but kernel_trace/stats with PID prefix or under hostname/pid.
	for base in agent_info domain_stats kernel_stats kernel_trace hip_api_trace counter_collection; do
		canonical="${OUTPUT}/rocprofv3_output_${base}.csv"
		if [ -f "$canonical" ]; then
			continue
		fi
		first=$(find . -maxdepth 4 -name "*${base}.csv" -type f 2>/dev/null | head -1)
		if [ -n "$first" ]; then
			cp -v "$first" "$canonical"
		fi
	done
	
	# Generate instruction_histogram.json from counter/domain_stats CSV so MAD-agent gets real instruction mix.
	if [ -f "${OUTPUT}/rocprofv3_output_counter_collection.csv" ] || [ -f "${OUTPUT}/rocprofv3_output_domain_stats.csv" ]; then
		CONVERTER="$(cd "$(dirname "$0")/../tools" 2>/dev/null && pwd)/rocprof_counter_csv_to_instruction_histogram.py"
		if [ -n "$CONVERTER" ] && [ -f "$CONVERTER" ]; then
			python3 "$CONVERTER" "$OUTPUT" || true
		fi
	fi

	# Copy output directory (even if empty - non-critical)
	cp -vLR --preserve=all "$OUTPUT" "$SAVESPACE" || echo "Note: Output directory may be empty (profiling was passive)"
	;;

rocm_trace_lite)
	# OUTPUT is rocm_trace_lite_output (same dir rtl_trace_wrapper.sh writes to).
	echo "rocm-trace-lite post-script: Collecting RTL outputs under ${OUTPUT}..."
	mkdir -p "$OUTPUT"
	if [ -f "trace.db" ] && [ ! -f "${OUTPUT}/trace.db" ]; then
		mv -v "trace.db" "$OUTPUT/" 2>/dev/null || cp -v "trace.db" "$OUTPUT/" || true
	fi
	for f in trace.json.gz trace.json; do
		if [ -f "$f" ]; then
			mv -v "$f" "$OUTPUT/" 2>/dev/null || cp -v "$f" "$OUTPUT/" || true
		fi
	done
	if [ ! -f "${OUTPUT}/trace.db" ]; then
		echo "WARNING: ${OUTPUT}/trace.db not found (rtl may have failed or used a different path)."
	fi
	cp -vLR --preserve=all "$OUTPUT" "$SAVESPACE" || echo "Note: rocm_trace_lite output directory may be empty"
	;;

rocm_trace_lite_fast)
	# Env-var mode: per-process trace_*.db files written by RTL's atexit handler.
	# This post-script runs AFTER madengine has extracted the performance metric,
	# so merge + summary + Perfetto do NOT affect wall time or throughput measurement.
	echo "rocm-trace-lite post-script (fast mode): merging per-process traces..."
	mkdir -p "$OUTPUT"

	# Merge per-process trace_<PID>.db files into a single trace.db
	if ls rocm_trace_lite_output/trace_*.db 1>/dev/null 2>&1; then
		python3 << 'PYEOF'
import glob, os, re, shutil, sys
files = sorted([
    f for f in glob.glob('rocm_trace_lite_output/trace_*.db')
    if re.match(r'^trace_\d+\.db$', os.path.basename(f))
])
if not files:
    print('No per-process trace files found.')
elif len(files) == 1:
    shutil.copy2(files[0], 'rocm_trace_lite_output/trace.db')
    print('Single process trace -> trace.db')
else:
    merged = False
    try:
        from rocm_trace_lite.cmd_trace import _merge_traces
        _merge_traces(files, 'rocm_trace_lite_output/trace.db')
        print(f'Merged {len(files)} per-process traces -> trace.db')
        merged = True
    except Exception as e:
        sys.stderr.write(f'merge warning: {e}\n')
    if not merged:
        shutil.copy2(files[0], 'rocm_trace_lite_output/trace.db')
        print('Warning: merge unavailable, using first trace file')
PYEOF
		rc=$?; [ $rc -ne 0 ] && echo "Warning: trace merge failed (non-fatal)"
	else
		echo "Warning: no per-process trace_*.db files found in rocm_trace_lite_output/"
	fi

	# Generate summary + Perfetto conversion (best-effort)
	if [ -f "rocm_trace_lite_output/trace.db" ]; then
		if command -v rtl >/dev/null 2>&1; then
			rtl summary rocm_trace_lite_output/trace.db 2>/dev/null | tee rocm_trace_lite_output/trace_summary.txt || true
			rtl convert rocm_trace_lite_output/trace.db -o rocm_trace_lite_output/trace.json.gz 2>&1 || true
		elif python3 -c 'import rocm_trace_lite' 2>/dev/null; then
			python3 -c "
from rocm_trace_lite.cmd_summary import summary
from rocm_trace_lite.cmd_convert import convert
summary('rocm_trace_lite_output/trace.db', 'rocm_trace_lite_output/trace_summary.txt')
convert('rocm_trace_lite_output/trace.db', 'rocm_trace_lite_output/trace.json.gz')
" 2>/dev/null || true
		fi
	else
		echo "WARNING: trace.db not found after merge."
	fi

	# Move all outputs into the canonical output directory
	if [ -d "rocm_trace_lite_output" ] && [ "rocm_trace_lite_output" != "$OUTPUT" ]; then
		cp -vLR --preserve=all rocm_trace_lite_output/* "$OUTPUT/" 2>/dev/null || true
	fi

	cp -vLR --preserve=all "$OUTPUT" "$SAVESPACE" || echo "Note: rocm_trace_lite output may be empty"
	;;

esac

chmod -R a+rw "${SAVESPACE}/${OUTPUT}"
