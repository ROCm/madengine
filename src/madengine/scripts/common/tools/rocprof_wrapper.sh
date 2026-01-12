#!/bin/bash
# 
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
# 
# ROCm Profiler Wrapper - Intelligently select between rocprof (legacy) and rocprofv3 (new)
#
# This wrapper handles the transition from rocprof to rocprofv3 across ROCm versions.
# It automatically detects the available profiler and uses the appropriate one.
#
# ROCm Version Support:
#   - ROCm < 7.0: Uses rocprof (legacy)
#   - ROCm >= 7.0: Prefers rocprofv3, falls back to rocprof if not available
#

# Function to detect ROCm version
get_rocm_version() {
    # Try multiple methods to detect ROCm version
    local version=""
    
    # Method 1: Check rocm-smi output
    if command -v rocm-smi &> /dev/null; then
        version=$(rocm-smi --version 2>/dev/null | grep -oP 'ROCm version: \K[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    fi
    
    # Method 2: Check /opt/rocm/.info/version file
    if [ -z "$version" ] && [ -f /opt/rocm/.info/version ]; then
        version=$(cat /opt/rocm/.info/version)
    fi
    
    # Method 3: Check ROCM_PATH or default ROCm installation
    if [ -z "$version" ]; then
        local rocm_path="${ROCM_PATH:-/opt/rocm}"
        if [ -f "$rocm_path/.info/version" ]; then
            version=$(cat "$rocm_path/.info/version")
        fi
    fi
    
    echo "$version"
}

# Function to compare version strings (returns 0 if v1 >= v2)
version_gte() {
    # Convert version strings to comparable numbers
    local v1=$(echo "$1" | awk -F. '{ printf("%d%03d%03d\n", $1,$2,$3); }')
    local v2=$(echo "$2" | awk -F. '{ printf("%d%03d%03d\n", $1,$2,$3); }')
    [ "$v1" -ge "$v2" ]
}

# Function to detect available profiler
detect_profiler() {
    local rocm_version=$(get_rocm_version)
    
    # Check if rocprofv3 is available
    if command -v rocprofv3 &> /dev/null; then
        echo "rocprofv3"
        return 0
    fi
    
    # Check if rocprof (legacy) is available
    if command -v rocprof &> /dev/null; then
        # For ROCm >= 7.0, warn that rocprofv3 should be available
        if [ -n "$rocm_version" ] && version_gte "$rocm_version" "7.0.0"; then
            echo "Warning: ROCm $rocm_version detected but rocprofv3 not found, using legacy rocprof" >&2
        fi
        echo "rocprof"
        return 0
    fi
    
    # No profiler found
    echo "Error: Neither rocprofv3 nor rocprof found in PATH" >&2
    echo "Please ensure ROCm profiler tools are installed" >&2
    return 1
}

# Main execution
main() {
    local profiler=$(detect_profiler)
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        return 1
    fi
    
    # Execute the detected profiler with all passed arguments
    if [ "$profiler" = "rocprof" ]; then
        # Legacy rocprof syntax: rocprof [options] <app> [args]
        # All arguments can be passed directly
        # Filter deprecation warnings while preserving stdout and exit code
        { rocprof "$@" 2>&1 1>&3 | grep -v "WARNING: We are phasing out" | grep -v "roctracer/rocprofiler" | grep -v "rocprofv2 in favor" >&2; } 3>&1
        return ${PIPESTATUS[0]}
    else
        # New rocprofv3 syntax: rocprofv3 [options] -- <app> [args]
        # Need to separate profiler options from application command
        local profiler_opts=()
        local app_cmd=()
        local found_separator=false
        
        for arg in "$@"; do
            if [ "$arg" = "--" ]; then
                # Found the separator, everything after this is the application command
                found_separator=true
                continue
            fi
            
            if [ "$found_separator" = true ]; then
                app_cmd+=("$arg")
            else
                profiler_opts+=("$arg")
            fi
        done
        
        # Build command with proper argument placement
        if [ "${#profiler_opts[@]}" -gt 0 ]; then
            # Has profiler options: rocprofv3 <opts> -- <app>
            rocprofv3 "${profiler_opts[@]}" -- "${app_cmd[@]}"
        else
            # No profiler options: rocprofv3 -- <app>
            rocprofv3 -- "${app_cmd[@]}"
        fi
        return $?
    fi
}

# Run main function
main "$@"

