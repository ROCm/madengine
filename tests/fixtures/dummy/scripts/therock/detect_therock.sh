#!/bin/sh
#
# Quick TheRock ROCm Detection Script
# 
# This script checks if TheRock is installed on the system.
# TheRock does NOT use apt - it uses Python pip or tarballs.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

FOUND=0

echo "=================================================="
echo "TheRock ROCm Distribution Detection"
echo "=================================================="
echo ""

# Function to check if path contains TheRock installation
check_therock_path() {
    path="$1"
    label="$2"
    
    if [ ! -d "$path" ]; then
        return 1
    fi
    
    manifest="$path/share/therock/therock_manifest.json"
    dist_info="$path/share/therock/dist_info.json"
    
    if [ -f "$manifest" ]; then
        printf "${GREEN}✓ Found TheRock installation${NC}\n"
        echo "  Type: $label"
        echo "  Path: $path"
        
        if [ -f "$dist_info" ]; then
            targets=$(grep -oP '(?<="dist_amdgpu_targets": ")[^"]*' "$dist_info" 2>/dev/null || echo "unknown")
            echo "  GPU Targets: $targets"
        fi
        
        if command -v jq > /dev/null 2>&1; then
            commit=$(jq -r '.the_rock_commit' "$manifest" 2>/dev/null || echo "unknown")
            echo "  Commit: $commit"
        fi
        
        echo ""
        FOUND=$((FOUND + 1))
        return 0
    fi
    
    return 1
}

# Check 1: rocm-sdk command (Python installation)
printf "${BLUE}[1] Checking for rocm-sdk command...${NC}\n"
if command -v rocm-sdk > /dev/null 2>&1; then
    printf "${GREEN}✓ Found rocm-sdk command${NC}\n"
    
    # Get version
    version=$(rocm-sdk version 2>/dev/null || echo "unknown")
    echo "  Version: $version"
    
    # Get root path
    if root_path=$(rocm-sdk path --root 2>/dev/null); then
        echo "  Root: $root_path"
        check_therock_path "$root_path" "Python Package"
    fi
else
    echo "  ✗ rocm-sdk command not found"
fi
echo ""

# Check 2: Python site-packages
printf "${BLUE}[2] Checking Python site-packages...${NC}\n"
if python3 -c "import rocm_sdk" 2>/dev/null; then
    version=$(python3 -c "import rocm_sdk; print(rocm_sdk.__version__)" 2>/dev/null || echo "unknown")
    printf "${GREEN}✓ Found rocm_sdk Python package${NC}\n"
    echo "  Version: $version"
    
    # Try to find the package path
    pkg_path=$(python3 -c "
import importlib.util
import pathlib
spec = importlib.util.find_spec('_rocm_sdk_core')
if spec and spec.origin:
    print(pathlib.Path(spec.origin).parent)
" 2>/dev/null || echo "")
    
    if [ -n "$pkg_path" ]; then
        check_therock_path "$pkg_path" "Python Package"
    fi
else
    echo "  ✗ rocm_sdk Python package not found"
fi
echo ""

# Check 3: Common installation paths
printf "${BLUE}[3] Checking common installation paths...${NC}\n"
for path in "$HOME/rocm" "$HOME/therock" "/opt/rocm" "/usr/local/rocm" "$HOME/.local/rocm"; do
    if check_therock_path "$path" "Tarball Installation"; then
        :  # Found, already printed
    fi
done

# Check 4: Environment variables
printf "${BLUE}[4] Checking environment variables...${NC}\n"
env_found=0
for var in ROCM_PATH ROCM_HOME HIP_PATH; do
    eval "var_value=\$$var"
    if [ -n "$var_value" ]; then
        echo "  Checking \$$var = $var_value"
        if check_therock_path "$var_value" "Environment Variable (\$$var)"; then
            env_found=1
        fi
    fi
done

if [ $env_found -eq 0 ]; then
    echo "  ✗ No TheRock installations found via environment variables"
fi
echo ""

# Check 5: Local build directory
printf "${BLUE}[5] Checking for local build directory...${NC}\n"
if [ -f "version.json" ] && [ -f "CMakeLists.txt" ]; then
    if grep -q "rocm-version" version.json 2>/dev/null; then
        printf "${YELLOW}✓ Found TheRock source directory${NC}\n"
        echo "  Path: $(pwd)"
        
        if [ -d "build/dist" ]; then
            for dist_dir in build/dist/*; do
                if [ -d "$dist_dir" ]; then
                    check_therock_path "$dist_dir" "Local Build"
                fi
            done
        else
            echo "  (No build/dist directory found - not yet built)"
        fi
    fi
else
    echo "  ✗ Not in a TheRock source directory"
fi
echo ""

# Summary
echo "=================================================="
echo "Summary"
echo "=================================================="

if [ $FOUND -gt 0 ]; then
    printf "${GREEN}Found $FOUND TheRock installation(s)${NC}\n"
    echo ""
    echo "TheRock is installed on this system!"
    exit 0
else
    printf "${RED}No TheRock installations detected${NC}\n"
    echo ""
    echo "TheRock does NOT use apt/system packages."
    echo "It installs via:"
    echo "  1. Python pip (recommended)"
    echo "  2. Standalone tarballs"
    echo "  3. Build from source"
    echo ""
    echo "To install TheRock:"
    echo "  pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ 'rocm[libraries,devel]'"
    echo ""
    echo "More info: https://github.com/ROCm/TheRock/blob/main/RELEASES.md"
    exit 1
fi

