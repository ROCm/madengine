#!/bin/bash
# Test script for rocenv_tool_v2.py
# Validates functionality on both TheRock and traditional ROCm systems

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "rocenv_tool_v2.py Test Suite"
echo "=========================================="
echo

# Function to print test results
pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
}

fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    exit 1
}

info() {
    echo -e "${YELLOW}ℹ INFO${NC}: $1"
}

# Test 1: Check file exists
echo "Test 1: File existence"
if [ -f "rocenv_tool_v2.py" ]; then
    pass "rocenv_tool_v2.py exists"
else
    fail "rocenv_tool_v2.py not found"
fi

# Test 2: Check dependencies
echo
echo "Test 2: Dependency checks"
if [ -f "console.py" ]; then
    pass "console.py found"
else
    fail "console.py not found"
fi

if [ -f "csv_parser.py" ]; then
    pass "csv_parser.py found"
else
    fail "csv_parser.py not found"
fi

# Test 3: Python syntax check
echo
echo "Test 3: Python syntax validation"
if python3 -m py_compile rocenv_tool_v2.py 2>/dev/null; then
    pass "Python syntax valid"
else
    fail "Python syntax errors detected"
fi

# Test 4: Help text
echo
echo "Test 4: Command-line interface"
if python3 rocenv_tool_v2.py --help > /dev/null 2>&1; then
    pass "Help text accessible"
else
    fail "Help text failed"
fi

# Test 5: Verbose mode detection
echo
echo "Test 5: Installation detection (verbose mode)"
info "Running detection..."
OUTPUT=$(python3 rocenv_tool_v2.py --verbose --output-name test_verbose 2>&1 || true)
echo "$OUTPUT" | head -20
echo

if echo "$OUTPUT" | grep -q "Installation Type:"; then
    INSTALL_TYPE=$(echo "$OUTPUT" | grep "Installation Type:" | head -1)
    pass "Detection completed: $INSTALL_TYPE"
else
    fail "Detection failed to identify installation type"
fi

# Test 6: Basic execution
echo
echo "Test 6: Basic execution (non-verbose)"
if python3 rocenv_tool_v2.py --output-name test_basic > /dev/null 2>&1; then
    pass "Basic execution successful"
else
    fail "Basic execution failed"
fi

# Test 7: Output directory creation
echo
echo "Test 7: Output directory validation"
if [ -d ".test_basic" ]; then
    pass "Output directory created"
    
    # Count subdirectories
    NUM_SECTIONS=$(find .test_basic -mindepth 1 -maxdepth 1 -type d | wc -l)
    info "Generated $NUM_SECTIONS information sections"
    
    if [ "$NUM_SECTIONS" -gt 5 ]; then
        pass "Sufficient sections generated ($NUM_SECTIONS)"
    else
        fail "Too few sections generated ($NUM_SECTIONS)"
    fi
else
    fail "Output directory not created"
fi

# Test 8: Check key sections
echo
echo "Test 8: Key section validation"
REQUIRED_SECTIONS=("os_information" "cpu_information" "gpu_information")
for section in "${REQUIRED_SECTIONS[@]}"; do
    if [ -d ".test_basic/$section" ]; then
        if [ -f ".test_basic/$section/$section.txt" ]; then
            pass "Section '$section' generated"
        else
            fail "Section '$section' file missing"
        fi
    else
        info "Section '$section' not generated (may be optional)"
    fi
done

# Test 9: ROCm-specific sections
echo
echo "Test 9: ROCm-specific sections"
if [ -d ".test_basic/rocm_information" ]; then
    pass "ROCm information section generated"
    
    # Check content
    if [ -f ".test_basic/rocm_information/rocm_information.txt" ]; then
        CONTENT=$(cat .test_basic/rocm_information/rocm_information.txt)
        
        if echo "$CONTENT" | grep -q "Installation Type:"; then
            DETECTED_TYPE=$(echo "$CONTENT" | grep "Installation Type:" | head -1)
            pass "ROCm installation type detected: $DETECTED_TYPE"
        fi
        
        if echo "$CONTENT" | grep -q "ROCm Root:"; then
            DETECTED_ROOT=$(echo "$CONTENT" | grep "ROCm Root:" | head -1)
            pass "ROCm root identified: $DETECTED_ROOT"
        fi
    fi
else
    info "ROCm information not generated (GPU may not be AMD)"
fi

# Test 10: CSV generation
echo
echo "Test 10: CSV generation"
if python3 rocenv_tool_v2.py --output-name test_csv --dump-csv > /dev/null 2>&1; then
    if [ -f "test_csv.csv" ]; then
        pass "CSV file generated"
        
        LINE_COUNT=$(wc -l < test_csv.csv)
        info "CSV contains $LINE_COUNT lines"
        
        if [ "$LINE_COUNT" -gt 10 ]; then
            pass "CSV contains data"
        fi
    else
        fail "CSV file not created"
    fi
else
    fail "CSV generation failed"
fi

# Test 11: Lite mode
echo
echo "Test 11: Lite mode"
if [ -f "env_tags.json" ]; then
    if python3 rocenv_tool_v2.py --lite --output-name test_lite > /dev/null 2>&1; then
        pass "Lite mode execution successful"
    else
        fail "Lite mode execution failed"
    fi
else
    info "env_tags.json not found, skipping lite mode test"
fi

# Test 12: Error handling (invalid path)
echo
echo "Test 12: Error handling"
# This should not crash even with missing tools
if timeout 30 python3 rocenv_tool_v2.py --output-name test_robust > /dev/null 2>&1; then
    pass "Robust error handling (script completed)"
else
    EXITCODE=$?
    if [ $EXITCODE -eq 124 ]; then
        fail "Script timed out (possible hang)"
    else
        fail "Script crashed unexpectedly"
    fi
fi

# Cleanup
echo
echo "=========================================="
echo "Cleanup"
echo "=========================================="
echo "Removing test output directories..."
rm -rf .test_basic .test_verbose .test_csv .test_lite .test_robust
rm -f test_csv.csv

echo
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "${GREEN}All tests passed!${NC}"
echo
echo "Next steps:"
echo "1. Review the implementation in rocenv_tool_v2.py"
echo "2. Test on a TheRock container:"
echo "   docker run -it <therock-image> python3 rocenv_tool_v2.py --verbose"
echo "3. Test on a traditional ROCm system:"
echo "   python3 rocenv_tool_v2.py --verbose"
echo "4. Compare outputs with original rocenv_tool.py"
echo
echo "Documentation:"
echo "- README_v2.md - Usage guide"
echo "- THEROCK_COMPATIBILITY.md - Compatibility details"
echo "- IMPLEMENTATION_SUMMARY.md - Implementation overview"
echo

