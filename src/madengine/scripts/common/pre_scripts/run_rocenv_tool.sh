#!/usr/bin/env bash
# 
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
# 

OUTPUT_FILE_NAME=${1:-"sys_config_info"}
ROCENV_MODE=${2:-"lite"}
# Third arg or MAD_GUEST_OS: must match madengine additional_context guest_os (UBUNTU, CENTOS, ...)
_GUEST_RAW=${3:-${MAD_GUEST_OS:-}}
if [ -z "${_GUEST_RAW}" ]; then
    ROCENV_GUEST_OS="UBUNTU"
else
    ROCENV_GUEST_OS=$(printf '%s' "${_GUEST_RAW}" | tr '[:lower:]' '[:upper:]')
fi

# Best-effort install for rocenv "full" mode, keyed off madengine guest_os (not /etc/os-release).
rocenv_install_diagnostic_packages() {
    local guest="$1"
    local pkgs="$2"

    case "${guest}" in
        UBUNTU)
            if ! command -v apt-get >/dev/null 2>&1; then
                echo "Warning: guest_os is UBUNTU but apt-get not found in this image; skipping package install for:${pkgs}"
                return 1
            fi
            apt-get update -qq >/dev/null 2>&1 && \
                apt-get install -y -qq --no-install-recommends ${pkgs} >/dev/null 2>&1
            ;;
        CENTOS)
            if command -v microdnf >/dev/null 2>&1; then
                microdnf install -y -q ${pkgs} >/dev/null 2>&1 && return 0
            fi
            if command -v dnf >/dev/null 2>&1; then
                dnf install -y -q ${pkgs} >/dev/null 2>&1 && return 0
            fi
            if command -v yum >/dev/null 2>&1; then
                yum install -y -q ${pkgs} >/dev/null 2>&1 && return 0
            fi
            echo "Warning: guest_os is CENTOS but no microdnf, dnf, or yum found; skipping package install for:${pkgs}"
            return 1
            ;;
        *)
            echo "Warning: rocenv full mode auto-install is not implemented for guest_os=${guest} (supported: UBUNTU, CENTOS). Missing tools:${pkgs}"
            return 1
            ;;
    esac
}

LITE_FLAG="--lite"
if [ "$ROCENV_MODE" = "full" ]; then
    LITE_FLAG=""
    # Install diagnostic tools on-demand if missing (best-effort)
    # These are needed for hardware_information, bios_settings,
    # dmsg_gpu_drm_atom_logs, and amdgpu_modinfo sections
    MISSING_PKGS=""
    command -v lshw      >/dev/null 2>&1 || MISSING_PKGS="${MISSING_PKGS:+$MISSING_PKGS }lshw"
    command -v dmidecode >/dev/null 2>&1 || MISSING_PKGS="${MISSING_PKGS:+$MISSING_PKGS }dmidecode"
    command -v modinfo   >/dev/null 2>&1 || MISSING_PKGS="${MISSING_PKGS:+$MISSING_PKGS }kmod"
    command -v dmesg     >/dev/null 2>&1 || MISSING_PKGS="${MISSING_PKGS:+$MISSING_PKGS }util-linux"
    if [ -n "$MISSING_PKGS" ]; then
        echo "rocenv full mode (guest_os=${ROCENV_GUEST_OS}): installing missing diagnostic tools: ${MISSING_PKGS}"
        rocenv_install_diagnostic_packages "${ROCENV_GUEST_OS}" "${MISSING_PKGS}" || \
            echo "Warning: could not install some diagnostic tools (network, permissions, or unsupported guest_os)"
    fi
fi

# Determine the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if rocEnvTool is in the same directory (K8s execution)
if [ -d "$SCRIPT_DIR/rocEnvTool" ]; then
    # K8s execution: rocEnvTool is already in place
    cd "$SCRIPT_DIR/rocEnvTool"
    python3 rocenv_tool.py $LITE_FLAG --dump-csv --print-csv --output-name $OUTPUT_FILE_NAME
    out_dir="."$OUTPUT_FILE_NAME
    out_csv=$OUTPUT_FILE_NAME".csv"
    # Copy results back to workspace root
    if [ -d "$out_dir" ]; then
        cp -r "$out_dir" /workspace/
    fi
    if [ -f "$out_csv" ]; then
        cp "$out_csv" /workspace/
    fi
else
    # Local execution: copy rocEnvTool from relative path
    cp -r ../scripts/common/pre_scripts/rocEnvTool .
    cd rocEnvTool
    python3 rocenv_tool.py $LITE_FLAG --dump-csv --print-csv --output-name $OUTPUT_FILE_NAME
    out_dir="."$OUTPUT_FILE_NAME
    out_csv=$OUTPUT_FILE_NAME".csv"
    cp -r $out_dir ../../
    cp $out_csv ../../
    cd ..
fi
