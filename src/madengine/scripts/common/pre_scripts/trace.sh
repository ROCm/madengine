#!/usr/bin/env bash
# 
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
# 

set -e
set -x

tool=$1

case "$tool" in

rpd)
	# OS packages only needed for RPD build; other tools (e.g. rocm_trace_lite) skip this.
	os=''
	if command -v apt >/dev/null 2>&1; then
		os=ubuntu
	elif command -v yum >/dev/null 2>&1; then
		os=centos
	else
		echo 'Unable to detect Host OS in pre_script (need apt or yum for RPD dependencies)' >&2
		exit 1
	fi
	if [ "$os" == 'ubuntu' ]; then
		sudo apt update
		sudo apt install -y sqlite3 libsqlite3-dev libfmt-dev python3-pip nlohmann-json3-dev
	elif [ "$os" == 'centos' ]; then
		sudo yum install -y libsqlite3x-devel.x86_64 fmt-devel python3-pip json-devel
	else
		echo "Unable to detect Host OS in trace pre-script"
	fi
	# Clone rocmProfileData repository
	if [ ! -d "rocmProfileData" ]; then
		git clone https://github.com/ROCm/rocmProfileData.git rocmProfileData
		if [ $? -ne 0 ]; then
			echo "Error: Failed to clone rocmProfileData repository"
			exit 1
		fi
	else
		echo "rocmProfileData directory already exists, skipping clone"
	fi
	
	# Build RPD tracer locally without system install
	cd ./rocmProfileData
	# Workaround for upstream rocmProfileData Makefile typo: UStringTable.o -> StringTable.o
	if [ -f rpd_tracer/Makefile ]; then
		sed -i 's/UStringTable\.o/StringTable.o/g' rpd_tracer/Makefile
	fi
	make rpd
	if [ $? -ne 0 ]; then
		echo "Error: Failed to build RPD tracer"
		exit 1
	fi
	
	# Install rocpd Python module locally
	cd rocpd_python
	python3 setup.py install
	if [ $? -ne 0 ]; then
		echo "Error: Failed to install rocpd Python module"
		exit 1
	fi
	cd ../..
	
	echo "RPD setup completed successfully"
	;;

rocm_trace_lite)
	# rocm-trace-lite ships as GitHub Release wheels (linux_x86_64), not on PyPI.
	# https://github.com/sunway513/rocm-trace-lite#installation
	# Wheel resolution (first match wins):
	#   1) ROCM_TRACE_LITE_WHEEL_URL — direct .whl URL (air-gapped / custom)
	#   2) ROCM_TRACE_LITE_FOLLOW_LATEST=1 — resolve latest linux_x86_64 wheel via GitHub API (needs curl)
	#   3) Pinned release below — reproducible default (no API; bump when upgrading RTL)
	_ROTL_PINNED_WHEEL='https://github.com/sunway513/rocm-trace-lite/releases/download/v0.3.3/rocm_trace_lite-0.3.3-py3-none-linux_x86_64.whl'
	if ! command -v python3 >/dev/null 2>&1; then
		echo "Error: rocm_trace_lite pre-script requires python3 on PATH." >&2
		exit 1
	fi
	if ! python3 -m pip --version >/dev/null 2>&1; then
		echo "Error: rocm_trace_lite pre-script requires pip (python3 -m pip failed)." >&2
		exit 1
	fi
	# ROCM_TRACE_LITE_WHEEL_URL may embed credentials; avoid leaking it via `set -x` and stderr.
	_rocm_trace_lite_restore_x=0
	case $- in *x*) _rocm_trace_lite_restore_x=1 ;; esac
	set +x
	_rtl_wheel="${ROCM_TRACE_LITE_WHEEL_URL:-}"
	if [ -z "$_rtl_wheel" ] && [ "${ROCM_TRACE_LITE_FOLLOW_LATEST:-}" = "1" ] && command -v curl >/dev/null 2>&1; then
		_rtl_wheel=$(curl -fsSL 'https://api.github.com/repos/sunway513/rocm-trace-lite/releases/latest' 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    for a in d.get("assets", []):
        n = a.get("name", "")
        if n.endswith("-py3-none-linux_x86_64.whl"):
            print(a["browser_download_url"])
            break
except (json.JSONDecodeError, KeyError, TypeError, ValueError):
    pass
' 2>/dev/null) || true
	fi
	if [ -z "$_rtl_wheel" ]; then
		_rtl_wheel="$_ROTL_PINNED_WHEEL"
	fi
	if ! python3 -m pip install --upgrade "$_rtl_wheel"; then
		if ! python3 -m pip install --user --upgrade "$_rtl_wheel"; then
			echo "Error: pip could not install rocm-trace-lite wheel (URL omitted from logs)." >&2
			echo "Check network, pip, ROCM_TRACE_LITE_WHEEL_URL / ROCM_TRACE_LITE_FOLLOW_LATEST, and trace.sh pinned wheel." >&2
			[ "$_rocm_trace_lite_restore_x" -eq 1 ] && set -x
			exit 1
		fi
	fi
	[ "$_rocm_trace_lite_restore_x" -eq 1 ] && set -x
	unset _rocm_trace_lite_restore_x
	if command -v rtl >/dev/null 2>&1; then
		echo "rocm-trace-lite: rtl is on PATH."
	elif python3 -c 'import rocm_trace_lite' 2>/dev/null; then
		echo "rocm-trace-lite: Python package import OK (use rtl or python3 -m rocm_trace_lite.cli)."
	else
		echo "Error: rocm-trace-lite wheel installed but neither 'rtl' nor import rocm_trace_lite works." >&2
		exit 1
	fi
	;;

esac
