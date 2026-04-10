#!/usr/bin/env bash
# 
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
# 

set -e
set -x

os=''
if [ -n "$(command -v apt)" ]; then
	os=ubuntu
elif [ -n "$(command -v yum)" ]; then
	os=centos
else
	echo 'Unable to detect Host OS in pre_script'
	exit 1
fi

tool=$1

case "$tool" in

rpd)
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
	# Override wheel: export ROCM_TRACE_LITE_WHEEL_URL='https://.../file.whl'
	if command -v python3 >/dev/null 2>&1; then
		_rtl_wheel="${ROCM_TRACE_LITE_WHEEL_URL:-}"
		if [ -z "$_rtl_wheel" ] && command -v curl >/dev/null 2>&1; then
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
			_rtl_wheel='https://github.com/sunway513/rocm-trace-lite/releases/download/v0.3.3/rocm_trace_lite-0.3.3-py3-none-linux_x86_64.whl'
		fi
		if ! python3 -m pip install --upgrade "$_rtl_wheel"; then
			python3 -m pip install --user --upgrade "$_rtl_wheel" || true
		fi
	fi
	;;

esac
