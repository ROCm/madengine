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
	# Docker madengine runs often use root with no sudo — use apt-get/yum directly when uid==0.
	os=''
	if command -v apt-get >/dev/null 2>&1; then
		os=ubuntu
	elif command -v yum >/dev/null 2>&1; then
		os=centos
	else
		echo 'Unable to detect Host OS in pre_script (need apt-get or yum for RPD dependencies)' >&2
		exit 1
	fi
	if [ "$os" == 'ubuntu' ]; then
		if [ "$(id -u)" -eq 0 ]; then
			apt-get update -qq
			DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
				sqlite3 libsqlite3-dev libfmt-dev python3-pip nlohmann-json3-dev \
				git build-essential pkg-config xxd
		elif command -v sudo >/dev/null 2>&1; then
			sudo apt-get update -qq
			sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
				sqlite3 libsqlite3-dev libfmt-dev python3-pip nlohmann-json3-dev \
				git build-essential pkg-config xxd
		else
			echo 'RPD pre-script: need root or sudo for apt-get' >&2
			exit 1
		fi
	elif [ "$os" == 'centos' ]; then
		if [ "$(id -u)" -eq 0 ]; then
			yum install -y gcc gcc-c++ make git \
				libsqlite3x-devel.x86_64 fmt-devel python3-pip json-devel vim-common
		elif command -v sudo >/dev/null 2>&1; then
			sudo yum install -y gcc gcc-c++ make git \
				libsqlite3x-devel.x86_64 fmt-devel python3-pip json-devel vim-common
		else
			echo 'RPD pre-script: need root or sudo for yum' >&2
			exit 1
		fi
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

dynolog)
	# dynolog is the profiling daemon that lets us drive torch.profiler on an
	# unmodified workload: PyTorch/Kineto registers with it when KINETO_USE_DAEMON=1,
	# and `dyno gputrace` then configures the profiler over IPC.
	# https://github.com/facebookincubator/dynolog/blob/main/docs/pytorch_profiler.md
	if command -v dynolog >/dev/null 2>&1 && command -v dyno >/dev/null 2>&1; then
		echo "dynolog: dynolog and dyno already on PATH, skipping install."
		exit 0
	fi

	# Only x86_64 debian packages are published upstream.
	_arch=$(uname -m)
	if [ "$_arch" != "x86_64" ]; then
		echo "Error: dynolog pre-script only supports x86_64 (found $_arch)." >&2
		echo "Build dynolog from source and put dynolog/dyno on PATH, or use a" >&2
		echo "model-side torch.profiler instead." >&2
		exit 1
	fi
	if ! command -v dpkg >/dev/null 2>&1; then
		echo "Error: dynolog pre-script needs dpkg (Debian/Ubuntu base image)." >&2
		exit 1
	fi

	_DYNOLOG_PINNED_DEB='https://github.com/facebookincubator/dynolog/releases/download/v0.5.0/dynolog_0.5.0-0-amd64.deb'
	_dynolog_deb="${DYNOLOG_DEB_URL:-$_DYNOLOG_PINNED_DEB}"
	_dynolog_tmp="/tmp/dynolog.deb"

	if command -v curl >/dev/null 2>&1; then
		curl -fsSL -o "$_dynolog_tmp" "$_dynolog_deb"
	elif command -v wget >/dev/null 2>&1; then
		wget -q -O "$_dynolog_tmp" "$_dynolog_deb"
	else
		echo "Error: dynolog pre-script needs curl or wget to download the package." >&2
		exit 1
	fi

	# The package ships a systemd unit; enabling it fails in a container, which is
	# harmless because we run the daemon directly. Tolerate a non-zero dpkg exit
	# and verify by checking for the binaries instead.
	if [ "$(id -u)" -eq 0 ]; then
		dpkg -i "$_dynolog_tmp" || apt-get install -f -y -qq || true
	elif command -v sudo >/dev/null 2>&1; then
		sudo dpkg -i "$_dynolog_tmp" || sudo apt-get install -f -y -qq || true
	else
		echo "Error: dynolog pre-script needs root or sudo to install the package." >&2
		exit 1
	fi
	rm -f "$_dynolog_tmp"

	if ! command -v dynolog >/dev/null 2>&1 || ! command -v dyno >/dev/null 2>&1; then
		echo "Error: dynolog package installed but dynolog/dyno are not on PATH." >&2
		exit 1
	fi
	echo "dynolog: installed $(dynolog --help 2>&1 | head -1 || echo 'ok')"

	# Kineto's daemon registration landed in torch 1.13; warn rather than fail so
	# the tool stays usable for diagnosing the environment.
	if ! python3 -c 'import torch' 2>/dev/null; then
		echo "Warning: torch is not importable here; on-demand tracing needs a PyTorch workload." >&2
	fi
	;;

tracelens)
	# TraceLens pins protobuf>=6.31 and xprof, which routinely conflicts with a
	# workload's own torch/tensorboard stack. Install it into a fully isolated
	# venv (no --system-site-packages) so the model environment is untouched.
	_tl_venv="${TRACELENS_VENV:-/opt/madengine-tracelens-venv}"
	_TRACELENS_PINNED_REF='6f9bcdbf6cc9911eb650de57b345917ea4d31a17'
	_tl_ref="${TRACELENS_GIT_REF:-$_TRACELENS_PINNED_REF}"
	_tl_spec="${TRACELENS_PIP_SPEC:-git+https://github.com/AMD-AGI/TraceLens.git@${_tl_ref}}"

	if [ -x "${_tl_venv}/bin/python3" ] && "${_tl_venv}/bin/python3" -c 'import TraceLens' 2>/dev/null; then
		echo "TraceLens: already installed in ${_tl_venv}, skipping."
		exit 0
	fi

	if ! python3 -m venv "$_tl_venv" 2>/dev/null; then
		echo "python3 -m venv failed; attempting to install the venv module..." >&2
		if [ "$(id -u)" -eq 0 ] && command -v apt-get >/dev/null 2>&1; then
			apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv
		elif command -v sudo >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
			sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv
		fi
		if ! python3 -m venv "$_tl_venv"; then
			echo "Error: could not create a virtualenv at ${_tl_venv}." >&2
			echo "Install python3-venv, or set TRACELENS_VENV to an existing venv." >&2
			exit 1
		fi
	fi

	"${_tl_venv}/bin/python3" -m pip install --upgrade -q pip
	# TRACELENS_PIP_SPEC may embed credentials for a private mirror; keep it out of
	# the `set -x` trace and out of stderr.
	_tl_restore_x=0
	case $- in *x*) _tl_restore_x=1 ;; esac
	set +x
	if ! "${_tl_venv}/bin/python3" -m pip install -q "$_tl_spec"; then
		echo "Error: pip could not install TraceLens (spec omitted from logs)." >&2
		echo "Check network access, or override TRACELENS_PIP_SPEC / TRACELENS_GIT_REF." >&2
		[ "$_tl_restore_x" -eq 1 ] && set -x
		exit 1
	fi
	[ "$_tl_restore_x" -eq 1 ] && set -x
	unset _tl_restore_x
	"${_tl_venv}/bin/python3" -c 'import TraceLens; print("TraceLens import OK")'

	# .pftrace input needs traceconv. TraceLens downloads it on demand, which fails
	# in an air-gapped container, so pre-stage it here when we still have network.
	if ! command -v traceconv >/dev/null 2>&1 && command -v curl >/dev/null 2>&1; then
		if curl -fsSL -o /usr/local/bin/traceconv https://get.perfetto.dev/traceconv 2>/dev/null; then
			chmod +x /usr/local/bin/traceconv
			echo "TraceLens: pre-staged traceconv for .pftrace input."
		else
			echo "TraceLens: could not pre-stage traceconv (only needed for .pftrace input)."
		fi
	fi
	;;

esac
