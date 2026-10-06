#!/usr/bin/env bash
# 
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
# 

set -e
set -x

# rtl trace (v0.3.3) sets HSA_TOOLS_LIB and LD_PRELOAD to get_lib_path(), which
# prefers the wheel's lib/librtl.so over /usr/local/lib. A native rebuild that
# only lands in /usr/local/lib is never the library that traces. Replace the
# wheel copy with the library built for this image.
_rtl_publish_native() {
	_built="$1"
	_pkg=$(python3 -c 'from rocm_trace_lite import get_lib_path; print(get_lib_path())' 2>/dev/null || true)
	if [ -n "$_pkg" ] && [ "$_pkg" != "$_built" ] && [ -f "$_pkg" ]; then
		install -m 755 "$_built" "$_pkg"
		echo "rocm-trace-lite: replaced ${_pkg} with the library built for this image."
	fi
}
if [ "${1:-}" = "--publish-native-librtl" ]; then
	_rtl_publish_native "${2:?built librtl.so}"
	exit 0
fi

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
				git build-essential pkg-config xxd cmake
		elif command -v sudo >/dev/null 2>&1; then
			sudo apt-get update -qq
			sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
				sqlite3 libsqlite3-dev libfmt-dev python3-pip nlohmann-json3-dev \
				git build-essential pkg-config xxd cmake
		else
			echo 'RPD pre-script: need root or sudo for apt-get' >&2
			exit 1
		fi
	elif [ "$os" == 'centos' ]; then
		if [ "$(id -u)" -eq 0 ]; then
			yum install -y gcc gcc-c++ make git cmake \
				libsqlite3x-devel.x86_64 fmt-devel python3-pip json-devel vim-common
		elif command -v sudo >/dev/null 2>&1; then
			sudo yum install -y gcc gcc-c++ make git cmake \
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
	
	# Build RPD tracer via upstream's CMake build (the repo replaced its old
	# per-directory Makefiles / `make rlog rpd` targets with this).
	cd ./rocmProfileData
	# rpd_tracer links against rlog; it's vendored as a self-referencing git
	# submodule (same repo, "rlog" branch) rather than an external project.
	git submodule update --init rlog
	# `make install` configures+builds via CMake and installs rlog, rpd_tracer,
	# and rocpd_python (via pip) into /usr/local. Skip the remote helper and
	# web viewer — neither was built by the old make-based flow.
	make install CMAKE_ARGS="-DRPD_BUILD_REMOTE=OFF -DRPD_BUILD_VIEWER=OFF"
	if [ $? -ne 0 ]; then
		echo "Error: Failed to build RPD tracer"
		exit 1
	fi
	# `cmake --install` drops librlog.so into /usr/local/lib but leaves the image's
	# ld.so cache stale, so LD_PRELOAD of librpd_tracer.so aborts with
	# "librlog.so: cannot open shared object file". Refresh the cache.
	if command -v ldconfig >/dev/null 2>&1; then
		ldconfig /usr/local/lib || true
	fi
	cd ..
	
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

	# The release wheel's librtl.so is built against one ROCm's HSA headers. On a
	# TheRock image there is no /opt/rocm, and that .so records 0 GPU ops (the
	# queue-intercept function is at a different offset). Rebuild against this
	# image's headers. /opt/rocm images keep the wheel.
	if [ ! -f /opt/rocm/include/hsa/hsa.h ] && [ "${ROCM_TRACE_LITE_SKIP_NATIVE_BUILD:-0}" != "1" ]; then
		_rtl_hdr=""
		for _root in ${ROCM_PATH:-} /opt/rocm; do
			if [ -n "$_root" ] && [ -f "$_root/include/hsa/hsa.h" ]; then
				_rtl_hdr="$_root"
				break
			fi
		done
		if [ -z "$_rtl_hdr" ]; then
			echo "Warning: no HSA headers found; keeping the prebuilt librtl.so." >&2
		else
			_rtl_ref="${ROCM_TRACE_LITE_GIT_REF:-v0.3.3}"
			# amdq1 hooks hsa_amd_queue_create. HIP 7.15 does not use hsa_queue_create.
			_rtl_build_id="${_rtl_ref}+amdq1"
			_rtl_stamp=/usr/local/lib/librtl.so.madengine-ref
			if [ -f /usr/local/lib/librtl.so ] && [ "$(cat "$_rtl_stamp" 2>/dev/null || true)" = "$_rtl_build_id" ]; then
				echo "rocm-trace-lite: native librtl.so for ${_rtl_ref} already installed."
				_rtl_publish_native /usr/local/lib/librtl.so
			else
				_rtl_libdir=$(find "$(dirname "$_rtl_hdr")" -maxdepth 4 -name 'libhsa-runtime64.so' -printf '%h\n' -quit 2>/dev/null || true)
				if [ -z "$_rtl_libdir" ]; then
					echo "Error: HSA headers at ${_rtl_hdr} but libhsa-runtime64.so was not found nearby." >&2
					echo "The prebuilt librtl.so will not record kernel dispatches on this runtime." >&2
					exit 1
				fi
				if [ ! -f /usr/include/sqlite3.h ] || ! command -v g++ >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
					if [ "$(id -u)" -eq 0 ] && command -v apt-get >/dev/null 2>&1; then
						apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq g++ make git libsqlite3-dev
					elif command -v sudo >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
						sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq g++ make git libsqlite3-dev
					fi
				fi
				if ! command -v g++ >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1 || [ ! -f /usr/include/sqlite3.h ]; then
					echo "Error: building librtl.so needs g++, git, and sqlite3.h." >&2
					exit 1
				fi
				_rtl_src=/opt/madengine-rocm-trace-lite
				rm -rf "$_rtl_src"
				git clone --depth 1 --branch "$_rtl_ref" https://github.com/sunway513/rocm-trace-lite.git "$_rtl_src"
				cat > /tmp/rtl-amd-queue-create.patch << 'ENDPATCH'
diff --git a/src/hsa_intercept.cpp b/src/hsa_intercept.cpp
index 18f055b..fa5f4da 100644
--- a/src/hsa_intercept.cpp
+++ b/src/hsa_intercept.cpp
@@ -521,6 +521,81 @@ static void queue_intercept_cb(const void* in_packets, uint64_t count,
 
 // ---- HSA API table replacement ----
 
+// HIP 7.15 creates compute queues through hsa_amd_queue_create. The export
+// jumps through amdExtTable, so replacing only hsa_queue_create never sees them.
+static hsa_status_t my_hsa_amd_queue_create(
+    hsa_agent_t agent, hsa_amd_queue_create_desc_t* descs, uint32_t num_descs) {
+
+    if (!g_intercept_available || descs == nullptr || num_descs != 1 ||
+        descs[0].engine_type != 0 || descs[0].flags != 0) {
+        return g_orig_ext.hsa_amd_queue_create_fn(agent, descs, num_descs);
+    }
+
+    hsa_amd_queue_create_desc_t& d = descs[0];
+    static std::atomic<int> logged_amd{0};
+    if (logged_amd.exchange(1) == 0) {
+        fprintf(stderr, "rtl: hsa_amd_queue_create bytes=%u priv=%u\n",
+                d.queue_size_bytes, d.engine.compute.private_segment_size);
+    }
+
+    uint32_t packets = d.queue_size_bytes / 64;
+    if (packets < 2) {
+        packets = 64;
+    }
+    uint32_t priv = d.engine.compute.private_segment_size;
+    if (priv == UINT32_MAX) {
+        priv = 0;
+    }
+    hsa_queue_t* queue = nullptr;
+    hsa_status_t status = g_orig_ext.hsa_amd_queue_intercept_create_fn(
+        agent, packets, d.engine.compute.type, d.callback, d.callback_data,
+        priv, 0, &queue);
+    if (status != HSA_STATUS_SUCCESS || queue == nullptr) {
+        fprintf(stderr,
+                "rtl: amd intercept_create failed (0x%x), using runtime queue\n",
+                (unsigned)status);
+        return g_orig_ext.hsa_amd_queue_create_fn(agent, descs, num_descs);
+    }
+    d.queue = queue;
+
+    if (g_orig_ext.hsa_amd_profiling_set_profiler_enabled_fn != nullptr) {
+        hsa_status_t prof_status =
+            g_orig_ext.hsa_amd_profiling_set_profiler_enabled_fn(queue, true);
+        if (prof_status != HSA_STATUS_SUCCESS) {
+            fprintf(stderr,
+                    "rtl: warning: failed to enable profiling on amd queue (status=%d)\n",
+                    (int)prof_status);
+        }
+    }
+    if (g_orig_ext.hsa_amd_queue_set_priority_fn != nullptr) {
+        g_orig_ext.hsa_amd_queue_set_priority_fn(queue, d.priority);
+    }
+    if (d.engine.compute.cu_mask_count != 0 && d.engine.compute.cu_mask != nullptr &&
+        g_orig_ext.hsa_amd_queue_cu_set_mask_fn != nullptr) {
+        g_orig_ext.hsa_amd_queue_cu_set_mask_fn(
+            queue, d.engine.compute.cu_mask_count, d.engine.compute.cu_mask);
+    }
+
+    auto* qi = new QueueInfo;
+    qi->device_id = 0;
+    qi->queue_handle = (uint64_t)queue;
+    {
+        std::lock_guard<std::mutex> lock(g_agent_mutex);
+        for (size_t i = 0; i < g_gpu_agents.size(); i++) {
+            if (g_gpu_agents[i].handle == agent.handle) {
+                qi->device_id = (int)i;
+                break;
+            }
+        }
+    }
+    {
+        std::lock_guard<std::mutex> lock(g_queue_mutex);
+        g_queue_map[(uint64_t)queue] = *qi;
+    }
+    g_orig_ext.hsa_amd_queue_intercept_register_fn(queue, queue_intercept_cb, qi);
+    return HSA_STATUS_SUCCESS;
+}
+
 static hsa_status_t my_hsa_queue_create(
     hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
     void (*callback)(hsa_status_t, hsa_queue_t*, void*),
@@ -707,7 +782,6 @@ extern "C" bool OnLoad(void* pTable,
     // Replace queue creation and executable freeze
     table->core_->hsa_queue_create_fn = my_hsa_queue_create;
     table->core_->hsa_executable_freeze_fn = my_hsa_executable_freeze;
-
     // Discover GPU agents (immutable after this point)
     hsa_iterate_agents(agent_iterate_cb, nullptr);
     fprintf(stderr, "rtl: found %zu GPU agent(s)\n", g_gpu_agents.size());
@@ -746,6 +820,11 @@ extern "C" bool OnLoad(void* pTable,
         }
     }
 
+    if (g_intercept_available && table->amd_ext_->hsa_amd_queue_create_fn != nullptr) {
+        table->amd_ext_->hsa_amd_queue_create_fn = my_hsa_amd_queue_create;
+        fprintf(stderr, "rtl: hooked hsa_amd_queue_create\n");
+    }
+
     // Initialize lock-free ring buffer (full reset for re-load safety)
     g_central_head.store(0, std::memory_order_relaxed);
     g_central_tail.store(0, std::memory_order_relaxed);
ENDPATCH
				git -C "$_rtl_src" apply /tmp/rtl-amd-queue-create.patch
				_rtl_stage=/tmp/rtl-rocm-prefix
				rm -rf "$_rtl_stage"
				mkdir -p "$_rtl_stage"
				ln -sfn "$_rtl_hdr/include" "$_rtl_stage/include"
				ln -sfn "$_rtl_libdir" "$_rtl_stage/lib"
				make -C "$_rtl_src" -j"$(nproc 2>/dev/null || echo 2)" HIP_PATH="$_rtl_stage"
				install -d /usr/local/lib
				install -m 755 "$_rtl_src/librtl.so" /usr/local/lib/librtl.so
				echo "$_rtl_build_id" > "$_rtl_stamp"
				_rtl_publish_native /usr/local/lib/librtl.so
				if command -v ldconfig >/dev/null 2>&1; then
					ldconfig /usr/local/lib || true
				fi
				echo "rocm-trace-lite: built librtl.so against ${_rtl_hdr} (libhsa ${_rtl_libdir})."
			fi
		fi
	fi

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
	_dynolog_tmp="/tmp/dynolog.deb"

	# DYNOLOG_DEB_URL may embed credentials for a private mirror; keep it out of
	# the `set -x` trace, both where it is read and where it is used.
	_dynolog_restore_x=0
	case $- in *x*) _dynolog_restore_x=1 ;; esac
	set +x
	_dynolog_deb="${DYNOLOG_DEB_URL:-$_DYNOLOG_PINNED_DEB}"
	if command -v curl >/dev/null 2>&1; then
		curl -fsSL -o "$_dynolog_tmp" "$_dynolog_deb"
	elif command -v wget >/dev/null 2>&1; then
		wget -q -O "$_dynolog_tmp" "$_dynolog_deb"
	else
		echo "Error: dynolog pre-script needs curl or wget to download the package." >&2
		exit 1
	fi
	[ "$_dynolog_restore_x" -eq 1 ] && set -x
	unset _dynolog_restore_x _dynolog_deb

	# The package ships a systemd unit; enabling it fails in a container, which is
	# harmless because we run the daemon directly. Tolerate a non-zero dpkg exit
	# and verify by checking for the binaries instead.
	if [ "$(id -u)" -eq 0 ]; then
		dpkg -i "$_dynolog_tmp" || { apt-get update -qq && apt-get install -f -y -qq; } || true
	elif command -v sudo >/dev/null 2>&1; then
		sudo dpkg -i "$_dynolog_tmp" || { sudo apt-get update -qq && sudo apt-get install -f -y -qq; } || true
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

	# TraceLens's pftrace reports convert with Perfetto's traceconv, a launcher that
	# downloads its binary with curl on first use. Slim framework images (e.g.
	# rocm/primus) ship without curl, which fails every pftrace report.
	if ! command -v curl >/dev/null 2>&1; then
		echo "TraceLens: curl not found; installing it for traceconv (pftrace reports)..."
		if [ "$(id -u)" -eq 0 ] && command -v apt-get >/dev/null 2>&1; then
			apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl
		elif [ "$(id -u)" -eq 0 ] && command -v yum >/dev/null 2>&1; then
			yum install -y -q curl
		elif command -v sudo >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
			sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl
		fi
		command -v curl >/dev/null 2>&1 || \
			echo "Warning: curl unavailable; TraceLens pftrace reports will fail (traceconv download)." >&2
	fi

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
	_tl_spec="${TRACELENS_PIP_SPEC:-git+https://github.com/AMD-AGI/TraceLens.git@${_tl_ref}}"
	if ! "${_tl_venv}/bin/python3" -m pip install -q "$_tl_spec"; then
		echo "Error: pip could not install TraceLens (spec omitted from logs)." >&2
		echo "Check network access, or override TRACELENS_PIP_SPEC / TRACELENS_GIT_REF." >&2
		[ "$_tl_restore_x" -eq 1 ] && set -x
		exit 1
	fi
	[ "$_tl_restore_x" -eq 1 ] && set -x
	unset _tl_restore_x _tl_spec
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
