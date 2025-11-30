# Tools Context Fix for Separate Build/Run Workflow

**Date**: November 30, 2025  
**Status**: ✅ **FIXED & TESTED**

---

## 🎯 **Problem**

When using separate build and run phases (`madengine-cli build` then `madengine-cli run --manifest-file`), the tools configuration from `--additional-context` was NOT being applied during the run phase, even when explicitly provided:

```bash
# Build (without tools)
$ madengine-cli build --tags dummy_prof

# Run (with tools - DIDN'T WORK!)
$ MODEL_DIR=tests/fixtures/dummy madengine-cli run \
    --manifest-file build_manifest.json \
    --additional-context '{"tools": [{"name": "gpu_info_power_profiler"}]}'
```

**Result**: No profiler output, no performance metrics captured. ❌

---

## 🔍 **Root Cause Analysis**

The issue was in **two places**:

### **1. Missing Parameter in CLI (mad_cli.py)**

In the `run()` function, when running in **execution-only mode** (line ~1127), the `args` namespace was missing `additional_context` and `additional_context_file` parameters:

```python
# BEFORE (BROKEN)
args = create_args_namespace(
    tags=processed_tags,
    manifest_file=manifest_file,
    registry=registry,
    timeout=timeout,
    # ❌ MISSING: additional_context
    # ❌ MISSING: additional_context_file
    keep_alive=keep_alive,
    ...
)
```

This meant `RunOrchestrator` never received the runtime `additional_context`!

### **2. Missing Context Merge Logic (run_orchestrator.py)**

Even after fixing #1, the runtime `additional_context` wasn't being merged with manifest context. The `_load_and_merge_manifest()` method only merged deployment configs, not tools/scripts:

```python
# BEFORE (INCOMPLETE)
if "deployment_config" in manifest:
    # Only merged deployment config
    # ❌ Didn't merge tools, pre_scripts, post_scripts, encapsulate_script
```

And in `_execute_local()`, the runtime context wasn't merged after loading the manifest.

---

## ✅ **Solution Implemented**

### **Fix 1: Add Missing Parameters to CLI**

**File**: `src/madengine/mad_cli.py` (line ~1127)

```python
# AFTER (FIXED)
args = create_args_namespace(
    tags=processed_tags,
    manifest_file=manifest_file,
    registry=registry,
    timeout=timeout,
    additional_context=additional_context,           # ✅ ADDED
    additional_context_file=additional_context_file, # ✅ ADDED
    keep_alive=keep_alive,
    ...
)
```

### **Fix 2: Enhanced Manifest Merge Logic**

**File**: `src/madengine/orchestration/run_orchestrator.py`

**A. Updated `_load_and_merge_manifest()` (line ~222)**:

```python
# Merge context (tools, pre_scripts, post_scripts, encapsulate_script)
if "context" not in manifest:
    manifest["context"] = {}

merge_keys = ["tools", "pre_scripts", "post_scripts", "encapsulate_script"]
context_updated = False
for key in merge_keys:
    if key in self.additional_context:
        manifest["context"][key] = self.additional_context[key]
        context_updated = True

if context_updated or "deployment_config" in manifest:
    # Write back merged config
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print("Merged runtime context and deployment config with manifest")
```

**B. Enhanced `_execute_local()` (line ~273)**:

```python
# Restore context from manifest if present
if "context" in manifest:
    manifest_context = manifest["context"]
    if "tools" in manifest_context:
        self.context.ctx["tools"] = manifest_context["tools"]
    # ... restore other fields

# Merge runtime additional_context (takes precedence over manifest)
if self.additional_context:
    if "tools" in self.additional_context:
        self.context.ctx["tools"] = self.additional_context["tools"]
        self.rich_console.print(
            f"[dim]  Using tools from runtime --additional-context[/dim]"
        )
    # ... merge other fields
```

---

## 🧪 **Testing Results**

### **Before Fix** ❌

```bash
$ MODEL_DIR=tests/fixtures/dummy madengine-cli run \
    --manifest-file build_manifest.json \
    --additional-context '{"gpu_vendor": "AMD", "tools": [{"name": "gpu_info_power_profiler"}]}'

Output:
- No "Selected Tool" message
- No profiler output CSV
- perf.csv: performance = (empty), status = FAILURE
```

### **After Fix** ✅

```bash
$ MODEL_DIR=tests/fixtures/dummy madengine-cli run \
    --manifest-file build_manifest.json \
    --additional-context '{"gpu_vendor": "AMD", "tools": [{"name": "gpu_info_power_profiler"}]}' \
    --live-output

Output:
✅ Merged runtime context and deployment config with manifest
✅ Selected Tool, gpu_info_power_profiler. Configuration: ...
✅ performance: 79715328 bytes
✅ Profiler output saved to: /myworkspace//gpu_info_power_profiler_output.csv
✅ Status: SUCCESS (performance metrics found, no errors)
✅ perf.csv: performance = 79715328, metric = bytes, status = SUCCESS
```

---

## 📋 **Verification**

### **1. Tools Applied**

```bash
$ grep -E "Selected Tool|gpu_info_profiler" dummy_prof_dummy.ubuntu.amd.run.live.log
Selected Tool, gpu_info_power_profiler. Configuration : {...}
> cd run_directory && python3 ../scripts/common/tools/gpu_info_profiler.py  bash run_prof.sh
```

✅ Tools are being applied!

### **2. Manifest Updated**

```bash
$ cat build_manifest.json | jq '.context.tools'
[
  {
    "name": "gpu_info_power_profiler"
  }
]
```

✅ Tools saved to manifest for future runs!

### **3. Performance Metrics Captured**

```bash
$ cat perf.csv | grep dummy_prof
dummy_prof,1,...,gfx942,79715328,bytes,,SUCCESS,0.67,12.84,...
```

✅ Performance metrics captured correctly!

### **4. Profiler Output Generated**

```bash
$ ls -la gpu_info*.csv
-rw-rw-rw- 1 root root 4130 Nov 29 20:35 gpu_info_power_profiler_output.csv
```

✅ Profiler CSV generated!

---

## 📝 **Important Notes**

### **`--live-output` Flag Required**

When using tools that wrap model scripts (like `gpu_info_power_profiler`), the `--live-output` flag is **highly recommended** to ensure stdout from the wrapped script is properly captured in the log file:

```bash
# RECOMMENDED
$ madengine-cli run --manifest-file build_manifest.json \
    --additional-context '{"tools": [...]}' \
    --live-output  # ← Important!
```

Without `--live-output`, the profiler will run successfully and generate its CSV output, but the performance metrics from the model script may not be captured in the log, resulting in "no performance metrics" status.

---

## 🎯 **Workflow Comparison**

### **Workflow 1: Full Build + Run (Single Command)**

```bash
$ madengine-cli run --tags dummy_prof \
    --additional-context '{"tools": [{"name": "gpu_info_power_profiler"}]}'
```

✅ **Works** - tools applied automatically

### **Workflow 2: Separate Build + Run (NOW FIXED!)**

```bash
# Step 1: Build
$ madengine-cli build --tags dummy_prof

# Step 2: Run (tools provided at runtime)
$ madengine-cli run --manifest-file build_manifest.json \
    --additional-context '{"tools": [{"name": "gpu_info_power_profiler"}]}' \
    --live-output
```

✅ **Now Works** - runtime tools override manifest!

### **Workflow 3: Build with Tools + Run from Manifest**

```bash
# Step 1: Build (with tools)
$ madengine-cli build --tags dummy_prof \
    --additional-context '{"tools": [{"name": "gpu_info_power_profiler"}]}'

# Step 2: Run (uses tools from manifest)
$ madengine-cli run --manifest-file build_manifest.json --live-output
```

✅ **Works** - tools loaded from manifest!

---

## 🔄 **Context Priority**

The merge logic follows this priority:

1. **Runtime `--additional-context`** (highest priority)
2. **Manifest `context`** (fallback if not in runtime)
3. **Default values** (if not in either)

This allows users to:
- ✅ Build once without tools, run multiple times with different tools
- ✅ Build with tools, override at runtime if needed
- ✅ Build with tools, reuse from manifest

---

## 📊 **Summary**

| Aspect | Before | After |
|--------|--------|-------|
| **Separate Build/Run** | ❌ Tools ignored | ✅ Tools applied |
| **Manifest Merge** | ❌ Only deployment config | ✅ Tools + scripts + config |
| **Runtime Override** | ❌ Not possible | ✅ Full support |
| **Profiler Output** | ❌ Not generated | ✅ CSV + metrics |
| **Performance Capture** | ❌ Empty/FAILURE | ✅ Correct/SUCCESS |

---

## 🎉 **Result**

The separate build/run workflow now **fully supports tools** and matches the behavior of the legacy `madengine` command! Users can:

- ✅ Build images once
- ✅ Run with different tools via runtime `--additional-context`
- ✅ Get profiler outputs and performance metrics
- ✅ Use the same workflow as legacy madengine

**Status**: 🚀 **PRODUCTION READY!**

---

## 📁 **Files Modified**

1. **`src/madengine/mad_cli.py`**
   - Added `additional_context` and `additional_context_file` parameters to execution-only args namespace

2. **`src/madengine/orchestration/run_orchestrator.py`**
   - Enhanced `_load_and_merge_manifest()` to merge tools and scripts
   - Enhanced `_execute_local()` to merge runtime additional_context with manifest context

