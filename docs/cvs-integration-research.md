# CVS Integration Research

Research only — no code changes made. Findings below are grounded in files actually
read in both repos as of 2026-09-21.

## 1. Architecture summary

**madengine** (`/home/ysha/codebase/madengine`) is a single-process CLI that a user
runs directly on a target host (a workstation, a K8s-API-reachable box, or — for
SLURM — the login node itself; see the module docstring in
`src/madengine/deployment/slurm.py:1-12`: "User has already SSH'd to SLURM login node
manually. madengine is executed ON the login node, not remotely."). It never opens SSH
connections to compute nodes itself. For SLURM it shells out locally to `sbatch`/
`squeue`/`scontrol`/`sinfo`/`srun` (`slurm.py:203-233`, `slurm_node_selector.py`); for
Kubernetes it talks to the K8s API (`kubernetes.py:193-245`) and lets the scheduler place
pods. GPU/ROCm validation (`src/madengine/utils/gpu_validator.py`) and vendor/arch
detection (`src/madengine/core/context.py:26,185`) only ever run **locally**, on the
one machine `madengine` is invoked from — there is no equivalent check against the
actual remote compute nodes a distributed job will land on. Log-based failure
detection is a static substring/regex scan (`src/madengine/execution/container_runner_helpers.py:16-30,111-153`)
against the container's own stdout/stderr — no dmesg/journalctl/kernel-log
correlation.

**cvs** (`/home/ysha/codebase/cvs`) is the inverse shape: a pytest-based validation
suite that is inherently multi-node and SSH/orchestrator-driven. Per its README
(`/home/ysha/codebase/cvs/README.md:6`) it "requires only SSH connectivity to cluster
nodes (no Slurm or Kubernetes)". Cluster topology + SSH creds live in a JSON file
(`cvs/input/cluster_file/cluster.json`: `username`, `priv_key_file`, `head_node_dict`,
`node_dict` keyed by IP/hostname). Execution goes through an `orch` fixture
(`cvs/tests/conftest.py`, described in `AGENTS.md:14,49-53`) that wraps either baremetal
SSH (`cvs/core/orchestrators/baremetal.py`) or container backends
(`cvs/core/orchestrators/container.py`), with a legacy parallel-SSH path
(`cvs/lib/parallel_ssh_lib.py`, `phdl`/`shdl`) still used by many suites. Test suites are
organized by concern under `cvs/tests/`: `health/` (RVS GPU stress/enumeration/PCIe/
babel-stream — see `cvs/tests/health/rvs_cvs.py`), `preflight/` (node/fabric/RDMA
connectivity checks, `cvs/lib/preflight/*.py`), `rccl/`, `ibperf/`, `platform/`,
`training/`, `inference/`, `benchmark/`. The CLI (`cvs/cli_plugins/`, documented in
`docs/reference/cli/cvs-run.rst`) exposes `cvs run` (pytest wrapper), `cvs exec`
(arbitrary command across all nodes, `cvs/cli_plugins/exec_plugin.py`), `cvs scp`,
`cvs monitor check_cluster_health` (`cvs/monitors/check_cluster_health.py` — PCIe link
width, dmesg scan, driver errors, journalctl scan, NIC link-flap detection, all via
`cvs/lib/verify_lib.py`), and `cvs list`/`cvs generate`/`cvs config`.

The core asymmetry that motivates integration: madengine assumes the node(s) it will
run on (or submit to) are healthy and only checks the *local* machine; cvs exists
specifically to answer "are these N nodes healthy" over SSH, with mature GPU
burn-in, connectivity, and dmesg/journalctl tooling madengine does not have.

## 2. Integration ideas (highest value/feasibility first)

### 2.1 Replace/augment `SlurmNodeSelector`'s heuristic GPU health check with a cvs health run

**madengine problem:** `src/madengine/deployment/slurm_node_selector.py:166-291`
(`check_node_health`) determines whether a SLURM candidate node is "clean" using a
crude heuristic: it `srun`s a script that greps `ps aux` for `ray::`/`vllm`/`raylet`
process names, then **estimates** GPU memory usage as
`process_count * 45.0` GB (line 255, comment: "Rough heuristic... observed from Job
2437") against a **hardcoded** assumed total of `192.0 * 4` GB (line 246, "Assume 4
GPUs"). This is fragile — it doesn't query actual GPU memory, doesn't detect
degraded/failed GPUs, ECC errors, thermal throttling, PCIe link degradation, or driver
issues; it only detects stale Ray/vLLM processes by name-matching. It's invoked
unconditionally in `deploy()` before every multi-node sbatch submission
(`slurm.py:1231-1296`, gated by `enable_node_check`, default `True`).

**cvs capability:** `cvs/tests/health/rvs_cvs.py` runs AMD's RVS (ROCm Validation
Suite) modules — GPU enumeration (`test_rvs_gpu_enumeration`), memory test
(`test_rvs_mem_test`), GST stress (`test_rvs_gst_single`), PCIe bandwidth
(`test_rvs_pebb_single`), P2P (`test_rvs_pbqt_single`) — over the `orch` fixture, with
real device detection (`get_gpu_device_name`, using `amd-smi static -a -g 0 --json`,
lines 187-237) and pass/fail parsing against RVS's own diagnostic output. This is
actual GPU health signal, not a process-count proxy. `cvs/monitors/check_cluster_health.py`
additionally offers a lighter-weight, faster check (PCIe link width, dmesg scan,
driver-error scan, journalctl scan, NIC flap) that would be cheaper to run per-sbatch-submission
than a full RVS pass.

**Feasibility:** Medium. cvs requires its own cluster JSON (SSH creds + node list) and
a separate install (`pip install -e .` / `make install`), so the cleanest integration
is not a Python import but shelling out to `cvs monitor check_cluster_health
--cluster_file <generated>` (or `cvs run health rvs_cvs::test_rvs_gpu_enumeration`) from
`SlurmNodeSelector.select_nodes()`, mapping its PASS/FAIL-per-node output back into the
existing `NodeStatus`/`NodeHealth` enum. Need to auto-generate the cvs cluster JSON
from the SLURM candidate node list (cvs already has a generator for this:
`cvs/input/generate/cluster_json.py`, exposed as `cvs generate cluster_json`). Risk:
adds an external dependency and non-trivial latency (RVS GST is a stress test, not
instant) to every job submission; would need a "light" mode (the `check_cluster_health`
monitor path, not full RVS) to stay fast enough for a preflight gate. Open question:
does the SLURM login node have SSH access to the same compute nodes cvs would target,
or does it need srun-only reachability (as today)? If SSH isn't available from the
login node, this integration is blocked outright.

### 2.2 Pre-flight GPU/ROCm validation on remote nodes, not just the local one, before distributed runs

**madengine problem:** `src/madengine/core/context.py:26,185` and
`src/madengine/utils/gpu_validator.py` (`validate_gpu_installation`,
`ROCmValidator.validate()`) only ever inspect the machine the `madengine` process is
running on (checks `/dev/kfd`, `rocminfo`, `amd-smi`/`rocm-smi` presence, KFD topology
— all local filesystem/subprocess calls, no remote target parameter exists anywhere in
the class). For a SLURM job requesting N nodes, or a K8s job scheduled onto arbitrary
GPU-labeled nodes (`kubernetes.py:216-245` only checks that nodes advertise the GPU
*resource count* via the K8s API, not that ROCm is actually healthy on them), there is
no equivalent "is ROCm installed and healthy on the nodes this job will actually run on"
gate. A node could pass K8s's device-plugin resource check yet have a broken ROCm
userspace stack, and the failure would only surface mid-job.

**cvs capability:** `cvs/tests/health/install/install_*.py` and
`cvs/tests/preflight/` cover exactly this class of check across arbitrary node sets via
SSH — no Slurm/K8s coupling needed (`README.md:6`). `cvs/lib/preflight/version_check.py`
and `cvs/lib/preflight/tier3_info.py` (seen in the `lib/preflight` directory listing)
suggest existing ROCm/driver version consistency checks across nodes, which
madengine's local-only `ROCmValidator` cannot do (it can't compare versions *across*
nodes at all).

**Feasibility:** Medium-high value, medium complexity. Same shelling-out approach as
2.1: generate a cluster JSON scoped to the job's candidate nodes and invoke a cvs
preflight suite as a gate. For K8s specifically this is harder because pod placement
isn't known until scheduling; would need to run this against the *labeled node pool*
before job creation, not against pods.

### 2.3 Use cvs `preflight` fabric/RDMA connectivity checks before multi-node distributed launches

**madengine problem:** For multi-node torchrun/deepspeed/megatron jobs, `slurm.py`
generates node-IP-resolution logic inline in the sbatch template (e.g. the SGLang
disaggregated launcher's node-IP resolution loop at `slurm.py:986-1037`, which has
extensive hand-rolled logic and comments about loopback-address pitfalls with
`/etc/hosts` on Ubuntu). This is IP *resolution*, not connectivity *validation* — there
is no check that RDMA/RoCE interfaces between the selected nodes are actually up and
performant before launching a job that depends on them (NCCL/RCCL init failures at
job start are a common distributed-training failure mode this would catch earlier).

**cvs capability:** `cvs/lib/preflight/rdma_connectivity.py`, `ifoe_l2_connectivity.py`,
`gid_consistency.py`, `interface_consistency.py`, `scaleup_fabric.py` (all in
`cvs/lib/preflight/`) exist specifically to validate RDMA/RoCE fabric health and GID
consistency across a node set before a distributed job — precisely the class of
failure that currently only surfaces as an opaque NCCL timeout deep into a madengine
multi-node run.

**Feasibility:** Medium. Same integration shape as 2.1/2.2 (shell out, map results).
Higher payoff for large multi-node RCCL-sensitive training/inference jobs; lower payoff
for single-node or small (2-4 node) jobs where fabric issues are rarer. Risk: RDMA
checks can be slow and require elevated privileges/interface names that must be
supplied per-cluster (`cvs/lib/preflight/node_smoke.py:63-76` shows RDMA interface
names must be configured, e.g. via `connectivity_check.rdma.interfaces`) — this is
extra config surface madengine's `additional_context` schema would need to carry or
proxy.

### 2.4 Invoke cvs as a `pre_scripts` hook (lowest-friction, opt-in integration)

**madengine problem:** madengine already has a first-class hook point for exactly this
kind of "run something before the model" need. Per the CLAUDE.md architecture summary:
"Keys like `k8s`, `slurm`, `distributed`, `tools`, `pre_scripts`, `post_scripts` drive
behavior" and "During run, `scripts/common/` is populated from the madengine package
(pre_scripts, post_scripts, tools)". This is a generic script-injection mechanism, not
specific to any deployment target.

**cvs capability:** Any `cvs run <suite>` or `cvs monitor check_cluster_health`
invocation is just a CLI command with a cluster/config JSON — trivially wrappable as a
`pre_scripts` entry that runs before the model container starts, gating the run if cvs
reports failures (non-zero exit).

**Feasibility:** Low complexity, but lower value than 2.1-2.3 because it's *user*
opt-in per model/run rather than an engine-level reliability improvement — it doesn't
change madengine's own defaults or fix `SlurmNodeSelector`'s heuristic. Good as a
stopgap or for users who want cvs gating without engine code changes: essentially "free"
today, since `pre_scripts` already exists. Worth documenting as a pattern even without
code changes, but doesn't address the SlurmNodeSelector reliability gap directly since
that logic runs before node *selection*, not before the model script.

**Status:** Implemented. See `src/madengine/scripts/common/pre_scripts/cvs_health_gate.sh` and the "Gating a run on cvs cluster health" section in `docs/configuration.md`.

### 2.5 Correlate container run failures with cvs's dmesg/journalctl scanning conventions

**madengine problem:** `container_runner_helpers.py:16-30` (`DEFAULT_LOG_ERROR_PATTERNS`)
and `log_text_has_error_pattern` (lines 111-153) only scan the container's own
stdout/stderr for substrings like "CUDA out of memory" / "Traceback" / "FAILED". A
class of distributed-training failures (GPU falls off the bus, Xid errors, PCIe AER,
NIC flap) shows up in the **host kernel log**, not container stdout, and madengine has
no host-side dmesg/journalctl correlation at all.

**cvs capability:** `cvs/lib/verify_lib.py`'s `full_dmesg_scan`, `full_journalctl_scan`,
`verify_driver_errors`, `verify_nic_link_flap` (used in
`check_cluster_health.py:82-100`) and the time-bounded dmesg pattern documented in
`AGENTS.md:62-69` (`start_time`/`end_time` bracketing via `orch.exec('date ...')`,
then `verify_dmesg_for_errors(handle, start_time, end_time, till_end_flag=False)`) are
a mature, already-time-bounded pattern for exactly this.

**Feasibility:** Medium-high complexity to integrate directly (would require SSH
access from wherever madengine runs, which container_runner.py — a purely local
Docker wrapper — does not have architecturally), but the *pattern* (time-bounded
dmesg/journalctl scan bracketing the run) is portable even without depending on cvs
code: madengine could adopt the same technique for local host dmesg in
`container_runner.py`, independent of any cvs dependency. Listed here because the
value is real but the "integration" here is more "borrow the pattern" than "call cvs
code" — flagged as lower-confidence in scope.

## 3. Not recommended / out of scope

- **Replacing madengine's SLURM/K8s deployment logic with cvs's orchestrator layer
  (`cvs/core/orchestrators/`).** cvs's orchestrator abstracts SSH/baremetal/container
  execution for running *test suites*, not for submitting production training/inference
  jobs with madengine's launcher-specific templating (torchrun/vLLM/SGLang/DeepSpeed/
  Megatron/TorchTitan/Primus — `slurm.py:764-1215`). The two systems solve different
  problems (job orchestration vs. cluster validation); merging them would be a rewrite,
  not an integration, and cvs's own `AGENTS.md` explicitly scopes it to "certify [cluster]
  readiness for production workloads," not run them.
- **Sharing a single config schema/file format between the two tools.** madengine's
  `additional_context` (parsed via `ast.literal_eval`, per `context.py` and the project
  CLAUDE.md) and cvs's `cluster_file`/`config_file` JSON (parsed via `json.load`, per
  `cvs/tests/health/rvs_cvs.py:36-37` and `cluster.json`) are different formats for
  different purposes (deployment/model config vs. cluster topology/SSH creds). Unifying
  them would churn both tools' user-facing config surfaces for marginal benefit; better
  to keep them separate and only pass a generated cvs cluster JSON through as an
  implementation detail of whichever integration idea above is chosen.
- **Continuous cluster monitoring (`cvs monitor`, `cluster-mon/` dashboard) as part of
  madengine's `run` flow.** `cvs/monitors/cluster-mon/` is a standing web
  service/dashboard (Docker Compose, frontend/backend) meant to run independently and
  continuously, not as a per-invocation gate inside a CLI tool's synchronous run flow.
  Fine to recommend running it *alongside* madengine-driven clusters operationally, but
  it isn't something madengine's `build`/`run`/`report` commands should invoke or block on.
