#!/usr/bin/env python3
"""
SLURM Node Selector with GPU Cleanup

Helps SLURM select clean GPU nodes by checking for stale processes before
job submission. Prevents "out of memory" errors in multi-node vLLM/Ray jobs.

Uses srun (not SSH) to check and clean nodes - works from SLURM login node.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import re
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from rich.console import Console
from rich.table import Table


# rocminfo lists the CPU agents first ("Name: AMD EPYC ...") and the GPU agents
# after them ("Name: gfx942"), so the first gfx token is the GPU's ISA name. The
# suffix is hex -- gfx90a is a real target -- so [0-9] alone would read it as gfx90.
_GFX_ARCH_RE = re.compile(r"\bgfx[0-9a-f]+\b")


def parse_gpu_arch(output: Optional[str]) -> Optional[str]:
    """Return the first gfx architecture named in rocminfo output, or None."""
    if not output:
        return None
    match = _GFX_ARCH_RE.search(output)
    return match.group(0) if match else None


def _first_plain_node(nodelist: Optional[str]) -> Optional[str]:
    """First node of a comma-separated nodelist, or None if it is not plain.

    A bracketed range ("node[01-04]") cannot be split on commas, and passing the
    whole list with -N1 is rejected by srun, so such a list is left out and the
    probe falls back to the partition's choice.
    """
    if not nodelist or "[" in nodelist:
        return None
    first = nodelist.split(",")[0].strip()
    return first or None


class NodeHealth(Enum):
    """Health status of a compute node."""

    CLEAN = "clean"  # No stale processes, ready to use
    DIRTY = "dirty"  # Has stale Ray/vLLM processes
    UNREACHABLE = "unreachable"  # Cannot connect to node
    UNKNOWN = "unknown"  # Status check failed


@dataclass
class NodeStatus:
    """Status of a compute node's GPUs."""

    node: str
    health: NodeHealth
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
    process_count: int
    error_message: Optional[str] = None

    @property
    def memory_free_gb(self) -> float:
        """Calculate free GPU memory."""
        return self.gpu_memory_total_gb - self.gpu_memory_used_gb

    @property
    def memory_usage_percent(self) -> float:
        """Calculate memory usage percentage."""
        if self.gpu_memory_total_gb == 0:
            return 0.0
        return (self.gpu_memory_used_gb / self.gpu_memory_total_gb) * 100


class SlurmNodeSelector:
    """
    Selects clean GPU nodes for SLURM job allocation.

    Checks candidate nodes for stale Ray/vLLM processes that would cause
    OOM errors. Can automatically clean dirty nodes or recommend exclusion.
    """

    # Memory threshold: nodes with >50GB used are considered dirty
    MEMORY_THRESHOLD_GB = 50.0

    # Process patterns that indicate stale processes
    STALE_PATTERNS = ["ray::", "RayWorkerWrapper", "raylet", "vllm"]

    def __init__(
        self,
        console: Optional[Console] = None,
        auto_cleanup: bool = False,
        verbose: bool = False,
        timeout: int = 120,
        reservation: Optional[str] = None,
    ):
        """
        Initialize node selector.

        Args:
            console: Rich console for output
            auto_cleanup: Automatically clean dirty nodes
            verbose: Enable verbose logging
            timeout: Seconds to wait for a probe srun. This is a QUEUE wait, not a
                command runtime: the probe cannot start until the scheduler gives it
                a slot, so the value has to cover how long that takes on a busy
                cluster. Override with slurm.node_check_timeout.
            reservation: SLURM reservation name (passed through to srun health/cleanup)
        """
        self.console = console or Console()
        # Set when select_nodes runs. The probe below needs it: a login node with no
        # default partition rejects a bare srun, and the whole check then reports
        # every node unreachable.
        self.partition: Optional[str] = None
        self.auto_cleanup = auto_cleanup
        self.verbose = verbose
        self.timeout = timeout
        self.reservation = reservation

    # Max candidates to check (avoids excessive checks on large clusters)
    MAX_CANDIDATES_CAP = 100

    def get_candidate_nodes(
        self,
        partition: str,
        count: int,
        exclude: Optional[str] = None,
        constraint: Optional[str] = None,
    ) -> Optional[List[str]]:
        """
        Query SLURM for idle candidate nodes in partition.

        Args:
            partition: SLURM partition name
            count: Number of nodes needed (used for optional cap)
            exclude: Comma-separated nodes to exclude
            constraint: SLURM constraint filter

        Returns:
            List of idle node names (all idle, up to MAX_CANDIDATES_CAP)
        """
        cmd = [
            "sinfo",
            "-p",
            partition,
            "-N",  # Node-oriented format
            "-h",  # No header
            "-o",
            "%N",  # Node name only
            "-t",
            "idle",  # Idle nodes only
        ]

        if constraint:
            cmd.extend(["-C", constraint])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                if self.verbose:
                    self.console.print(
                        f"[yellow]⚠ sinfo failed: {result.stderr}[/yellow]"
                    )
                return None

            # Parse nodes
            all_nodes = set()
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line:
                    all_nodes.add(line)

            # Remove excluded nodes
            if exclude:
                excluded = set(exclude.split(","))
                all_nodes -= excluded

            # Return all idle nodes, capped to avoid excessive checks
            candidates = sorted(list(all_nodes))[: self.MAX_CANDIDATES_CAP]
            return candidates

        except subprocess.TimeoutExpired:
            self.console.print("[yellow]⚠ sinfo timed out[/yellow]")
            return None
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow]⚠ Query failed: {e}[/yellow]")
            return None

    def check_node_health(
        self, node: str, job_name: Optional[str] = None
    ) -> NodeStatus:
        """
        Check GPU health on a node using srun.

        Uses srun to execute GPU check on the node without SSH.
        Checks for stale Ray/vLLM processes and GPU memory usage.

        Args:
            node: Node name to check
            job_name: Optional SLURM job name for this srun (enables cleanup of health-check jobs)

        Returns:
            NodeStatus with health information
        """
        # GPU check script (runs on compute node)
        check_script = """
set -e

# Try amd-smi first, then rocm-smi
if command -v amd-smi &> /dev/null; then
    GPU_TOOL="amd-smi"
    GPU_INFO=$(amd-smi list 2>/dev/null || echo "GPU_CHECK_FAILED")
elif command -v rocm-smi &> /dev/null; then
    GPU_TOOL="rocm-smi"
    GPU_INFO=$(rocm-smi 2>/dev/null || echo "GPU_CHECK_FAILED")
else
    echo "NO_GPU_TOOL_FOUND"
    exit 1
fi

echo "===GPU_INFO==="
echo "$GPU_INFO"
echo "===END_GPU_INFO==="

# Check for stale processes
echo "===PROCESSES==="
ps aux | grep -E "(ray::|RayWorkerWrapper|raylet|vllm)" | grep -v grep || echo "NO_PROCESSES"
echo "===END_PROCESSES==="
"""
        srun_cmd = [
            "srun",
            f"--nodelist={node}",
            "--ntasks=1",
            "--time=00:01:00",
            "--overlap",  # Allow overlap with running jobs
            "--quiet",
        ]
        # Without this the probe inherits whatever default partition the login node
        # has -- on OCI amd-rccl there is none, so srun answers "Invalid
        # specification" and every node is recorded UNREACHABLE. Five builds
        # reported all 46-50 nodes unreachable for this reason, which is also why
        # the check has been standing down instead of doing its job.
        if self.partition:
            srun_cmd.append(f"--partition={self.partition}")
        if job_name:
            srun_cmd.append(f"--job-name={job_name}")
        if self.reservation:
            srun_cmd.append(f"--reservation={self.reservation}")
        srun_cmd.extend(["bash", "-c", check_script])

        try:
            result = subprocess.run(
                srun_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode != 0:
                return NodeStatus(
                    node=node,
                    health=NodeHealth.UNREACHABLE,
                    gpu_memory_used_gb=0.0,
                    gpu_memory_total_gb=0.0,
                    process_count=0,
                    error_message=f"srun failed: {result.stderr[:100]}",
                )

            # Parse output
            output = result.stdout

            # Extract GPU info
            gpu_info = self._extract_section(
                output, "===GPU_INFO===", "===END_GPU_INFO==="
            )
            processes = self._extract_section(
                output, "===PROCESSES===", "===END_PROCESSES==="
            )

            # Parse GPU memory (simplified - in production would parse actual output)
            # For MI300X: typically 192GB per GPU
            total_memory_gb = 192.0 * 4  # Assume 4 GPUs

            # Count processes
            process_count = 0
            if processes and "NO_PROCESSES" not in processes:
                process_count = len([l for l in processes.split("\n") if l.strip()])

            # Estimate memory usage
            # Rough heuristic: each process uses ~45GB (observed from Job 2437)
            used_memory_gb = process_count * 45.0

            # Determine health
            if process_count == 0:
                health = NodeHealth.CLEAN
            elif used_memory_gb > self.MEMORY_THRESHOLD_GB:
                health = NodeHealth.DIRTY
            else:
                health = NodeHealth.CLEAN  # Minor processes, should be OK

            return NodeStatus(
                node=node,
                health=health,
                gpu_memory_used_gb=used_memory_gb,
                gpu_memory_total_gb=total_memory_gb,
                process_count=process_count,
            )

        except subprocess.TimeoutExpired:
            return NodeStatus(
                node=node,
                health=NodeHealth.UNREACHABLE,
                gpu_memory_used_gb=0.0,
                gpu_memory_total_gb=0.0,
                process_count=0,
                error_message="Timeout",
            )
        except Exception as e:
            return NodeStatus(
                node=node,
                health=NodeHealth.UNKNOWN,
                gpu_memory_used_gb=0.0,
                gpu_memory_total_gb=0.0,
                process_count=0,
                error_message=str(e)[:100],
            )

    def probe_gpu_arch(
        self,
        partition: str,
        constraint: Optional[str] = None,
        exclude: Optional[str] = None,
        nodelist: Optional[str] = None,
    ) -> Optional[str]:
        """
        Ask one compute node of the partition which GPU architecture it has.

        madengine runs SLURM deployments from the login node, which has no GPUs,
        so the architecture a model card's skip_gpu_arch is compared against has
        to come from the nodes the job will land on. One srun of rocminfo on the
        same partition, reservation and constraint the job uses answers that.

        Uses the same queue-wait timeout as the health probe: it is an srun too,
        and on a busy cluster it waits for a slot before it runs anything.

        Returns:
            The gfx name (e.g. "gfx942"), or None if the probe could not run or
            printed no gfx agent. None means UNKNOWN, never "not this arch".
        """
        job_name = f"madengine_archprobe_{os.getpid()}_{int(time.time())}"
        # The KFD topology in sysfs first, rocminfo only as a fallback. This step
        # asks for no GPUs, and where SLURM constrains devices a GPU-less step
        # cannot open /dev/kfd: rocminfo then lists no GPU agent and the probe
        # came back empty in 1.5s (build 133), so a gfx942-excluded card was
        # submitted to gfx942 nodes. The topology files need no device access.
        # Same decode as MAD's scripts/common/cluster.sh, which reports
        # "GPU arch: gfx942" on these nodes from inside the job.
        # rocminfo is often not on a non-login PATH; /opt/rocm/bin is where ROCm
        # installs it.
        probe_script = (
            'for p in /sys/class/kfd/kfd/topology/nodes/*/properties; do '
            'v=$(awk \'$1=="gfx_target_version"{print $2}\' "$p" 2>/dev/null); '
            '[ -n "$v" ] && [ "$v" != 0 ] && '
            '{ printf "gfx%d%x%x\\n" $((v/10000)) $(((v/100)%100)) $((v%100)); exit 0; }; '
            'done; '
            'PATH="$PATH:/opt/rocm/bin" rocminfo 2>/dev/null'
        )
        srun_cmd = [
            "srun",
            "--nodes=1",
            "--ntasks=1",
            "--time=00:01:00",
            "--overlap",
            "--quiet",
            f"--job-name={job_name}",
        ]
        # Named for the same reason check_node_health names it: a login node with
        # no default partition rejects a bare srun.
        if partition:
            srun_cmd.append(f"--partition={partition}")
        if self.reservation:
            srun_cmd.append(f"--reservation={self.reservation}")
        if constraint:
            srun_cmd.append(f"--constraint={constraint}")
        node = _first_plain_node(nodelist)
        if node:
            srun_cmd.append(f"--nodelist={node}")
        elif exclude:
            srun_cmd.append(f"--exclude={exclude}")
        srun_cmd.extend(["bash", "-c", probe_script])

        try:
            result = subprocess.run(
                srun_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            self.console.print(
                f"[yellow]⚠ GPU architecture probe timed out after {self.timeout}s "
                f"waiting for a node in {partition}[/yellow]"
            )
            return None
        except Exception as e:
            self.console.print(f"[yellow]⚠ GPU architecture probe failed: {e}[/yellow]")
            return None
        finally:
            # A probe that timed out is still queued; do not leave it behind.
            SlurmNodeSelector.cancel_health_check_jobs(job_name, self.console)

        if result.returncode != 0:
            self.console.print(
                f"[yellow]⚠ GPU architecture probe srun failed: "
                f"{(result.stderr or '').strip()[:200]}[/yellow]"
            )
            return None
        return parse_gpu_arch(result.stdout)

    def cleanup_node(self, node: str, job_name: Optional[str] = None) -> bool:
        """
        Clean up stale processes on a node using srun.

        Args:
            node: Node name to clean
            job_name: Optional SLURM job name for this srun (enables cleanup of health-check jobs)

        Returns:
            True if cleanup successful
        """
        # Cleanup script (consolidated from bash scripts)
        cleanup_script = """
# Containers first. These workloads run inside docker, and a container outlives
# the job that started it: scancel kills the job's shell, but the container
# belongs to the docker daemon and keeps its GPU memory. That is how a node ends
# up occupied with no SLURM job on it, and why killing host processes alone left
# build 93 looking at busy GPUs.
if command -v docker >/dev/null 2>&1; then
    docker ps -q | xargs --no-run-if-empty docker stop --time 10 2>/dev/null || true
fi

# Kill Ray processes
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "RayWorkerWrapper" 2>/dev/null || true
pkill -9 -f "raylet" 2>/dev/null || true

# Kill vLLM and SGLang processes. SGLang was missing entirely, so a node dirtied
# by the sglang_disagg workloads was never cleaned by this.
pkill -9 -f "vllm" 2>/dev/null || true
pkill -9 -f "sglang" 2>/dev/null || true

# Kill Ray Python workers
pgrep -f "ray/_private/workers" | xargs -r kill -9 2>/dev/null || true

# Give processes time to die and the driver time to release GPU memory. A
# container stopped a moment ago still shows its memory as used.
sleep 5

echo "CLEANUP_OK"
"""
        srun_cmd = [
            "srun",
            f"--nodelist={node}",
            "--ntasks=1",
            "--time=00:01:00",
            "--overlap",
            "--quiet",
        ]
        # Same omission the probe had: without the partition a login node with no
        # default rejects this outright, so cleanup silently never happens.
        if self.partition:
            srun_cmd.append(f"--partition={self.partition}")
        if job_name:
            srun_cmd.append(f"--job-name={job_name}")
        if self.reservation:
            srun_cmd.append(f"--reservation={self.reservation}")
        srun_cmd.extend(["bash", "-c", cleanup_script])

        try:
            result = subprocess.run(
                srun_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            success = result.returncode == 0 and "CLEANUP_OK" in result.stdout

            if success and self.verbose:
                self.console.print(f"[green]    ✓ Cleaned {node}[/green]")

            return success

        except Exception as e:
            if self.verbose:
                self.console.print(
                    f"[yellow]    ⚠ Cleanup failed for {node}: {e}[/yellow]"
                )
            return False

    def select_nodes(
        self,
        partition: str,
        nodes_needed: int,
        exclude: Optional[str] = None,
        constraint: Optional[str] = None,
    ) -> Tuple[List[str], str]:
        """
        Select clean nodes for SLURM job.

        Checks idle nodes on-demand and stops as soon as enough clean nodes
        are found. Excludes dirty, unreachable, and unknown nodes from allocation.

        Args:
            partition: SLURM partition name
            nodes_needed: Number of nodes required for job
            exclude: Current exclude list (comma-separated)
            constraint: SLURM constraint filter

        Returns:
            Tuple of (clean_nodes, updated_exclude_list)
            - clean_nodes: List of clean node names (may be empty)
            - updated_exclude_list: Comma-separated list to pass to sbatch
        """
        # The probe runs before any allocation exists, so it has to name the partition.
        self.partition = partition

        self.console.print("\n[bold cyan]🔍 Checking GPU Node Health[/bold cyan]")
        self.console.print(
            f"Partition: [cyan]{partition}[/cyan] | "
            f"Nodes needed: [cyan]{nodes_needed}[/cyan]\n"
        )

        # Unique job name for all health-check srun invocations (enables cleanup)
        self._health_check_job_name = (
            f"madengine_nodecheck_{os.getpid()}_{int(time.time())}"
        )

        # Get all idle candidate nodes
        candidates = self.get_candidate_nodes(
            partition, nodes_needed, exclude, constraint
        )

        if not candidates:
            self.console.print(
                "[yellow]⚠ Cannot query candidate nodes, skipping preflight check[/yellow]\n"
            )
            self._health_check_job_name = None
            return [], exclude or ""

        if self.verbose:
            self.console.print(
                f"[dim]Idle candidates: {len(candidates)} (checking on-demand until {nodes_needed} clean)[/dim]\n"
            )

        # On-demand check: stop as soon as we have enough clean nodes
        statuses: List[NodeStatus] = []
        clean_nodes: List[str] = []
        for node in candidates:
            if self.verbose:
                self.console.print(f"  Checking {node}...", end="")
            status = self.check_node_health(node, job_name=self._health_check_job_name)
            statuses.append(status)
            if self.verbose:
                emoji = {
                    NodeHealth.CLEAN: "✓",
                    NodeHealth.DIRTY: "⚠",
                    NodeHealth.UNREACHABLE: "✗",
                    NodeHealth.UNKNOWN: "?",
                }[status.health]
                self.console.print(f" {emoji}")
            if status.health == NodeHealth.CLEAN:
                clean_nodes.append(node)
                if len(clean_nodes) >= nodes_needed:
                    break

        # Display summary table (only nodes we checked)
        self._display_status_table(statuses)

        # Nodes to exclude: DIRTY, UNREACHABLE, and UNKNOWN
        dirty_nodes = [s for s in statuses if s.health == NodeHealth.DIRTY]
        unreachable_nodes = [s for s in statuses if s.health == NodeHealth.UNREACHABLE]
        unknown_nodes = [s for s in statuses if s.health == NodeHealth.UNKNOWN]
        nodes_to_exclude = set()
        nodes_to_exclude.update(s.node for s in dirty_nodes)
        nodes_to_exclude.update(s.node for s in unreachable_nodes)
        nodes_to_exclude.update(s.node for s in unknown_nodes)

        # Handle dirty nodes (optional auto-cleanup)
        if dirty_nodes:
            self.console.print(
                f"\n[yellow]⚠ Found {len(dirty_nodes)} dirty node(s) "
                f"with stale Ray/vLLM processes[/yellow]"
            )
            if self.auto_cleanup:
                self.console.print("[yellow]Running automatic cleanup...[/yellow]\n")
                for status in dirty_nodes:
                    self.console.print(f"  Cleaning {status.node}...")
                    if self.cleanup_node(
                        status.node, job_name=self._health_check_job_name
                    ):
                        time.sleep(2)
                        new_status = self.check_node_health(
                            status.node, job_name=self._health_check_job_name
                        )
                        if new_status.health == NodeHealth.CLEAN:
                            clean_nodes.append(new_status.node)
                            nodes_to_exclude.discard(status.node)
                            self.console.print(
                                f"    [green]✓ {status.node} is now clean[/green]"
                            )
                        else:
                            self.console.print(
                                f"    [red]✗ {status.node} still dirty[/red]"
                            )
                    else:
                        self.console.print(f"    [red]✗ Cleanup failed[/red]")

        # Build updated exclude list (dirty + unreachable + unknown)
        existing_exclude = set(exclude.split(",")) if exclude else set()
        existing_exclude.update(nodes_to_exclude)
        updated_exclude = ",".join(sorted(existing_exclude))

        if unreachable_nodes or unknown_nodes:
            bad = [s.node for s in unreachable_nodes] + [s.node for s in unknown_nodes]
            self.console.print(
                f"\n[yellow]Excluding unreachable/unknown nodes: {', '.join(bad)}[/yellow]"
            )
        if dirty_nodes and not self.auto_cleanup:
            self.console.print(
                f"\n[yellow]Adding dirty nodes to exclude list: "
                f"{', '.join(s.node for s in dirty_nodes)}[/yellow]"
            )

        # Final summary
        if len(clean_nodes) >= nodes_needed:
            self.console.print(
                f"\n[bold green]✅ Found {len(clean_nodes)} clean nodes "
                f"(need {nodes_needed})[/bold green]\n"
            )
        elif len(clean_nodes) > 0:
            self.console.print(
                f"\n[yellow]⚠ Only {len(clean_nodes)} clean nodes found "
                f"(need {nodes_needed})[/yellow]"
            )
            self.console.print(
                "[yellow]Job may wait for additional nodes to become available[/yellow]\n"
            )
        else:
            self.console.print("\n[red]❌ No clean nodes available[/red]")
            self.console.print(
                "[yellow]Recommendation: Wait for nodes to be cleaned or run manual cleanup[/yellow]\n"
            )

        # A check that condemns the WHOLE partition has failed; the partition has
        # not. Returning an exclude list naming every node hands sbatch a job that
        # can never be scheduled, which is strictly worse than not checking: the
        # job does not wait for a node, it waits forever.
        #
        # Seen on OCI amd-rccl, where all 51 nodes came back unreachable/unknown
        # from the login node -- the probe could not reach any of them, which says
        # something about the probe, not about 51 machines. The value of this check
        # is steering around a few known-bad nodes; with no node left standing it
        # has no signal to offer, so it stands down and lets SLURM schedule.
        examined = {st.node for st in statuses}
        if examined and examined.issubset(set(existing_exclude)):
            # Flag it for the caller as well. Dropping the exclude list alone is half a
            # stand-down: the caller also gates submission on len(clean_nodes), which is
            # zero for exactly the same reason the exclude list named everything. Build 87
            # cleared the exclusion and then failed anyway on "Not enough clean nodes:
            # need 2, found 0" -- the check still deciding the outcome after announcing it
            # had nothing to say.
            self.health_check_inconclusive = True
            self.console.print(
                f"[yellow]⚠ The health check excluded all {len(examined)} node(s) it "
                f"examined, so it has no usable signal and is being disregarded.[/yellow]"
            )
            self.console.print(
                "[yellow]  Submitting without a node exclusion and letting SLURM "
                "schedule. Set slurm.enable_node_check=false to skip this check "
                "entirely.[/yellow]\n"
            )
            return clean_nodes, exclude

        return clean_nodes, updated_exclude

    def _extract_section(self, text: str, start_marker: str, end_marker: str) -> str:
        """Extract section between markers."""
        try:
            start = text.index(start_marker) + len(start_marker)
            end = text.index(end_marker, start)
            return text[start:end].strip()
        except ValueError:
            return ""

    def _display_status_table(self, statuses: List[NodeStatus]):
        """Display node status in a table."""
        table = Table(title="Node Health Status")

        table.add_column("Node", style="cyan", no_wrap=True)
        table.add_column("Health", style="bold")
        table.add_column("Memory Used", justify="right")
        table.add_column("Processes", justify="right")
        table.add_column("Notes", style="dim")

        for status in statuses:
            health_style = {
                NodeHealth.CLEAN: "green",
                NodeHealth.DIRTY: "yellow",
                NodeHealth.UNREACHABLE: "red",
                NodeHealth.UNKNOWN: "dim",
            }[status.health]

            health_text = {
                NodeHealth.CLEAN: "✓ Clean",
                NodeHealth.DIRTY: "⚠ Dirty",
                NodeHealth.UNREACHABLE: "✗ Unreachable",
                NodeHealth.UNKNOWN: "? Unknown",
            }[status.health]

            memory_text = (
                f"{status.gpu_memory_used_gb:.0f} GB"
                if status.gpu_memory_used_gb > 0
                else "-"
            )
            processes_text = (
                str(status.process_count) if status.process_count > 0 else "-"
            )
            notes = status.error_message if status.error_message else ""

            table.add_row(
                status.node,
                f"[{health_style}]{health_text}[/{health_style}]",
                memory_text,
                processes_text,
                notes,
            )

        self.console.print(table)
        self.console.print()

    @staticmethod
    def cancel_health_check_jobs(
        job_name: Optional[str], console: Optional[Console] = None
    ) -> None:
        """
        Cancel any SLURM jobs created by the node health check (srun invocations).

        Call this after select_nodes() so pending health-check jobs do not stay in the queue.

        Args:
            job_name: Job name used for health-check srun (e.g. selector._health_check_job_name)
            console: Optional Rich console for messages
        """
        if not job_name:
            return
        _console = console or Console()
        try:
            user = os.environ.get("USER", "")
            if not user:
                return
            result = subprocess.run(
                ["squeue", "-u", user, "-n", job_name, "-h", "-o", "%i"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return
            job_ids = result.stdout.strip().split()
            for jid in job_ids:
                if jid.isdigit():
                    subprocess.run(
                        ["scancel", jid],
                        capture_output=True,
                        timeout=5,
                    )
            if job_ids and _console:
                _console.print(
                    f"[dim]Cancelled {len(job_ids)} health-check job(s)[/dim]"
                )
        except Exception:
            pass
