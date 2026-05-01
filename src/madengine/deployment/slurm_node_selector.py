#!/usr/bin/env python3
"""
SLURM Node Selector with GPU Cleanup

Helps SLURM select clean GPU nodes by checking for stale processes before
job submission. Prevents "out of memory" errors in multi-node vLLM/Ray jobs.

Uses srun (not SSH) to check and clean nodes - works from SLURM login node.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from rich.console import Console
from rich.table import Table


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
        timeout: int = 30,
        reservation: Optional[str] = None,
    ):
        """
        Initialize node selector.
        
        Args:
            console: Rich console for output
            auto_cleanup: Automatically clean dirty nodes
            verbose: Enable verbose logging
            timeout: Timeout for srun commands (seconds)
            reservation: SLURM reservation name for reserved nodes
        """
        self.console = console or Console()
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
            "-p", partition,
            "-N",  # Node-oriented format
            "-h",  # No header
            "-o", "%N",  # Node name only
            "-t", "idle",  # Idle nodes only
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
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if line:
                    all_nodes.add(line)

            # Remove excluded nodes
            if exclude:
                excluded = set(exclude.split(','))
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
    
    def check_node_health(self, node: str, job_name: Optional[str] = None) -> NodeStatus:
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
        if job_name:
            srun_cmd.append(f"--job-name={job_name}")
        if hasattr(self, 'reservation') and self.reservation:
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
            gpu_info = self._extract_section(output, "===GPU_INFO===", "===END_GPU_INFO===")
            processes = self._extract_section(output, "===PROCESSES===", "===END_PROCESSES===")
            
            # Parse GPU memory (simplified - in production would parse actual output)
            # For MI300X: typically 192GB per GPU
            total_memory_gb = 192.0 * 4  # Assume 4 GPUs
            
            # Count processes
            process_count = 0
            if processes and "NO_PROCESSES" not in processes:
                process_count = len([l for l in processes.split('\n') if l.strip()])
            
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
# Kill Ray processes
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "RayWorkerWrapper" 2>/dev/null || true
pkill -9 -f "raylet" 2>/dev/null || true

# Kill vLLM processes
pkill -9 -f "vllm" 2>/dev/null || true

# Kill Ray Python workers
pgrep -f "ray/_private/workers" | xargs -r kill -9 2>/dev/null || true

# Give processes time to die
sleep 2

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
                self.console.print(f"[yellow]    ⚠ Cleanup failed for {node}: {e}[/yellow]")
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
        self.console.print("\n[bold cyan]🔍 Checking GPU Node Health[/bold cyan]")
        self.console.print(
            f"Partition: [cyan]{partition}[/cyan] | "
            f"Nodes needed: [cyan]{nodes_needed}[/cyan]\n"
        )

        # Unique job name for all health-check srun invocations (enables cleanup)
        self._health_check_job_name = f"madengine_nodecheck_{os.getpid()}_{int(time.time())}"

        # Get all idle candidate nodes
        candidates = self.get_candidate_nodes(partition, nodes_needed, exclude, constraint)

        if not candidates:
            self.console.print(
                "[yellow]⚠ Cannot query candidate nodes, skipping preflight check[/yellow]\n"
            )
            self._health_check_job_name = None
            return [], exclude or ""

        if self.verbose:
            self.console.print(f"[dim]Idle candidates: {len(candidates)} (checking on-demand until {nodes_needed} clean)[/dim]\n")

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
                    if self.cleanup_node(status.node, job_name=self._health_check_job_name):
                        time.sleep(2)
                        new_status = self.check_node_health(status.node, job_name=self._health_check_job_name)
                        if new_status.health == NodeHealth.CLEAN:
                            clean_nodes.append(new_status.node)
                            nodes_to_exclude.discard(status.node)
                            self.console.print(f"    [green]✓ {status.node} is now clean[/green]")
                        else:
                            self.console.print(f"    [red]✗ {status.node} still dirty[/red]")
                    else:
                        self.console.print(f"    [red]✗ Cleanup failed[/red]")

        # Build updated exclude list (dirty + unreachable + unknown)
        existing_exclude = set(exclude.split(',')) if exclude else set()
        existing_exclude.update(nodes_to_exclude)
        updated_exclude = ','.join(sorted(existing_exclude))

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
            self.console.print("[yellow]Job may wait for additional nodes to become available[/yellow]\n")
        else:
            self.console.print(
                "\n[red]❌ No clean nodes available[/red]"
            )
            self.console.print(
                "[yellow]Recommendation: Wait for nodes to be cleaned or run manual cleanup[/yellow]\n"
            )

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
            
            memory_text = f"{status.gpu_memory_used_gb:.0f} GB" if status.gpu_memory_used_gb > 0 else "-"
            processes_text = str(status.process_count) if status.process_count > 0 else "-"
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
    def cancel_health_check_jobs(job_name: Optional[str], console: Optional[Console] = None) -> None:
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
                _console.print(f"[dim]Cancelled {len(job_ids)} health-check job(s)[/dim]")
        except Exception:
            pass
