#!/usr/bin/env python3
"""
SLURM Deployment - HPC cluster deployment using CLI commands.

Uses subprocess to call SLURM CLI commands (sbatch, squeue, scancel).
No Python SLURM library required (zero dependencies).

**Assumption**: User has already SSH'd to SLURM login node manually.
madengine-cli is executed ON the login node, not remotely.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import subprocess
from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader

from .base import BaseDeployment, DeploymentConfig, DeploymentResult, DeploymentStatus
from .config_loader import ConfigLoader
from madengine.utils.gpu_config import resolve_runtime_gpus


class SlurmDeployment(BaseDeployment):
    """
    SLURM HPC cluster deployment using CLI commands.

    **Workflow**:
    1. User: ssh login_node@hpc.example.com
    2. User: madengine-cli run --tags model --additional-context '{"deploy": "slurm", ...}'
    3. madengine-cli: Runs sbatch locally (no SSH needed)

    Uses subprocess to call SLURM CLI commands locally:
    - sbatch: Submit jobs to SLURM scheduler
    - squeue: Monitor job status
    - scancel: Cancel jobs
    - scontrol: Get cluster info

    No Python SLURM library required (zero dependencies).
    No SSH handling needed (user is already on login node).
    """

    DEPLOYMENT_TYPE = "slurm"
    REQUIRED_TOOLS = ["sbatch", "squeue", "scontrol"]  # Must be available locally

    def __init__(self, config: DeploymentConfig):
        """
        Initialize SLURM deployment.

        Args:
            config: Deployment configuration
        """
        # Apply intelligent defaults using ConfigLoader
        # This merges built-in presets with user configuration
        full_config = ConfigLoader.load_slurm_config(config.additional_context)
        config.additional_context = full_config

        super().__init__(config)

        # Parse SLURM configuration (now with defaults applied)
        self.slurm_config = config.additional_context.get("slurm", {})
        self.distributed_config = config.additional_context.get("distributed", {})

        # SLURM parameters
        self.partition = self.slurm_config.get("partition", "gpu")
        self.nodes = self.slurm_config.get("nodes", 1)
        self.gpus_per_node = self.slurm_config.get("gpus_per_node", 8)
        self.time_limit = self.slurm_config.get("time", "24:00:00")
        self.output_dir = Path(self.slurm_config.get("output_dir", "./slurm_output"))

        # Setup Jinja2 template engine
        template_dir = Path(__file__).parent / "templates" / "slurm"
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        
        # Register custom Jinja2 filters
        self.jinja_env.filters['dirname'] = lambda path: str(Path(path).parent)
        self.jinja_env.filters['basename'] = lambda path: str(Path(path).name)

        # Generated script path
        self.script_path = None

    def validate(self) -> bool:
        """Validate SLURM commands are available locally."""
        # Check required SLURM CLI tools
        for tool in self.REQUIRED_TOOLS:
            result = subprocess.run(
                ["which", tool], capture_output=True, timeout=5
            )
            if result.returncode != 0:
                self.console.print(
                    f"[red]✗ Required tool not found: {tool}[/red]\n"
                    f"[yellow]Make sure you are on a SLURM login node[/yellow]"
                )
                return False

        # Verify we can query SLURM cluster
        result = subprocess.run(["sinfo", "-h"], capture_output=True, timeout=10)
        if result.returncode != 0:
            self.console.print("[red]✗ Cannot query SLURM (sinfo failed)[/red]")
            return False

        # Validate configuration
        if self.nodes < 1:
            self.console.print(f"[red]✗ Invalid nodes: {self.nodes}[/red]")
            return False

        if self.gpus_per_node < 1:
            self.console.print(f"[red]✗ Invalid GPUs per node: {self.gpus_per_node}[/red]")
            return False

        self.console.print("[green]✓ SLURM environment validated[/green]")
        return True

    def prepare(self) -> bool:
        """Generate sbatch script from template."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Get model info from manifest
            model_keys = list(self.manifest["built_models"].keys())
            if not model_keys:
                raise ValueError("No models in manifest")

            model_key = model_keys[0]
            model_info = self.manifest["built_models"][model_key]

            # Prepare template context
            context = self._prepare_template_context(model_info)

            # Render template
            template = self.jinja_env.get_template("job.sh.j2")
            script_content = template.render(**context)

            # Save script
            self.script_path = self.output_dir / f"madengine_{model_info['name']}.sh"
            self.script_path.write_text(script_content)
            self.script_path.chmod(0o755)

            self.console.print(
                f"[green]✓ Generated sbatch script: {self.script_path}[/green]"
            )
            return True

        except Exception as e:
            self.console.print(f"[red]✗ Failed to generate script: {e}[/red]")
            return False

    def _prepare_template_context(self, model_info: Dict) -> Dict[str, Any]:
        """Prepare context for Jinja2 template rendering."""
        # Use hierarchical GPU resolution: runtime > deployment > model > default
        additional_context = self.config.additional_context.copy()
        additional_context["slurm"] = self.slurm_config
        resolved_gpus_per_node = resolve_runtime_gpus(model_info, additional_context)
        
        return {
            "model_name": model_info["name"],
            "manifest_file": os.path.abspath(self.config.manifest_file),
            "partition": self.partition,
            "nodes": self.nodes,
            "gpus_per_node": resolved_gpus_per_node,  # Use resolved GPU count
            "time_limit": self.time_limit,
            "output_dir": str(self.output_dir),
            "master_port": self.distributed_config.get("port", 29500),
            "distributed_backend": self.distributed_config.get("backend", "nccl"),
            "network_interface": self.slurm_config.get("network_interface"),
            "exclusive": self.slurm_config.get("exclusive", True),
            "qos": self.slurm_config.get("qos"),
            "account": self.slurm_config.get("account"),
            "modules": self.slurm_config.get("modules", []),
            "env_vars": self.config.additional_context.get("env_vars", {}),
            "shared_workspace": self.slurm_config.get("shared_workspace"),
            "shared_data": self.config.additional_context.get("shared_data"),
            "results_dir": self.slurm_config.get("results_dir"),
            "timeout": self.config.timeout,
            "live_output": self.config.additional_context.get("live_output", False),
            "tags": " ".join(model_info.get("tags", [])),
            "credential_file": "credential.json"
            if Path("credential.json").exists()
            else None,
            "data_file": "data.json" if Path("data.json").exists() else None,
        }

    def deploy(self) -> DeploymentResult:
        """Submit sbatch script to SLURM scheduler (locally)."""
        if not self.script_path or not self.script_path.exists():
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id="",
                message="Script not generated. Run prepare() first.",
            )

        try:
            # Submit job to SLURM (runs locally on login node)
            result = subprocess.run(
                ["sbatch", str(self.script_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                # Parse job ID: "Submitted batch job 12345"
                job_id = result.stdout.strip().split()[-1]

                self.console.print(f"[green]✓ Submitted SLURM job: {job_id}[/green]")
                self.console.print(f"  Nodes: {self.nodes} x {self.gpus_per_node} GPUs")
                self.console.print(f"  Partition: {self.partition}")

                return DeploymentResult(
                    status=DeploymentStatus.SUCCESS,
                    deployment_id=job_id,
                    message=f"SLURM job {job_id} submitted successfully",
                    logs_path=str(self.output_dir),
                )
            else:
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    deployment_id="",
                    message=f"sbatch failed: {result.stderr}",
                )

        except subprocess.TimeoutExpired:
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id="",
                message="sbatch submission timed out",
            )
        except Exception as e:
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id="",
                message=f"Deployment error: {str(e)}",
            )

    def monitor(self, deployment_id: str) -> DeploymentResult:
        """Check SLURM job status (locally)."""
        try:
            # Query job status using squeue (runs locally)
            result = subprocess.run(
                ["squeue", "-j", deployment_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0 or not result.stdout.strip():
                # Job not found in queue - likely completed or failed
                return self._check_job_completion(deployment_id)

            status = result.stdout.strip().upper()
            
            # Check if live output is enabled
            live_output = self.config.additional_context.get("live_output", False)

            # Stream work node output if live_output is enabled and job is running
            if status == "RUNNING" and live_output:
                self._stream_job_output(deployment_id)

            if status in ["RUNNING", "PENDING", "CONFIGURING", "COMPLETING"]:
                # COMPLETING is a transient state before COMPLETED - treat as running
                return DeploymentResult(
                    status=DeploymentStatus.RUNNING,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} is {status.lower()}",
                )
            elif status in ["COMPLETED"]:
                # Show final output only if live_output is enabled
                if live_output:
                    self._stream_job_output(deployment_id, final=True)
                else:
                    self._show_log_summary(deployment_id, success=True)
                return DeploymentResult(
                    status=DeploymentStatus.SUCCESS,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} completed successfully",
                )
            else:  # FAILED, CANCELLED, TIMEOUT, NODE_FAIL, etc.
                # Show output on failure or show summary
                if live_output:
                    self._stream_job_output(deployment_id, final=True)
                else:
                    self._show_log_summary(deployment_id, success=False)
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} {status.lower()}",
                )

        except Exception as e:
            self.console.print(f"[red]Monitor exception for job {deployment_id}: {e}[/red]")
            import traceback
            self.console.print(f"[dim red]{traceback.format_exc()}[/dim red]")
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id=deployment_id,
                message=f"Monitor error: {str(e)}",
            )

    def _stream_job_output(self, job_id: str, final: bool = False):
        """Stream output from SLURM job output file."""
        # Track last position read from output file
        if not hasattr(self, '_output_positions'):
            self._output_positions = {}
        
        # Find output file
        output_dir = self.slurm_config.get("output_dir", "./slurm_output")
        output_pattern = f"{output_dir}/madengine-*_{job_id}_*.out"
        
        try:
            import glob
            output_files = glob.glob(output_pattern)
            
            if not output_files:
                return  # Output file not created yet
            
            output_file = output_files[0]  # Use first match
            
            # Read new content from file
            try:
                with open(output_file, 'r') as f:
                    # Seek to last position
                    last_pos = self._output_positions.get(job_id, 0)
                    f.seek(last_pos)
                    
                    # Read new lines
                    new_content = f.read()
                    
                    if new_content:
                        # Print new output with prefix
                        for line in new_content.splitlines():
                            if line.strip():  # Skip empty lines
                                self.console.print(f"[dim cyan]│[/dim cyan] {line}")
                    
                    # Update position
                    self._output_positions[job_id] = f.tell()
                    
            except FileNotFoundError:
                pass  # File not ready yet
                
        except Exception as e:
            # Silently ignore streaming errors to not disrupt monitoring
            if final:
                self.console.print(f"[dim yellow]Note: Could not stream output: {e}[/dim yellow]")

    def _show_log_summary(self, job_id: str, success: bool = True):
        """Show a summary with pointers to log files instead of streaming verbose output."""
        output_dir = self.slurm_config.get("output_dir", "./slurm_output")
        
        try:
            import glob
            # Find output and error files for this job
            output_files = glob.glob(f"{output_dir}/madengine-*_{job_id}_*.out")
            error_files = glob.glob(f"{output_dir}/madengine-*_{job_id}_*.err")
            
            if output_files or error_files:
                status_symbol = "✓" if success else "✗"
                status_color = "green" if success else "red"
                
                self.console.print(f"[{status_color}]{status_symbol}[/{status_color}] SLURM job {job_id} logs saved to:")
                
                for out_file in output_files:
                    self.console.print(f"  [cyan]→[/cyan] Output: {out_file}")
                    
                for err_file in error_files:
                    # Check if error file has content
                    if os.path.exists(err_file) and os.path.getsize(err_file) > 0:
                        self.console.print(f"  [yellow]→[/yellow] Errors: {err_file}")
                
                if not success and error_files:
                    # Show last few lines of error file for failed jobs
                    for err_file in error_files:
                        if os.path.exists(err_file) and os.path.getsize(err_file) > 0:
                            self.console.print(f"\n[yellow]Last 10 lines of error log:[/yellow]")
                            try:
                                with open(err_file, 'r') as f:
                                    lines = f.readlines()
                                    for line in lines[-10:]:
                                        if line.strip():
                                            self.console.print(f"  {line.rstrip()}")
                            except Exception:
                                pass
                            break  # Only show first error file
            else:
                self.console.print(f"[dim yellow]Note: Log files for job {job_id} not found in {output_dir}[/dim yellow]")
                
        except Exception as e:
            self.console.print(f"[dim yellow]Note: Could not locate log files: {e}[/dim yellow]")

    def _check_job_completion(self, job_id: str) -> DeploymentResult:
        """Check completed job status using sacct (locally)."""
        try:
            result = subprocess.run(
                ["sacct", "-j", job_id, "-n", "-X", "-o", "State"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                status = result.stdout.strip().upper()
                self.console.print(f"[dim]SLURM job {job_id} final status: {status}[/dim]")
                
                # Check if live output is enabled
                live_output = self.config.additional_context.get("live_output", False)
                
                if "COMPLETED" in status:
                    # Show final output or summary based on live_output flag
                    if live_output:
                        self._stream_job_output(job_id, final=True)
                    else:
                        self._show_log_summary(job_id, success=True)
                    return DeploymentResult(
                        status=DeploymentStatus.SUCCESS,
                        deployment_id=job_id,
                        message=f"Job {job_id} completed successfully",
                    )
                else:
                    # Show output on failure or summary
                    if live_output:
                        self._stream_job_output(job_id, final=True)
                    else:
                        self._show_log_summary(job_id, success=False)
                    return DeploymentResult(
                        status=DeploymentStatus.FAILED,
                        deployment_id=job_id,
                        message=f"Job {job_id} failed: {status}",
                    )

            # Fallback - assume completed
            self.console.print(f"[dim yellow]Warning: Could not get status for job {job_id}, assuming success[/dim yellow]")
            return DeploymentResult(
                status=DeploymentStatus.SUCCESS,
                deployment_id=job_id,
                message=f"Job {job_id} completed (assumed)",
            )

        except Exception as e:
            self.console.print(f"[dim yellow]Warning: Exception checking job {job_id}: {e}[/dim yellow]")
            return DeploymentResult(
                status=DeploymentStatus.SUCCESS,
                deployment_id=job_id,
                message=f"Job {job_id} completed (status unavailable)",
            )

    def collect_results(self, deployment_id: str) -> Dict[str, Any]:
        """Collect performance results from SLURM output files.
        
        Args:
            deployment_id: SLURM job ID
        """
        # Get session_start_row from config (passed from orchestrator)
        session_start_row = self.config.additional_context.get("session_start_row")
        
        results = {
            "job_id": deployment_id,
            "nodes": self.nodes,
            "gpus_per_node": self.gpus_per_node,
            "perf_files": [],
            "logs": [],
            "successful_runs": [],
            "failed_runs": [],
            "session_start_row": session_start_row,  # Track for downstream filtering
        }

        try:
            # Find output files
            output_pattern = f"madengine-*_{deployment_id}_*.out"
            output_files = list(self.output_dir.glob(output_pattern))

            results["logs"] = [str(f) for f in output_files]

            # Find performance CSV files
            # Strategy 1: Check results_dir if configured
            if self.slurm_config.get("results_dir"):
                results_dir = Path(self.slurm_config["results_dir"])
                perf_pattern = f"perf_{deployment_id}_*.csv"
                perf_files = list(results_dir.glob(perf_pattern))
                results["perf_files"] = [str(f) for f in perf_files]
            
            # Strategy 2: Check shared workspace (NFS) for perf.csv
            # When using shared storage, perf.csv is written directly to workspace
            if not results["perf_files"]:
                workspace_perf = Path("perf.csv")
                if workspace_perf.exists():
                    results["perf_files"] = [str(workspace_perf)]
                    self.console.print("[dim]Note: Using perf.csv from shared workspace[/dim]")
            
            # Parse perf.csv to populate successful_runs and failed_runs
            # Filter based on session_start_row passed as parameter (no external files!)
            if results["perf_files"]:
                perf_file = Path(results["perf_files"][0])
                try:
                    import csv
                    
                    with open(perf_file, 'r') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        
                        # Filter to only include rows from current session if session_start_row provided
                        if session_start_row is not None and session_start_row < len(rows):
                            rows = rows[session_start_row:]
                            self.console.print(f"[cyan]📊 Filtered to current session: {len(rows)} runs (from row {session_start_row} of {len(rows) + session_start_row} total)[/cyan]")
                        elif session_start_row is not None:
                            # Session start equals or exceeds current rows - no new runs yet
                            self.console.print(f"[yellow]⚠️  No new runs in this session (session started at row {session_start_row}, CSV has {len(rows)} rows)[/yellow]")
                            rows = []
                        else:
                            # No session info provided - show all rows (for backward compatibility)
                            self.console.print(f"[dim]Showing all {len(rows)} runs from perf.csv (no session filtering)[/dim]")
                        
                        for row in rows:
                            run_data = {
                                "model": row.get("model", ""),
                                "status": row.get("status", ""),
                                "performance": row.get("performance", ""),
                                "metric": row.get("metric", ""),
                                "duration": row.get("test_duration", ""),
                                "gpu_arch": row.get("gpu_architecture", ""),
                                "deployment": row.get("deployment_type", ""),
                                "machine": row.get("machine_name", ""),
                            }
                            
                            if row.get("status") == "SUCCESS":
                                results["successful_runs"].append(run_data)
                            else:
                                results["failed_runs"].append(run_data)
                except Exception as parse_error:
                    import traceback
                    self.console.print(f"[red]ERROR parsing perf.csv: {parse_error}[/red]")
                    self.console.print(f"[dim]{traceback.format_exc()}[/dim]")

            self.console.print(
                f"[green]✓ Collected results: {len(results['perf_files'])} perf files, "
                f"{len(results['logs'])} log files[/green]"
            )

        except Exception as e:
            self.console.print(f"[yellow]⚠ Results collection incomplete: {e}[/yellow]")

        return results

    def cleanup(self, deployment_id: str) -> bool:
        """Cancel SLURM job if still running (locally)."""
        try:
            subprocess.run(
                ["scancel", deployment_id], capture_output=True, timeout=10
            )
            self.console.print(f"[yellow]Cancelled SLURM job: {deployment_id}[/yellow]")
            return True

        except Exception as e:
            self.console.print(f"[yellow]⚠ Cleanup warning: {e}[/yellow]")
            return False

