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
        super().__init__(config)

        # Parse SLURM configuration
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
        return {
            "model_name": model_info["name"],
            "manifest_file": os.path.abspath(self.config.manifest_file),
            "partition": self.partition,
            "nodes": self.nodes,
            "gpus_per_node": self.gpus_per_node,
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

            if result.returncode != 0:
                # Job not found - likely completed or failed
                return self._check_job_completion(deployment_id)

            status = result.stdout.strip().upper()

            if status in ["RUNNING", "PENDING", "CONFIGURING"]:
                return DeploymentResult(
                    status=DeploymentStatus.RUNNING,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} is {status.lower()}",
                )
            elif status in ["COMPLETED"]:
                return DeploymentResult(
                    status=DeploymentStatus.SUCCESS,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} completed successfully",
                )
            else:  # FAILED, CANCELLED, TIMEOUT, etc.
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} {status.lower()}",
                )

        except Exception as e:
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id=deployment_id,
                message=f"Monitor error: {str(e)}",
            )

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
                if "COMPLETED" in status:
                    return DeploymentResult(
                        status=DeploymentStatus.SUCCESS,
                        deployment_id=job_id,
                        message=f"Job {job_id} completed",
                    )
                else:
                    return DeploymentResult(
                        status=DeploymentStatus.FAILED,
                        deployment_id=job_id,
                        message=f"Job {job_id} failed: {status}",
                    )

            # Fallback - assume completed
            return DeploymentResult(
                status=DeploymentStatus.SUCCESS,
                deployment_id=job_id,
                message=f"Job {job_id} completed (assumed)",
            )

        except Exception:
            return DeploymentResult(
                status=DeploymentStatus.SUCCESS,
                deployment_id=job_id,
                message=f"Job {job_id} completed (status unavailable)",
            )

    def collect_results(self, deployment_id: str) -> Dict[str, Any]:
        """Collect performance results from SLURM output files."""
        results = {
            "job_id": deployment_id,
            "nodes": self.nodes,
            "gpus_per_node": self.gpus_per_node,
            "perf_files": [],
            "logs": [],
        }

        try:
            # Find output files
            output_pattern = f"madengine-*_{deployment_id}_*.out"
            output_files = list(self.output_dir.glob(output_pattern))

            results["logs"] = [str(f) for f in output_files]

            # Find performance CSV files
            if self.slurm_config.get("results_dir"):
                results_dir = Path(self.slurm_config["results_dir"])
                perf_pattern = f"perf_{deployment_id}_*.csv"
                perf_files = list(results_dir.glob(perf_pattern))
                results["perf_files"] = [str(f) for f in perf_files]

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

