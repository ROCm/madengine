#!/usr/bin/env python3
"""
SLURM Distributed Runner for MADEngine

This module implements SLURM-based distributed execution using
SLURM workload manager for orchestrated parallel execution across HPC clusters.
"""

import json
import logging
import os
import subprocess
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    import paramiko
    from scp import SCPClient
except ImportError:
    raise ImportError(
        "SLURM runner requires paramiko and scp for SSH connections. "
        "Install with: pip install paramiko scp"
    )

from madengine.runners.base import (
    BaseDistributedRunner,
    NodeConfig,
    WorkloadSpec,
    ExecutionResult,
    DistributedResult,
)
from madengine.core.errors import (
    ConnectionError as MADConnectionError,
    AuthenticationError,
    TimeoutError as MADTimeoutError,
    RunnerError,
    create_error_context
)


@dataclass
class SlurmNodeConfig(NodeConfig):
    """SLURM-specific node configuration."""
    partition: str = "gpu"
    qos: Optional[str] = None
    account: Optional[str] = None
    constraint: Optional[str] = None
    exclusive: bool = False
    mem_per_gpu: Optional[str] = None
    max_time: str = "24:00:00"


@dataclass
class SlurmExecutionError(RunnerError):
    """SLURM execution specific errors."""

    job_id: str
    
    def __init__(self, message: str, job_id: str, **kwargs):
        self.job_id = job_id
        context = create_error_context(
            operation="slurm_execution",
            component="SlurmRunner",
            additional_info={"job_id": job_id}
        )
        super().__init__(f"SLURM job {job_id}: {message}", context=context, **kwargs)


class SlurmConnection:
    """Manages SSH connection to SLURM login node."""

    def __init__(self, login_node: Dict[str, Any], timeout: int = 30):
        """Initialize SSH connection to SLURM login node.

        Args:
            login_node: Login node configuration
            timeout: Connection timeout in seconds
        """
        self.login_node = login_node
        self.timeout = timeout
        self.ssh_client = None
        self.sftp_client = None
        self.logger = logging.getLogger(f"SlurmConnection.{login_node['hostname']}")
        self._connected = False

    def connect(self) -> bool:
        """Establish SSH connection to SLURM login node.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Connection parameters
            connect_params = {
                "hostname": self.login_node["address"],
                "port": self.login_node.get("port", 22),
                "username": self.login_node["username"],
                "timeout": self.timeout,
            }

            # Use SSH key if provided
            if self.login_node.get("ssh_key_path"):
                expanded_key_path = os.path.expanduser(self.login_node["ssh_key_path"])
                if os.path.exists(expanded_key_path):
                    connect_params["key_filename"] = expanded_key_path
                    os.chmod(expanded_key_path, 0o600)

            self.ssh_client.connect(**connect_params)
            self.sftp_client = self.ssh_client.open_sftp()

            self._connected = True
            self.logger.info(f"Successfully connected to SLURM login node {self.login_node['hostname']}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to SLURM login node: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if connection is active."""
        return (
            self._connected
            and self.ssh_client
            and self.ssh_client.get_transport()
            and self.ssh_client.get_transport().is_active()
        )

    def execute_command(self, command: str, timeout: int = 300) -> Tuple[int, str, str]:
        """Execute command on SLURM login node.

        Args:
            command: Command to execute
            timeout: Command timeout in seconds

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        if not self.is_connected():
            raise MADConnectionError("Connection not established")

        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(command, timeout=timeout)
            exit_code = stdout.channel.recv_exit_status()
            stdout_str = stdout.read().decode("utf-8", errors="replace")
            stderr_str = stderr.read().decode("utf-8", errors="replace")

            return exit_code, stdout_str, stderr_str

        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return 1, "", str(e)

    def copy_file(self, local_path: str, remote_path: str, create_dirs: bool = True) -> bool:
        """Copy file to SLURM login node.

        Args:
            local_path: Local file path
            remote_path: Remote file path
            create_dirs: Whether to create remote directories

        Returns:
            True if copy successful, False otherwise
        """
        if not self.is_connected():
            raise MADConnectionError("Connection not established")

        try:
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Local file not found: {local_path}")

            # Create directory if needed
            if create_dirs:
                remote_dir = os.path.dirname(remote_path)
                if remote_dir:
                    self.execute_command(f"mkdir -p {remote_dir}")

            # Copy file
            self.sftp_client.put(local_path, remote_path)
            self.sftp_client.chmod(remote_path, 0o644)

            self.logger.debug(f"Successfully copied {local_path} to {remote_path}")
            return True

        except Exception as e:
            self.logger.error(f"File copy failed: {e}")
            return False

    def close(self):
        """Close SSH connection."""
        try:
            if self.sftp_client:
                self.sftp_client.close()
                self.sftp_client = None
            if self.ssh_client:
                self.ssh_client.close()
                self.ssh_client = None
            self._connected = False
            self.logger.debug(f"Closed connection to {self.login_node['hostname']}")
        except Exception as e:
            self.logger.warning(f"Error closing connection: {e}")

    def __enter__(self):
        """Context manager entry."""
        if not self.connect():
            raise MADConnectionError("Failed to establish SLURM connection")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class SlurmDistributedRunner(BaseDistributedRunner):
    """Distributed runner using SLURM workload manager."""

    def __init__(self, inventory_path: str, job_scripts_dir: str = None, **kwargs):
        """Initialize SLURM distributed runner.

        Args:
            inventory_path: Path to SLURM inventory configuration file
            job_scripts_dir: Directory containing pre-generated job scripts
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(inventory_path, **kwargs)
        self.job_scripts_dir = Path(job_scripts_dir) if job_scripts_dir else None
        self.slurm_connection: Optional[SlurmConnection] = None
        self.submitted_jobs: List[str] = []
        self.cleanup_handlers: List[callable] = []
        
        # Load SLURM-specific configuration
        self.slurm_config = self._load_slurm_config()

    def _load_slurm_config(self) -> Dict[str, Any]:
        """Load SLURM-specific configuration from inventory."""
        if not os.path.exists(self.inventory_path):
            raise FileNotFoundError(f"Inventory file not found: {self.inventory_path}")

        with open(self.inventory_path, "r") as f:
            if self.inventory_path.endswith(".json"):
                inventory_data = json.load(f)
            else:
                inventory_data = yaml.safe_load(f)

        if "slurm_cluster" not in inventory_data:
            raise ValueError("Invalid SLURM inventory: missing 'slurm_cluster' section")

        return inventory_data["slurm_cluster"]

    def _parse_inventory(self, inventory_data: Dict[str, Any]) -> List[NodeConfig]:
        """Parse SLURM inventory data into NodeConfig objects.

        For SLURM, nodes represent logical execution units (partitions/resources)
        rather than individual physical nodes.

        Args:
            inventory_data: Raw inventory data

        Returns:
            List of NodeConfig objects representing SLURM partitions
        """
        nodes = []

        if "slurm_cluster" in inventory_data:
            slurm_config = inventory_data["slurm_cluster"]
            
            # Create logical nodes from partitions
            for partition in slurm_config.get("partitions", []):
                node = SlurmNodeConfig(
                    hostname=partition["name"],
                    address="slurm-partition",  # Logical address
                    partition=partition["name"],
                    gpu_count=partition.get("default_gpu_count", 1),
                    gpu_vendor=partition.get("gpu_vendor", "AMD"),
                    labels={"partition": partition["name"]},
                    qos=partition.get("qos"),
                    account=partition.get("account"),
                    max_time=partition.get("max_time", "24:00:00"),
                )
                nodes.append(node)

        if not nodes:
            raise ValueError("No SLURM partitions found in inventory")

        return nodes

    def setup_infrastructure(self, workload: WorkloadSpec) -> bool:
        """Setup SLURM infrastructure for distributed execution.

        Args:
            workload: Workload specification

        Returns:
            True if setup successful, False otherwise
        """
        try:
            self.logger.info("Setting up SLURM infrastructure for distributed execution")

            # Validate pre-generated job scripts exist
            if not self._validate_job_scripts():
                self.logger.error("Pre-generated job scripts not found")
                return False

            # Establish connection to SLURM login node
            login_node = self.slurm_config["login_node"]
            self.slurm_connection = SlurmConnection(login_node)
            
            if not self.slurm_connection.connect():
                self.logger.error("Failed to connect to SLURM login node")
                return False

            # Validate SLURM cluster access
            if not self._validate_slurm_access():
                self.logger.error("SLURM cluster validation failed")
                return False

            # Copy job scripts to SLURM login node
            if not self._copy_job_scripts():
                self.logger.error("Failed to copy job scripts to SLURM cluster")
                return False

            self.logger.info("SLURM infrastructure setup completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"SLURM infrastructure setup failed: {e}")
            return False

    def _validate_job_scripts(self) -> bool:
        """Validate that pre-generated job scripts exist."""
        if not self.job_scripts_dir or not self.job_scripts_dir.exists():
            self.logger.error(f"Job scripts directory not found: {self.job_scripts_dir}")
            return False

        # Check for job array script
        job_array_script = self.job_scripts_dir / "madengine_job_array.sh"
        if not job_array_script.exists():
            self.logger.error(f"Job array script not found: {job_array_script}")
            return False

        # Check for setup script
        setup_script = self.job_scripts_dir / "setup_environment.sh"
        if not setup_script.exists():
            self.logger.error(f"Setup script not found: {setup_script}")
            return False

        return True

    def _validate_slurm_access(self) -> bool:
        """Validate SLURM cluster access and permissions."""
        try:
            # Test basic SLURM commands
            exit_code, stdout, stderr = self.slurm_connection.execute_command("sinfo --version")
            if exit_code != 0:
                self.logger.error(f"SLURM not available: {stderr}")
                return False

            # Check available partitions
            exit_code, stdout, stderr = self.slurm_connection.execute_command("sinfo -h -o '%P'")
            if exit_code != 0:
                self.logger.error(f"Failed to query SLURM partitions: {stderr}")
                return False

            available_partitions = [p.strip('*') for p in stdout.strip().split('\n') if p.strip()]
            self.logger.info(f"Available SLURM partitions: {available_partitions}")

            return True

        except Exception as e:
            self.logger.error(f"SLURM access validation failed: {e}")
            return False

    def _copy_job_scripts(self) -> bool:
        """Copy job scripts to SLURM login node."""
        try:
            workspace_path = self.slurm_config.get("workspace", {}).get("shared_filesystem", "/shared/madengine")
            scripts_dir = f"{workspace_path}/job_scripts"

            # Create remote scripts directory
            self.slurm_connection.execute_command(f"mkdir -p {scripts_dir}")

            # Copy all job scripts
            for script_file in self.job_scripts_dir.glob("*.sh"):
                remote_path = f"{scripts_dir}/{script_file.name}"
                if not self.slurm_connection.copy_file(str(script_file), remote_path):
                    return False
                # Make scripts executable
                self.slurm_connection.execute_command(f"chmod +x {remote_path}")

            # Copy Python submission script if exists
            submit_script = self.job_scripts_dir / "submit_jobs.py"
            if submit_script.exists():
                remote_path = f"{workspace_path}/submit_jobs.py"
                if not self.slurm_connection.copy_file(str(submit_script), remote_path):
                    return False
                self.slurm_connection.execute_command(f"chmod +x {remote_path}")

            self.logger.info("Successfully copied job scripts to SLURM cluster")
            return True

        except Exception as e:
            self.logger.error(f"Failed to copy job scripts: {e}")
            return False

    def execute_workload(self, workload: WorkloadSpec) -> DistributedResult:
        """Execute workload using pre-generated SLURM job scripts.

        Args:
            workload: Workload specification (minimal, most config is in scripts)

        Returns:
            Distributed execution result
        """
        try:
            self.logger.info("Starting SLURM distributed execution using pre-generated job scripts")

            # Validate job scripts exist
            if not self._validate_job_scripts():
                return DistributedResult(
                    total_nodes=0,
                    successful_executions=0,
                    failed_executions=1,
                    total_duration=0.0,
                    node_results=[],
                )

            # Submit environment setup job first
            setup_job_id = self._submit_setup_job()
            if setup_job_id:
                self.logger.info(f"Submitted setup job: {setup_job_id}")
                self.submitted_jobs.append(setup_job_id)

            # Submit main job array with dependency on setup job
            main_job_id = self._submit_job_array(setup_job_id)
            if not main_job_id:
                return DistributedResult(
                    total_nodes=0,
                    successful_executions=0,
                    failed_executions=1,
                    total_duration=0.0,
                    node_results=[],
                )

            self.logger.info(f"Submitted main job array: {main_job_id}")
            self.submitted_jobs.append(main_job_id)

            # Monitor job execution
            results = self._monitor_job_execution([main_job_id], workload.timeout)

            # Create distributed result
            distributed_result = DistributedResult(
                total_nodes=len(results),
                successful_executions=sum(1 for r in results if r.status == "SUCCESS"),
                failed_executions=sum(1 for r in results if r.status != "SUCCESS"),
                total_duration=max([r.duration for r in results], default=0.0),
                node_results=results,
            )

            self.logger.info("SLURM distributed execution completed")
            return distributed_result

        except Exception as e:
            self.logger.error(f"SLURM distributed execution failed: {e}")
            return DistributedResult(
                total_nodes=0,
                successful_executions=0,
                failed_executions=1,
                total_duration=0.0,
                node_results=[],
            )

    def _submit_setup_job(self) -> Optional[str]:
        """Submit environment setup job."""
        try:
            workspace_path = self.slurm_config.get("workspace", {}).get("shared_filesystem", "/shared/madengine")
            setup_script = f"{workspace_path}/job_scripts/setup_environment.sh"

            # Submit setup job
            cmd = f"sbatch {setup_script}"
            exit_code, stdout, stderr = self.slurm_connection.execute_command(cmd)

            if exit_code == 0:
                # Extract job ID from sbatch output
                job_id = stdout.strip().split()[-1]
                return job_id
            else:
                self.logger.error(f"Failed to submit setup job: {stderr}")
                return None

        except Exception as e:
            self.logger.error(f"Setup job submission failed: {e}")
            return None

    def _submit_job_array(self, dependency_job_id: Optional[str] = None) -> Optional[str]:
        """Submit main job array."""
        try:
            workspace_path = self.slurm_config.get("workspace", {}).get("shared_filesystem", "/shared/madengine")
            job_array_script = f"{workspace_path}/job_scripts/madengine_job_array.sh"

            # Build sbatch command
            cmd = "sbatch"
            if dependency_job_id:
                cmd += f" --dependency=afterok:{dependency_job_id}"
            cmd += f" {job_array_script}"

            # Submit job array
            exit_code, stdout, stderr = self.slurm_connection.execute_command(cmd)

            if exit_code == 0:
                # Extract job ID from sbatch output
                job_id = stdout.strip().split()[-1]
                return job_id
            else:
                self.logger.error(f"Failed to submit job array: {stderr}")
                return None

        except Exception as e:
            self.logger.error(f"Job array submission failed: {e}")
            return None

    def _monitor_job_execution(self, job_ids: List[str], timeout: int) -> List[ExecutionResult]:
        """Monitor SLURM job execution until completion."""
        results = []
        start_time = time.time()

        self.logger.info(f"Monitoring SLURM jobs: {job_ids}")

        while job_ids and (time.time() - start_time) < timeout:
            completed_jobs = []

            for job_id in job_ids:
                try:
                    # Check job status
                    status = self._get_job_status(job_id)
                    
                    if status in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"]:
                        # Job completed, collect results
                        job_results = self._collect_job_results(job_id, status)
                        results.extend(job_results)
                        completed_jobs.append(job_id)
                        
                        self.logger.info(f"Job {job_id} completed with status: {status}")

                except Exception as e:
                    self.logger.error(f"Error checking job {job_id}: {e}")
                    # Create failed result
                    result = ExecutionResult(
                        node_id=job_id,
                        model_tag="unknown",
                        status="FAILURE",
                        duration=time.time() - start_time,
                        error_message=str(e),
                    )
                    results.append(result)
                    completed_jobs.append(job_id)

            # Remove completed jobs
            for job_id in completed_jobs:
                job_ids.remove(job_id)

            if job_ids:
                time.sleep(30)  # Check every 30 seconds

        # Handle timeout for remaining jobs
        for job_id in job_ids:
            result = ExecutionResult(
                node_id=job_id,
                model_tag="timeout",
                status="TIMEOUT",
                duration=timeout,
                error_message=f"Job monitoring timed out after {timeout} seconds",
            )
            results.append(result)

        return results

    def _get_job_status(self, job_id: str) -> str:
        """Get SLURM job status."""
        try:
            cmd = f"squeue -j {job_id} -h -o '%T'"
            exit_code, stdout, stderr = self.slurm_connection.execute_command(cmd)

            if exit_code == 0 and stdout.strip():
                return stdout.strip()
            else:
                # Job not in queue, check if completed
                cmd = f"sacct -j {job_id} -n -o 'State' | head -1"
                exit_code, stdout, stderr = self.slurm_connection.execute_command(cmd)
                
                if exit_code == 0 and stdout.strip():
                    return stdout.strip()
                else:
                    return "UNKNOWN"

        except Exception as e:
            self.logger.error(f"Failed to get job status for {job_id}: {e}")
            return "ERROR"

    def _collect_job_results(self, job_id: str, status: str) -> List[ExecutionResult]:
        """Collect results from completed SLURM job."""
        results = []
        
        try:
            # For job arrays, get results for each array task
            if "_" in job_id:  # Job array format: jobid_arrayindex
                # This is a single array task
                result = self._get_single_job_result(job_id, status)
                results.append(result)
            else:
                # This is a job array, get results for all tasks
                cmd = f"sacct -j {job_id} -n -o 'JobID,State,ExitCode' | grep '{job_id}_'"
                exit_code, stdout, stderr = self.slurm_connection.execute_command(cmd)
                
                if exit_code == 0:
                    for line in stdout.strip().split('\n'):
                        if line.strip():
                            parts = line.strip().split()
                            array_job_id = parts[0]
                            array_status = parts[1]
                            
                            result = self._get_single_job_result(array_job_id, array_status)
                            results.append(result)
                else:
                    # Fallback: create single result
                    result = self._get_single_job_result(job_id, status)
                    results.append(result)

        except Exception as e:
            self.logger.error(f"Failed to collect results for job {job_id}: {e}")
            result = ExecutionResult(
                node_id=job_id,
                model_tag="error",
                status="FAILURE",
                duration=0.0,
                error_message=str(e),
            )
            results.append(result)

        return results

    def _get_single_job_result(self, job_id: str, status: str) -> ExecutionResult:
        """Get result for a single SLURM job."""
        try:
            # Get job details
            cmd = f"sacct -j {job_id} -n -o 'JobName,State,ExitCode,Elapsed,NodeList'"
            exit_code, stdout, stderr = self.slurm_connection.execute_command(cmd)
            
            job_name = "unknown"
            elapsed_time = 0.0
            node_list = "unknown"
            exit_code_val = "0:0"
            
            if exit_code == 0 and stdout.strip():
                parts = stdout.strip().split()
                if len(parts) >= 5:
                    job_name = parts[0]
                    exit_code_val = parts[2]
                    elapsed_str = parts[3]
                    node_list = parts[4]
                    
                    # Parse elapsed time (format: HH:MM:SS or MM:SS)
                    time_parts = elapsed_str.split(':')
                    if len(time_parts) == 3:
                        elapsed_time = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                    elif len(time_parts) == 2:
                        elapsed_time = int(time_parts[0]) * 60 + int(time_parts[1])

            # Extract model tag from job name
            model_tag = job_name.replace("madengine-", "").replace("-", "_")
            if not model_tag or model_tag == "unknown":
                model_tag = f"task_{job_id.split('_')[-1] if '_' in job_id else '0'}"

            # Determine success based on SLURM status and exit code
            success = status == "COMPLETED" and exit_code_val.startswith("0:")

            return ExecutionResult(
                node_id=node_list,
                model_tag=model_tag,
                status="SUCCESS" if success else "FAILURE",
                duration=elapsed_time,
                performance_metrics={"slurm_job_id": job_id, "slurm_status": status},
                error_message=None if success else f"SLURM status: {status}, Exit code: {exit_code_val}",
            )

        except Exception as e:
            self.logger.error(f"Failed to get job result for {job_id}: {e}")
            return ExecutionResult(
                node_id=job_id,
                model_tag="error",
                status="FAILURE",
                duration=0.0,
                error_message=str(e),
            )

    def cleanup_infrastructure(self, workload: WorkloadSpec) -> bool:
        """Cleanup SLURM infrastructure after execution.

        Args:
            workload: Workload specification

        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            self.logger.info("Cleaning up SLURM infrastructure")

            # Cancel any remaining/running jobs
            for job_id in self.submitted_jobs:
                try:
                    cmd = f"scancel {job_id}"
                    self.slurm_connection.execute_command(cmd)
                    self.logger.info(f"Cancelled SLURM job: {job_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to cancel job {job_id}: {e}")

            # Run custom cleanup handlers
            for cleanup_handler in self.cleanup_handlers:
                try:
                    cleanup_handler()
                except Exception as e:
                    self.logger.warning(f"Cleanup handler failed: {e}")

            # Close SLURM connection
            if self.slurm_connection:
                self.slurm_connection.close()
                self.slurm_connection = None

            self.logger.info("SLURM infrastructure cleanup completed")
            return True

        except Exception as e:
            self.logger.error(f"SLURM cleanup failed: {e}")
            return False

    def add_cleanup_handler(self, handler: callable):
        """Add a cleanup handler to be called during cleanup."""
        self.cleanup_handlers.append(handler)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_infrastructure(None)