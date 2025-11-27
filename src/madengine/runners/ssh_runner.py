#!/usr/bin/env python3
"""
SSH Distributed Runner for MADEngine

This module implements SSH-based distributed execution using paramiko
for secure remote execution across multiple nodes.
"""

import json
import logging
import os
import time
import contextlib
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

try:
    import paramiko
    from scp import SCPClient
except ImportError:
    raise ImportError(
        "SSH runner requires paramiko and scp. Install with: pip install paramiko scp"
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


# Legacy error classes - use unified error system instead
# Kept for backward compatibility but deprecated

@dataclass
class SSHConnectionError(MADConnectionError):
    """Deprecated: Use MADConnectionError instead."""

    hostname: str
    error_type: str
    message: str

    def __init__(self, hostname: str, error_type: str, message: str):
        self.hostname = hostname
        self.error_type = error_type
        self.message = message
        context = create_error_context(
            operation="ssh_connection",
            component="SSHRunner",
            node_id=hostname,
            additional_info={"error_type": error_type}
        )
        super().__init__(f"SSH {error_type} error on {hostname}: {message}", context=context)


class TimeoutError(MADTimeoutError):
    """Deprecated: Use MADTimeoutError instead."""
    
    def __init__(self, message: str, **kwargs):
        context = create_error_context(operation="ssh_execution", component="SSHRunner")
        super().__init__(message, context=context, **kwargs)


@contextlib.contextmanager
def timeout_context(seconds: int):
    """Context manager for handling timeouts."""

    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class SSHConnection:
    """Manages SSH connection to a single node with enhanced error handling."""

    def __init__(self, node: NodeConfig, timeout: int = 30):
        """Initialize SSH connection.

        Args:
            node: Node configuration
            timeout: Connection timeout in seconds
        """
        self.node = node
        self.timeout = timeout
        self.ssh_client = None
        self.sftp_client = None
        self.logger = logging.getLogger(f"SSHConnection.{node.hostname}")
        self._connected = False
        self._connection_attempts = 0
        self._max_connection_attempts = 3

    def connect(self) -> bool:
        """Establish SSH connection to node with retry logic.

        Returns:
            True if connection successful, False otherwise
        """
        for attempt in range(self._max_connection_attempts):
            try:
                self._connection_attempts = attempt + 1
                self.ssh_client = paramiko.SSHClient()
                self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

                # Connection parameters
                connect_params = {
                    "hostname": self.node.address,
                    "port": self.node.port,
                    "username": self.node.username,
                    "timeout": self.timeout,
                }

                # Use SSH key if provided - expand path
                if self.node.ssh_key_path:
                    expanded_key_path = os.path.expanduser(self.node.ssh_key_path)
                    if os.path.exists(expanded_key_path):
                        connect_params["key_filename"] = expanded_key_path
                        # Ensure proper permissions
                        os.chmod(expanded_key_path, 0o600)
                    else:
                        self.logger.warning(
                            f"SSH key file not found: {expanded_key_path}"
                        )

                # Test connection with timeout
                with timeout_context(self.timeout):
                    self.ssh_client.connect(**connect_params)
                    self.sftp_client = self.ssh_client.open_sftp()

                self._connected = True
                self.logger.info(f"Successfully connected to {self.node.hostname}")
                return True

            except TimeoutError:
                self.logger.warning(f"Connection attempt {attempt + 1} timed out")
                if attempt < self._max_connection_attempts - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                continue

            except paramiko.AuthenticationException as e:
                raise SSHConnectionError(
                    self.node.hostname, "authentication", f"Authentication failed: {e}"
                )

            except paramiko.SSHException as e:
                self.logger.warning(f"SSH error on attempt {attempt + 1}: {e}")
                if attempt < self._max_connection_attempts - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                continue

            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < self._max_connection_attempts - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                continue

        self.logger.error(
            f"Failed to connect to {self.node.hostname} after {self._max_connection_attempts} attempts"
        )
        return False

    def is_connected(self) -> bool:
        """Check if connection is active."""
        return (
            self._connected
            and self.ssh_client
            and self.ssh_client.get_transport().is_active()
        )

    def close(self):
        """Close SSH connection safely."""
        try:
            if self.sftp_client:
                self.sftp_client.close()
                self.sftp_client = None
            if self.ssh_client:
                self.ssh_client.close()
                self.ssh_client = None
            self._connected = False
            self.logger.debug(f"Closed connection to {self.node.hostname}")
        except Exception as e:
            self.logger.warning(f"Error closing connection: {e}")

    def __enter__(self):
        """Context manager entry."""
        if not self.connect():
            raise SSHConnectionError(
                self.node.hostname, "connection", "Failed to establish connection"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def execute_command(self, command: str, timeout: int = 300) -> tuple:
        """Execute command on remote node with enhanced error handling.

        Args:
            command: Command to execute
            timeout: Command timeout in seconds

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        if not self.is_connected():
            raise SSHConnectionError(
                self.node.hostname, "connection", "Connection not established"
            )

        try:
            with timeout_context(timeout):
                stdin, stdout, stderr = self.ssh_client.exec_command(
                    command, timeout=timeout
                )

                # Wait for command completion
                exit_code = stdout.channel.recv_exit_status()

                stdout_str = stdout.read().decode("utf-8", errors="replace")
                stderr_str = stderr.read().decode("utf-8", errors="replace")

                return exit_code, stdout_str, stderr_str

        except TimeoutError:
            raise SSHConnectionError(
                self.node.hostname,
                "timeout",
                f"Command timed out after {timeout} seconds: {command}",
            )
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return 1, "", str(e)

    def copy_file(
        self, local_path: str, remote_path: str, create_dirs: bool = True
    ) -> bool:
        """Copy file to remote node with enhanced error handling.

        Args:
            local_path: Local file path
            remote_path: Remote file path
            create_dirs: Whether to create remote directories

        Returns:
            True if copy successful, False otherwise
        """
        if not self.is_connected():
            raise SSHConnectionError(
                self.node.hostname, "connection", "Connection not established"
            )

        try:
            # Validate local file exists
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Local file not found: {local_path}")

            # Create directory if needed
            if create_dirs:
                remote_dir = os.path.dirname(remote_path)
                if remote_dir:
                    self.execute_command(f"mkdir -p {remote_dir}")

            # Copy file
            self.sftp_client.put(local_path, remote_path)

            # Set proper permissions
            self.sftp_client.chmod(remote_path, 0o644)

            self.logger.debug(f"Successfully copied {local_path} to {remote_path}")
            return True

        except Exception as e:
            self.logger.error(f"File copy failed: {e}")
            return False

    def copy_directory(self, local_path: str, remote_path: str) -> bool:
        """Copy directory to remote node with enhanced error handling.

        Args:
            local_path: Local directory path
            remote_path: Remote directory path

        Returns:
            True if copy successful, False otherwise
        """
        if not self.is_connected():
            raise SSHConnectionError(
                self.node.hostname, "connection", "Connection not established"
            )

        try:
            # Validate local directory exists
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Local directory not found: {local_path}")

            # Use SCP for directory transfer
            with SCPClient(self.ssh_client.get_transport()) as scp:
                scp.put(local_path, remote_path, recursive=True)

            self.logger.debug(
                f"Successfully copied directory {local_path} to {remote_path}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Directory copy failed: {e}")
            return False


class SSHDistributedRunner(BaseDistributedRunner):
    """Distributed runner using SSH connections with enhanced error handling."""

    def __init__(self, inventory_path: str, **kwargs):
        """Initialize SSH distributed runner.

        Args:
            inventory_path: Path to inventory configuration file
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(inventory_path, **kwargs)
        self.connections: Dict[str, SSHConnection] = {}
        self.connection_pool: Optional[ThreadPoolExecutor] = None
        self.cleanup_handlers: List[callable] = []

    def _create_connection(self, node: NodeConfig) -> Optional[SSHConnection]:
        """Create SSH connection to node with proper error handling.

        Args:
            node: Node configuration

        Returns:
            SSH connection instance or None if failed
        """
        try:
            connection = SSHConnection(node, timeout=30)
            if connection.connect():
                self.connections[node.hostname] = connection
                return connection
            return None
        except SSHConnectionError as e:
            self.logger.error(f"SSH connection error: {e}")
            return None
        except Exception as e:
            self.logger.error(
                f"Unexpected error creating connection to {node.hostname}: {e}"
            )
            return None

    def setup_infrastructure(self, workload: WorkloadSpec) -> bool:
        """Setup SSH infrastructure for distributed execution with enhanced error handling.

        Args:
            workload: Workload specification

        Returns:
            True if setup successful, False otherwise
        """
        try:
            self.logger.info("Setting up SSH infrastructure for distributed execution")

            # Filter nodes based on workload requirements
            target_nodes = self.filter_nodes(workload.node_selector)
            if not target_nodes:
                self.logger.error("No nodes match the workload requirements")
                return False

            # Create connection pool
            self.connection_pool = ThreadPoolExecutor(max_workers=len(target_nodes))

            # Setup connections and environment in parallel
            setup_futures = []

            for node in target_nodes:
                future = self.connection_pool.submit(self._setup_node, node, workload)
                setup_futures.append((node, future))

            # Collect results
            success_count = 0
            failed_nodes = []

            for node, future in setup_futures:
                try:
                    if future.result(timeout=600):  # 10 minute timeout per node
                        success_count += 1
                    else:
                        failed_nodes.append(node.hostname)
                except Exception as e:
                    self.logger.error(f"Setup failed for {node.hostname}: {e}")
                    failed_nodes.append(node.hostname)

            if failed_nodes:
                self.logger.warning(f"Failed to setup nodes: {failed_nodes}")

            if success_count == 0:
                self.logger.error("Failed to setup any nodes")
                return False

            self.logger.info(
                f"Successfully setup infrastructure on {success_count} nodes"
            )
            return True

        except Exception as e:
            self.logger.error(f"Infrastructure setup failed: {e}")
            return False

    def _setup_node(self, node: NodeConfig, workload: WorkloadSpec) -> bool:
        """Setup a single node for execution - simplified to focus on manifest distribution."""
        try:
            # Create connection
            connection = self._create_connection(node)
            if not connection:
                return False

            # Setup MAD environment (clone/update repository and install)
            if not self._setup_mad_environment(connection, node.hostname):
                return False

            # Copy build manifest - this is the key file we need
            if not self._copy_build_manifest(connection, workload.manifest_file):
                self.logger.error(f"Failed to copy manifest to {node.hostname}")
                return False

            # Copy any supporting files that might be needed (credential.json, data.json, etc.)
            if not self._copy_supporting_files(connection):
                self.logger.warning(
                    f"Failed to copy some supporting files to {node.hostname}"
                )
                # Don't fail for supporting files, just warn

            return True

        except Exception as e:
            self.logger.error(f"Node setup failed for {node.hostname}: {e}")
            return False

    def _copy_supporting_files(self, connection: SSHConnection) -> bool:
        """Copy supporting files that might be needed for execution."""
        supporting_files = ["credential.json", "data.json", "models.json"]
        success = True

        for file_name in supporting_files:
            if os.path.exists(file_name):
                try:
                    remote_path = f"MAD/{file_name}"
                    if not connection.copy_file(file_name, remote_path):
                        self.logger.warning(f"Failed to copy {file_name}")
                        success = False
                except Exception as e:
                    self.logger.warning(f"Error copying {file_name}: {e}")
                    success = False

        return success

    def _setup_mad_environment(self, connection: SSHConnection, hostname: str) -> bool:
        """Setup MAD repository and madengine-cli on a remote node with retry logic."""
        self.logger.info(f"Setting up MAD environment on {hostname}")

        max_retries = 3

        # Enhanced setup commands for madengine-cli
        setup_commands = [
            # Clone or update MAD repository
            (
                "if [ -d MAD ]; then cd MAD && git pull origin main; "
                "else git clone https://github.com/ROCm/MAD.git; fi"
            ),
            # Setup Python environment and install madengine
            "cd MAD",
            "python3 -m venv venv || true",
            "source venv/bin/activate",
            # Install dependencies and madengine
            "pip install --upgrade pip",
            "pip install -r requirements.txt",
            "pip install -e .",
            # Verify madengine-cli is installed and working
            "which madengine-cli",
            "madengine-cli --help > /dev/null",
        ]

        for attempt in range(max_retries):
            try:
                for i, command in enumerate(setup_commands):
                    self.logger.debug(
                        f"Executing setup command {i+1}/{len(setup_commands)} on {hostname}"
                    )
                    exit_code, stdout, stderr = connection.execute_command(
                        command, timeout=300
                    )
                    if exit_code != 0:
                        self.logger.warning(
                            f"MAD setup command failed on attempt {attempt + 1} "
                            f"on {hostname}: {command}\nStderr: {stderr}"
                        )
                        if attempt == max_retries - 1:
                            self.logger.error(
                                f"Failed to setup MAD environment on {hostname} "
                                f"after {max_retries} attempts"
                            )
                            return False
                        break
                else:
                    # All commands succeeded
                    self.logger.info(
                        f"Successfully set up MAD environment on {hostname}"
                    )
                    return True

            except SSHConnectionError as e:
                self.logger.warning(f"SSH error during MAD setup on {hostname}: {e}")
                if attempt == max_retries - 1:
                    return False
                time.sleep(2**attempt)  # Exponential backoff

            except Exception as e:
                self.logger.warning(
                    f"MAD setup attempt {attempt + 1} exception on " f"{hostname}: {e}"
                )
                if attempt == max_retries - 1:
                    self.logger.error(
                        f"Failed to setup MAD environment on {hostname} "
                        f"after {max_retries} attempts"
                    )
                    return False
                time.sleep(2**attempt)  # Exponential backoff

        return False

    def _copy_build_manifest(
        self, connection: SSHConnection, manifest_file: str
    ) -> bool:
        """Copy build manifest to remote node with error handling."""
        try:
            if not manifest_file or not os.path.exists(manifest_file):
                self.logger.error(f"Build manifest file not found: {manifest_file}")
                return False

            remote_path = "MAD/build_manifest.json"
            success = connection.copy_file(manifest_file, remote_path)

            if success:
                self.logger.info(
                    f"Successfully copied build manifest to {connection.node.hostname}"
                )

            return success

        except Exception as e:
            self.logger.error(f"Failed to copy build manifest: {e}")
            return False

    def execute_workload(self, workload: WorkloadSpec) -> DistributedResult:
        """Execute workload across distributed nodes using build manifest.

        This method distributes the pre-built manifest to remote nodes and
        executes 'madengine-cli run' on each node.

        Args:
            workload: Workload specification containing manifest file path

        Returns:
            Distributed execution result
        """
        try:
            self.logger.info("Starting SSH distributed execution using build manifest")

            # Validate manifest file exists
            if not workload.manifest_file or not os.path.exists(workload.manifest_file):
                return DistributedResult(
                    success=False,
                    node_results=[],
                    error_message=f"Build manifest file not found: {workload.manifest_file}",
                )

            # Load manifest to get model tags and configuration
            try:
                with open(workload.manifest_file, "r") as f:
                    manifest_data = json.load(f)

                # Extract model tags from manifest
                model_tags = []
                if "models" in manifest_data:
                    model_tags = list(manifest_data["models"].keys())
                elif "model_tags" in manifest_data:
                    model_tags = manifest_data["model_tags"]

                if not model_tags:
                    self.logger.warning("No model tags found in manifest")
                    model_tags = ["dummy"]  # fallback

            except Exception as e:
                return DistributedResult(
                    success=False,
                    node_results=[],
                    error_message=f"Failed to parse manifest: {e}",
                )

            # Get target nodes
            target_nodes = self.filter_nodes(workload.node_selector)
            if not target_nodes:
                return DistributedResult(
                    success=False,
                    node_results=[],
                    error_message="No nodes match the workload requirements",
                )

            # Setup infrastructure
            if not self.setup_infrastructure(workload):
                return DistributedResult(
                    success=False,
                    node_results=[],
                    error_message="Failed to setup SSH infrastructure",
                )

            # Execute in parallel across nodes and models
            execution_futures = []

            for node in target_nodes:
                # Execute all models on this node (or distribute models across nodes)
                future = self.connection_pool.submit(
                    self._execute_models_on_node_safe, node, model_tags, workload
                )
                execution_futures.append((node, future))

            # Collect results
            results = []

            for node, future in execution_futures:
                try:
                    node_results = future.result(
                        timeout=workload.timeout + 120
                    )  # Extra buffer
                    results.extend(node_results)
                except Exception as e:
                    self.logger.error(f"Execution failed on {node.hostname}: {e}")
                    # Create failed result for all models on this node
                    for model_tag in model_tags:
                        failed_result = ExecutionResult(
                            node_id=node.hostname,
                            model_tag=model_tag,
                            success=False,
                            error_message=str(e),
                        )
                        results.append(failed_result)

            # Aggregate results
            distributed_result = DistributedResult(
                success=any(r.success for r in results), node_results=results
            )

            self.logger.info("SSH distributed execution completed")
            return distributed_result

        except Exception as e:
            self.logger.error(f"Distributed execution failed: {e}")
            return DistributedResult(
                success=False, node_results=[], error_message=str(e)
            )

    def _execute_models_on_node_safe(
        self, node: NodeConfig, model_tags: List[str], workload: WorkloadSpec
    ) -> List[ExecutionResult]:
        """Execute all models on a specific node with comprehensive error handling."""
        try:
            return self._execute_models_on_node(node, model_tags, workload)
        except Exception as e:
            self.logger.error(f"Models execution failed on {node.hostname}: {e}")
            # Return failed results for all models
            results = []
            for model_tag in model_tags:
                results.append(
                    ExecutionResult(
                        node_id=node.hostname,
                        model_tag=model_tag,
                        success=False,
                        error_message=str(e),
                    )
                )
            return results

    def _execute_models_on_node(
        self, node: NodeConfig, model_tags: List[str], workload: WorkloadSpec
    ) -> List[ExecutionResult]:
        """Execute models on a specific node using 'madengine-cli run'."""
        results = []

        try:
            connection = self.connections.get(node.hostname)
            if not connection or not connection.is_connected():
                raise SSHConnectionError(
                    node.hostname, "connection", "Connection not available"
                )

            # Execute madengine-cli run with the manifest
            start_time = time.time()

            # Build command to run madengine-cli with the manifest
            command = self._build_execution_command(workload)

            self.logger.info(f"Executing on {node.hostname}: {command}")

            exit_code, stdout, stderr = connection.execute_command(
                command, timeout=workload.timeout
            )

            execution_time = time.time() - start_time

            # Parse output to extract per-model results
            # For now, create results for all models with the same status
            for model_tag in model_tags:
                result = ExecutionResult(
                    node_id=node.hostname,
                    model_tag=model_tag,
                    success=(exit_code == 0),
                    output=stdout,
                    error_message=stderr if exit_code != 0 else None,
                    execution_time=execution_time
                    / len(model_tags),  # Distribute time across models
                )
                results.append(result)

                if exit_code == 0:
                    self.logger.info(
                        f"Successfully executed {model_tag} on {node.hostname}"
                    )
                else:
                    self.logger.warning(
                        f"Execution failed for {model_tag} on {node.hostname}"
                    )

            return results

        except SSHConnectionError as e:
            # Return failed results for all models
            for model_tag in model_tags:
                results.append(
                    ExecutionResult(
                        node_id=node.hostname,
                        model_tag=model_tag,
                        success=False,
                        error_message=str(e),
                        execution_time=0,
                    )
                )
            return results
        except Exception as e:
            # Return failed results for all models
            for model_tag in model_tags:
                results.append(
                    ExecutionResult(
                        node_id=node.hostname,
                        model_tag=model_tag,
                        success=False,
                        error_message=str(e),
                        execution_time=0,
                    )
                )
            return results

    def _build_execution_command(self, workload: WorkloadSpec) -> str:
        """Build the madengine-cli run command with the manifest file.

        Args:
            workload: Workload specification containing manifest file

        Returns:
            Command string to execute on remote node
        """
        # The basic command structure
        cmd_parts = [
            "cd MAD",
            "source venv/bin/activate",
            f"madengine-cli run --manifest-file build_manifest.json",
        ]

        # Add timeout if specified (and not default)
        if workload.timeout and workload.timeout > 0 and workload.timeout != 3600:
            cmd_parts[-1] += f" --timeout {workload.timeout}"

        # Add registry if specified
        if workload.registry:
            cmd_parts[-1] += f" --registry {workload.registry}"

        # Add live output for better monitoring
        cmd_parts[-1] += " --live-output"

        # Combine all commands
        return " && ".join(cmd_parts)

    def _execute_model_on_node_safe(
        self, node: NodeConfig, model_tag: str, workload: WorkloadSpec
    ) -> ExecutionResult:
        """Execute a model on a specific node with comprehensive error handling."""
        try:
            return self._execute_model_on_node(node, model_tag, workload)
        except Exception as e:
            self.logger.error(f"Model execution failed on {node.hostname}: {e}")
            return ExecutionResult(
                node_id=node.hostname,
                model_tag=model_tag,
                success=False,
                error_message=str(e),
            )

    def _execute_model_on_node(
        self, node: NodeConfig, model_tag: str, workload: WorkloadSpec
    ) -> ExecutionResult:
        """Execute a model on a specific node with timeout and error handling."""
        start_time = time.time()

        try:
            connection = self.connections.get(node.hostname)
            if not connection or not connection.is_connected():
                raise SSHConnectionError(
                    node.hostname, "connection", "Connection not available"
                )

            # Build and execute command
            command = self._build_execution_command(node, model_tag, workload)

            exit_code, stdout, stderr = connection.execute_command(
                command, timeout=workload.timeout
            )

            execution_time = time.time() - start_time

            # Create execution result
            result = ExecutionResult(
                node_id=node.hostname,
                model_tag=model_tag,
                success=(exit_code == 0),
                output=stdout,
                error_message=stderr if exit_code != 0 else None,
                execution_time=execution_time,
            )

            if exit_code == 0:
                self.logger.info(
                    f"Successfully executed {model_tag} on {node.hostname}"
                )
            else:
                self.logger.warning(
                    f"Execution failed for {model_tag} on {node.hostname}"
                )

            return result

        except SSHConnectionError as e:
            return ExecutionResult(
                node_id=node.hostname,
                model_tag=model_tag,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return ExecutionResult(
                node_id=node.hostname,
                model_tag=model_tag,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
            )

    def cleanup_infrastructure(self, workload: WorkloadSpec) -> bool:
        """Cleanup infrastructure after execution with comprehensive cleanup.

        Args:
            workload: Workload specification

        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            self.logger.info("Cleaning up SSH infrastructure")

            # Run custom cleanup handlers
            for cleanup_handler in self.cleanup_handlers:
                try:
                    cleanup_handler()
                except Exception as e:
                    self.logger.warning(f"Cleanup handler failed: {e}")

            # Close all connections
            for hostname, connection in self.connections.items():
                try:
                    connection.close()
                except Exception as e:
                    self.logger.warning(f"Error closing connection to {hostname}: {e}")

            self.connections.clear()

            # Shutdown connection pool
            if self.connection_pool:
                self.connection_pool.shutdown(wait=True)
                self.connection_pool = None

            self.logger.info("SSH infrastructure cleanup completed")
            return True

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
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

    # ...existing methods remain the same...
