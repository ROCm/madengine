#!/usr/bin/env python3
"""
Ansible Distributed Runner for MADEngine

This module implements Ansible-based distributed execution using
the ansible-runner library for orchestrated parallel execution.
"""

import json
import os
import tempfile
import time
import yaml
from typing import List, Optional, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

try:
    import ansible_runner
except ImportError:
    raise ImportError(
        "Ansible runner requires ansible-runner. "
        "Install with: pip install ansible-runner"
    )

from madengine.runners.base import (
    BaseDistributedRunner,
    NodeConfig,
    WorkloadSpec,
    ExecutionResult,
    DistributedResult,
)
from madengine.core.errors import (
    RunnerError,
    ConfigurationError,
    create_error_context
)


@dataclass
class AnsibleExecutionError(RunnerError):
    """Ansible execution specific errors."""

    playbook_path: str
    
    def __init__(self, message: str, playbook_path: str, **kwargs):
        self.playbook_path = playbook_path
        context = create_error_context(
            operation="ansible_execution",
            component="AnsibleRunner",
            file_path=playbook_path
        )
        super().__init__(message, context=context, **kwargs)


class AnsibleDistributedRunner(BaseDistributedRunner):
    """Distributed runner using Ansible with enhanced error handling."""

    def __init__(self, inventory_path: str, playbook_path: str = None, **kwargs):
        """Initialize Ansible distributed runner.

        Args:
            inventory_path: Path to Ansible inventory file
            playbook_path: Path to pre-generated Ansible playbook file
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(inventory_path, **kwargs)
        self.playbook_path = playbook_path or "madengine_distributed.yml"
        self.playbook_dir = kwargs.get("playbook_dir", "/tmp/madengine_ansible")
        self.cleanup_handlers: List[callable] = []
        self.created_files: List[str] = []
        self.executor: Optional[ThreadPoolExecutor] = None

    def _validate_inventory(self) -> bool:
        """Validate Ansible inventory file."""
        try:
            if not os.path.exists(self.inventory_path):
                self.logger.error(f"Inventory file not found: {self.inventory_path}")
                return False

            # Try to parse inventory
            with open(self.inventory_path, "r") as f:
                content = f.read()

            # Basic validation - should contain host information
            if not content.strip():
                self.logger.error("Inventory file is empty")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Invalid inventory file: {e}")
            return False

    def _ensure_playbook_directory(self) -> bool:
        """Ensure playbook directory exists and is writable."""
        try:
            os.makedirs(self.playbook_dir, exist_ok=True)

            # Test write permissions
            test_file = os.path.join(self.playbook_dir, ".test_write")
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                return True
            except Exception as e:
                self.logger.error(f"Playbook directory not writable: {e}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to create playbook directory: {e}")
            return False

    def _create_ansible_inventory(self, target_nodes: List[NodeConfig]) -> str:
        """Create Ansible inventory file from node configurations.

        Args:
            target_nodes: List of target nodes

        Returns:
            Path to created inventory file
        """
        inventory_data = {
            "gpu_nodes": {
                "hosts": {},
                "vars": {
                    "ansible_user": "root",
                    "ansible_ssh_common_args": "-o StrictHostKeyChecking=no",
                },
            }
        }

        for node in target_nodes:
            host_vars = {
                "ansible_host": node.address,
                "ansible_port": node.port,
                "ansible_user": node.username,
                "gpu_count": node.gpu_count,
                "gpu_vendor": node.gpu_vendor,
            }

            # Add SSH key if provided
            if node.ssh_key_path:
                host_vars["ansible_ssh_private_key_file"] = node.ssh_key_path

            # Add custom labels as variables
            host_vars.update(node.labels)

            inventory_data["gpu_nodes"]["hosts"][node.hostname] = host_vars

        # Write inventory file
        inventory_file = os.path.join(self.playbook_dir, "inventory.yml")
        with open(inventory_file, "w") as f:
            yaml.dump(inventory_data, f, default_flow_style=False)

        return inventory_file

    def setup_infrastructure(self, workload: WorkloadSpec) -> bool:
        """Setup Ansible infrastructure for distributed execution.

        Args:
            workload: Workload specification

        Returns:
            True if setup successful, False otherwise
        """
        try:
            self.logger.info("Setting up Ansible infrastructure")

            # Validate prerequisites
            if not self._validate_inventory():
                return False

            if not self._ensure_playbook_directory():
                return False

            # Validate that the pre-generated playbook exists
            if not os.path.exists(self.playbook_path):
                self.logger.error(
                    f"Playbook file not found: {self.playbook_path}. "
                    f"Generate it first using 'madengine-cli generate ansible'"
                )
                return False

            # Create executor
            self.executor = ThreadPoolExecutor(max_workers=4)

            self.logger.info("Ansible infrastructure setup completed")
            return True

        except Exception as e:
            self.logger.error(f"Ansible infrastructure setup failed: {e}")
            return False

    def _execute_playbook(self) -> bool:
        """Execute the pre-generated Ansible playbook."""
        try:
            self.logger.info(f"Executing Ansible playbook: {self.playbook_path}")

            # Use ansible-runner for execution
            result = ansible_runner.run(
                private_data_dir=self.playbook_dir,
                playbook=os.path.basename(self.playbook_path),
                inventory=self.inventory_path,
                suppress_env_files=True,
                quiet=False,
            )

            if result.status == "successful":
                self.logger.info("Ansible playbook completed successfully")
                return True
            else:
                self.logger.error(
                    f"Ansible playbook failed with status: {result.status}"
                )

                # Log detailed error information
                if hasattr(result, "stderr") and result.stderr:
                    self.logger.error(f"Stderr: {result.stderr}")

                return False

        except Exception as e:
            self.logger.error(f"Playbook execution failed: {e}")
            return False

    def execute_workload(self, workload: WorkloadSpec) -> DistributedResult:
        """Execute workload using pre-generated Ansible playbook.

        Args:
            workload: Minimal workload specification (most config is in playbook)

        Returns:
            Distributed execution result
        """
        try:
            self.logger.info("Starting Ansible distributed workload execution")

            # Validate that the pre-generated playbook exists
            if not os.path.exists(self.playbook_path):
                return DistributedResult(
                    success=False,
                    node_results=[],
                    error_message=f"Playbook file not found: {self.playbook_path}. "
                    f"Generate it first using 'madengine-cli generate ansible'",
                )

            # Execute the pre-generated playbook directly
            if not self._execute_playbook():
                return DistributedResult(
                    success=False,
                    node_results=[],
                    error_message="Playbook execution failed",
                )

            # Parse results
            results = self._parse_execution_results()

            distributed_result = DistributedResult(
                success=any(r.success for r in results), node_results=results
            )

            self.logger.info("Ansible distributed workload execution completed")
            return distributed_result

        except Exception as e:
            self.logger.error(f"Distributed execution failed: {e}")
            return DistributedResult(
                success=False, node_results=[], error_message=str(e)
            )

    def _parse_execution_results(self) -> List[ExecutionResult]:
        """Parse execution results from Ansible output."""
        results = []

        try:
            # Parse results from ansible-runner output
            artifacts_dir = os.path.join(self.playbook_dir, "artifacts")
            if not os.path.exists(artifacts_dir):
                self.logger.warning("No artifacts directory found")
                return results

            # Look for job events or stdout
            stdout_file = os.path.join(artifacts_dir, "stdout")
            if os.path.exists(stdout_file):
                with open(stdout_file, "r") as f:
                    output = f.read()

                # Create a basic result based on overall success
                result = ExecutionResult(
                    node_id="ansible-execution",
                    model_tag="playbook",
                    success=True,  # If we got here, basic execution succeeded
                    output=output,
                    error_message=None,
                    execution_time=0,
                )
                results.append(result)
            else:
                # No output found - assume failed
                result = ExecutionResult(
                    node_id="ansible-execution",
                    model_tag="playbook",
                    success=False,
                    error_message="No output artifacts found",
                )
                results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"Failed to parse execution results: {e}")
            return [
                ExecutionResult(
                    node_id="ansible-execution",
                    model_tag="playbook",
                    success=False,
                    error_message=f"Result parsing failed: {e}",
                )
            ]

    def cleanup_infrastructure(self, workload: WorkloadSpec) -> bool:
        """Cleanup infrastructure after execution.

        Args:
            workload: Workload specification

        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            self.logger.info("Cleaning up Ansible infrastructure")

            # Run custom cleanup handlers
            for cleanup_handler in self.cleanup_handlers:
                try:
                    cleanup_handler()
                except Exception as e:
                    self.logger.warning(f"Cleanup handler failed: {e}")

            # Clean up created files
            for file_path in self.created_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    self.logger.warning(f"Failed to remove {file_path}: {e}")

            self.created_files.clear()

            # Shutdown executor
            if self.executor:
                self.executor.shutdown(wait=True)
                self.executor = None

            # Optionally clean up playbook directory
            if os.path.exists(self.playbook_dir):
                try:
                    import shutil

                    shutil.rmtree(self.playbook_dir)
                except Exception as e:
                    self.logger.warning(f"Failed to remove playbook directory: {e}")

            self.logger.info("Ansible infrastructure cleanup completed")
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
