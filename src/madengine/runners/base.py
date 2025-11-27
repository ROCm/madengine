#!/usr/bin/env python3
"""
Base Distributed Runner for MADEngine

This module provides the abstract base class for distributed runners
that orchestrate workload execution across multiple nodes and clusters.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from madengine.core.console import Console


@dataclass
class NodeConfig:
    """Configuration for a single node in the distributed system."""

    hostname: str
    address: str
    port: int = 22
    username: str = "root"
    ssh_key_path: Optional[str] = None
    gpu_count: int = 1
    gpu_vendor: str = "AMD"
    labels: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate node configuration."""
        if not self.hostname or not self.address:
            raise ValueError("hostname and address are required")
        if self.gpu_vendor not in ["AMD", "NVIDIA", "INTEL"]:
            raise ValueError(f"Invalid gpu_vendor: {self.gpu_vendor}")


@dataclass
class WorkloadSpec:
    """Specification for a distributed workload."""

    model_tags: List[str]
    manifest_file: str
    timeout: int = 3600
    registry: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)
    node_selector: Dict[str, str] = field(default_factory=dict)
    parallelism: int = 1

    def __post_init__(self):
        """Validate workload specification."""
        if not self.model_tags:
            raise ValueError("model_tags cannot be empty")
        if not os.path.exists(self.manifest_file):
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_file}")


@dataclass
class ExecutionResult:
    """Result of a distributed execution."""

    node_id: str
    model_tag: str
    status: str  # SUCCESS, FAILURE, TIMEOUT, SKIPPED
    duration: float
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "node_id": self.node_id,
            "model_tag": self.model_tag,
            "status": self.status,
            "duration": self.duration,
            "performance_metrics": self.performance_metrics,
            "error_message": self.error_message,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass
class DistributedResult:
    """Overall result of a distributed execution."""

    total_nodes: int
    successful_executions: int
    failed_executions: int
    total_duration: float
    node_results: List[ExecutionResult] = field(default_factory=list)

    def add_result(self, result: ExecutionResult):
        """Add a node execution result."""
        self.node_results.append(result)
        if result.status == "SUCCESS":
            self.successful_executions += 1
        else:
            self.failed_executions += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_nodes": self.total_nodes,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "total_duration": self.total_duration,
            "node_results": [result.to_dict() for result in self.node_results],
        }


class BaseDistributedRunner(ABC):
    """Abstract base class for distributed runners."""

    def __init__(
        self,
        inventory_path: str,
        console: Optional[Console] = None,
        verbose: bool = False,
    ):
        """Initialize the distributed runner.

        Args:
            inventory_path: Path to inventory configuration file
            console: Console instance for output
            verbose: Enable verbose logging
        """
        self.inventory_path = inventory_path
        self.console = console or Console()
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load inventory configuration
        self.nodes = self._load_inventory(inventory_path)

        # Initialize result tracking
        self.results = DistributedResult(
            total_nodes=len(self.nodes),
            successful_executions=0,
            failed_executions=0,
            total_duration=0.0,
        )

    def _load_inventory(self, inventory_path: str) -> List[NodeConfig]:
        """Load inventory from configuration file.

        Args:
            inventory_path: Path to inventory file

        Returns:
            List of NodeConfig objects
        """
        if not os.path.exists(inventory_path):
            raise FileNotFoundError(f"Inventory file not found: {inventory_path}")

        with open(inventory_path, "r") as f:
            if inventory_path.endswith(".json"):
                inventory_data = json.load(f)
            elif inventory_path.endswith((".yml", ".yaml")):
                import yaml

                inventory_data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported inventory format: {inventory_path}")

        return self._parse_inventory(inventory_data)

    def _parse_inventory(self, inventory_data: Dict[str, Any]) -> List[NodeConfig]:
        """Parse inventory data into NodeConfig objects.

        Args:
            inventory_data: Raw inventory data

        Returns:
            List of NodeConfig objects
        """
        nodes = []

        # Support different inventory formats
        if "nodes" in inventory_data:
            # Simple format: {"nodes": [{"hostname": "...", ...}]}
            for node_data in inventory_data["nodes"]:
                nodes.append(NodeConfig(**node_data))
        elif "gpu_nodes" in inventory_data:
            # Ansible-style format: {"gpu_nodes": {...}}
            for node_data in inventory_data["gpu_nodes"]:
                nodes.append(NodeConfig(**node_data))
        else:
            # Auto-detect format
            for key, value in inventory_data.items():
                if isinstance(value, list):
                    for node_data in value:
                        if isinstance(node_data, dict) and "hostname" in node_data:
                            nodes.append(NodeConfig(**node_data))

        if not nodes:
            raise ValueError("No valid nodes found in inventory")

        return nodes

    def filter_nodes(self, node_selector: Dict[str, str]) -> List[NodeConfig]:
        """Filter nodes based on selector criteria.

        Args:
            node_selector: Key-value pairs for node selection

        Returns:
            Filtered list of nodes
        """
        if not node_selector:
            return self.nodes

        filtered_nodes = []
        for node in self.nodes:
            match = True
            for key, value in node_selector.items():
                if key == "gpu_vendor" and node.gpu_vendor != value:
                    match = False
                    break
                elif key in node.labels and node.labels[key] != value:
                    match = False
                    break

            if match:
                filtered_nodes.append(node)

        return filtered_nodes

    def validate_workload(self, workload: WorkloadSpec) -> bool:
        """Validate workload specification.

        Args:
            workload: Workload specification to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check manifest file exists
            if not os.path.exists(workload.manifest_file):
                self.logger.error(f"Manifest file not found: {workload.manifest_file}")
                return False

            # Load and validate manifest
            with open(workload.manifest_file, "r") as f:
                manifest = json.load(f)

            if "built_images" not in manifest:
                self.logger.error("Invalid manifest: missing built_images")
                return False

            # Filter nodes based on selector
            target_nodes = self.filter_nodes(workload.node_selector)
            if not target_nodes:
                self.logger.error("No nodes match the selector criteria")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Workload validation failed: {e}")
            return False

    def prepare_execution_context(self, workload: WorkloadSpec) -> Dict[str, Any]:
        """Prepare execution context for distributed execution.

        Args:
            workload: Workload specification

        Returns:
            Execution context dictionary
        """
        # Load manifest
        with open(workload.manifest_file, "r") as f:
            manifest = json.load(f)

        # Prepare context
        context = {
            "manifest": manifest,
            "registry": workload.registry or manifest.get("registry", ""),
            "timeout": workload.timeout,
            "additional_context": workload.additional_context,
            "model_tags": workload.model_tags,
            "parallelism": workload.parallelism,
        }

        return context

    @abstractmethod
    def setup_infrastructure(self, workload: WorkloadSpec) -> bool:
        """Setup infrastructure for distributed execution.

        Args:
            workload: Workload specification

        Returns:
            True if setup successful, False otherwise
        """
        pass

    @abstractmethod
    def execute_workload(self, workload: WorkloadSpec) -> DistributedResult:
        """Execute workload across distributed nodes.

        Args:
            workload: Workload specification

        Returns:
            Distributed execution result
        """
        pass

    @abstractmethod
    def cleanup_infrastructure(self, workload: WorkloadSpec) -> bool:
        """Cleanup infrastructure after execution.

        Args:
            workload: Workload specification

        Returns:
            True if cleanup successful, False otherwise
        """
        pass

    def run(self, workload: WorkloadSpec) -> DistributedResult:
        """Run the complete distributed execution workflow.

        Args:
            workload: Workload specification

        Returns:
            Distributed execution result
        """
        import time

        start_time = time.time()

        try:
            # Validate workload
            if not self.validate_workload(workload):
                raise ValueError("Invalid workload specification")

            # Setup infrastructure
            if not self.setup_infrastructure(workload):
                raise RuntimeError("Failed to setup infrastructure")

            # Execute workload
            result = self.execute_workload(workload)

            # Cleanup infrastructure
            self.cleanup_infrastructure(workload)

            # Update total duration
            result.total_duration = time.time() - start_time

            return result

        except Exception as e:
            self.logger.error(f"Distributed execution failed: {e}")
            # Ensure cleanup even on failure
            try:
                self.cleanup_infrastructure(workload)
            except Exception as cleanup_error:
                self.logger.error(f"Cleanup failed: {cleanup_error}")

            # Return failure result
            self.results.total_duration = time.time() - start_time
            return self.results

    def generate_report(self, output_file: str = "distributed_report.json") -> str:
        """Generate execution report.

        Args:
            output_file: Output file path

        Returns:
            Path to generated report
        """
        report_data = self.results.to_dict()

        with open(output_file, "w") as f:
            json.dump(report_data, f, indent=2)

        return output_file
