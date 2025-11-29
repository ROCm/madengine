#!/usr/bin/env python3
"""
Base classes for deployment layer.

Defines abstract base class for all deployment targets (SLURM, Kubernetes).
Implements Template Method pattern for deployment workflow.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console


class DeploymentStatus(Enum):
    """Deployment status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DeploymentConfig:
    """Configuration for distributed deployment."""

    target: str  # "slurm", "k8s" (NOT "local" - that uses container_runner)
    manifest_file: str
    additional_context: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 3600
    monitor: bool = True
    cleanup_on_failure: bool = True


@dataclass
class DeploymentResult:
    """Result of deployment operation."""

    status: DeploymentStatus
    deployment_id: str
    message: str
    metrics: Optional[Dict[str, Any]] = None
    logs_path: Optional[str] = None
    artifacts: Optional[List[str]] = None

    @property
    def is_success(self) -> bool:
        """Check if deployment succeeded."""
        return self.status == DeploymentStatus.SUCCESS

    @property
    def is_failed(self) -> bool:
        """Check if deployment failed."""
        return self.status == DeploymentStatus.FAILED


class BaseDeployment(ABC):
    """
    Abstract base class for all deployment targets.

    Implements Template Method pattern for deployment workflow.
    Subclasses implement specific deployment logic for SLURM, Kubernetes, etc.

    Workflow:
    1. Validate environment and configuration
    2. Prepare deployment artifacts (scripts, manifests)
    3. Deploy to target infrastructure
    4. Monitor until completion (if enabled)
    5. Collect results and metrics
    6. Cleanup (if needed)
    """

    DEPLOYMENT_TYPE: str = "base"
    REQUIRED_TOOLS: List[str] = []  # e.g., ["sbatch", "squeue"] for SLURM

    def __init__(self, config: DeploymentConfig):
        """
        Initialize deployment.

        Args:
            config: Deployment configuration
        """
        self.config = config
        self.manifest = self._load_manifest(config.manifest_file)
        self.console = Console()

    def _load_manifest(self, manifest_file: str) -> Dict:
        """
        Load and validate build manifest.

        Args:
            manifest_file: Path to build_manifest.json

        Returns:
            Loaded manifest dict

        Raises:
            FileNotFoundError: If manifest doesn't exist
            ValueError: If manifest is invalid
        """
        manifest_path = Path(manifest_file)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Validate required fields
        required = ["built_images", "built_models", "context"]
        missing = [f for f in required if f not in manifest]
        if missing:
            raise ValueError(f"Invalid manifest, missing: {missing}")

        return manifest

    # Template Method - defines workflow
    def execute(self) -> DeploymentResult:
        """
        Execute full deployment workflow (Template Method).

        This method orchestrates the entire deployment process by calling
        abstract methods that subclasses must implement.

        Returns:
            DeploymentResult with status and metrics
        """
        try:
            # Step 1: Validate
            self.console.print(
                f"[blue]Validating {self.DEPLOYMENT_TYPE} deployment...[/blue]"
            )
            if not self.validate():
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    deployment_id="",
                    message=f"{self.DEPLOYMENT_TYPE} validation failed",
                )

            # Step 2: Prepare
            self.console.print("[blue]Preparing deployment artifacts...[/blue]")
            if not self.prepare():
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    deployment_id="",
                    message="Preparation failed",
                )

            # Step 3: Deploy
            self.console.print(f"[blue]Deploying to {self.DEPLOYMENT_TYPE}...[/blue]")
            result = self.deploy()

            if not result.is_success:
                if self.config.cleanup_on_failure:
                    self.cleanup(result.deployment_id)
                return result

            # Step 4: Monitor (optional)
            if self.config.monitor:
                result = self._monitor_until_complete(result.deployment_id)

            # Step 5: Collect Results
            if result.is_success:
                metrics = self.collect_results(result.deployment_id)
                result.metrics = metrics

            return result

        except Exception as e:
            self.console.print(f"[red]Deployment error: {e}[/red]")
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id="",
                message=f"Exception: {str(e)}",
            )

    def _monitor_until_complete(self, deployment_id: str) -> DeploymentResult:
        """
        Monitor deployment until completion.

        Args:
            deployment_id: Deployment ID to monitor

        Returns:
            Final deployment status
        """
        self.console.print("[blue]Monitoring deployment...[/blue]")

        while True:
            status = self.monitor(deployment_id)

            if status.status in [DeploymentStatus.SUCCESS, DeploymentStatus.FAILED]:
                return status

            # Still running, wait and check again
            self.console.print(
                f"  Status: {status.status.value} - {status.message}"
            )
            time.sleep(30)  # Check every 30 seconds

    # Abstract methods to be implemented by subclasses

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate deployment environment and configuration.

        Should check:
        - Required tools are available (sbatch, kubectl, etc.)
        - Credentials/access are valid
        - Configuration parameters are correct
        - Connectivity to target system

        Returns:
            True if validation passes, False otherwise
        """
        pass

    @abstractmethod
    def prepare(self) -> bool:
        """
        Prepare deployment artifacts.

        Should generate:
        - Deployment scripts (sbatch scripts, K8s Job manifests)
        - Configuration files
        - Environment setup

        Returns:
            True if preparation succeeds, False otherwise
        """
        pass

    @abstractmethod
    def deploy(self) -> DeploymentResult:
        """
        Execute deployment to target infrastructure.

        Should:
        - Submit job to scheduler (sbatch, kubectl apply)
        - Return immediately with deployment_id
        - Not wait for completion (use monitor() for that)

        Returns:
            DeploymentResult with status and deployment_id
        """
        pass

    @abstractmethod
    def monitor(self, deployment_id: str) -> DeploymentResult:
        """
        Check deployment status.

        Should query:
        - SLURM job status (squeue)
        - K8s Job status (kubectl get job)
        - etc.

        Args:
            deployment_id: ID returned from deploy()

        Returns:
            Current deployment status
        """
        pass

    @abstractmethod
    def collect_results(self, deployment_id: str) -> Dict[str, Any]:
        """
        Collect results and metrics from completed deployment.

        Should gather:
        - Performance metrics
        - Log files
        - Output artifacts
        - Error information (if any)

        Args:
            deployment_id: ID of completed deployment

        Returns:
            Dict with metrics and results
        """
        pass

    @abstractmethod
    def cleanup(self, deployment_id: str) -> bool:
        """
        Cleanup deployment resources.

        Should:
        - Cancel running jobs
        - Delete temporary files
        - Release resources

        Args:
            deployment_id: ID of deployment to clean up

        Returns:
            True if cleanup succeeds, False otherwise
        """
        pass

