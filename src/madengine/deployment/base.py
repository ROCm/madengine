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

from jinja2 import Environment, FileSystemLoader
from rich.console import Console


# Regex for parsing "performance: <value> <metric>" log lines.
# Value: optional sign, integer/decimal, scientific notation (e or E).
# Separator: optional unit suffix (/[a-zA-Z]+) and/or comma, in any order —
#   "123/s, metric", "123, metric", "123,/s metric", "123, /s metric".
# Metric: any non-whitespace, non-comma token (e.g. loss, latency_ms,
#   samples/sec, tokens/sec).
PERFORMANCE_LOG_PATTERN = (
    r"performance:\s+"
    r"([+\-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+\-]?[0-9]+)?)"
    r"(?:,?\s*/[a-zA-Z]+\s*,?|,)?\s+"
    r"([^\s,]+)"
)


def create_jinja_env(template_dir: Path) -> Environment:
    """Create a Jinja2 Environment with common filters for deployment templates.

    Args:
        template_dir: Path to the template directory (e.g. deployment/templates/slurm).

    Returns:
        Jinja2 Environment with dirname and basename filters registered.
    """
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    env.filters["dirname"] = lambda path: str(Path(path).parent)
    env.filters["basename"] = lambda path: str(Path(path).name)
    return env


class DeploymentStatus(Enum):
    """Deployment status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


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
        result = None
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

            # Step 5: Collect Results (always collect, even on failure to record failed runs)
            if result.deployment_id:
                try:
                    metrics = self.collect_results(result.deployment_id)
                    result.metrics = metrics
                except Exception as e:
                    self.console.print(f"[yellow]Warning: Could not collect results for {result.deployment_id}: {e}[/yellow]")
                    # Ensure empty metrics dict exists even if collection fails
                    result.metrics = {"successful_runs": [], "failed_runs": []}

            return result

        except KeyboardInterrupt:
            if result is not None and getattr(result, "deployment_id", None):
                self.cleanup(result.deployment_id)
                self.console.print("\n[yellow]Cancelled deployment and cleaned up resources.[/yellow]")
            raise
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

            if status.status in [DeploymentStatus.SUCCESS, DeploymentStatus.FAILED, DeploymentStatus.UNKNOWN]:
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

    # -------------------------------------------------------------------------
    # Shared multi-node metrics aggregation (used by SLURM and Kubernetes)
    # -------------------------------------------------------------------------

    def _parse_performance_from_log(
        self, log_content: str, model_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse node-local performance from log content.

        Expected format (from training scripts):
            performance: <value>[<unit>][,] <metric>
            node_id: <id>
            local_gpus: <num_gpus>
            test_duration: <value>s

        <value> may include an optional sign, decimal point, and scientific notation
        (e.g. 1.23e+4 or 1.23E+4).  A unit suffix such as /s and/or a comma separator
        between the value and metric name are accepted in either order, e.g.:
            performance: 14164/s, samples_per_second
            performance: 14164, /s samples_per_second

        Args:
            log_content: Raw log text (e.g. node stdout)
            model_name: Model name for the record

        Returns:
            Dict with performance, metric, node_id, local_gpus, duration, gpu_architecture, etc.
            None if no performance line found.
        """
        import re

        match = re.search(PERFORMANCE_LOG_PATTERN, log_content)
        if not match:
            return None

        value = float(match.group(1))
        metric = match.group(2).rstrip(',')

        node_id_pattern = r"node_id:\s*(\d+)"
        node_match = re.search(node_id_pattern, log_content)
        node_id = int(node_match.group(1)) if node_match else None

        local_gpus_pattern = r"local_gpus:\s*(\d+)"
        gpus_match = re.search(local_gpus_pattern, log_content)
        local_gpus = int(gpus_match.group(1)) if gpus_match else 1

        duration_pattern = r"test_duration:\s*([\d.]+)s"
        duration_match = re.search(duration_pattern, log_content)
        if duration_match:
            duration = f"{duration_match.group(1)}s"
        else:
            # Also match container output: "Test Duration: 12.34 seconds"
            alt_duration_pattern = r"Test Duration:\s*([\d.]+)\s*seconds"
            alt_match = re.search(alt_duration_pattern, log_content, re.IGNORECASE)
            duration = f"{alt_match.group(1)}s" if alt_match else "N/A"

        gpu_arch_pattern = r"(?:🔹\s*)?Name\s*:\s*(gfx\w+)"
        gpu_arch_match = re.search(gpu_arch_pattern, log_content)
        gpu_arch = gpu_arch_match.group(1) if gpu_arch_match else "N/A"

        return {
            "model": model_name,
            "performance": value,
            "metric": metric,
            "node_id": node_id,
            "local_gpus": local_gpus,
            "duration": duration,
            "test_duration": duration,
            "gpu_architecture": gpu_arch,
            "data_name": "N/A",
            "data_provider": "N/A",
        }

    def _determine_aggregation_method(self, metric_name: str) -> str:
        """
        Determine how to aggregate a metric based on its name/type.

        Returns:
            "sum", "average", or "max"
        """
        metric_lower = metric_name.lower()
        if any(
            keyword in metric_lower
            for keyword in [
                "throughput",
                "samples_per_second",
                "tokens_per_second",
                "images_per_second",
                "requests_per_second",
                "qps",
                "bandwidth",
                "ops_per_second",
                "samples/sec",
                "tokens/sec",
            ]
        ):
            return "sum"
        if any(
            keyword in metric_lower
            for keyword in [
                "latency",
                "time",
                "duration",
                "milliseconds",
                "seconds",
                "ttft",
                "tpot",
                "response_time",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "loss",
            ]
        ):
            return "average"
        if any(
            keyword in metric_lower
            for keyword in ["memory", "bytes", "ram", "vram", "gb", "mb"]
        ):
            return "max"
        self.console.print(
            f"[yellow]⚠ Unknown metric type '{metric_name}', using sum aggregation[/yellow]"
        )
        return "sum"

    def _aggregate_node_metrics(
        self,
        per_node_metrics: List[Dict[str, Any]],
        nnodes: int,
        launcher_type: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Aggregate per-node metrics into a single job-level record.

        Throughput metrics are summed; latency/accuracy averaged; memory max.
        Used by both SLURM and Kubernetes for multi-node result aggregation.
        """
        import statistics

        if not per_node_metrics:
            return None

        first_metric = per_node_metrics[0]
        metric_name = first_metric["metric"]
        aggregation_method = self._determine_aggregation_method(metric_name)

        if aggregation_method == "sum":
            aggregated_value = sum(m["performance"] for m in per_node_metrics)
            method_desc = "sum_across_nodes"
        elif aggregation_method == "average":
            aggregated_value = statistics.mean(m["performance"] for m in per_node_metrics)
            method_desc = "average_across_nodes"
        elif aggregation_method == "max":
            aggregated_value = max(m["performance"] for m in per_node_metrics)
            method_desc = "max_across_nodes"
        else:
            aggregated_value = sum(m["performance"] for m in per_node_metrics)
            method_desc = "sum_across_nodes (default)"

        perfs = [m["performance"] for m in per_node_metrics]
        if len(perfs) > 1:
            statistics_dict = {
                "mean": statistics.mean(perfs),
                "std_dev": statistics.stdev(perfs),
                "min": min(perfs),
                "max": max(perfs),
                "coefficient_variation": (
                    statistics.stdev(perfs) / statistics.mean(perfs)
                    if statistics.mean(perfs) > 0
                    else 0
                ),
            }
        else:
            statistics_dict = {
                "mean": perfs[0],
                "std_dev": 0,
                "min": perfs[0],
                "max": perfs[0],
                "coefficient_variation": 0,
            }

        gpu_arch = "N/A"
        for m in per_node_metrics:
            if m.get("gpu_architecture") and m["gpu_architecture"] != "N/A":
                gpu_arch = m["gpu_architecture"]
                break

        durations = [
            m.get("duration", m.get("test_duration", "N/A"))
            for m in per_node_metrics
            if m.get("duration", "N/A") != "N/A" or m.get("test_duration", "N/A") != "N/A"
        ]
        if durations:
            duration_values = []
            for d in durations:
                if isinstance(d, str) and d.endswith("s"):
                    try:
                        duration_values.append(float(d[:-1]))
                    except ValueError:
                        pass
            duration = f"{max(duration_values):.2f}s" if duration_values else "N/A"
        else:
            duration = "N/A"

        total_gpus = sum(m.get("local_gpus", 1) for m in per_node_metrics)
        gpus_per_node = per_node_metrics[0].get("local_gpus", 1) if per_node_metrics else 1

        aggregated_record = {
            "model": first_metric["model"],
            "n_gpus": total_gpus,
            "nnodes": nnodes,
            "gpus_per_node": gpus_per_node,
            "performance": aggregated_value,
            "metric": metric_name,
            "status": "SUCCESS",
            "topology": f"{nnodes}N×{gpus_per_node}G",
            "launcher": launcher_type or "N/A",
            "deployment_type": getattr(self, "DEPLOYMENT_TYPE", "base"),
            "gpu_architecture": gpu_arch,
            "test_duration": duration,
            "data_name": first_metric.get("data_name", "N/A"),
            "data_provider": first_metric.get("data_provider", "N/A"),
            "aggregation_method": method_desc,
            "nodes_contributing": len(per_node_metrics),
            "per_node_mean": statistics_dict["mean"],
            "per_node_std_dev": statistics_dict["std_dev"],
            "per_node_cv": statistics_dict["coefficient_variation"],
        }
        return aggregated_record

    def _ensure_perf_csv_exists(self) -> None:
        """Ensure perf.csv exists with standard header (for appending aggregated rows)."""
        perf_csv_path = Path("perf.csv")
        if perf_csv_path.exists():
            return
        standard_header = (
            "model,n_gpus,nnodes,gpus_per_node,training_precision,pipeline,args,tags,"
            "docker_file,base_docker,docker_sha,docker_image,git_commit,machine_name,"
            "deployment_type,launcher,gpu_architecture,performance,metric,relative_change,"
            "status,build_duration,test_duration,dataname,data_provider_type,data_size,"
            "data_download_duration,build_number,additional_docker_run_options"
        )
        perf_csv_path.write_text(standard_header + "\n", encoding="utf-8")

    def _write_to_perf_csv(self, perf_data: Dict[str, Any]) -> None:
        """
        Append one row to perf.csv. Uses existing header if file exists for column order.
        """
        import csv

        perf_csv_path = Path("perf.csv")
        default_headers = [
            "model",
            "n_gpus",
            "nnodes",
            "gpus_per_node",
            "training_precision",
            "pipeline",
            "args",
            "tags",
            "docker_file",
            "base_docker",
            "docker_sha",
            "docker_image",
            "git_commit",
            "machine_name",
            "deployment_type",
            "launcher",
            "gpu_architecture",
            "performance",
            "metric",
            "relative_change",
            "status",
            "build_duration",
            "test_duration",
            "dataname",
            "data_provider_type",
            "data_size",
            "data_download_duration",
            "build_number",
            "additional_docker_run_options",
        ]
        file_exists = perf_csv_path.exists()
        existing_header = None
        if file_exists:
            with open(
                perf_csv_path, "r", newline="", encoding="utf-8", errors="replace"
            ) as rf:
                reader = csv.reader(rf)
                existing_header = next(reader, None)
        headers = existing_header if existing_header else default_headers
        if file_exists and existing_header:
            row_by_name = {k: perf_data.get(k, "") for k in headers}
            row_to_write = [str(row_by_name.get(h, "")) for h in headers]
        else:
            row_to_write = perf_data

        with open(perf_csv_path, "a", newline="", encoding="utf-8") as f:
            if file_exists and existing_header:
                # File already has a header — write a plain row using csv.writer
                # to preserve the exact column order captured in row_to_write
                csv.writer(f).writerow(row_to_write)
            else:
                # New file — write header then the data row via DictWriter
                writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
                writer.writeheader()
                writer.writerow(row_to_write)

