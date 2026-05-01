"""
Kubernetes results collection and performance reporting mixin.

Handles collecting pod logs, PVC artifacts, parsing performance metrics,
aggregating multi-node results, and writing to perf.csv / perf_super.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import normalize_launcher
from madengine.utils.path_utils import scripts_base_dir_from
from madengine.utils.run_details import flatten_tags_in_place, get_build_number, get_pipeline

try:
    from kubernetes.client.rest import ApiException

    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

try:
    from madengine.reporting.update_perf_csv import update_perf_csv
    from madengine.reporting.update_perf_super import update_perf_super_json, update_perf_super_csv
    REPORTING_AVAILABLE = True
except ImportError:
    REPORTING_AVAILABLE = False


def _pod_job_name_label_selector(deployment_id: str) -> str:
    """Selector for the ``job-name`` pod label; value must be a valid <=63-char label value."""
    from .k8s_names import sanitize_k8s_label_value
    return f"job-name={sanitize_k8s_label_value(deployment_id)}"


class KubernetesResultsMixin:
    """Results collection and performance reporting for Kubernetes deployments."""

    # Standard perf.csv header (must match container_runner.ensure_perf_csv_exists)
    _PERF_CSV_HEADER = (
        "model,n_gpus,nnodes,gpus_per_node,training_precision,pipeline,args,tags,"
        "docker_file,base_docker,docker_sha,docker_image,git_commit,machine_name,"
        "deployment_type,launcher,gpu_architecture,performance,metric,relative_change,"
        "status,build_duration,test_duration,dataname,data_provider_type,data_size,"
        "data_download_duration,build_number,additional_docker_run_options"
    )

    def collect_results(self, deployment_id: str) -> Dict[str, Any]:
        """
        Enhanced results collection from K8s pods following vLLM multi-node best practices.

        For Data Parallel deployments (vLLM, SGLang):
        - Each pod runs an independent replica
        - Only pod-0 reports metrics to avoid duplicates
        - Total throughput = pod-0 throughput x num_replicas

        Collects:
        1. Pod logs (``k8s_results/<job>/<pod>/pod.log``)
        2. PVC mirror per pod (``.../<pod>/pvc/``), mapped from ``/results/<subdir>/``
        3. File artifacts via kubectl cp when pods are still running (keep-alive path)

        Returns:
            Dict with logs, artifacts, and performance results
        """
        results = {
            "job_name": deployment_id,
            "namespace": self.namespace,
            "logs": [],
            "artifacts": [],
            "successful_runs": [],
            "failed_runs": [],
        }

        # Create results directory for this deployment
        results_dir = Path(f"./k8s_results/{deployment_id}")
        results_dir.mkdir(parents=True, exist_ok=True)

        self.console.print(f"[cyan]📦 Collecting results from K8s job: {deployment_id}[/cyan]")

        try:
            # Get pods for this job
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=_pod_job_name_label_selector(deployment_id),
            )

            # Get model info and build info from manifest
            model_keys = list(self.manifest["built_models"].keys())
            if model_keys:
                model_key = model_keys[0]
                model_info = self.manifest["built_models"][model_key]
            else:
                model_info = {}

            # Get build info from built_images
            image_keys = list(self.manifest.get("built_images", {}).keys())
            if image_keys:
                image_key = image_keys[0]
                build_info = self.manifest["built_images"][image_key]
            else:
                build_info = {}

            # Check if this is a multi-node distributed job
            deployment_config = self.manifest.get("deployment_config", {})
            distributed_config = deployment_config.get("distributed", {})
            is_distributed = distributed_config.get("enabled", False)
            nnodes = distributed_config.get("nnodes", 1)
            is_multinode = is_distributed and nnodes > 1

            # Determine launcher_type the same way as _prepare_template_context does
            # (deployment_config doesn't store launcher_type directly)
            launcher_config = self.config.additional_context.get("launcher", {})
            launcher_type = (
                launcher_config.get("type")
                if launcher_config.get("type") is not None
                else distributed_config.get("launcher")
            )

            # Normalize launcher based on deployment type and validity
            launcher_type = normalize_launcher(launcher_type, "kubernetes")

            is_ray_launcher = launcher_type in ["vllm", "sglang"]

            # Sort pods by name to ensure consistent ordering (pod-0 is master)
            sorted_pods = sorted(pods.items, key=lambda p: p.metadata.name)

            # ========================================================================
            # Per-Node Collection Strategy
            # Collect logs and artifacts from ALL nodes
            # Parse performance from ALL nodes (each reports node-local metrics)
            # Aggregate metrics based on type (sum for throughput, etc.)
            # ========================================================================

            per_node_metrics = []  # Store performance from each node
            results["nodes"] = []  # Store per-node details for display

            # Special handling for Ray-based launchers (vLLM, SGLang)
            # These report per-replica metrics, need scaling
            if is_multinode and is_ray_launcher:
                self.console.print(
                    f"[cyan]Multi-node Ray deployment: {nnodes} nodes (Data Parallel mode)[/cyan]"
                )

            # Collect from ALL pods
            for pod_index, pod in enumerate(sorted_pods):
                pod_name = pod.metadata.name
                pod_dir = results_dir / pod_name
                pod_dir.mkdir(exist_ok=True)

                # Extract node rank from pod name (e.g., madengine-dummy-torchrun-0 -> 0)
                try:
                    node_rank = int(pod_name.rsplit('-', 1)[-1])
                except (ValueError, IndexError):
                    node_rank = pod_index

                self.console.print(f"[dim]  Collecting from pod: {pod_name} (node-{node_rank})[/dim]")

                try:
                    # 1. Collect pod logs
                    log = self.core_v1.read_namespaced_pod_log(
                        name=pod_name, namespace=self.namespace
                    )
                    log_file = pod_dir / "pod.log"
                    log_file.write_text(log)
                    results["logs"].append({
                        "pod": pod_name,
                        "log": log,
                        "file": str(log_file)
                    })

                    # 2. Parse NODE-LOCAL performance from log
                    perf_data = self._parse_performance_from_log(
                        log, model_info.get("name", "")
                    )

                    # Pod phase/exit can lag right after Job success; poll until terminal or timeout
                    pod = self._refresh_pod_until_terminal_phase(pod_name)
                    pod_status = pod.status.phase if pod else "Unknown"
                    pod_exit_code = (
                        self._primary_workload_container_exit_code(pod) if pod else -1
                    )

                    # Store per-node info for display table
                    node_info = {
                        "node_id": node_rank,
                        "pod_name": pod_name,
                        "status": "SUCCESS" if pod_status == "Succeeded" and pod_exit_code == 0 else "FAILED",
                        "exit_code": pod_exit_code,
                        "performance": perf_data.get("performance") if perf_data else None,
                        "metric": perf_data.get("metric") if perf_data else None,
                        "duration": perf_data.get("duration") if perf_data else None,
                        "log_file": str(log_file)
                    }
                    results["nodes"].append(node_info)

                    if perf_data:
                        # For Ray launchers, this is per-replica metric
                        if is_multinode and is_ray_launcher:
                            perf_data["is_per_replica"] = True
                        per_node_metrics.append(perf_data)
                        self.console.print(
                            f"[green]  ✓ Parsed performance: {perf_data['performance']:.2f} "
                            f"{perf_data['metric']} (node-{node_rank})[/green]"
                        )
                    else:
                        self.console.print(
                            f"[dim]  No performance metric found in node-{node_rank} log[/dim]"
                        )

                except ApiException as e:
                    self.console.print(
                        f"[red]✗ Failed to get logs for pod {pod_name}: {e.reason}[/red]"
                    )
                    results["nodes"].append({
                        "node_id": node_rank,
                        "pod_name": pod_name,
                        "status": "FAILED",
                        "exit_code": -1,
                        "performance": None,
                        "metric": None,
                        "error": f"Failed to get logs: {e.reason}"
                    })
                except Exception as e:
                    self.console.print(
                        f"[red]✗ Error collecting from pod {pod_name}: {e}[/red]"
                    )
                    results["nodes"].append({
                        "node_id": node_rank,
                        "pod_name": pod_name,
                        "status": "FAILED",
                        "exit_code": -1,
                        "performance": None,
                        "metric": None,
                        "error": str(e)
                    })

            self.console.print(
                f"[green]✓ Collected logs from {len(results['logs'])} pods[/green]"
            )

            # Collect artifacts from PVC before deciding success/failure (needed for multiple_results fallback)
            k8s_pod_names = [p.metadata.name for p in sorted_pods]
            self._collect_from_pvc(deployment_id, results_dir, results, pod_names=k8s_pod_names)

            # ========================================================================
            # Aggregate per-node metrics
            # ========================================================================
            if per_node_metrics:
                # Special handling for Ray launchers - multiply by nnodes
                if is_multinode and is_ray_launcher:
                    original_perf = per_node_metrics[0]["performance"]
                    aggregated_perf = original_perf * nnodes
                    self.console.print(
                        f"[green]  Per-replica: {original_perf:.1f} req/s[/green]"
                    )
                    self.console.print(
                        f"[green]  Total capacity: {aggregated_perf:.1f} req/s ({nnodes} nodes)[/green]"
                    )

                    # Create aggregated record manually for Ray
                    aggregated_record = {
                        "model": per_node_metrics[0]["model"],
                        "performance": aggregated_perf,
                        "metric": per_node_metrics[0]["metric"],
                        "status": "SUCCESS",
                        "topology": f"{nnodes}N×{per_node_metrics[0].get('local_gpus', 1)}G",
                        "nnodes": nnodes,
                        "launcher": launcher_type or "N/A",
                        "deployment_type": "kubernetes",
                        "gpu_architecture": per_node_metrics[0].get("gpu_architecture", "N/A"),
                        "duration": per_node_metrics[0].get("duration", "N/A"),
                        "data_name": per_node_metrics[0].get("data_name", "N/A"),
                        "data_provider": per_node_metrics[0].get("data_provider", "N/A"),
                        "aggregation_method": "scaled_by_nnodes",
                        "nodes_contributing": nnodes
                    }
                else:
                    # Use new aggregation logic for other launchers
                    aggregated_record = self._aggregate_node_metrics(
                        per_node_metrics,
                        nnodes,
                        launcher_type
                    )

                if aggregated_record:
                    # Full reporting pipeline: perf_entry at project root, then update_* (same as local/SLURM)
                    self._ensure_perf_csv_exists()
                    run_details_dict = self._build_perf_entry_from_aggregated(
                        aggregated_record, model_info, build_info, deployment_id
                    )
                    perf_entry_path = Path("perf_entry.json")
                    with open(perf_entry_path, "w", encoding="utf-8") as f:
                        json.dump(run_details_dict, f, indent=2)
                    if run_details_dict.get("status") == "SUCCESS":
                        update_perf_csv(perf_csv="perf.csv", single_result=str(perf_entry_path))
                    else:
                        update_perf_csv(perf_csv="perf.csv", exception_result=str(perf_entry_path))
                    scripts_path = model_info.get("scripts", "")
                    scripts_base_dir = scripts_base_dir_from(scripts_path)
                    try:
                        if run_details_dict.get("status") == "SUCCESS":
                            num_entries = update_perf_super_json(
                                single_result=str(perf_entry_path),
                                perf_super_json="perf_super.json",
                                scripts_base_dir=scripts_base_dir,
                            )
                        else:
                            num_entries = update_perf_super_json(
                                exception_result=str(perf_entry_path),
                                perf_super_json="perf_super.json",
                                scripts_base_dir=scripts_base_dir,
                            )
                        update_perf_super_csv(
                            perf_super_json="perf_super.json",
                            perf_super_csv="perf_super.csv",
                            num_entries=num_entries,
                        )
                    except Exception as e:
                        self.console.print(f"[yellow]⚠ Could not update perf_super: {e}[/yellow]")
                    results["successful_runs"].append({
                        "model": model_info.get("name"),
                        "perf_data": aggregated_record,
                        "nodes": results["nodes"],
                        "per_node_metrics": per_node_metrics
                    })
                    self.console.print(
                        f"[green]✓ Aggregated performance from {len(per_node_metrics)} nodes[/green]"
                    )
                    self.console.print(
                        f"[green]✓ Updated perf_entry.json, perf.csv, perf_super.* (Docker-compatible)[/green]"
                    )
            else:
                # No performance from log: try multiple_results CSV (same contract as local Docker)
                # Resolve single CSV path (one pod) or merged CSV path (multi-pod with sum/avg rules)
                resolved_csv_path = self._resolve_multiple_results_csv(
                    results_dir, results, model_info
                )
                if resolved_csv_path and REPORTING_AVAILABLE:
                    # Docker-compatible flow: produce perf.csv, perf_entry.*, perf_super.*
                    gpu_arch = "N/A"
                    if results.get("logs"):
                        log_content = results["logs"][0].get("log", "")
                        m = re.search(r"(?:🔹\s*)?Name\s*:\s*(gfx\w+)", log_content)
                        if m:
                            gpu_arch = m.group(1)
                    self._ensure_perf_csv_exists()
                    common_info = self._build_common_info_dict(
                        model_info, build_info, deployment_id, gpu_arch
                    )
                    common_info_path = Path("common_info.json")
                    with open(common_info_path, "w", encoding="utf-8") as f:
                        json.dump(common_info, f, indent=2)
                    update_perf_csv(
                        perf_csv="perf.csv",
                        multiple_results=str(resolved_csv_path),
                        common_info=str(common_info_path),
                        model_name=model_info.get("name", ""),
                    )
                    scripts_path = model_info.get("scripts", "")
                    scripts_base_dir = scripts_base_dir_from(scripts_path)
                    num_entries = update_perf_super_json(
                        perf_super_json="perf_super.json",
                        multiple_results=str(resolved_csv_path),
                        common_info=str(common_info_path),
                        model_name=model_info.get("name", ""),
                        scripts_base_dir=scripts_base_dir,
                    )
                    update_perf_super_csv(
                        perf_super_json="perf_super.json",
                        perf_super_csv="perf_super.csv",
                        num_entries=num_entries,
                    )
                    # Build successful_runs for display (one entry per CSV row)
                    import csv as _csv
                    model_name = model_info.get("name", "")
                    with open(resolved_csv_path, "r", encoding="utf-8", errors="ignore") as f:
                        reader = _csv.DictReader(f)
                        for row in reader:
                            row = {k.strip(): v for k, v in row.items() if k}
                            if row.get("performance") and row.get("metric"):
                                display_model = f"{model_name}_{row.get('model', '')}"
                                record = self._create_multiple_result_row_record(
                                    model_info, build_info, deployment_id,
                                    {
                                        "model": display_model,
                                        "performance": row.get("performance"),
                                        "metric": row.get("metric", ""),
                                        "gpu_architecture": gpu_arch,
                                        "duration": row.get("test_duration", "N/A"),
                                    },
                                )
                                if record:
                                    results["successful_runs"].append({
                                        "model": display_model,
                                        "perf_data": record,
                                        "nodes": [],
                                        "per_node_metrics": [{"model": display_model, "performance": row.get("performance"), "metric": row.get("metric", "")}],
                                    })
                    self.console.print(
                        f"[green]✓ Updated perf.csv, perf_entry.*, perf_super.* (Docker-compatible)[/green]"
                    )
                elif resolved_csv_path and not REPORTING_AVAILABLE:
                    # Fallback when reporting module not available: legacy row-by-row write
                    fallback_metrics = self._parse_multiple_results_from_artifacts(
                        results_dir, results, model_info, build_info
                    )
                    if fallback_metrics:
                        for item in fallback_metrics:
                            record = self._create_multiple_result_row_record(
                                model_info, build_info, deployment_id, item
                            )
                            if record:
                                self._write_to_perf_csv(record)
                                results["successful_runs"].append({
                                    "model": item["model"],
                                    "perf_data": record,
                                    "nodes": [],
                                    "per_node_metrics": [item],
                                })
                        self.console.print(
                            f"[green]✓ Wrote {len(fallback_metrics)} row(s) from multiple_results to perf.csv[/green]"
                        )
                if not resolved_csv_path:
                    # No multiple_results CSV found: record failure
                    error_msg = "No performance metrics found from any node"
                    failure_record = self._create_failure_record(
                        model_info, build_info, deployment_id, error_msg
                    )
                    self._write_to_perf_csv(failure_record)
                    results["failed_runs"].append({
                        "model": model_info.get("name", "Unknown"),
                        "error": error_msg,
                        "nodes": results["nodes"]
                    })
                    self.console.print(
                        f"[yellow]⚠ No performance metrics found, recorded as FAILED[/yellow]"
                    )
                elif resolved_csv_path and not REPORTING_AVAILABLE and not results.get("successful_runs"):
                    # Legacy path ran but produced no valid rows
                    error_msg = "No performance metrics found from any node"
                    failure_record = self._create_failure_record(
                        model_info, build_info, deployment_id, error_msg
                    )
                    self._write_to_perf_csv(failure_record)
                    results["failed_runs"].append({
                        "model": model_info.get("name", "Unknown"),
                        "error": error_msg,
                        "nodes": results["nodes"]
                    })
                    self.console.print(
                        f"[yellow]⚠ No performance metrics found, recorded as FAILED[/yellow]"
                    )

            # 4. Generate summary
            self._generate_results_summary(results, results_dir)

        except Exception as e:
            self.console.print(f"[yellow]⚠ Results collection incomplete: {e}[/yellow]")

        return results

    def _collect_artifacts_immediately(self, deployment_id: str, pod_name: str) -> None:
        """
        Collect artifacts immediately from a running pod during the sleep period.
        This is called when we detect the "Keeping pod alive" message in logs.
        """
        try:
            # Create results directory
            results_dir = Path("k8s_results") / deployment_id
            results_dir.mkdir(parents=True, exist_ok=True)

            pod_dir = results_dir / pod_name
            pod_dir.mkdir(exist_ok=True)

            # Collect artifacts
            artifacts = self._collect_pod_artifacts(pod_name, pod_dir)

            if artifacts:
                self.console.print(f"[green]✓ Collected {len(artifacts)} artifacts from {pod_name}[/green]")
            else:
                self.console.print(f"[yellow]⚠ No artifacts collected from {pod_name}[/yellow]")

        except Exception as e:
            self.console.print(f"[yellow]⚠ Error collecting artifacts: {e}[/yellow]")

    def _collect_pod_artifacts(self, pod_name: str, dest_dir: Path) -> List[Dict]:
        """
        Collect file artifacts from pod using kubectl cp.

        Collects:
        - perf.csv (performance results)
        - *_env.csv (environment details from rocEnvTool)
        - profiling outputs (rocprof*, results*, *.db)
        - tracing outputs (*_output/ directories)
        - tool-specific outputs

        Args:
            pod_name: Name of the Kubernetes pod
            dest_dir: Local directory to save artifacts

        Returns:
            List of collected artifact metadata
        """
        artifacts = []

        # Define artifact patterns to collect
        artifact_patterns = [
            {"pattern": "perf.csv", "type": "performance"},
            {"pattern": "*_env.csv", "type": "environment"},
            {"pattern": "results*", "type": "profiling"},
            {"pattern": "*.db", "type": "profiling"},
            {"pattern": "trace.*", "type": "tracing"},
            {"pattern": "prof.csv", "type": "profiling"},  # Raw profiler output before post-script renames it
            {"pattern": "gpu_info_*.csv", "type": "profiling"},
            {"pattern": "library_trace.csv", "type": "tracing"},
        ]

        for artifact_def in artifact_patterns:
            pattern = artifact_def["pattern"]
            artifact_type = artifact_def["type"]

            try:
                # Try direct kubectl cp without exec (works during the sleep period)
                # For patterns with wildcards, try common specific filenames
                if '*' in pattern:
                    # Expand pattern to specific known files
                    if pattern == "*_env.csv":
                        specific_files = ["dummy_prof_env.csv", "dummy_data_minio_env.csv"]
                    elif pattern == "gpu_info_*.csv":
                        specific_files = ["gpu_info_power_profiler_output.csv", "gpu_info_vram_profiler_output.csv"]
                    elif pattern == "results*":
                        specific_files = ["results.csv", "results.txt", "results.json"]
                    elif pattern == "trace.*":
                        specific_files = ["trace.txt", "trace.csv", "trace.json"]
                    else:
                        specific_files = []

                    for filename in specific_files:
                        local_path = dest_dir / filename
                        cp_cmd = [
                            "kubectl", "cp",
                            f"{self.namespace}/{pod_name}:/workspace/{filename}",
                            str(local_path)
                        ]

                        cp_result = subprocess.run(
                            cp_cmd, capture_output=True, text=True, timeout=30
                        )

                        if cp_result.returncode == 0 and local_path.exists():
                            artifacts.append({
                                "pod": pod_name,
                                "type": artifact_type,
                                "source": f"/workspace/{filename}",
                                "local_path": str(local_path),
                                "size": local_path.stat().st_size
                            })
                            self.console.print(
                                f"[dim]    ✓ Collected {artifact_type}: {filename}[/dim]"
                            )
                        elif cp_result.stderr and "No such file" not in cp_result.stderr:
                            # Log unexpected errors (but not "file not found")
                            self.console.print(
                                f"[yellow]    ⚠ Failed to collect {filename}: {cp_result.stderr.strip()}[/yellow]"
                            )
                else:
                    # Direct file - try to copy it
                    local_path = dest_dir / pattern
                    cp_cmd = [
                        "kubectl", "cp",
                        f"{self.namespace}/{pod_name}:/workspace/{pattern}",
                        str(local_path)
                    ]

                    cp_result = subprocess.run(
                        cp_cmd, capture_output=True, text=True, timeout=30
                    )

                    if cp_result.returncode == 0 and local_path.exists():
                        artifacts.append({
                            "pod": pod_name,
                            "type": artifact_type,
                            "source": f"/workspace/{pattern}",
                            "local_path": str(local_path),
                            "size": local_path.stat().st_size
                        })
                        self.console.print(
                            f"[dim]    ✓ Collected {artifact_type}: {pattern}[/dim]"
                        )
                    elif cp_result.stderr and "No such file" not in cp_result.stderr:
                        # Log unexpected errors (but not "file not found")
                        self.console.print(
                            f"[yellow]    ⚠ Failed to collect {pattern}: {cp_result.stderr.strip()}[/yellow]"
                        )

            except subprocess.TimeoutExpired:
                pass  # Timeout - skip this file
            except Exception:
                pass  # File not found or not accessible - this is expected

        # Try to collect known output directories using kubectl cp directly (during sleep period)
        output_directories = ["rocprof_output", "rpd_output", "trace_output"]
        for dir_name in output_directories:
            try:
                local_dir = dest_dir / dir_name
                cp_cmd = [
                    "kubectl", "cp",
                    f"{self.namespace}/{pod_name}:/workspace/{dir_name}",
                    str(local_dir)
                ]

                cp_result = subprocess.run(
                    cp_cmd, capture_output=True, text=True, timeout=60
                )

                if cp_result.returncode == 0 and local_dir.exists():
                    # Count files in directory
                    file_count = sum(1 for _ in local_dir.rglob('*') if _.is_file())
                    if file_count > 0:
                        total_size = sum(f.stat().st_size for f in local_dir.rglob('*') if f.is_file())
                        artifacts.append({
                            "pod": pod_name,
                            "type": "tool_output_directory",
                            "source": f"/workspace/{dir_name}",
                            "local_path": str(local_dir),
                            "file_count": file_count,
                            "size": total_size
                        })
                        self.console.print(
                            f"[dim]    ✓ Collected directory: {dir_name} ({file_count} files, {total_size} bytes)[/dim]"
                        )
            except Exception:
                pass  # Directory not found - this is expected

        return artifacts

    def _collect_from_pvc(
        self,
        deployment_id: str,
        results_dir: Path,
        results: Dict,
        pod_names: Optional[List[str]] = None,
    ):
        """
        Collect all artifacts from the PVC using a temporary busybox pod.

        This is the best practice for collecting results from completed K8s jobs.
        kubectl cp doesn't work on completed pods, so we use a helper pod.

        When ``pod_names`` is provided, each ``/results/<subdir>/`` is copied to
        ``results_dir/<k8s_pod_name>/pvc/`` by matching subdir to pod name (exact or
        ``pod.startswith(subdir + "-")``). Unmatched subdirs go under
        ``results_dir/pvc_unmapped/<subdir>/``. When ``pod_names`` is omitted, the
        legacy layout ``results_dir/<subdir>/`` is used.

        Args:
            deployment_id: Job deployment ID
            results_dir: Local directory to save results
            results: Results dict to update
            pod_names: Full Kubernetes pod names for this job (ordered)
        """
        from .kubernetes import assign_pvc_subdirs_to_pods

        pvc_name = f"{deployment_id}-results"

        try:
            # Create a temporary pod to access PVC
            collector_pod_name = f"collector-{deployment_id[:15]}"

            self.console.print(f"[dim]📦 Collecting artifacts from PVC: {pvc_name}[/dim]")

            collector_spec: Dict[str, Any] = {
                "restartPolicy": "Never",
                "containers": [{
                    "name": "collector",
                    "image": "busybox:latest",
                    "command": ["sh", "-c", "sleep 600"],
                    "volumeMounts": [{"name": "results", "mountPath": "/results"}]
                }],
                "volumes": [{"name": "results", "persistentVolumeClaim": {"claimName": pvc_name}}]
            }
            ips = getattr(self, "_image_pull_secrets_for_pods", None) or []
            if ips:
                collector_spec["imagePullSecrets"] = ips

            collector_pod_spec = {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {"name": collector_pod_name, "namespace": self.namespace},
                "spec": collector_spec,
            }

            # Delete existing collector pod if it exists (prevents 409 Conflict)
            try:
                self.core_v1.delete_namespaced_pod(
                    collector_pod_name, self.namespace, grace_period_seconds=0
                )
                time.sleep(2)  # Wait for pod to be deleted
            except ApiException as e:
                if e.status != 404:  # 404 means pod doesn't exist, which is fine
                    pass

            # Create collector pod
            self.core_v1.create_namespaced_pod(self.namespace, collector_pod_spec)

            # Wait for pod to be ready
            for _ in range(30):  # Wait up to 30 seconds
                try:
                    pod_status = self.core_v1.read_namespaced_pod_status(
                        collector_pod_name, self.namespace
                    )
                    if pod_status.status.phase == "Running":
                        break
                except ApiException as e:
                    # Pod not found yet or not ready - this is expected during startup
                    if e.status != 404:
                        self.console.print(f"[dim]Waiting for collector pod (status: {e.status})...[/dim]")
                time.sleep(1)
            else:
                raise Exception("Collector pod did not start in time")

            # Mount / NFS may need a moment before another pod sees prior job writes.
            time.sleep(2)

            # List pod result directories in PVC (retry: NFS can lag right after Job completion)
            list_cmd = [
                "kubectl",
                "exec",
                collector_pod_name,
                "-n",
                self.namespace,
                "-c",
                "collector",
                "--",
                "ls",
                "-1",
                "/results/",
            ]
            list_result = subprocess.CompletedProcess(
                args=list_cmd, returncode=-1, stdout="", stderr=""
            )
            pod_dirs: List[str] = []
            for attempt in range(45):
                list_result = subprocess.run(
                    list_cmd, capture_output=True, text=True, timeout=30
                )
                if list_result.returncode == 0 and list_result.stdout.strip():
                    pod_dirs = [
                        d
                        for d in list_result.stdout.strip().split("\n")
                        if d and d != "lost+found"
                    ]
                    if pod_dirs:
                        break
                if list_result.stderr.strip():
                    self.console.print(
                        f"[dim]    PVC ls attempt {attempt + 1} (rc={list_result.returncode}): "
                        f"{list_result.stderr.strip()[:300]}[/dim]"
                    )
                time.sleep(1)

            if pod_dirs:
                pvc_map: Dict[str, str] = {}
                if pod_names:
                    pvc_map = assign_pvc_subdirs_to_pods(pod_dirs, pod_names)

                for pod_dir_name in pod_dirs:
                    if not pod_dir_name:
                        continue

                    matched_pod = pvc_map.get(pod_dir_name) if pod_names else None
                    if pod_names:
                        if matched_pod:
                            local_pod_dir = results_dir / matched_pod / "pvc"
                        else:
                            local_pod_dir = results_dir / "pvc_unmapped" / pod_dir_name
                    else:
                        local_pod_dir = results_dir / pod_dir_name

                    local_pod_dir.mkdir(parents=True, exist_ok=True)

                    cp_cmd = [
                        "kubectl",
                        "cp",
                        "-c",
                        "collector",
                        f"{self.namespace}/{collector_pod_name}:/results/{pod_dir_name}",
                        str(local_pod_dir),
                    ]

                    cp_result = subprocess.run(cp_cmd, capture_output=True, text=True, timeout=60)

                    if cp_result.returncode == 0:
                        # Count collected files
                        file_count = sum(1 for _ in local_pod_dir.rglob('*') if _.is_file())
                        if file_count > 0:
                            art: Dict[str, Any] = {
                                "source": f"PVC:{pvc_name}/{pod_dir_name}",
                                "local_path": str(local_pod_dir),
                                "file_count": file_count,
                                "type": "pvc_collection",
                                "pvc_subdir": pod_dir_name,
                            }
                            if pod_names:
                                art["k8s_pod"] = matched_pod
                            results["artifacts"].append(art)
                            if matched_pod:
                                dest_hint = f"{matched_pod}/pvc"
                            elif pod_names:
                                dest_hint = f"pvc_unmapped/{pod_dir_name}"
                            else:
                                dest_hint = pod_dir_name
                            self.console.print(
                                f"[dim]    ✓ Collected {file_count} files from {pod_dir_name} → {dest_hint}[/dim]"
                            )

                self.console.print(f"[green]✓ Collected artifacts from PVC[/green]")
            else:
                hint = ""
                if list_result.returncode != 0 or list_result.stderr.strip():
                    hint = (
                        f" (kubectl exec rc={list_result.returncode}"
                        + (
                            f", stderr={list_result.stderr.strip()[:400]!r}"
                            if list_result.stderr.strip()
                            else ""
                        )
                        + ")"
                    )
                self.console.print(
                    f"[yellow]⚠ No results found in PVC after retries{hint}[/yellow]"
                )

            # Cleanup collector pod
            self.core_v1.delete_namespaced_pod(
                collector_pod_name, self.namespace, grace_period_seconds=0
            )

        except Exception as e:
            self.console.print(f"[yellow]⚠ Could not collect from PVC: {e}[/yellow]")

    def _generate_results_summary(self, results: Dict, results_dir: Path):
        """
        Generate a summary JSON of all collected artifacts.

        Args:
            results: Results dict with logs and artifacts
            results_dir: Directory where results are saved
        """
        summary = {
            "job_name": results["job_name"],
            "namespace": results["namespace"],
            "collected_at": datetime.now().isoformat(),
            "k8s_results_layout": (
                "Per pod: <job>/<pod_name>/pod.log (API log) and "
                "<job>/<pod_name>/pvc/ (mirror of /results/<subdir>/). "
                "Unmatched PVC subdirs: <job>/pvc_unmapped/<subdir>/."
            ),
            "layout_version": 2,
            "pods": len(results["logs"]),
            "total_artifacts": len(results["artifacts"]),
            "artifacts_by_type": {},
            "artifacts": results["artifacts"],
            "successful_runs": len(results["successful_runs"]),
            "failed_runs": len(results["failed_runs"]),
        }

        # Group artifacts by type
        for artifact in results["artifacts"]:
            artifact_type = artifact.get("type", "unknown")
            summary["artifacts_by_type"][artifact_type] = summary["artifacts_by_type"].get(artifact_type, 0) + 1

        summary_file = results_dir / "results_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))

        self.console.print(f"[green]✓ Results summary: {summary_file}[/green]")

        # Print summary table if artifacts were collected
        if summary["artifacts_by_type"]:
            from rich.table import Table
            table = Table(title="Collected Artifacts")
            table.add_column("Type", style="cyan")
            table.add_column("Count", justify="right", style="green")

            for artifact_type, count in sorted(summary["artifacts_by_type"].items()):
                table.add_row(artifact_type, str(count))

            self.console.print(table)

    def _create_failure_record(self, model_info: Dict, build_info: Dict, pod_name: str, error_msg: str) -> Dict:
        """
        Create a failure record for perf.csv when performance metrics are missing.

        Args:
            model_info: Model information from manifest
            build_info: Build information from manifest
            pod_name: Kubernetes pod name
            error_msg: Error message describing the failure

        Returns:
            Dict with all perf.csv fields marked as FAILED
        """
        # Get topology information for failure record
        deployment_config = self.manifest.get("deployment_config", {})
        distributed_config = deployment_config.get("distributed", {})
        nnodes = distributed_config.get("nnodes", 1)
        nproc_per_node = distributed_config.get("nproc_per_node")
        if nproc_per_node is None:
            nproc_per_node = int(model_info.get("n_gpus", 1))
        # Launcher: use distributed.launcher when set, otherwise "native" for k8s
        launcher = normalize_launcher(distributed_config.get("launcher"), "kubernetes")

        # Create a record with the same structure as successful runs
        # but with performance=0, metric="", and status="FAILED"
        result = {
            # Core identification
            "model": model_info.get("name", ""),
            "n_gpus": str(nnodes * nproc_per_node),
            "nnodes": str(nnodes),
            "gpus_per_node": str(nproc_per_node),

            # Model configuration
            "training_precision": model_info.get("training_precision", ""),
            "pipeline": get_pipeline(),
            "args": model_info.get("args", ""),
            "tags": model_info.get("tags", ""),

            # Build information
            "docker_file": build_info.get("dockerfile", ""),
            "base_docker": build_info.get("base_docker", ""),
            "docker_sha": build_info.get("docker_sha", ""),
            "docker_image": build_info.get("docker_image", ""),

            # Runtime information
            "git_commit": "",
            "machine_name": pod_name,
            "deployment_type": "kubernetes",
            "launcher": launcher,
            "gpu_architecture": "",

            # Performance metrics - FAILED
            "performance": "0",
            "metric": error_msg,  # Store error message in metric field
            "relative_change": "",
            "status": "FAILURE",  # Use "FAILURE" to match CSV schema

            # Timing
            "build_duration": build_info.get("build_duration", ""),
            "test_duration": "",

            # Data information
            "dataname": model_info.get("data", ""),
            "data_provider_type": "",
            "data_size": "",
            "data_download_duration": "",

            # Build tracking
            "build_number": get_build_number(),
            "additional_docker_run_options": model_info.get("additional_docker_run_options", ""),
        }
        flatten_tags_in_place(result)
        return result

    def _ensure_perf_csv_exists(self) -> None:
        """Ensure perf.csv exists with standard header (same as Docker container_runner)."""
        perf_csv_path = Path("perf.csv")
        if not perf_csv_path.exists():
            perf_csv_path.write_text(self._PERF_CSV_HEADER + "\n", encoding="utf-8")
            self.console.print("[dim]Created perf.csv with standard header[/dim]")

    def _build_perf_entry_from_aggregated(
        self,
        aggregated_record: Dict[str, Any],
        model_info: Dict[str, Any],
        build_info: Dict[str, Any],
        deployment_id: str,
    ) -> Dict[str, Any]:
        """Build full run_details dict from aggregated record for perf_entry and update_* pipeline."""
        from madengine.utils.config_parser import ConfigParser

        deployment_config = self.manifest.get("deployment_config", {})
        distributed_config = deployment_config.get("distributed", {})
        nnodes = distributed_config.get("nnodes", 1)
        nproc_per_node = distributed_config.get("nproc_per_node")
        if nproc_per_node is None:
            nproc_per_node = int(model_info.get("n_gpus", 1))
        launcher = normalize_launcher(distributed_config.get("launcher"), "kubernetes")
        test_duration = aggregated_record.get("test_duration") or aggregated_record.get("duration", "")
        run_details = {
            "model": model_info.get("name", aggregated_record.get("model", "")),
            "n_gpus": str(aggregated_record.get("n_gpus", nnodes * nproc_per_node)),
            "nnodes": str(aggregated_record.get("nnodes", nnodes)),
            "gpus_per_node": str(aggregated_record.get("gpus_per_node", nproc_per_node)),
            "training_precision": model_info.get("training_precision", ""),
            "pipeline": get_pipeline(),
            "args": model_info.get("args", ""),
            "tags": model_info.get("tags", ""),
            "docker_file": build_info.get("dockerfile", ""),
            "base_docker": build_info.get("base_docker", ""),
            "docker_sha": build_info.get("docker_sha", ""),
            "docker_image": build_info.get("docker_image", ""),
            "git_commit": "",
            "machine_name": deployment_id,
            "deployment_type": "kubernetes",
            "launcher": launcher,
            "gpu_architecture": aggregated_record.get("gpu_architecture", ""),
            "performance": str(aggregated_record.get("performance", "")),
            "metric": aggregated_record.get("metric", ""),
            "relative_change": "",
            "status": aggregated_record.get("status", "SUCCESS"),
            "build_duration": build_info.get("build_duration", ""),
            "test_duration": test_duration,
            "dataname": aggregated_record.get("data_name", model_info.get("data", "")),
            "data_provider_type": aggregated_record.get("data_provider", ""),
            "data_size": "",
            "data_download_duration": "",
            "build_number": get_build_number(),
            "additional_docker_run_options": model_info.get("additional_docker_run_options", ""),
        }
        flatten_tags_in_place(run_details)
        try:
            scripts_path = model_info.get("scripts", "")
            scripts_base_dir = scripts_base_dir_from(scripts_path)
            config_parser = ConfigParser(scripts_base_dir=scripts_base_dir)
            run_details["configs"] = config_parser.parse_and_load(
                model_info.get("args", ""), scripts_path
            )
        except Exception:
            run_details["configs"] = None
        return run_details

    def _build_common_info_dict(
        self,
        model_info: Dict,
        build_info: Dict,
        deployment_id: str,
        gpu_architecture: str = "",
    ) -> Dict:
        """
        Build common_info dict for update_perf_csv / update_perf_super (Docker-compatible).
        Same shape as container_runner create_run_details_dict; model/performance/metric
        are omitted so they are filled from the multiple_results CSV.
        """
        deployment_config = self.manifest.get("deployment_config", {})
        distributed_config = deployment_config.get("distributed", {})
        nnodes = distributed_config.get("nnodes", 1)
        nproc_per_node = distributed_config.get("nproc_per_node")
        if nproc_per_node is None:
            nproc_per_node = int(model_info.get("n_gpus", 1))
        total_gpus = nnodes * nproc_per_node
        gpus_per_node = str(nproc_per_node)
        nnodes_str = str(nnodes)
        # Launcher: use distributed.launcher when set, otherwise "native" for k8s
        launcher = normalize_launcher(distributed_config.get("launcher"), "kubernetes")
        result = {
            "n_gpus": str(total_gpus),
            "nnodes": nnodes_str,
            "gpus_per_node": gpus_per_node,
            "training_precision": model_info.get("training_precision", ""),
            "pipeline": get_pipeline(),
            "args": model_info.get("args", ""),
            "tags": model_info.get("tags", ""),
            "docker_file": build_info.get("dockerfile", ""),
            "base_docker": build_info.get("base_docker", ""),
            "docker_sha": build_info.get("docker_sha", ""),
            "docker_image": build_info.get("docker_image", ""),
            "git_commit": "",
            "machine_name": deployment_id,
            "deployment_type": "kubernetes",
            "launcher": launcher,
            "gpu_architecture": gpu_architecture,
            "relative_change": "",
            "build_duration": build_info.get("build_duration", ""),
            "test_duration": "",
            "dataname": model_info.get("data", ""),
            "data_provider_type": "",
            "data_size": "",
            "data_download_duration": "",
            "build_number": get_build_number(),
            "additional_docker_run_options": model_info.get("additional_docker_run_options", ""),
        }
        flatten_tags_in_place(result)
        return result

    def _create_multiple_result_row_record(
        self,
        model_info: Dict,
        build_info: Dict,
        deployment_id: str,
        item: Dict,
    ) -> Dict:
        """
        Build one perf.csv row for a single row from a multiple_results CSV.
        Same shape as _create_failure_record but with SUCCESS and item's performance/metric/model.
        """
        deployment_config = self.manifest.get("deployment_config", {})
        distributed_config = deployment_config.get("distributed", {})
        nnodes = distributed_config.get("nnodes", 1)
        nproc_per_node = distributed_config.get("nproc_per_node")
        if nproc_per_node is None:
            nproc_per_node = int(model_info.get("n_gpus", 1))

        # Launcher: use distributed.launcher when set, otherwise "native" for k8s
        launcher = normalize_launcher(distributed_config.get("launcher"), "kubernetes")
        result = {
            "model": item.get("model", model_info.get("name", "")),
            "n_gpus": str(nnodes * nproc_per_node),
            "nnodes": str(nnodes),
            "gpus_per_node": str(nproc_per_node),
            "training_precision": model_info.get("training_precision", ""),
            "pipeline": get_pipeline(),
            "args": model_info.get("args", ""),
            "tags": model_info.get("tags", ""),
            "docker_file": build_info.get("dockerfile", ""),
            "base_docker": build_info.get("base_docker", ""),
            "docker_sha": build_info.get("docker_sha", ""),
            "docker_image": build_info.get("docker_image", ""),
            "git_commit": "",
            "machine_name": deployment_id,
            "deployment_type": "kubernetes",
            "launcher": launcher,
            "gpu_architecture": item.get("gpu_architecture", ""),
            "performance": str(item.get("performance", "")),
            "metric": item.get("metric", ""),
            "relative_change": "",
            "status": "SUCCESS",
            "build_duration": build_info.get("build_duration", ""),
            "test_duration": item.get("duration", ""),
            "dataname": model_info.get("data", ""),
            "data_provider_type": "",
            "data_size": "",
            "data_download_duration": "",
            "build_number": get_build_number(),
            "additional_docker_run_options": model_info.get("additional_docker_run_options", ""),
        }
        flatten_tags_in_place(result)
        return result

    def _parse_multiple_results_from_artifacts(
        self,
        results_dir: Path,
        results: Dict,
        model_info: Dict,
        build_info: Dict,
    ) -> List[Dict]:
        """
        Parse performance from a multiple_results CSV (e.g. perf_dummy.csv) collected from PVC.
        Used when the model only writes CSV and does not print 'performance: X Y' to the log
        (same contract as local container_runner multiple_results handling).

        Returns:
            List of perf_data dicts (same shape as _parse_node_performance), or empty list.
        """
        import csv as csv_module
        multiple_results_file = model_info.get("multiple_results")
        filename = Path(multiple_results_file).name if multiple_results_file else None
        # Try to get gpu_architecture from first pod log
        gpu_arch = "N/A"
        if results.get("logs"):
            log_content = results["logs"][0].get("log", "")
            gpu_arch_match = re.search(r"(?:🔹\s*)?Name\s*:\s*(gfx\w+)", log_content)
            if gpu_arch_match:
                gpu_arch = gpu_arch_match.group(1)
        parsed_list = []
        for art in results.get("artifacts", []):
            if art.get("type") != "pvc_collection":
                continue
            local_path = Path(art.get("local_path", ""))
            if not local_path.is_dir():
                continue
            # Prefer exact filename (same as Docker multiple_results); fallback to any perf_*.csv
            csv_path = (local_path / filename) if filename else None
            if not csv_path or not csv_path.is_file():
                perf_csvs = sorted(local_path.glob("perf_*.csv"))
                csv_path = perf_csvs[0] if perf_csvs else None
            if not csv_path or not csv_path.is_file():
                continue
            try:
                with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv_module.DictReader(f)
                    reader.fieldnames = [f.strip() for f in (reader.fieldnames or [])]
                    if not reader.fieldnames or "performance" not in reader.fieldnames or "metric" not in reader.fieldnames:
                        continue
                    for row_idx, row in enumerate(reader):
                        perf_val = row.get("performance", "").strip()
                        metric_val = row.get("metric", "").strip()
                        if not perf_val or not metric_val:
                            continue
                        try:
                            perf_float = float(perf_val)
                        except (ValueError, TypeError):
                            continue
                        # Same model naming as local handle_multiple_results: model_name + "_" + str(model)
                        row_model = row.get("model", row_idx)
                        display_model = f"{model_info.get('name')}_{row_model}"
                        parsed_list.append({
                            "model": display_model,
                            "performance": perf_float,
                            "metric": metric_val,
                            "node_id": row_idx,
                            "local_gpus": 1,
                            "duration": "N/A",
                            "gpu_architecture": gpu_arch,
                            "data_name": "N/A",
                            "data_provider": "N/A",
                        })
                if parsed_list:
                    self.console.print(
                        f"[green]  ✓ Parsed performance from {csv_path.name} ({len(parsed_list)} row(s))[/green]"
                    )
                    return parsed_list
            except Exception as e:
                self.console.print(
                    f"[dim]  Could not parse {csv_path.name} from PVC: {e}[/dim]"
                )
        return []

    def _aggregation_for_extra_column(self, column_name: str) -> str:
        """
        Return how to aggregate an extra CSV column when merging multi-node results.
        Best practice: throughput/counts -> sum; latencies/utilization -> average;
        duration/capacity -> max; identifiers -> first.
        """
        col = column_name.lower().strip()
        # Sum: counts, totals, throughput-like
        if any(k in col for k in [
            "count", "total", "samples", "tokens", "throughput",
            "requests", "images", "bandwidth", "ops"
        ]):
            return "sum"
        # Average: rates per unit, utilization, ratios
        if any(k in col for k in [
            "utilization", "usage", "percent", "ratio", "latency",
            "time_ms", "ttft", "tpot", "accuracy", "loss"
        ]):
            return "average"
        # Max: duration (slowest node), memory, capacity
        if any(k in col for k in [
            "duration", "time", "seconds", "memory", "bytes", "mb", "gb"
        ]):
            return "max"
        return "first"

    def _merge_multi_node_multiple_results_csv(
        self, csv_paths: List[Path], output_path: Path
    ) -> bool:
        """
        Merge multiple pod multiple_results CSVs into one with sum/average rules.
        Rows are aligned by index (row 0 from each pod -> one merged row 0).
        - performance: aggregated by _determine_aggregation_method(metric) (sum or average).
        - Other numeric columns: by _aggregation_for_extra_column (sum/average/max).
        - model, metric: taken from first CSV.
        """
        import csv as csv_module
        import statistics

        required = ["model", "performance", "metric"]
        rows_by_index: Dict[int, List[Dict]] = {}

        for path in csv_paths:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv_module.DictReader(f)
                    fieldnames = [c.strip() for c in (reader.fieldnames or [])]
                    if not all(h in fieldnames for h in required):
                        continue
                    for idx, row in enumerate(reader):
                        row = {k.strip(): v for k, v in row.items() if k}
                        if not row.get("performance") or not row.get("metric"):
                            continue
                        try:
                            float(str(row["performance"]).strip())
                        except (ValueError, TypeError):
                            continue
                        if idx not in rows_by_index:
                            rows_by_index[idx] = []
                        rows_by_index[idx].append(row)
            except Exception as e:
                self.console.print(f"[dim]  Could not read {path.name}: {e}[/dim]")
                continue

        if not rows_by_index:
            return False

        # Build union of columns (required first, then rest)
        extra_cols = set()
        for group in rows_by_index.values():
            for row in group:
                extra_cols.update(k for k in row if k not in required)
        all_columns = list(required) + sorted(extra_cols)
        merged_rows = []
        for idx in sorted(rows_by_index.keys()):
            group = rows_by_index[idx]
            first = group[0]
            metric_name = (first.get("metric") or "").strip()
            perf_agg = self._determine_aggregation_method(metric_name)
            perf_values = []
            for r in group:
                try:
                    perf_values.append(float(str(r.get("performance", "")).strip()))
                except (ValueError, TypeError):
                    pass
            if not perf_values:
                continue
            if perf_agg == "sum":
                performance = sum(perf_values)
            elif perf_agg == "average":
                performance = statistics.mean(perf_values)
            elif perf_agg == "max":
                performance = max(perf_values)
            else:
                performance = sum(perf_values)
            merged = {
                "model": first.get("model", ""),
                "performance": performance,
                "metric": first.get("metric", ""),
            }
            for col in all_columns:
                if col in merged:
                    continue
                values = [r.get(col) for r in group]
                try:
                    nums = [float(str(v).strip()) for v in values if v is not None and str(v).strip()]
                except (ValueError, TypeError):
                    nums = []
                if nums:
                    extra_agg = self._aggregation_for_extra_column(col)
                    if extra_agg == "sum":
                        merged[col] = sum(nums)
                    elif extra_agg == "average":
                        merged[col] = statistics.mean(nums)
                    elif extra_agg == "max":
                        merged[col] = max(nums)
                    else:
                        merged[col] = first.get(col, "")
                else:
                    merged[col] = first.get(col, "")
            merged_rows.append(merged)

        if not merged_rows:
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv_module.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(merged_rows)
        self.console.print(
            f"[green]  ✓ Merged {len(csv_paths)} pod CSV(s) into {len(merged_rows)} row(s) → {output_path.name}[/green]"
        )
        return True

    def _resolve_multiple_results_csv(
        self, results_dir: Path, results: Dict, model_info: Dict
    ) -> Optional[Path]:
        """
        Resolve path to a single multiple_results CSV for update_perf_csv.
        Single pod: return that CSV path. Multi-pod: merge all pod CSVs with
        sum/average rules and return path to merged file.
        """
        multiple_results_file = model_info.get("multiple_results")
        filename = Path(multiple_results_file).name if multiple_results_file else None
        csv_paths: List[Path] = []
        for art in results.get("artifacts", []):
            if art.get("type") != "pvc_collection":
                continue
            local_path = Path(art.get("local_path", ""))
            if not local_path.is_dir():
                continue
            csv_path = (local_path / filename) if filename else None
            if not csv_path or not csv_path.is_file():
                perf_csvs = sorted(local_path.glob("perf_*.csv"))
                csv_path = perf_csvs[0] if perf_csvs else None
            if csv_path and csv_path.is_file():
                csv_paths.append(csv_path)
        if not csv_paths:
            return None
        if len(csv_paths) == 1:
            return csv_paths[0]
        merged_path = results_dir / "multiple_results_merged.csv"
        if self._merge_multi_node_multiple_results_csv(csv_paths, merged_path):
            return merged_path
        return csv_paths[0]
