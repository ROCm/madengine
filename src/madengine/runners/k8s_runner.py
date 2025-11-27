#!/usr/bin/env python3
"""
Kubernetes Distributed Runner for MADEngine

This module implements Kubernetes-based distributed execution using
the kubernetes Python client for orchestrated parallel execution.
"""

import json
import os
import time
import yaml
from typing import Dict, List, Any, Optional
import contextlib
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
except ImportError:
    raise ImportError(
        "Kubernetes runner requires kubernetes. Install with: pip install kubernetes"
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
    ConnectionError as MADConnectionError,
    create_error_context
)


@dataclass
class KubernetesExecutionError(RunnerError):
    """Kubernetes execution specific errors."""

    resource_type: str
    resource_name: str
    
    def __init__(self, message: str, resource_type: str, resource_name: str, **kwargs):
        self.resource_type = resource_type
        self.resource_name = resource_name
        context = create_error_context(
            operation="kubernetes_execution",
            component="KubernetesRunner",
            additional_info={
                "resource_type": resource_type,
                "resource_name": resource_name
            }
        )
        super().__init__(
            f"Kubernetes error in {resource_type}/{resource_name}: {message}", 
            context=context, 
            **kwargs
        )


class KubernetesDistributedRunner(BaseDistributedRunner):
    """Distributed runner using Kubernetes with enhanced error handling."""

    def __init__(self, inventory_path: str, manifests_dir: str, **kwargs):
        """Initialize Kubernetes distributed runner.

        The runner only executes pre-generated Kubernetes manifests created by the generate command.
        It does not create or modify any Kubernetes resources dynamically.

        Args:
            inventory_path: Path to Kubernetes inventory/configuration file
            manifests_dir: Directory containing pre-generated Kubernetes manifests
            **kwargs: Additional arguments (kubeconfig_path, namespace, etc.)
        """
        super().__init__(inventory_path, **kwargs)
        self.manifests_dir = manifests_dir
        self.kubeconfig_path = kwargs.get("kubeconfig_path")
        self.namespace = kwargs.get("namespace", "default")
        self.cleanup_handlers: List[callable] = []
        self.created_resources: List[Dict[str, str]] = []
        self.executor: Optional[ThreadPoolExecutor] = None
        self.k8s_client = None
        self.batch_client = None
        self._connection_validated = False

    def _validate_kubernetes_connection(self) -> bool:
        """Validate Kubernetes connection and permissions."""
        try:
            if self._connection_validated:
                return True

            # Test basic connectivity
            version = self.k8s_client.get_version()
            self.logger.info(f"Connected to Kubernetes cluster version: {version}")

            # Test namespace access
            try:
                self.k8s_client.read_namespace(name=self.namespace)
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    self.logger.error(f"Namespace '{self.namespace}' not found")
                    return False
                elif e.status == 403:
                    self.logger.error(f"No access to namespace '{self.namespace}'")
                    return False
                raise

            # Test job creation permissions
            try:
                # Try to list jobs to check permissions
                self.batch_client.list_namespaced_job(namespace=self.namespace, limit=1)
            except client.exceptions.ApiException as e:
                if e.status == 403:
                    self.logger.error("No permission to create jobs")
                    return False
                raise

            self._connection_validated = True
            return True

        except Exception as e:
            self.logger.error(f"Kubernetes connection validation failed: {e}")
            return False

    def _ensure_namespace_exists(self) -> bool:
        """Ensure the target namespace exists."""
        try:
            self.k8s_client.read_namespace(name=self.namespace)
            return True
        except client.exceptions.ApiException as e:
            if e.status == 404:
                # Try to create namespace
                try:
                    namespace = client.V1Namespace(
                        metadata=client.V1ObjectMeta(name=self.namespace)
                    )
                    self.k8s_client.create_namespace(body=namespace)
                    self.logger.info(f"Created namespace: {self.namespace}")
                    return True
                except client.exceptions.ApiException as create_e:
                    self.logger.error(f"Failed to create namespace: {create_e}")
                    return False
            else:
                self.logger.error(f"Namespace access error: {e}")
                return False
        except Exception as e:
            self.logger.error(f"Namespace validation failed: {e}")
            return False

    def _init_kubernetes_client(self):
        """Initialize Kubernetes client."""
        try:
            if self.kubeconfig_path:
                config.load_kube_config(config_file=self.kubeconfig_path)
            else:
                # Try in-cluster config first, fallback to default kubeconfig
                try:
                    config.load_incluster_config()
                except config.ConfigException:
                    config.load_kube_config()

            self.k8s_client = client.CoreV1Api()
            self.batch_client = client.BatchV1Api()

            # Test connection
            self.k8s_client.get_api_resources()
            self.logger.info("Successfully connected to Kubernetes cluster")

        except Exception as e:
            self.logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise

    def _parse_inventory(self, inventory_data: Dict[str, Any]) -> List[NodeConfig]:
        """Parse Kubernetes inventory data.

        For Kubernetes, inventory represents node selectors and resource requirements
        rather than individual nodes.

        Args:
            inventory_data: Raw inventory data

        Returns:
            List of NodeConfig objects (representing logical nodes/pods)
        """
        nodes = []

        # Support Kubernetes-specific inventory format
        if "pods" in inventory_data:
            for pod_spec in inventory_data["pods"]:
                node = NodeConfig(
                    hostname=pod_spec.get("name", f"pod-{len(nodes)}"),
                    address=pod_spec.get("node_selector", {}).get(
                        "kubernetes.io/hostname", ""
                    ),
                    gpu_count=pod_spec.get("resources", {})
                    .get("requests", {})
                    .get("nvidia.com/gpu", 1),
                    gpu_vendor=pod_spec.get("gpu_vendor", "NVIDIA"),
                    labels=pod_spec.get("node_selector", {}),
                    environment=pod_spec.get("environment", {}),
                )
                nodes.append(node)
        elif "node_selectors" in inventory_data:
            # Alternative format with explicit node selectors
            for i, selector in enumerate(inventory_data["node_selectors"]):
                node = NodeConfig(
                    hostname=f"pod-{i}",
                    address="",
                    gpu_count=selector.get("gpu_count", 1),
                    gpu_vendor=selector.get("gpu_vendor", "NVIDIA"),
                    labels=selector.get("labels", {}),
                    environment=selector.get("environment", {}),
                )
                nodes.append(node)
        else:
            # Fallback to base class parsing
            return super()._parse_inventory(inventory_data)

        return nodes

    def _create_namespace(self) -> bool:
        """Create namespace if it doesn't exist.

        Returns:
            True if namespace exists or was created, False otherwise
        """
        try:
            self.k8s_client.read_namespace(name=self.namespace)
            self.logger.info(f"Namespace '{self.namespace}' already exists")
            return True
        except ApiException as e:
            if e.status == 404:
                # Namespace doesn't exist, create it
                namespace = client.V1Namespace(
                    metadata=client.V1ObjectMeta(name=self.namespace)
                )
                self.k8s_client.create_namespace(body=namespace)
                self.logger.info(f"Created namespace '{self.namespace}'")
                return True
            else:
                self.logger.error(f"Failed to check namespace: {e}")
                return False

    def _create_configmap(self, workload: WorkloadSpec) -> bool:
        """Create ConfigMap with manifest and configuration.

        Args:
            workload: Workload specification

        Returns:
            True if ConfigMap created successfully, False otherwise
        """
        try:
            # Read manifest file
            with open(workload.manifest_file, "r") as f:
                manifest_content = f.read()

            # Create ConfigMap data
            config_data = {
                "build_manifest.json": manifest_content,
                "additional_context.json": json.dumps(workload.additional_context),
                "config.json": json.dumps(
                    {
                        "timeout": workload.timeout,
                        "registry": workload.registry,
                        "model_tags": workload.model_tags,
                    }
                ),
            }

            # Add supporting files if they exist
            supporting_files = ["credential.json", "data.json", "models.json"]
            for file_name in supporting_files:
                if os.path.exists(file_name):
                    try:
                        with open(file_name, "r") as f:
                            config_data[file_name] = f.read()
                        self.logger.info(f"Added {file_name} to ConfigMap")
                    except Exception as e:
                        self.logger.warning(f"Failed to read {file_name}: {e}")

            # Create ConfigMap
            configmap = client.V1ConfigMap(
                metadata=client.V1ObjectMeta(
                    name=self.configmap_name, namespace=self.namespace
                ),
                data=config_data,
            )

            # Delete existing ConfigMap if it exists
            try:
                self.k8s_client.delete_namespaced_config_map(
                    name=self.configmap_name, namespace=self.namespace
                )
            except ApiException as e:
                if e.status != 404:
                    self.logger.warning(f"Failed to delete existing ConfigMap: {e}")

            # Create new ConfigMap
            self.k8s_client.create_namespaced_config_map(
                namespace=self.namespace, body=configmap
            )

            self.created_resources.append(("ConfigMap", self.configmap_name))
            self.logger.info(f"Created ConfigMap '{self.configmap_name}'")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create ConfigMap: {e}")
            return False

    def _create_job(
        self, node: NodeConfig, model_tag: str, workload: WorkloadSpec
    ) -> str:
        """Create Kubernetes Job for a specific model on a node.

        Args:
            node: Node configuration
            model_tag: Model tag to execute
            workload: Workload specification

        Returns:
            Job name if created successfully, None otherwise
        """
        job_name = f"{self.job_name_prefix}-{node.hostname}-{model_tag}".replace(
            "_", "-"
        ).lower()

        try:
            # Create container spec
            container = client.V1Container(
                name="madengine-runner",
                image=self.container_image,
                command=["sh", "-c"],
                args=[
                    f"""
                    # Setup MAD environment
                    if [ -d MAD ]; then
                        cd MAD && git pull origin main
                    else
                        git clone https://github.com/ROCm/MAD.git
                    fi

                    cd MAD
                    python3 -m venv venv || true
                    source venv/bin/activate
                    pip install -r requirements.txt
                    pip install paramiko scp ansible-runner kubernetes PyYAML || true

                    # Copy config files from mounted volume
                    cp /workspace/build_manifest.json .
                    cp /workspace/credential.json . 2>/dev/null || true
                    cp /workspace/data.json . 2>/dev/null || true
                    cp /workspace/models.json . 2>/dev/null || true

                    # Execute madengine from MAD directory
                    madengine-cli run \\
                        --manifest-file build_manifest.json \\
                        --timeout {workload.timeout} \\
                        --tags {model_tag} \\
                        --registry {workload.registry or ''} \\
                        --additional-context "$(cat /workspace/additional_context.json 2>/dev/null || echo '{{}}')"  # noqa: E501
                """
                ],
                volume_mounts=[
                    client.V1VolumeMount(name="config-volume", mount_path="/workspace")
                ],
                env=[
                    client.V1EnvVar(name=k, value=v)
                    for k, v in node.environment.items()
                ],
                resources=client.V1ResourceRequirements(
                    requests=(
                        {"nvidia.com/gpu": str(node.gpu_count)}
                        if node.gpu_vendor == "NVIDIA"
                        else (
                            {"amd.com/gpu": str(node.gpu_count)}
                            if node.gpu_vendor == "AMD"
                            else {}
                        )
                    )
                ),
            )

            # Create pod spec
            pod_spec = client.V1PodSpec(
                containers=[container],
                restart_policy="Never",
                volumes=[
                    client.V1Volume(
                        name="config-volume",
                        config_map=client.V1ConfigMapVolumeSource(
                            name=self.configmap_name
                        ),
                    )
                ],
                node_selector=node.labels if node.labels else None,
            )

            # Create job spec
            job_spec = client.V1JobSpec(
                template=client.V1PodTemplateSpec(spec=pod_spec),
                backoff_limit=3,
                ttl_seconds_after_finished=300,
            )

            # Create job
            job = client.V1Job(
                metadata=client.V1ObjectMeta(name=job_name, namespace=self.namespace),
                spec=job_spec,
            )

            # Submit job
            self.batch_client.create_namespaced_job(namespace=self.namespace, body=job)

            self.created_resources.append(("Job", job_name))
            self.logger.info(f"Created job '{job_name}'")
            return job_name

        except Exception as e:
            self.logger.error(f"Failed to create job '{job_name}': {e}")
            return None

    def _wait_for_jobs(
        self, job_names: List[str], timeout: int = 3600
    ) -> Dict[str, Any]:
        """Wait for jobs to complete.

        Args:
            job_names: List of job names to wait for
            timeout: Timeout in seconds

        Returns:
            Dictionary mapping job names to their results
        """
        job_results = {}
        start_time = time.time()

        while job_names and (time.time() - start_time) < timeout:
            completed_jobs = []

            for job_name in job_names:
                try:
                    job = self.batch_client.read_namespaced_job(
                        name=job_name, namespace=self.namespace
                    )

                    if job.status.completion_time:
                        # Job completed successfully
                        job_results[job_name] = {
                            "status": "SUCCESS",
                            "completion_time": job.status.completion_time,
                            "start_time": job.status.start_time,
                        }
                        completed_jobs.append(job_name)
                    elif job.status.failed:
                        # Job failed
                        job_results[job_name] = {
                            "status": "FAILURE",
                            "failed_pods": job.status.failed,
                            "start_time": job.status.start_time,
                        }
                        completed_jobs.append(job_name)

                except ApiException as e:
                    self.logger.error(f"Failed to get job status for {job_name}: {e}")
                    job_results[job_name] = {"status": "FAILURE", "error": str(e)}
                    completed_jobs.append(job_name)

            # Remove completed jobs from the list
            for job_name in completed_jobs:
                job_names.remove(job_name)

            if job_names:
                time.sleep(10)  # Wait 10 seconds before checking again

        # Mark remaining jobs as timed out
        for job_name in job_names:
            job_results[job_name] = {
                "status": "TIMEOUT",
                "message": f"Job did not complete within {timeout} seconds",
            }

        return job_results

    def _create_configmaps(self, workload: WorkloadSpec) -> bool:
        """Create ConfigMaps for workload data with size validation."""
        try:
            # Create ConfigMap for additional context
            if workload.additional_context:
                context_data = workload.additional_context

                # Validate ConfigMap size (1MB limit)
                if len(json.dumps(context_data).encode("utf-8")) > 1024 * 1024:
                    self.logger.error("Additional context too large for ConfigMap")
                    return False

                configmap_name = f"{self.job_name_prefix}-context"
                configmap = client.V1ConfigMap(
                    metadata=client.V1ObjectMeta(
                        name=configmap_name, namespace=self.namespace
                    ),
                    data={"additional_context.json": json.dumps(context_data)},
                )

                try:
                    self.k8s_client.create_namespaced_config_map(
                        namespace=self.namespace, body=configmap
                    )
                    self.created_resources.append(
                        {
                            "type": "configmap",
                            "name": configmap_name,
                            "namespace": self.namespace,
                        }
                    )
                    self.logger.info(f"Created ConfigMap: {configmap_name}")

                except client.exceptions.ApiException as e:
                    if e.status == 409:  # Already exists
                        self.logger.info(f"ConfigMap {configmap_name} already exists")
                    else:
                        self.logger.error(f"Failed to create ConfigMap: {e}")
                        return False

            # Create ConfigMap for manifest file
            if workload.manifest_file and os.path.exists(workload.manifest_file):
                with open(workload.manifest_file, "r") as f:
                    manifest_data = f.read()

                # Validate size
                if len(manifest_data.encode("utf-8")) > 1024 * 1024:
                    self.logger.error("Manifest file too large for ConfigMap")
                    return False

                configmap_name = f"{self.job_name_prefix}-manifest"
                configmap = client.V1ConfigMap(
                    metadata=client.V1ObjectMeta(
                        name=configmap_name, namespace=self.namespace
                    ),
                    data={"build_manifest.json": manifest_data},
                )

                try:
                    self.k8s_client.create_namespaced_config_map(
                        namespace=self.namespace, body=configmap
                    )
                    self.created_resources.append(
                        {
                            "type": "configmap",
                            "name": configmap_name,
                            "namespace": self.namespace,
                        }
                    )
                    self.logger.info(f"Created ConfigMap: {configmap_name}")

                except client.exceptions.ApiException as e:
                    if e.status == 409:  # Already exists
                        self.logger.info(f"ConfigMap {configmap_name} already exists")
                    else:
                        self.logger.error(f"Failed to create ConfigMap: {e}")
                        return False

            return True

        except Exception as e:
            self.logger.error(f"ConfigMap creation failed: {e}")
            return False

    def execute_workload(self, workload: WorkloadSpec = None) -> DistributedResult:
        """Execute workload using pre-generated Kubernetes manifests.

        This method applies pre-generated Kubernetes manifests from the manifests_dir
        and monitors the resulting jobs for completion.

        Args:
            workload: Legacy parameter, not used in simplified workflow

        Returns:
            Distributed execution result
        """
        try:
            self.logger.info(
                "Starting Kubernetes distributed execution using pre-generated manifests"
            )

            # Initialize Kubernetes client
            self._init_kubernetes_client()

            # Validate connection and permissions
            if not self._validate_kubernetes_connection():
                return DistributedResult(
                    success=False,
                    node_results=[],
                    error_message="Failed to validate Kubernetes connection",
                )

            # Apply manifests
            if not self._apply_manifests():
                return DistributedResult(
                    success=False,
                    node_results=[],
                    error_message="Failed to apply Kubernetes manifests",
                )

            # Monitor execution
            results = self._monitor_execution()

            distributed_result = DistributedResult(
                success=any(r.success for r in results) if results else False,
                node_results=results,
            )

            self.logger.info("Kubernetes distributed execution completed")
            return distributed_result

        except Exception as e:
            self.logger.error(f"Distributed execution failed: {e}")
            return DistributedResult(
                success=False, node_results=[], error_message=str(e)
            )

    def _apply_manifests(self) -> bool:
        """Apply pre-generated Kubernetes manifests from manifests_dir.

        Returns:
            True if manifests applied successfully, False otherwise
        """
        try:
            if not os.path.exists(self.manifests_dir):
                self.logger.error(
                    f"Manifests directory not found: {self.manifests_dir}"
                )
                return False

            # Find all YAML manifest files
            manifest_files = []
            for root, dirs, files in os.walk(self.manifests_dir):
                for file in files:
                    if file.endswith((".yaml", ".yml")):
                        manifest_files.append(os.path.join(root, file))

            if not manifest_files:
                self.logger.error(
                    f"No YAML manifest files found in {self.manifests_dir}"
                )
                return False

            self.logger.info(f"Applying {len(manifest_files)} manifest files")

            # Apply each manifest
            for manifest_file in manifest_files:
                if not self._apply_manifest_file(manifest_file):
                    return False

            self.logger.info("All manifests applied successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to apply manifests: {e}")
            return False

    def _apply_manifest_file(self, manifest_file: str) -> bool:
        """Apply a single manifest file.

        Args:
            manifest_file: Path to the manifest file

        Returns:
            True if applied successfully, False otherwise
        """
        try:
            with open(manifest_file, "r") as f:
                manifest_content = f.read()

            # Parse YAML documents (may contain multiple documents)
            for document in yaml.safe_load_all(manifest_content):
                if not document:
                    continue

                self._apply_manifest_object(document)

            self.logger.info(f"Applied manifest: {os.path.basename(manifest_file)}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to apply manifest {manifest_file}: {e}")
            return False

    def _apply_manifest_object(self, manifest: Dict[str, Any]) -> None:
        """Apply a single Kubernetes manifest object.

        Args:
            manifest: Kubernetes manifest as dictionary
        """
        try:
            kind = manifest.get("kind", "").lower()
            api_version = manifest.get("apiVersion", "")
            metadata = manifest.get("metadata", {})
            name = metadata.get("name", "unknown")

            # Track created resources for cleanup
            resource_info = {
                "kind": kind,
                "name": name,
                "namespace": metadata.get("namespace", self.namespace),
            }
            self.created_resources.append(resource_info)

            # Apply based on resource type
            if kind == "job":
                self.batch_client.create_namespaced_job(
                    namespace=resource_info["namespace"], body=manifest
                )
            elif kind == "configmap":
                self.k8s_client.create_namespaced_config_map(
                    namespace=resource_info["namespace"], body=manifest
                )
            elif kind == "namespace":
                self.k8s_client.create_namespace(body=manifest)
            # Add more resource types as needed
            else:
                self.logger.warning(f"Unsupported resource type: {kind}")

            self.logger.debug(f"Applied {kind}/{name}")

        except ApiException as e:
            if e.status == 409:  # Already exists
                self.logger.info(f"Resource {kind}/{name} already exists")
            else:
                raise
        except Exception as e:
            self.logger.error(f"Failed to apply {kind}/{name}: {e}")
            raise

    def _monitor_execution(self) -> List[ExecutionResult]:
        """Monitor execution of applied manifests.

        Returns:
            List of execution results
        """
        try:
            results = []

            # Find all job resources that were created
            job_resources = [r for r in self.created_resources if r["kind"] == "job"]

            if not job_resources:
                self.logger.warning("No jobs found to monitor")
                return results

            self.logger.info(f"Monitoring {len(job_resources)} jobs")

            # Monitor each job
            for job_resource in job_resources:
                result = self._get_job_result(
                    job_resource["name"],
                    job_resource["name"],  # Use job name as node_id
                    "unknown",  # Model tag not available in simplified workflow
                )
                results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"Failed to monitor execution: {e}")
            return []

    def _monitor_jobs(self, workload: WorkloadSpec) -> List[ExecutionResult]:
        """Monitor job execution with timeout and error handling."""
        results = []

        try:
            # Get target nodes
            target_nodes = self.filter_nodes(workload.node_selector)

            # Monitor jobs with timeout
            start_time = time.time()
            timeout = workload.timeout + 60  # Add buffer

            while (time.time() - start_time) < timeout:
                all_completed = True

                for node in target_nodes:
                    for model_tag in workload.model_tags:
                        job_name = f"{self.job_name_prefix}-{node.hostname}-{model_tag}".replace(
                            "_", "-"
                        ).lower()

                        try:
                            # Check if result already exists
                            if any(
                                r.node_id == node.hostname and r.model_tag == model_tag
                                for r in results
                            ):
                                continue

                            # Get job status
                            job = self.batch_client.read_namespaced_job(
                                name=job_name, namespace=self.namespace
                            )

                            if job.status.succeeded:
                                # Job completed successfully
                                result = self._get_job_result(
                                    job_name, node.hostname, model_tag
                                )
                                results.append(result)

                            elif job.status.failed:
                                # Job failed
                                result = ExecutionResult(
                                    node_id=node.hostname,
                                    model_tag=model_tag,
                                    success=False,
                                    error_message="Job failed",
                                )
                                results.append(result)

                            else:
                                # Job still running
                                all_completed = False

                        except client.exceptions.ApiException as e:
                            if e.status == 404:
                                # Job not found
                                result = ExecutionResult(
                                    node_id=node.hostname,
                                    model_tag=model_tag,
                                    success=False,
                                    error_message="Job not found",
                                )
                                results.append(result)
                            else:
                                self.logger.error(f"Error checking job {job_name}: {e}")
                                all_completed = False

                if all_completed:
                    break

                time.sleep(10)  # Check every 10 seconds

            # Handle timeout
            if (time.time() - start_time) >= timeout:
                self.logger.warning("Job monitoring timed out")
                # Add timeout results for missing jobs
                for node in target_nodes:
                    for model_tag in workload.model_tags:
                        if not any(
                            r.node_id == node.hostname and r.model_tag == model_tag
                            for r in results
                        ):
                            result = ExecutionResult(
                                node_id=node.hostname,
                                model_tag=model_tag,
                                success=False,
                                error_message="Job timed out",
                            )
                            results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"Job monitoring failed: {e}")
            return results

    def _get_job_result(
        self, job_name: str, node_id: str, model_tag: str
    ) -> ExecutionResult:
        """Get result from completed job."""
        try:
            # Get pod logs
            pods = self.k8s_client.list_namespaced_pod(
                namespace=self.namespace, label_selector=f"job-name={job_name}"
            )

            if not pods.items:
                return ExecutionResult(
                    node_id=node_id,
                    model_tag=model_tag,
                    success=False,
                    error_message="No pods found for job",
                )

            pod = pods.items[0]

            # Get pod logs
            logs = self.k8s_client.read_namespaced_pod_log(
                name=pod.metadata.name, namespace=self.namespace
            )

            # Parse result from logs
            success = "SUCCESS" in logs

            return ExecutionResult(
                node_id=node_id,
                model_tag=model_tag,
                success=success,
                output=logs,
                error_message=None if success else "Job failed",
            )

        except Exception as e:
            self.logger.error(f"Error getting job result: {e}")
            return ExecutionResult(
                node_id=node_id,
                model_tag=model_tag,
                success=False,
                error_message=str(e),
            )

    def cleanup_infrastructure(self, workload: WorkloadSpec) -> bool:
        """Cleanup infrastructure after execution.

        Args:
            workload: Workload specification

        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            self.logger.info("Cleaning up Kubernetes infrastructure")

            # Run custom cleanup handlers
            for cleanup_handler in self.cleanup_handlers:
                try:
                    cleanup_handler()
                except Exception as e:
                    self.logger.warning(f"Cleanup handler failed: {e}")

            # Clean up created resources
            for resource in self.created_resources:
                try:
                    if resource["type"] == "configmap":
                        self.k8s_client.delete_namespaced_config_map(
                            name=resource["name"], namespace=resource["namespace"]
                        )
                        self.logger.info(f"Deleted ConfigMap: {resource['name']}")
                    elif resource["type"] == "job":
                        self.batch_client.delete_namespaced_job(
                            name=resource["name"], namespace=resource["namespace"]
                        )
                        self.logger.info(f"Deleted Job: {resource['name']}")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to delete resource {resource['name']}: {e}"
                    )

            self.created_resources.clear()

            # Shutdown executor
            if self.executor:
                self.executor.shutdown(wait=True)
                self.executor = None

            self.logger.info("Kubernetes infrastructure cleanup completed")
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
