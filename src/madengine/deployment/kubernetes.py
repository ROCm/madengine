#!/usr/bin/env python3
"""
Kubernetes Deployment - Container orchestration using Python library.

Uses Kubernetes Python client library for type-safe, production-ready deployment.
Requires AMD GPU Device Plugin: https://github.com/ROCm/k8s-device-plugin

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException

    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

from .base import BaseDeployment, DeploymentConfig, DeploymentResult, DeploymentStatus


class KubernetesDeployment(BaseDeployment):
    """
    Kubernetes cluster deployment using Python client library.

    Uses kubernetes Python API for type-safe, production-ready deployment:
    - client.BatchV1Api(): Job creation and management
    - client.CoreV1Api(): Pod logs and status

    Requires AMD GPU Device Plugin: https://github.com/ROCm/k8s-device-plugin

    **Workflow**:
    1. User has kubeconfig configured (in-cluster or ~/.kube/config)
    2. madengine-cli run --tags model --additional-context '{"deploy": "k8s", ...}'
    3. Creates K8s Job using built Docker image from build phase
    4. Job runs madengine workflow inside container (no docker-in-docker)
    """

    DEPLOYMENT_TYPE = "k8s"
    REQUIRED_TOOLS = []  # No CLI tools needed, uses Python library

    def __init__(self, config: DeploymentConfig):
        """
        Initialize Kubernetes deployment.

        Args:
            config: Deployment configuration

        Raises:
            ImportError: If kubernetes Python library not installed
        """
        if not KUBERNETES_AVAILABLE:
            raise ImportError(
                "Kubernetes Python library not installed.\n"
                "Install with: pip install madengine[kubernetes]\n"
                "Or: pip install kubernetes"
            )

        super().__init__(config)

        # Parse K8s configuration
        self.k8s_config = config.additional_context.get("k8s", {})
        if not self.k8s_config:
            self.k8s_config = config.additional_context.get("kubernetes", {})

        self.namespace = self.k8s_config.get("namespace", "default")
        self.gpu_resource_name = self.k8s_config.get("gpu_resource_name", "amd.com/gpu")

        # Load Kubernetes configuration
        kubeconfig_path = self.k8s_config.get("kubeconfig")
        try:
            if kubeconfig_path:
                config.load_kube_config(config_file=kubeconfig_path)
            else:
                # Try in-cluster first, then default kubeconfig
                try:
                    config.load_incluster_config()
                except:
                    config.load_kube_config()
        except Exception as e:
            raise RuntimeError(f"Failed to load Kubernetes config: {e}")

        # Initialize API clients
        self.batch_v1 = client.BatchV1Api()
        self.core_v1 = client.CoreV1Api()

        # Generated Job name
        self.job_name = None
        self.job_manifest = None

    def validate(self) -> bool:
        """Validate Kubernetes cluster access and configuration."""
        try:
            # Test cluster connectivity
            version = client.VersionApi().get_code()
            self.console.print(
                f"[green]✓ Connected to K8s cluster (v{version.major}.{version.minor})[/green]"
            )

            # Check if namespace exists
            try:
                self.core_v1.read_namespace(self.namespace)
                self.console.print(
                    f"[green]✓ Namespace '{self.namespace}' exists[/green]"
                )
            except ApiException as e:
                if e.status == 404:
                    self.console.print(
                        f"[yellow]⚠ Namespace '{self.namespace}' not found[/yellow]"
                    )
                    return False
                raise

            # Validate AMD GPU Device Plugin is deployed
            nodes = self.core_v1.list_node()
            amd_gpu_nodes = [
                n
                for n in nodes.items
                if self.gpu_resource_name in n.status.allocatable
            ]

            if not amd_gpu_nodes:
                self.console.print(
                    f"[yellow]⚠ No nodes with {self.gpu_resource_name} found[/yellow]\n"
                    f"[yellow]  Ensure AMD GPU Device Plugin is deployed:[/yellow]\n"
                    f"[yellow]  kubectl create -f https://raw.githubusercontent.com/ROCm/k8s-device-plugin/master/k8s-ds-amdgpu-dp.yaml[/yellow]"
                )
                return False

            self.console.print(f"[green]✓ Found {len(amd_gpu_nodes)} AMD GPU nodes[/green]")
            return True

        except Exception as e:
            self.console.print(f"[red]✗ Validation failed: {e}[/red]")
            return False

    def prepare(self) -> bool:
        """Prepare K8s Job manifest."""
        try:
            # Get model info
            model_keys = list(self.manifest["built_models"].keys())
            if not model_keys:
                raise ValueError("No models in manifest")

            model_key = model_keys[0]
            model_info = self.manifest["built_models"][model_key]
            image_info = self.manifest["built_images"][model_key]

            # Generate job name (K8s compatible: lowercase, hyphens)
            self.job_name = f"madengine-{model_info['name'].lower().replace('_', '-')}"

            # Build Job manifest using Python objects
            self.job_manifest = self._build_job_manifest(model_info, image_info)

            self.console.print(
                f"[green]✓ Prepared Job manifest: {self.job_name}[/green]"
            )
            return True

        except Exception as e:
            self.console.print(f"[red]✗ Failed to prepare manifest: {e}[/red]")
            return False

    def _build_job_manifest(
        self, model_info: Dict, image_info: Dict
    ) -> Any:
        """Build K8s Job manifest using Python objects (returns client.V1Job)."""
        gpu_count = int(model_info.get("n_gpus", 1))

        # Container specification
        container = client.V1Container(
            name=self.job_name,
            image=image_info["registry_image"],
            image_pull_policy=self.k8s_config.get("image_pull_policy", "Always"),
            working_dir="/workspace",
            command=["/bin/bash", "-c"],
            args=[self._get_container_script(model_info)],
            resources=client.V1ResourceRequirements(
                requests={
                    self.gpu_resource_name: str(gpu_count),
                    "memory": self.k8s_config.get("memory", "128Gi"),
                    "cpu": self.k8s_config.get("cpu", "32"),
                },
                limits={
                    self.gpu_resource_name: str(gpu_count),
                    "memory": self.k8s_config.get("memory_limit", "256Gi"),
                    "cpu": self.k8s_config.get("cpu_limit", "64"),
                },
            ),
            volume_mounts=self._build_volume_mounts(),
        )

        # Pod specification
        pod_spec = client.V1PodSpec(
            restart_policy="Never",
            containers=[container],
            node_selector=self.k8s_config.get("node_selector", {}),
            tolerations=self._build_tolerations(),
            volumes=self._build_volumes(),
        )

        # Job specification
        job_spec = client.V1JobSpec(
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={"app": "madengine", "model": model_info["name"]}
                ),
                spec=pod_spec,
            ),
            backoff_limit=self.k8s_config.get("backoff_limit", 3),
            completions=1,
            parallelism=1,
        )

        # Complete Job object
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=self.job_name,
                namespace=self.namespace,
                labels={
                    "app": "madengine",
                    "model": model_info["name"],
                    "madengine-job": "true",
                },
            ),
            spec=job_spec,
        )

        return job

    def _get_container_script(self, model_info: Dict) -> str:
        """Generate container startup script."""
        return """
        set -e
        echo "MADEngine Kubernetes Job Starting..."
        
        # GPU visibility (AMD GPU Device Plugin handles allocation)
        export ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-0}
        
        # Run MAD model automation workflow
        cd /workspace
        bash run.sh
        
        # Copy results if configured
        if [ -f "perf.csv" ] && [ -d "/results" ]; then
            cp perf.csv /results/perf_${HOSTNAME}.csv
        fi
        
        echo "Job completed with exit code $?"
        """

    def _build_volume_mounts(self) -> List:
        """Build volume mounts from configuration."""
        mounts = []

        if self.k8s_config.get("results_pvc"):
            mounts.append(
                client.V1VolumeMount(name="results", mount_path="/results")
            )

        if self.k8s_config.get("data_pvc"):
            mounts.append(
                client.V1VolumeMount(
                    name="data", mount_path="/data", read_only=True
                )
            )

        return mounts

    def _build_volumes(self) -> List:
        """Build volumes from configuration."""
        volumes = []

        if self.k8s_config.get("results_pvc"):
            volumes.append(
                client.V1Volume(
                    name="results",
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                        claim_name=self.k8s_config["results_pvc"]
                    ),
                )
            )

        if self.k8s_config.get("data_pvc"):
            volumes.append(
                client.V1Volume(
                    name="data",
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                        claim_name=self.k8s_config["data_pvc"]
                    ),
                )
            )

        return volumes

    def _build_tolerations(self) -> List:
        """Build tolerations from configuration."""
        tolerations_config = self.k8s_config.get("tolerations", [])
        tolerations = []

        for tol in tolerations_config:
            tolerations.append(
                client.V1Toleration(
                    key=tol.get("key"),
                    operator=tol.get("operator", "Equal"),
                    value=tol.get("value", ""),
                    effect=tol.get("effect", "NoSchedule"),
                )
            )

        return tolerations

    def deploy(self) -> DeploymentResult:
        """Submit Job to Kubernetes cluster."""
        try:
            # Create Job using Python API
            job = self.batch_v1.create_namespaced_job(
                namespace=self.namespace, body=self.job_manifest
            )

            self.console.print(f"[green]✓ Submitted K8s Job: {self.job_name}[/green]")
            self.console.print(f"  Namespace: {self.namespace}")
            self.console.print(
                f"  Image: {self.job_manifest.spec.template.spec.containers[0].image}"
            )

            return DeploymentResult(
                status=DeploymentStatus.SUCCESS,
                deployment_id=self.job_name,
                message=f"Job {self.job_name} created successfully",
            )

        except ApiException as e:
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id="",
                message=f"K8s API error: {e.reason} - {e.body}",
            )
        except Exception as e:
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id="",
                message=f"Deployment error: {str(e)}",
            )

    def monitor(self, deployment_id: str) -> DeploymentResult:
        """Monitor Job status using Python API."""
        try:
            job = self.batch_v1.read_namespaced_job_status(
                name=deployment_id, namespace=self.namespace
            )

            # Check job conditions
            if job.status.succeeded:
                return DeploymentResult(
                    status=DeploymentStatus.SUCCESS,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} completed successfully",
                )

            if job.status.failed:
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} failed",
                )

            if job.status.active:
                return DeploymentResult(
                    status=DeploymentStatus.RUNNING,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} running ({job.status.active} active pods)",
                )

            return DeploymentResult(
                status=DeploymentStatus.PENDING,
                deployment_id=deployment_id,
                message=f"Job {deployment_id} pending",
            )

        except ApiException as e:
            if e.status == 404:
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} not found",
                )
            raise

    def collect_results(self, deployment_id: str) -> Dict[str, Any]:
        """Collect Job results and logs."""
        results = {
            "job_name": deployment_id,
            "namespace": self.namespace,
            "logs": [],
        }

        try:
            # Get pods for this job
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace, label_selector=f"job-name={deployment_id}"
            )

            # Collect logs from each pod
            for pod in pods.items:
                pod_name = pod.metadata.name
                try:
                    log = self.core_v1.read_namespaced_pod_log(
                        name=pod_name, namespace=self.namespace
                    )
                    results["logs"].append({"pod": pod_name, "log": log})
                except ApiException:
                    pass

            self.console.print(
                f"[green]✓ Collected logs from {len(results['logs'])} pods[/green]"
            )

        except Exception as e:
            self.console.print(f"[yellow]⚠ Results collection incomplete: {e}[/yellow]")

        return results

    def cleanup(self, deployment_id: str) -> bool:
        """Delete Job and associated pods."""
        try:
            # Delete Job (propagates to pods)
            self.batch_v1.delete_namespaced_job(
                name=deployment_id,
                namespace=self.namespace,
                propagation_policy="Background",
            )

            self.console.print(f"[yellow]Deleted K8s Job: {deployment_id}[/yellow]")
            return True

        except ApiException as e:
            if e.status == 404:
                return True  # Already deleted
            self.console.print(f"[yellow]⚠ Cleanup warning: {e.reason}[/yellow]")
            return False
        except Exception as e:
            self.console.print(f"[yellow]⚠ Cleanup error: {e}[/yellow]")
            return False

