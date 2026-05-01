#!/usr/bin/env python3
"""
Kubernetes Deployment - Container orchestration using Jinja2 templates + Python library.

Uses Jinja2 templates for manifest generation (industry best practice) and
Kubernetes Python client library for applying manifests.
Requires a GPU device plugin matching the configured resource name (AMD or NVIDIA).

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from kubernetes import client
    from kubernetes import config as k8s_config
    from kubernetes.client.rest import ApiException

    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .base import BaseDeployment, DeploymentConfig, DeploymentResult, DeploymentStatus, create_jinja_env
from .config_loader import ConfigLoader, apply_deployment_config
from .k8s_names import (
    sanitize_k8s_container_name,
    sanitize_k8s_label_value,
    sanitize_k8s_object_name,
)
from .k8s_pvc import KubernetesPVCMixin
from .k8s_results import KubernetesResultsMixin
from .k8s_scripts import KubernetesScriptsMixin
from .k8s_secrets import (
    SECRETS_STRATEGY_FROM_LOCAL,
    create_or_update_secrets_from_credentials,
    delete_job_secrets_if_exist,
    merge_secrets_config,
)
from .k8s_template_context import KubernetesTemplateContextMixin
from .kubernetes_launcher_mixin import KubernetesLauncherMixin


def _pod_job_name_label_selector(deployment_id: str) -> str:
    """Selector for the ``job-name`` pod label; value must be a valid ≤63-char label value."""
    return f"job-name={sanitize_k8s_label_value(deployment_id)}"


def match_pvc_subdir_to_k8s_pod(
    pvc_subdir: str,
    pod_names: List[str],
    assigned: set,
) -> Optional[str]:
    """
    Map one top-level name under /results/ to a full Kubernetes pod name.

    Matches ``pod == pvc_subdir`` or ``pod.startswith(pvc_subdir + "-")`` among pods
    not yet assigned. Prefer exact equality; if multiple prefix matches, pick the
    first sorted name (deterministic).
    """
    available = sorted(p for p in pod_names if p not in assigned)
    exact = [p for p in available if p == pvc_subdir]
    if exact:
        return exact[0]
    prefixed = [p for p in available if p.startswith(pvc_subdir + "-")]
    if not prefixed:
        return None
    return sorted(prefixed)[0]


def assign_pvc_subdirs_to_pods(pod_dirs: List[str], pod_names: List[str]) -> Dict[str, str]:
    """
    Assign each PVC subdir to at most one pod. Process longest names first so
    short prefixes do not steal pods (e.g. ``foo-0`` before ``foo``).
    """
    cleaned = [d.strip() for d in pod_dirs if d and d.strip()]
    assigned: set = set()
    mapping: Dict[str, str] = {}
    for pvc_subdir in sorted(cleaned, key=lambda x: (-len(x), x)):
        m = match_pvc_subdir_to_k8s_pod(pvc_subdir, pod_names, assigned)
        if m:
            mapping[pvc_subdir] = m
            assigned.add(m)
    return mapping


class KubernetesDeployment(
    KubernetesLauncherMixin,
    KubernetesResultsMixin,
    KubernetesTemplateContextMixin,
    KubernetesScriptsMixin,
    KubernetesPVCMixin,
    BaseDeployment,
):
    """
    Kubernetes cluster deployment using Python client library.

    Uses kubernetes Python API for type-safe, production-ready deployment:
    - client.BatchV1Api(): Job creation and management
    - client.CoreV1Api(): Pod logs and status

    Requires nodes advertising the configured GPU resource (AMD or NVIDIA device plugin).

    **Workflow**:
    1. User has kubeconfig configured (in-cluster or ~/.kube/config)
    2. madengine run --tags model --additional-context '{"deploy": "k8s", ...}'
    3. Creates K8s Job using built Docker image from build phase
    4. Job runs madengine workflow inside container (no docker-in-docker)
    """

    DEPLOYMENT_TYPE = "k8s"
    REQUIRED_TOOLS = []  # No CLI tools needed, uses Python library

    def __init__(self, config: DeploymentConfig):
        """
        Initialize Kubernetes deployment with Jinja2 templates.

        Args:
            config: Deployment configuration

        Raises:
            ImportError: If kubernetes or yaml Python libraries not installed
        """
        if not KUBERNETES_AVAILABLE:
            raise ImportError(
                "Kubernetes Python library not installed.\n"
                "Install with: pip install madengine[kubernetes]\n"
                "Or: pip install kubernetes"
            )

        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML library not installed.\n"
                "Install with: pip install pyyaml"
            )

        apply_deployment_config(config, ConfigLoader.load_k8s_config)
        super().__init__(config)

        # Parse K8s configuration (now with defaults applied)
        self.k8s_config = config.additional_context.get("k8s", {})
        if not self.k8s_config:
            self.k8s_config = config.additional_context.get("kubernetes", {})

        self.namespace = self.k8s_config.get("namespace", "default")
        self.gpu_resource_name = self.k8s_config.get("gpu_resource_name", "amd.com/gpu")

        # Setup Jinja2 template environment
        template_dir = Path(__file__).parent / "templates" / "kubernetes"
        self.jinja_env = create_jinja_env(template_dir)

        # Initialize data provider (will be used if models need data)
        self.data = None
        self.context_for_data = None

        # Load Kubernetes configuration
        kubeconfig_path = self.k8s_config.get("kubeconfig")
        try:
            if kubeconfig_path:
                k8s_config.load_kube_config(config_file=kubeconfig_path)
            else:
                # Try in-cluster first, then default kubeconfig
                try:
                    k8s_config.load_incluster_config()
                except (k8s_config.ConfigException, FileNotFoundError):
                    # Not running in-cluster, try default kubeconfig
                    k8s_config.load_kube_config()
        except Exception as e:
            raise RuntimeError(f"Failed to load Kubernetes config: {e}")

        # Initialize API clients
        self.batch_v1 = client.BatchV1Api()
        self.core_v1 = client.CoreV1Api()

        # Generated resources (see prepare(): K8s uses different constraints per field)
        self.job_name = None
        self.job_label = None  # pod label job-name + label selectors; ≤63 chars (sanitize_k8s_label_value)
        self.service_name = None  # headless Service metadata.name + Pod subdomain; DNS label ≤63 (no dots)
        self.main_container_name = None  # same string as service_name (container names are DNS labels)
        self.configmap_name = None
        self.configmap_yaml = None
        self.job_yaml = None
        self.service_yaml = None

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

            # Validate GPU device plugin: nodes expose the configured GPU resource
            nodes = self.core_v1.list_node()
            gpu_nodes = [
                n
                for n in nodes.items
                if self.gpu_resource_name in (n.status.allocatable or {})
            ]

            if not gpu_nodes:
                vendor = str(
                    self.config.additional_context.get("gpu_vendor", "AMD")
                ).upper()
                if vendor == "NVIDIA":
                    hint = (
                        "[yellow]  Ensure NVIDIA GPU device plugin is installed, e.g.:[/yellow]\n"
                        "[yellow]  https://github.com/NVIDIA/k8s-device-plugin[/yellow]"
                    )
                else:
                    hint = (
                        "[yellow]  Ensure AMD GPU device plugin is deployed, e.g.:[/yellow]\n"
                        "[yellow]  kubectl create -f https://raw.githubusercontent.com/ROCm/k8s-device-plugin/master/k8s-ds-amdgpu-dp.yaml[/yellow]"
                    )
                self.console.print(
                    f"[yellow]⚠ No nodes with {self.gpu_resource_name} found[/yellow]\n"
                    f"{hint}"
                )
                return False

            self.console.print(
                f"[green]✓ Found {len(gpu_nodes)} node(s) with {self.gpu_resource_name}[/green]"
            )
            return True

        except Exception as e:
            self.console.print(f"[red]✗ Validation failed: {e}[/red]")
            return False

    def prepare(self) -> bool:
        """Generate K8s manifests from Jinja2 templates."""
        try:
            # Get model info
            model_keys = list(self.manifest["built_models"].keys())
            if not model_keys:
                raise ValueError("No models in manifest")

            model_key = model_keys[0]
            model_info = self.manifest["built_models"][model_key]
            image_info = self.manifest["built_images"][model_key]

            # Resource names: one logical job, three string forms (job_name alone cannot satisfy every K8s limit).
            # - job_name: Job/PVC/ConfigMap metadata (RFC 1123 subdomain, up to 253 chars; may contain dots).
            # - job_label: value for label job-name on pods + Service selector + API label_selector queries
            #   (≤63; sanitize_k8s_label_value); must stay in sync with _pod_job_name_label_selector(deployment_id).
            # - service_name / main_container_name: single DNS label (≤63, no dots) from
            #   sanitize_k8s_container_name(job_name)—used for headless Service name, Pod subdomain, and the
            #   middle label in Indexed Job DNS ({job_name}-{i}.{service_name}.ns.svc...). Launcher mixin uses
            #   service_name for that subdomain segment via _k8s_headless_subdomain_label.
            raw_model_name = model_info["name"]
            self.job_name = sanitize_k8s_object_name("madengine", raw_model_name)
            self.job_label = sanitize_k8s_label_value(self.job_name)
            self.main_container_name = sanitize_k8s_container_name(self.job_name)
            self.service_name = self.main_container_name
            if any(ch in raw_model_name for ch in "/\\ "):
                self.console.print(
                    f"[dim]K8s resource name (sanitized from model name): {self.job_name}[/dim]"
                )
            if self.main_container_name != self.job_name:
                self.console.print(
                    f"[dim]Pod container name (DNS label, no dots): {self.main_container_name}[/dim]"
                )
            self.configmap_name = f"{self.job_name}-config"

            # Prepare template context
            context = self._prepare_template_context(model_info, image_info)
            self._image_pull_secrets_for_pods = context.get("image_pull_secrets") or []

            # Render ConfigMap template
            configmap_template = self.jinja_env.get_template("configmap.yaml.j2")
            self.configmap_yaml = configmap_template.render(**context)

            # Render Job template
            job_template = self.jinja_env.get_template("job.yaml.j2")
            self.job_yaml = job_template.render(**context)

            # Optionally render Service template (for multi-node torchrun)
            if context.get("create_headless_service"):
                service_template = self.jinja_env.get_template("service.yaml.j2")
                self.service_yaml = service_template.render(**context)

            # Debug mode: save rendered manifests
            if self.config.additional_context.get("debug", False):
                self._save_debug_manifests()

            self.console.print(
                f"[green]✓ Prepared K8s manifests: {self.job_name}[/green]"
            )
            return True

        except Exception as e:
            self.console.print(f"[red]✗ Failed to prepare manifests: {e}[/red]")
            import traceback

            traceback.print_exc()
            return False

    def _save_debug_manifests(self):
        """Save rendered manifests to disk for debugging."""
        output_dir = Path(self.k8s_config.get("output_dir", "./k8s_manifests"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save ConfigMap
        (output_dir / "configmap.yaml").write_text(self.configmap_yaml)

        # Save Job
        (output_dir / "job.yaml").write_text(self.job_yaml)

        # Save Service if exists
        if self.service_yaml:
            (output_dir / "service.yaml").write_text(self.service_yaml)

        self.console.print(
            f"[yellow]Debug: Manifests saved to {output_dir}[/yellow]"
        )

    
    def _cleanup_existing_resources(self):
        """Delete existing Job, ConfigMap, and Service if they exist."""
        # Delete existing Job
        try:
            self.batch_v1.delete_namespaced_job(
                name=self.job_name,
                namespace=self.namespace,
                propagation_policy="Background"
            )
            self.console.print(f"[dim]Deleted existing Job: {self.job_name}[/dim]")
        except ApiException as e:
            if e.status != 404:  # Ignore not found
                pass

        try:
            delete_job_secrets_if_exist(self.core_v1, self.namespace, self.job_name)
            self.console.print(
                f"[dim]Removed job-scoped Secrets for {self.job_name} (if present)[/dim]"
            )
        except ApiException:
            pass
        
        # Delete existing ConfigMap
        try:
            self.core_v1.delete_namespaced_config_map(
                name=self.configmap_name,
                namespace=self.namespace
            )
            self.console.print(f"[dim]Deleted existing ConfigMap: {self.configmap_name}[/dim]")
        except ApiException as e:
            if e.status != 404:
                pass
        
        # Delete existing Service
        if hasattr(self, 'service_yaml') and self.service_yaml:
            try:
                self.core_v1.delete_namespaced_service(
                    name=self.service_name,
                    namespace=self.namespace
                )
                self.console.print(f"[dim]Deleted existing Service: {self.service_name}[/dim]")
            except ApiException as e:
                if e.status != 404:
                    pass
        
        # Delete existing collector pod (must be done before PVC to allow PVC deletion)
        collector_pod_name = f"collector-{self.job_name}"
        try:
            self.core_v1.delete_namespaced_pod(
                name=collector_pod_name,
                namespace=self.namespace,
                grace_period_seconds=0
            )
            self.console.print(f"[dim]Deleted existing collector pod: {collector_pod_name}[/dim]")
            # Wait a moment for pod to release the PVC
            time.sleep(2)
        except ApiException as e:
            if e.status != 404:
                pass
        
        # Delete existing PVC
        pvc_name = f"{self.job_name}-results"
        try:
            self.core_v1.delete_namespaced_persistent_volume_claim(
                name=pvc_name,
                namespace=self.namespace
            )
            self.console.print(f"[dim]Deleted existing PVC: {pvc_name}[/dim]")
            
            # Wait for PVC to be fully deleted (not just marked for deletion)
            max_wait = 90  # Maximum 90 seconds (PV can take time to detach)
            wait_interval = 1  # Check every 1 second
            for i in range(max_wait):
                try:
                    self.core_v1.read_namespaced_persistent_volume_claim(
                        name=pvc_name,
                        namespace=self.namespace
                    )
                    if i > 0 and i % 10 == 0:
                        self.console.print(
                            f"[dim]Waiting for PVC {pvc_name} to be removed... ({i}s)[/dim]"
                        )
                    time.sleep(wait_interval)
                except ApiException as e:
                    if e.status == 404:
                        # PVC is fully deleted
                        break
        except ApiException as e:
            if e.status != 404:
                pass
        
        # Wait a moment for other resources to be deleted
        time.sleep(1)

    def deploy(self) -> DeploymentResult:
        """Apply rendered manifests using kubernetes Python client."""
        try:
            # Clean up any existing resources first
            self._cleanup_existing_resources()
            
            # 1. Create PVC for results storage
            self.console.print("[blue]Creating PVC for results storage...[/blue]")
            nnodes_deploy = getattr(self, "_nnodes", 1)
            pvc_name = self._create_results_pvc(nnodes=nnodes_deploy)
            self.console.print(f"[green]✓ Created PVC: {pvc_name}[/green]")
            
            # 1b. Create or reuse data PVC if data provider is configured and auto-creation was flagged
            if hasattr(self, '_data_config') and self._data_config:
                # Check if we set the PVC name during prepare (auto-creation case)
                data_pvc_name = self.k8s_config.get("data_pvc")
                if data_pvc_name == "madengine-shared-data":
                    # Auto-creation mode: create/reuse the PVC
                    nnodes = getattr(self, '_nnodes', 1)
                    self._create_or_get_data_pvc(nnodes=nnodes)
            
            # 2. Create Secrets from local credential.json (strategy: from_local_credentials)
            merged_sec = merge_secrets_config(self.k8s_config)
            strategy = merged_sec.get("strategy", SECRETS_STRATEGY_FROM_LOCAL)
            cred_path = Path("credential.json")
            if strategy == SECRETS_STRATEGY_FROM_LOCAL and cred_path.exists():
                self.console.print(
                    "[blue]Creating Kubernetes Secrets from credential.json...[/blue]"
                )
                create_or_update_secrets_from_credentials(
                    self.core_v1, self.namespace, self.job_name, cred_path
                )
                self.console.print("[green]✓ Applied registry/runtime Secrets[/green]")

            # 3. Create ConfigMap
            self.console.print("[blue]Creating ConfigMap...[/blue]")
            configmap_dict = yaml.safe_load(self.configmap_yaml)
            self.core_v1.create_namespaced_config_map(
                namespace=self.namespace, body=configmap_dict
            )
            self.console.print(
                f"[green]✓ Created ConfigMap: {self.configmap_name}[/green]"
            )

            # 4. Create Service (if needed for multi-node)
            if self.service_yaml:
                self.console.print("[blue]Creating headless Service...[/blue]")
                service_dict = yaml.safe_load(self.service_yaml)
                self.core_v1.create_namespaced_service(
                    namespace=self.namespace, body=service_dict
                )
                self.console.print(f"[green]✓ Created Service: {self.service_name}[/green]")

            # 5. Create Job
            self.console.print("[blue]Creating Job...[/blue]")
            job_dict = yaml.safe_load(self.job_yaml)
            job = self.batch_v1.create_namespaced_job(
                namespace=self.namespace, body=job_dict
            )

            # Extract image for display
            image = job_dict["spec"]["template"]["spec"]["containers"][0]["image"]

            self.console.print(f"[green]✓ Submitted K8s Job: {self.job_name}[/green]")
            self.console.print(f"  Namespace: {self.namespace}")
            self.console.print(f"  Image: {image}")

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
        """
        Monitor Job status using Python API.
        
        If live_output is enabled, streams pod logs in real-time.
        Otherwise, polls status periodically.
        """
        # Check if live output is requested
        live_output = self.config.additional_context.get("live_output", False)
        
        if live_output:
            return self._monitor_with_live_logs(deployment_id)
        else:
            return self._monitor_status_only(deployment_id)
    
    def _monitor_status_only(self, deployment_id: str) -> DeploymentResult:
        """Monitor Job status without streaming logs."""
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
                # Get pod logs to show error
                self._print_pod_logs_on_failure(deployment_id)
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
    
    def _monitor_with_live_logs(self, deployment_id: str) -> DeploymentResult:
        """Monitor Job and stream logs in real-time."""
        self.console.print(f"\n[cyan]═══ Streaming pod logs (--live-output) ═══[/cyan]\n")
        
        pod_name = None
        log_position = 0
        
        while True:
            try:
                # Check job status
                job = self.batch_v1.read_namespaced_job_status(
                    name=deployment_id, namespace=self.namespace
                )
                
                # Get pod if we don't have it yet
                if not pod_name:
                    pods = self.core_v1.list_namespaced_pod(
                        namespace=self.namespace,
                        label_selector=_pod_job_name_label_selector(deployment_id),
                    )
                    if pods.items:
                        pod_name = pods.items[0].metadata.name
                        self.console.print(f"[dim]Following logs from pod: {pod_name}[/dim]\n")
                
                # Stream logs if we have a pod
                if pod_name:
                    try:
                        # Get logs from current position
                        logs = self.core_v1.read_namespaced_pod_log(
                            name=pod_name,
                            namespace=self.namespace,
                            tail_lines=100 if log_position == 0 else None
                        )
                        
                        # Print new log lines and trigger artifact collection
                        if logs:
                            log_lines = logs.split('\n')
                            if len(log_lines) > log_position:
                                for line in log_lines[log_position:]:
                                    if line.strip():
                                        print(line)
                                log_position = len(log_lines)
                    
                    except ApiException as e:
                        if e.status != 400:  # Ignore "container not ready" errors
                            pass
                
                # Check if job completed
                if job.status.succeeded:
                    self.console.print(f"\n[green]✓ Job {deployment_id} completed successfully[/green]\n")
                    return DeploymentResult(
                        status=DeploymentStatus.SUCCESS,
                        deployment_id=deployment_id,
                        message=f"Job {deployment_id} completed successfully",
                    )
                
                if job.status.failed:
                    self.console.print(f"\n[red]✗ Job {deployment_id} failed[/red]\n")
                    # Print final logs
                    if pod_name:
                        self._print_pod_logs_on_failure(deployment_id)
                    return DeploymentResult(
                        status=DeploymentStatus.FAILED,
                        deployment_id=deployment_id,
                        message=f"Job {deployment_id} failed",
                    )
                
                time.sleep(2)  # Poll every 2 seconds
                
            except ApiException as e:
                if e.status == 404:
                    return DeploymentResult(
                        status=DeploymentStatus.FAILED,
                        deployment_id=deployment_id,
                        message=f"Job {deployment_id} not found",
                    )
                raise
    
    def _print_pod_logs_on_failure(self, deployment_id: str):
        """Print pod logs when job fails (for debugging)."""
        try:
            self.console.print(f"\n[yellow]═══ Pod logs (last 50 lines) ═══[/yellow]\n")
            
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=_pod_job_name_label_selector(deployment_id),
            )
            
            for pod in pods.items:
                pod_name = pod.metadata.name
                try:
                    logs = self.core_v1.read_namespaced_pod_log(
                        name=pod_name,
                        namespace=self.namespace,
                        tail_lines=50
                    )
                    self.console.print(f"[dim]Pod: {pod_name}[/dim]")
                    print(logs)
                    print()
                except ApiException:
                    pass
        except Exception:
            pass

    def _primary_workload_container_exit_code(self, pod: Any) -> int:
        """
        Exit code of the primary workload container (spec.containers[0]), matched by name
        against container_statuses (ordering-safe if sidecars are added later).
        """
        if not pod.spec or not pod.spec.containers:
            return 0
        primary_name = pod.spec.containers[0].name
        for cs in pod.status.container_statuses or []:
            if cs.name == primary_name and cs.state and cs.state.terminated:
                return cs.state.terminated.exit_code or 0
        # Fallback: first terminated container in spec order
        name_order = [c.name for c in pod.spec.containers]
        for want in name_order:
            for cs in pod.status.container_statuses or []:
                if cs.name == want and cs.state and cs.state.terminated:
                    return cs.state.terminated.exit_code or 0
        return 0

    def _refresh_pod_until_terminal_phase(
        self,
        pod_name: str,
        *,
        timeout_seconds: float = 30.0,
        interval_seconds: float = 0.5,
    ) -> Any:
        """
        Poll read_namespaced_pod until phase is Succeeded or Failed, or timeout.
        Avoids stale list/single-get right after Job completion (phase still Running).
        """
        deadline = time.monotonic() + timeout_seconds
        last: Any = None
        while time.monotonic() < deadline:
            last = self.core_v1.read_namespaced_pod(
                name=pod_name, namespace=self.namespace
            )
            phase = last.status.phase
            if phase in ("Succeeded", "Failed"):
                return last
            time.sleep(interval_seconds)
        return last

    def cleanup(self, deployment_id: str) -> bool:
        """Delete Job, ConfigMap, Service and associated pods."""
        success = True

        try:
            # Delete Job (propagates to pods)
            self.batch_v1.delete_namespaced_job(
                name=deployment_id,
                namespace=self.namespace,
                propagation_policy="Background",
            )
            self.console.print(f"[yellow]Deleted K8s Job: {deployment_id}[/yellow]")
        except ApiException as e:
            if e.status != 404:
                self.console.print(f"[yellow]⚠ Job cleanup warning: {e.reason}[/yellow]")
                success = False
        except Exception as e:
            self.console.print(f"[yellow]⚠ Job cleanup error: {e}[/yellow]")
            success = False

        # Delete ConfigMap
        try:
            configmap_name = f"{deployment_id}-config"
            self.core_v1.delete_namespaced_config_map(
                name=configmap_name, namespace=self.namespace
            )
            self.console.print(
                f"[yellow]Deleted ConfigMap: {configmap_name}[/yellow]"
            )
        except ApiException as e:
            if e.status != 404:
                self.console.print(
                    f"[yellow]⚠ ConfigMap cleanup warning: {e.reason}[/yellow]"
                )
        except Exception:
            pass

        # Delete Service (if exists); name is DNS-label headless svc, not always full Job name
        try:
            svc_name = sanitize_k8s_container_name(deployment_id)
            self.core_v1.delete_namespaced_service(
                name=svc_name, namespace=self.namespace
            )
            self.console.print(f"[yellow]Deleted Service: {svc_name}[/yellow]")
        except ApiException as e:
            if e.status != 404:
                pass  # Service may not exist for single-node jobs
        except Exception:
            pass

        return success

