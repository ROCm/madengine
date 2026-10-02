#!/usr/bin/env python3
"""
Kubernetes Deployment - Container orchestration using Jinja2 templates + Python library.

Uses Jinja2 templates for manifest generation (industry best practice) and
Kubernetes Python client library for applying manifests.
Requires a GPU device plugin matching the configured resource name (AMD or NVIDIA).

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
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
from .k8s_results import KubernetesResultsMixin, collector_pod_name
from .k8s_scripts import KubernetesScriptsMixin
from .k8s_secrets import (
    job_secret_names,
    SECRETS_STRATEGY_FROM_LOCAL,
    create_or_update_secrets_from_credentials,
    delete_job_secrets_if_exist,
    merge_secrets_config,
)
from .k8s_slurm_multi import KubernetesSlurmMultiMixin, is_slurm_multi
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
    KubernetesSlurmMultiMixin,
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
                "Reinstall madengine (pip install madengine) or install the "
                "library directly with: pip install kubernetes"
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

        # Load Kubernetes configuration. One kubeconfig file and one context are
        # resolved here and used by BOTH the Python client and every kubectl call
        # (see kubectl_cmd). They used to be chosen separately -- the client read
        # k8s.kubeconfig (default ~/.kube/config) at that file's current context,
        # while kubectl followed $KUBECONFIG and its own current context -- so on a
        # machine that can reach two clusters, a run could deploy to one and
        # copy results from, or exec into, the other.
        self.kubeconfig_path, self.kube_context = self._resolve_kube_target()
        try:
            if self.kubeconfig_path or self.kube_context:
                k8s_config.load_kube_config(
                    config_file=self.kubeconfig_path, context=self.kube_context
                )
            else:
                # No kubeconfig anywhere: running in-cluster on a service account.
                k8s_config.load_incluster_config()
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

    def _resolve_kube_target(self):
        """(kubeconfig file, context) this deployment talks to.

        File: k8s.kubeconfig if set, else $KUBECONFIG, else ~/.kube/config; None
        when none exists, which means in-cluster. Context: k8s.context if set --
        and it must exist in that file, or this refuses rather than fall back to
        whatever the file's current context is -- else the file's current context,
        pinned here so kubectl cannot drift from it.
        """
        path = self.k8s_config.get("kubeconfig") or os.environ.get("KUBECONFIG") or ""
        if not path and Path("~/.kube/config").expanduser().exists():
            path = "~/.kube/config"
        # $KUBECONFIG may list several files; the client and kubectl both merge them.
        path = os.pathsep.join(str(Path(p).expanduser()) for p in path.split(os.pathsep) if p)
        requested = self.k8s_config.get("context")
        if not path:
            if requested:
                raise RuntimeError(
                    f"k8s.context is '{requested}' but no kubeconfig was found "
                    "(k8s.kubeconfig, $KUBECONFIG, ~/.kube/config)."
                )
            return None, None
        single = None if os.pathsep in path else path
        try:
            contexts, current = k8s_config.list_kube_config_contexts(config_file=single)
        except Exception as e:
            raise RuntimeError(f"Failed to read kubeconfig {path}: {e}")
        names = [c["name"] for c in contexts or []]
        if requested:
            if requested not in names:
                raise RuntimeError(
                    f"k8s.context '{requested}' is not in {path}; it has: "
                    f"{', '.join(names) or '(none)'}. Refusing to fall back to another context."
                )
            return single, requested
        return single, (current or {}).get("name")

    def kubectl_cmd(self, *args: str) -> List[str]:
        """A kubectl command line pinned to this deployment's kubeconfig and context."""
        cmd = ["kubectl"]
        if self.kubeconfig_path:
            cmd += ["--kubeconfig", self.kubeconfig_path]
        if self.kube_context:
            cmd += ["--context", self.kube_context]
        return cmd + list(args)

    def _delete_if_ours(self, kind: str, name: str, read, delete, legacy_unlabelled: bool = False) -> bool:
        """Delete a same-named leftover from an earlier run, only if madengine made it.

        Everything madengine creates carries app=madengine. An object of the same
        name without it belongs to someone else sharing the namespace: stop with
        the reason instead of deleting it. legacy_unlabelled accepts objects of
        the two kinds older madengine created without the label (Secrets and the
        collector pod, both under madengine's own job-derived names).
        """
        try:
            obj = read()
        except ApiException as e:
            if e.status == 404:
                return False
            raise
        labels = (obj.metadata.labels or {}) if obj.metadata else {}
        if labels.get("app") != "madengine" and not (legacy_unlabelled and not labels):
            raise RuntimeError(
                f"{kind} '{name}' already exists in namespace '{self.namespace}' and was not "
                "created by madengine (no app=madengine label). Refusing to delete it: use "
                "another k8s.namespace, or remove it yourself if it is yours."
            )
        delete()
        return True

    def validate(self) -> bool:
        """Validate Kubernetes cluster access and configuration."""
        try:
            # Test cluster connectivity
            version = client.VersionApi().get_code()
            self.console.print(
                f"[green]✓ Connected to K8s cluster (v{version.major}.{version.minor})[/green]"
            )
            self.console.print(
                f"  context: {self.kube_context or '(in-cluster)'}"
                f"  kubeconfig: {self.kubeconfig_path or '(in-cluster)'}"
                f"  namespace: {self.namespace}"
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

            # A slurm_multi card runs its own SLURM launcher, unchanged, across one
            # pod per node (k8s_slurm_multi.py); it has its own manifests.
            self._slurm_multi = is_slurm_multi(
                model_info, self.config.additional_context.get("distributed") or {}
            )
            if self._slurm_multi:
                built = self._prepare_slurm_multi(model_key, model_info, image_info)
                self.configmap_yaml = yaml.safe_dump(built["configmap"], sort_keys=False)
                self.job_yaml = yaml.safe_dump(built["job"], sort_keys=False)
                self.service_yaml = yaml.safe_dump(built["service"], sort_keys=False)
                self._nnodes = built["nnodes"]
                if self.config.additional_context.get("debug", False):
                    self._save_debug_manifests()
                self.console.print(
                    f"[green]✓ Prepared K8s manifests: {self.job_name} (slurm_multi: "
                    f"{built['nnodes']} pod(s) x {built['gpus']} GPU(s), image {built['image']})[/green]"
                )
                return True

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
        """Delete this job's leftovers from an earlier run -- only what madengine created.

        Names are deterministic (madengine-<model>), so a rerun replaces its own
        Job, ConfigMap, Service, Secrets, collector pod and results PVC. An object
        with the same name that madengine did not create stops the deploy (see
        _delete_if_ours) instead of being deleted.
        """
        ns = self.namespace
        if self._delete_if_ours(
            "Job", self.job_name,
            lambda: self.batch_v1.read_namespaced_job(self.job_name, ns),
            lambda: self.batch_v1.delete_namespaced_job(
                name=self.job_name, namespace=ns, propagation_policy="Background"),
        ):
            self.console.print(f"[dim]Deleted existing Job: {self.job_name}[/dim]")

        for secret in job_secret_names(self.job_name):
            if self._delete_if_ours(
                "Secret", secret,
                lambda secret=secret: self.core_v1.read_namespaced_secret(secret, ns),
                lambda secret=secret: self.core_v1.delete_namespaced_secret(name=secret, namespace=ns),
                legacy_unlabelled=True,
            ):
                self.console.print(f"[dim]Deleted existing Secret: {secret}[/dim]")

        if self._delete_if_ours(
            "ConfigMap", self.configmap_name,
            lambda: self.core_v1.read_namespaced_config_map(self.configmap_name, ns),
            lambda: self.core_v1.delete_namespaced_config_map(name=self.configmap_name, namespace=ns),
        ):
            self.console.print(f"[dim]Deleted existing ConfigMap: {self.configmap_name}[/dim]")

        if getattr(self, "service_yaml", None):
            if self._delete_if_ours(
                "Service", self.service_name,
                lambda: self.core_v1.read_namespaced_service(self.service_name, ns),
                lambda: self.core_v1.delete_namespaced_service(name=self.service_name, namespace=ns),
            ):
                self.console.print(f"[dim]Deleted existing Service: {self.service_name}[/dim]")

        # The collector pod goes before the PVC it mounts, so the PVC can be released.
        coll_pod_name = collector_pod_name(self.job_name)
        if self._delete_if_ours(
            "Pod", coll_pod_name,
            lambda: self.core_v1.read_namespaced_pod(coll_pod_name, ns),
            lambda: self.core_v1.delete_namespaced_pod(
                name=coll_pod_name, namespace=ns, grace_period_seconds=0),
            legacy_unlabelled=True,
        ):
            self.console.print(f"[dim]Deleted existing collector pod: {coll_pod_name}[/dim]")
            time.sleep(2)

        pvc_name = f"{self.job_name}-results"
        if self._delete_if_ours(
            "PersistentVolumeClaim", pvc_name,
            lambda: self.core_v1.read_namespaced_persistent_volume_claim(pvc_name, ns),
            lambda: self.core_v1.delete_namespaced_persistent_volume_claim(name=pvc_name, namespace=ns),
        ):
            self.console.print(f"[dim]Deleted existing PVC: {pvc_name}[/dim]")
            # Wait for it to be gone, not just marked for deletion (a PV can take time to detach).
            for i in range(90):
                try:
                    self.core_v1.read_namespaced_persistent_volume_claim(name=pvc_name, namespace=ns)
                except ApiException as e:
                    if e.status == 404:
                        break
                if i > 0 and i % 10 == 0:
                    self.console.print(f"[dim]Waiting for PVC {pvc_name} to be removed... ({i}s)[/dim]")
                time.sleep(1)

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

    @staticmethod
    def _job_outcome(job) -> Optional[str]:
        """'succeeded', 'failed', or None while the Job is still running.

        Read from the Job's own conditions, not its pod counters: `succeeded` is
        non-zero as soon as ONE pod finishes (a multi-node Job is not done then),
        and `failed` is non-zero while retries under backoffLimit are still to
        come. Counter fallback for an API server that sets no conditions.
        """
        st = job.status
        for c in (st.conditions or []) if st else []:
            if c.status == "True" and c.type in ("Complete", "SuccessCriteriaMet"):
                return "succeeded"
            if c.status == "True" and c.type in ("Failed", "FailureTarget"):
                return "failed"
        completions = (job.spec.completions if job.spec and job.spec.completions else 1)
        if st and (st.succeeded or 0) >= completions:
            return "succeeded"
        backoff = job.spec.backoff_limit if job.spec and job.spec.backoff_limit is not None else 6
        if st and (st.failed or 0) > backoff:
            return "failed"
        return None

    def _monitor_status_only(self, deployment_id: str) -> DeploymentResult:
        """Monitor Job status without streaming logs."""
        try:
            job = self.batch_v1.read_namespaced_job_status(
                name=deployment_id, namespace=self.namespace
            )

            outcome = self._job_outcome(job)
            if outcome == "succeeded":
                return DeploymentResult(
                    status=DeploymentStatus.SUCCESS,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} completed successfully",
                )

            if outcome == "failed":
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
                outcome = self._job_outcome(job)
                if outcome == "succeeded":
                    self.console.print(f"\n[green]✓ Job {deployment_id} completed successfully[/green]\n")
                    return DeploymentResult(
                        status=DeploymentStatus.SUCCESS,
                        deployment_id=deployment_id,
                        message=f"Job {deployment_id} completed successfully",
                    )

                if outcome == "failed":
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
