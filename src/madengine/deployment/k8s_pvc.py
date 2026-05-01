"""
Kubernetes PVC lifecycle management mixin.

Handles PersistentVolumeClaim creation, deletion, and storage class
resolution for both per-job results and long-lived shared data volumes.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import time
from pathlib import Path
from typing import Optional

from jinja2 import Template

try:
    from kubernetes.client.rest import ApiException

    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class KubernetesPVCMixin:
    """PVC lifecycle management for Kubernetes deployments."""

    def _k8s_data_storage_class(self) -> Optional[str]:
        """StorageClass for long-lived ``madengine-shared-data`` (NFS RWX recommended)."""
        return (
            self.k8s_config.get("data_storage_class")
            or self.k8s_config.get("nfs_storage_class")
            or self.k8s_config.get("storage_class")
        )

    def _k8s_results_storage_class(self, nnodes: int) -> Optional[str]:
        """
        Per-job results: local-path (RWO) for single-node, NFS (RWX) for multi-node.

        Falls back to ``storage_class`` for backward compatibility.
        """
        if nnodes > 1:
            return (
                self.k8s_config.get("multi_node_results_storage_class")
                or self.k8s_config.get("nfs_storage_class")
                or self.k8s_config.get("storage_class")
            )
        return (
            self.k8s_config.get("single_node_results_storage_class")
            or self.k8s_config.get("local_path_storage_class")
            or self.k8s_config.get("storage_class")
        )

    def _create_results_pvc(self, nnodes: int = 1) -> str:
        """
        Create a PersistentVolumeClaim for per-job results.

        Single-node uses ReadWriteOnce (typically local-path). Multi-node uses
        ReadWriteMany (typically nfs-banff or other RWX class).
        """
        pvc_name = f"{self.job_name}-results"
        access_mode = "ReadWriteMany" if nnodes > 1 else "ReadWriteOnce"
        storage_class = self._k8s_results_storage_class(nnodes)

        template_dir = Path(__file__).parent / "templates" / "kubernetes"
        pvc_template = template_dir / "pvc.yaml.j2"

        with open(pvc_template, "r") as f:
            pvc_template_str = f.read()

        template = Template(pvc_template_str)
        self.console.print(
            f"[dim]  Results PVC: access={access_mode}, "
            f"storageClass={storage_class or '(cluster default)'}[/dim]"
        )
        if nnodes > 1 and not storage_class:
            self.console.print(
                "[yellow]⚠️  Multi-node: set k8s.nfs_storage_class or "
                "multi_node_results_storage_class to an RWX class (e.g. nfs-banff).[/yellow]"
            )
        pvc_yaml = template.render(
            pvc_name=pvc_name,
            namespace=self.namespace,
            access_mode=access_mode,
            storage_size=self.k8s_config.get("results_storage_size", "10Gi"),
            storage_class=storage_class,
        )

        # Create PVC (retry on 409 "object is being deleted" until it is gone)
        pvc_dict = yaml.safe_load(pvc_yaml)
        max_create_retries = 6
        create_wait_seconds = 5
        for attempt in range(max_create_retries):
            try:
                self.core_v1.create_namespaced_persistent_volume_claim(
                    namespace=self.namespace, body=pvc_dict
                )
                return pvc_name
            except ApiException as e:
                if e.status == 409 and e.body and "object is being deleted" in (e.body or ""):
                    if attempt < max_create_retries - 1:
                        self.console.print(
                            f"[dim]PVC still terminating, waiting {create_wait_seconds}s before retry ({attempt + 1}/{max_create_retries})[/dim]"
                        )
                        time.sleep(create_wait_seconds)
                    else:
                        raise
                else:
                    raise

    def _wait_for_pvc_deleted(self, pvc_name: str, max_wait: int = 90) -> None:
        """Block until the PVC is fully removed (or timeout)."""
        for i in range(max_wait):
            try:
                self.core_v1.read_namespaced_persistent_volume_claim(
                    name=pvc_name, namespace=self.namespace
                )
                if i > 0 and i % 10 == 0:
                    self.console.print(
                        f"[dim]Waiting for PVC {pvc_name} to be removed... ({i}s)[/dim]"
                    )
                time.sleep(1)
            except ApiException as e:
                if e.status == 404:
                    return
                raise

    def _create_or_get_data_pvc(self, nnodes: int = 1) -> str:
        """
        Create or reuse ``madengine-shared-data`` for long-lived datasets (cache).

        Always uses ReadWriteMany + an NFS-style StorageClass so the same PVC
        works for single- and multi-pod jobs. Use ``data_storage_class`` or
        ``nfs_storage_class`` (e.g. nfs-banff), not local-path.

        Args:
            nnodes: Reserved for logging (shared-data access mode does not depend on it).

        Returns:
            Name of the PVC (existing or newly created)
        """
        pvc_name = "madengine-shared-data"

        if self.k8s_config.get("recreate_shared_data_pvc"):
            try:
                self.core_v1.delete_namespaced_persistent_volume_claim(
                    name=pvc_name, namespace=self.namespace
                )
                self.console.print(
                    "[yellow]recreate_shared_data_pvc: deleted existing "
                    f"{pvc_name} (backup data first if needed)[/yellow]"
                )
                self._wait_for_pvc_deleted(pvc_name)
            except ApiException as e:
                if e.status != 404:
                    raise

        try:
            existing_pvc = self.core_v1.read_namespaced_persistent_volume_claim(
                name=pvc_name,
                namespace=self.namespace,
            )
            self.console.print(f"[dim]✓ Using existing data PVC: {pvc_name}[/dim]")

            access_modes = existing_pvc.spec.access_modes or []
            if "ReadWriteMany" not in access_modes:
                self.console.print(
                    f"[yellow]⚠️  Warning: {pvc_name} is not ReadWriteMany "
                    f"(modes: {access_modes}).[/yellow]"
                )
                self.console.print(
                    "[yellow]   For NFS-backed long-lived data, delete the PVC and re-run with "
                    "k8s.data_storage_class / nfs_storage_class set, or use "
                    "recreate_shared_data_pvc (after backup).[/yellow]"
                )
            return pvc_name

        except ApiException as e:
            if e.status != 404:
                raise

        access_mode = "ReadWriteMany"
        storage_class = self._k8s_data_storage_class()
        self.console.print(f"[blue]Creating shared data PVC: {pvc_name}...[/blue]")
        self.console.print(
            f"[dim]  Access mode: {access_mode}; storageClass={storage_class or '(cluster default)'}; "
            f"nnodes={nnodes}[/dim]"
        )
        if not storage_class:
            self.console.print(
                "[yellow]⚠️  Set k8s.nfs_storage_class or data_storage_class to an RWX class "
                "(e.g. nfs-banff) for shared-data. Default SC may be local-path (RWO-only).[/yellow]"
            )

        template_dir = Path(__file__).parent / "templates" / "kubernetes"
        pvc_template = template_dir / "pvc-data.yaml.j2"

        with open(pvc_template, "r") as f:
            pvc_template_str = f.read()

        template = Template(pvc_template_str)
        pvc_yaml = template.render(
            pvc_name=pvc_name,
            namespace=self.namespace,
            access_mode=access_mode,
            storage_size=self.k8s_config.get("data_storage_size", "100Gi"),
            storage_class=storage_class,
        )

        pvc_dict = yaml.safe_load(pvc_yaml)
        self.core_v1.create_namespaced_persistent_volume_claim(
            namespace=self.namespace, body=pvc_dict
        )

        self.console.print("[dim]Waiting for PVC to be bound...[/dim]")
        for _ in range(30):
            try:
                pvc = self.core_v1.read_namespaced_persistent_volume_claim(
                    name=pvc_name, namespace=self.namespace
                )
                if pvc.status.phase == "Bound":
                    self.console.print("[green]✓ PVC bound successfully[/green]")
                    break
            except ApiException:
                pass
            time.sleep(1)
        else:
            self.console.print(
                f"[yellow]⚠️  Warning: PVC created but not bound yet. "
                f"Check: kubectl describe pvc {pvc_name}[/yellow]"
            )

        return pvc_name
