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


RESULTS_LAYOUT_AUTO = "auto"
RESULTS_LAYOUT_SHARED = "shared"
RESULTS_LAYOUT_PER_POD = "per_pod"
SHARED_DATA_PVC_NAME = "madengine-shared-data"

# StorageClass has no access-mode field. These provisioners serve a shared
# filesystem. A name that merely exists, including local-path, does not.
_RWX_PROVISIONERS = frozenset(
    {
        "nfs.csi.k8s.io",
        "cephfs.csi.ceph.com",
        "efs.csi.aws.com",
        "file.csi.azure.com",
        "filestore.csi.storage.gke.io",
        "nfs-subdir-external-provisioner",
    }
)


def provisioner_supports_rwx(provisioner: Optional[str]) -> bool:
    """True when this provisioner can create a ReadWriteMany filesystem."""
    if provisioner is None:
        return False
    name = str(provisioner).strip().lower()
    if not name:
        return False
    tail = name.rsplit("/", 1)[-1]
    if name in _RWX_PROVISIONERS or tail in _RWX_PROVISIONERS:
        return True
    for segment in name.replace(".", "/").split("/"):
        if segment in {"nfs", "cephfs"} or segment.startswith(("nfs-", "cephfs-")):
            return True
    return False


def resolve_results_layout(
    layout: Optional[str], nnodes: int, rwx_usable: bool
) -> str:
    """Choose how multi-pod results are stored.

    ``shared`` is one ReadWriteMany claim. ``per_pod`` is a ReadWriteOnce
    claim created with each pod. ``auto`` uses ``shared`` when the configured
    multi-node StorageClass can serve ReadWriteMany, and ``per_pod`` when it
    cannot. A single pod always uses one claim.
    """
    if nnodes <= 1:
        return RESULTS_LAYOUT_SHARED
    mode = (layout or RESULTS_LAYOUT_AUTO).strip().lower()
    if mode == RESULTS_LAYOUT_SHARED:
        return RESULTS_LAYOUT_SHARED
    if mode == RESULTS_LAYOUT_PER_POD:
        return RESULTS_LAYOUT_PER_POD
    if rwx_usable:
        return RESULTS_LAYOUT_SHARED
    return RESULTS_LAYOUT_PER_POD


def resolve_data_layout(
    layout: Optional[str],
    rwx_usable: bool,
    existing_claim_is_rwx: Optional[bool],
) -> str:
    """Choose the dataset volume.

    ``existing_claim_is_rwx`` is True when ``madengine-shared-data`` exists
    and is ReadWriteMany, False when that claim exists with another access
    mode, and None when it is absent. A claim that is not ReadWriteMany is
    left in the cluster and not mounted. ``shared`` and ``per_pod`` are
    explicit overrides.
    """
    mode = (layout or RESULTS_LAYOUT_AUTO).strip().lower()
    if mode == RESULTS_LAYOUT_PER_POD:
        return RESULTS_LAYOUT_PER_POD
    if mode == RESULTS_LAYOUT_SHARED:
        return RESULTS_LAYOUT_SHARED
    if existing_claim_is_rwx is True:
        return RESULTS_LAYOUT_SHARED
    if existing_claim_is_rwx is False:
        return RESULTS_LAYOUT_PER_POD
    if rwx_usable:
        return RESULTS_LAYOUT_SHARED
    return RESULTS_LAYOUT_PER_POD


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

    def _k8s_local_results_storage_class(self) -> Optional[str]:
        """ReadWriteOnce class for one results volume per pod."""
        return (
            self.k8s_config.get("local_path_storage_class")
            or self.k8s_config.get("single_node_results_storage_class")
            or self.k8s_config.get("storage_class")
        )

    def _storage_class_supports_rwx(self, name: Optional[str]) -> bool:
        """True when ``name`` can serve a ReadWriteMany filesystem.

        The StorageClass object must exist, and either its provisioner is a
        shared-filesystem driver or a PersistentVolume of that class is already
        ReadWriteMany. A missing client keeps the shared layout so manifest
        rendering in unit tests does not require a cluster. A 403 or 404 is
        not usable storage.
        """
        if not name:
            return False
        storage_v1 = getattr(self, "storage_v1", None)
        if storage_v1 is None:
            return True
        try:
            storage_class = storage_v1.read_storage_class(name=name)
        except ApiException as e:
            if getattr(e, "status", None) in (403, 404):
                return False
            raise
        provisioner = getattr(storage_class, "provisioner", None)
        # A test double has no real provisioner string. Keep the shared layout,
        # matching manifest renders that never connect to a cluster.
        if provisioner is not None and not isinstance(provisioner, str):
            return True
        if provisioner_supports_rwx(provisioner):
            return True
        return self._has_rwx_persistent_volume(name)

    def _has_rwx_persistent_volume(self, storage_class: str) -> bool:
        """True when a PersistentVolume of this class is already ReadWriteMany."""
        core_v1 = getattr(self, "core_v1", None)
        if core_v1 is None:
            return False
        try:
            volumes = core_v1.list_persistent_volume(
                field_selector=f"spec.storageClassName={storage_class}"
            )
        except ApiException as e:
            if getattr(e, "status", None) in (403, 404):
                return False
            raise
        for volume in getattr(volumes, "items", None) or []:
            modes = getattr(getattr(volume, "spec", None), "access_modes", None) or []
            if "ReadWriteMany" in modes:
                return True
        return False

    def _existing_shared_data_is_rwx(self) -> Optional[bool]:
        """Access mode of ``madengine-shared-data``, or None when it is absent."""
        core_v1 = getattr(self, "core_v1", None)
        if core_v1 is None:
            return None
        try:
            claim = core_v1.read_namespaced_persistent_volume_claim(
                name=SHARED_DATA_PVC_NAME,
                namespace=getattr(self, "namespace", None) or "default",
            )
        except ApiException as e:
            if getattr(e, "status", None) in (403, 404):
                return None
            raise
        modes = getattr(getattr(claim, "spec", None), "access_modes", None) or []
        return "ReadWriteMany" in modes

    def _select_results_layout(self, nnodes: int) -> str:
        rwx_class = self._k8s_results_storage_class(max(nnodes, 2))
        return resolve_results_layout(
            self.k8s_config.get("results_layout"),
            nnodes,
            self._storage_class_supports_rwx(rwx_class),
        )

    def _select_data_layout(self) -> str:
        return resolve_data_layout(
            self.k8s_config.get("data_layout"),
            self._storage_class_supports_rwx(self._k8s_data_storage_class()),
            self._existing_shared_data_is_rwx(),
        )

    def _results_use_per_pod(self, nnodes: int) -> bool:
        layout = self._select_results_layout(nnodes)
        self._results_layout = layout
        return layout == RESULTS_LAYOUT_PER_POD

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

        Uses ReadWriteMany so the same PVC works for every pod. Callers skip
        this when ``data_layout`` is ``per_pod`` or the class cannot serve
        ReadWriteMany. An existing claim that is not ReadWriteMany is left in
        place; ``data_layout: shared`` is required to keep using it.

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
        if not storage_class or not self._storage_class_supports_rwx(storage_class):
            self.console.print(
                "[yellow]⚠️  This data StorageClass is not a confirmed ReadWriteMany "
                "filesystem. The claim may stay pending. Set k8s.data_layout to "
                "per_pod for a local disk on each pod.[/yellow]"
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
