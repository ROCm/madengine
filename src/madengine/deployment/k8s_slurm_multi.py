#!/usr/bin/env python3
"""slurm_multi cards on Kubernetes: the card's own SLURM launcher, unmodified.

A slurm_multi card (MAD's multinode vLLM / SGLang launchers) ships a SLURM batch
script that fans out with srun and starts one container per node with docker run.
On Kubernetes there is no srun and no docker, but every pod already runs the
card's image. So madengine runs one pod per node of the card -- an Indexed Job
with a headless Service, so pods reach each other by hostname -- and runs the
script, unchanged, on pod 0, with small stand-ins for srun, docker and scontrol
(k8s_shim/mad_k8s.py) first on PATH. See k8s_shim/mad_k8s_node.sh for what each
pod does.

What the card's script needs from the cluster is the pod spec's business, set in
the k8s config, never in the card:
  k8s.gpu_count                GPUs per pod (else the card's slurm.gpus_per_node)
  k8s.volumes                  host paths / PVCs / NFS the script reads (weights)
  k8s.extra_resources          other device-plugin resources (e.g. RDMA)
  k8s.security_context         container securityContext (e.g. IPC_LOCK for RDMA)
  k8s.shm_size                 /dev/shm size (default 64Gi)
  k8s.include_script_dirs      directories shipped with the card's (default: the
                               sibling "common" directory, when there is one)
  k8s.host_network             pods on the node's network (RoCE on the node's NICs);
                               implies one pod per node -- the launchers bind fixed
                               ports
  k8s.pod_annotations          e.g. a Multus `k8s.v1.cni.cncf.io/networks` entry, to
                               give each pod its own RDMA interface instead
"""

import base64
import io
import os
import re
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from madengine.core.timeout import resolve_run_timeout

from .common import is_self_managed_launcher
from .config_loader import PRESET_ENV_KEYS, PRESET_K8S_KEYS

SHIM_DIR = Path(__file__).parent / "k8s_shim"
AGENT_PORT = 29711
# ConfigMaps are capped at 1 MiB in total; leave room for the object's own fields.
BUNDLE_LIMIT_BYTES = 900 * 1024
_DNS_LABEL = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
_SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache"}


def card_launcher(model_info: Dict, distributed_config: Dict) -> Optional[str]:
    """The card's launcher. The card's own setting wins over the deployment's,
    whose presets default multi-node runs to torchrun."""
    return (model_info.get("distributed") or {}).get("launcher") or (distributed_config or {}).get("launcher")


def is_slurm_multi(model_info: Dict, distributed_config: Dict) -> bool:
    return is_self_managed_launcher(card_launcher(model_info, distributed_config))


def build_scripts_bundle(script_path: str, include_dirs: List[str]) -> Tuple[bytes, List[str]]:
    """tar.gz of the card's script directory, the extra dirs, and the shim.

    Paths in the archive are relative to the current directory (the MAD root), so
    the script finds its siblings (e.g. ../common/cluster.sh) where it would on a
    shared filesystem. Returns (archive, member names).
    """
    script_dir = Path(script_path).parent
    dirs = [script_dir] + [Path(d) for d in include_dirs if d]
    buf, names = io.BytesIO(), []
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for d in dirs:
            if not d.is_dir():
                raise FileNotFoundError(f"script directory {d} not found (cwd {Path.cwd()})")
            for f in sorted(d.rglob("*")):
                if not f.is_file() or _SKIP_DIRS & set(f.parts):
                    continue
                tar.add(str(f), arcname=str(f))
                names.append(str(f))
        for f in sorted(SHIM_DIR.iterdir()):
            if f.is_file() and f.suffix in (".py", ".sh"):
                tar.add(str(f), arcname=f".mad-k8s/{f.name}")
                names.append(f".mad-k8s/{f.name}")
    data = buf.getvalue()
    if len(base64.b64encode(data)) > BUNDLE_LIMIT_BYTES:
        raise ValueError(
            f"scripts bundle is {len(data)} bytes compressed ({len(names)} files from "
            f"{', '.join(map(str, dirs))}); a ConfigMap holds under 1 MiB. Narrow "
            "k8s.include_script_dirs."
        )
    return data, names


def pod_hostname_base(job_name: str, nnodes: int) -> str:
    """A run name whose pods' hostnames ("<name>-<index>") are valid DNS labels.

    The card's script addresses nodes by those hostnames. Card names carry dots
    (llama-3.1-8b) and can fill all 63 characters; dots become dashes, and a name
    still too long is cut and given a short hash of the full name, so two cards
    that share a long prefix stay distinct.
    """
    import hashlib

    name = re.sub(r"[^a-z0-9-]", "-", job_name.lower()).strip("-")
    room = 63 - len(f"-{max(nnodes - 1, 0)}")
    if len(name) > room:
        digest = hashlib.sha1(job_name.encode()).hexdigest()[:6]
        name = f"{name[: room - 7].rstrip('-')}-{digest}"
    return name


def default_include_dirs(script_path: str) -> List[str]:
    """The sibling `common` directory of the card's script directory, if any:
    the shared helpers MAD's launchers source as ${SCRIPT_DIR}/../common."""
    common = Path(script_path).parent.parent / "common"
    return [str(common)] if common.is_dir() else []


class KubernetesSlurmMultiMixin:
    """Prepare and collect a slurm_multi card on Kubernetes (see module docstring)."""

    def _slurm_multi_env(self, model_info: Dict, image: str) -> Dict[str, str]:
        """The card's env_vars plus the run's, without preset-only keys.

        On SLURM a slurm_multi card gets exactly this (SlurmDeployment._user_env_vars):
        presets carry settings for madengine's own launchers -- here the AMD
        multi-GPU preset's NCCL_IB_DISABLE=1 and NCCL_SOCKET_IFNAME=eth0 -- and
        handing them to a script that owns its environment changes its transport.
        """
        ctx = self.config.additional_context
        preset_only = set(ctx.get(PRESET_ENV_KEYS) or [])
        env = {k: str(v) for k, v in (model_info.get("env_vars") or {}).items()}
        env.update({k: str(v) for k, v in (ctx.get("env_vars") or {}).items() if k not in preset_only})
        env["DOCKER_IMAGE_NAME"] = image
        # The launchers coordinate through files under LOG_PATH, so it must be shared
        # by every pod: the results volume is (ReadWriteMany for multi-node).
        env.setdefault("LOG_PATH", "/results/logs")
        return env

    def _slurm_multi_image(self, model_info: Dict, image_info: Dict, model_key: str) -> str:
        for ref in (image_info.get("registry_image"), image_info.get("docker_image"),
                    model_info.get("docker_image"),
                    (model_info.get("env_vars") or {}).get("DOCKER_IMAGE_NAME")):
            if ref and not str(ref).startswith("<"):
                return str(ref)
        return model_key

    def _user_k8s(self, key: str):
        """k8s.<key> if the run set it; None if only a preset did.

        Presets size a pod from a profile guessed from the run config (single-gpu when
        it states no gpu_count or distributed block). A slurm_multi card sizes itself.
        """
        if key in set(self.config.additional_context.get(PRESET_K8S_KEYS) or []):
            return None
        return self.k8s_config.get(key)

    def _slurm_multi_gpus(self, model_info: Dict) -> int:
        for v in (self._user_k8s("gpu_count"), (model_info.get("slurm") or {}).get("gpus_per_node"),
                  model_info.get("n_gpus")):
            try:
                if v is not None and int(v) > 0:
                    return int(v)
            except (TypeError, ValueError):
                pass
        raise ValueError(
            "GPUs per pod unknown: set k8s.gpu_count (the card has no slurm.gpus_per_node "
            "and n_gpus is not a positive number)"
        )

    def _slurm_multi_nnodes(self, model_info: Dict) -> int:
        ctx_dist = self.config.additional_context.get("distributed") or {}
        for v in (ctx_dist.get("nnodes") if ctx_dist.get("launcher") in (None, "slurm_multi") else None,
                  (model_info.get("distributed") or {}).get("nnodes"),
                  (model_info.get("slurm") or {}).get("nodes")):
            if v:
                return int(v)
        return 1

    def _slurm_multi_volumes(self) -> Tuple[List[Dict], List[Dict]]:
        volumes, mounts = [], []
        for i, v in enumerate(self.k8s_config.get("volumes") or []):
            name = v.get("name") or f"mad-vol-{i}"
            if v.get("host_path"):
                src = {"hostPath": {"path": v["host_path"], "type": v.get("host_path_type", "Directory")}}
            elif v.get("pvc"):
                src = {"persistentVolumeClaim": {"claimName": v["pvc"]}}
            elif v.get("nfs"):
                src = {"nfs": {"server": v["nfs"]["server"], "path": v["nfs"]["path"]}}
            else:
                raise ValueError(f"k8s.volumes[{i}] needs host_path, pvc or nfs: {v}")
            volumes.append({"name": name, **src})
            mount = {"name": name, "mountPath": v.get("mount_path") or v.get("host_path")}
            if not mount["mountPath"]:
                raise ValueError(f"k8s.volumes[{i}] needs mount_path")
            if v.get("read_only"):
                mount["readOnly"] = True
            mounts.append(mount)
        return volumes, mounts

    def _prepare_slurm_multi(self, model_key: str, model_info: Dict, image_info: Dict) -> Dict[str, Any]:
        """Build the ConfigMap, headless Service and Indexed Job; return them as dicts."""
        nnodes = self._slurm_multi_nnodes(model_info)
        # Every name this run creates derives from one that makes valid hostnames.
        self.job_name = pod_hostname_base(self.job_name, nnodes)
        self.job_label = self.service_name = self.main_container_name = self.job_name
        self.configmap_name = f"{self.job_name}-config"
        gpus = self._slurm_multi_gpus(model_info)
        image = self._slurm_multi_image(model_info, image_info, model_key)
        script = model_info.get("scripts") or ""
        if not script:
            raise ValueError("slurm_multi card has no scripts entry")
        # Pod i's hostname is "<job>-<i>" (Indexed Job), and the script addresses
        # nodes by it, so the job name must make valid hostnames.
        prefix = self.job_name
        if not _DNS_LABEL.match(prefix) or len(f"{prefix}-{nnodes - 1}") > 63:
            raise ValueError(f"job name {prefix!r} does not make valid pod hostnames (DNS label, <= 63 chars)")

        k8s = self.k8s_config
        include = k8s.get("include_script_dirs")
        if include is None:
            include = default_include_dirs(script)
        bundle, members = build_scripts_bundle(script, include)
        self.console.print(
            f"[dim]slurm_multi bundle: {len(members)} files, {len(bundle)} bytes "
            f"({Path(script).parent}{', ' + ', '.join(include) if include else ''})[/dim]"
        )

        env = self._slurm_multi_env(model_info, image)
        # One id per run: SLURM_JOB_ID in the pods, and the run's own results directory.
        self._slurm_multi_run_id = str(int(time.time()) % 100000000)
        env.update({
            "MAD_K8S_NNODES": str(nnodes), "MAD_K8S_NODE_PREFIX": prefix,
            "MAD_K8S_JOB_ID": self._slurm_multi_run_id, "MAD_K8S_IMAGE": image,
            "MAD_K8S_SCRIPT": script, "MAD_K8S_ARGS": model_info.get("args") or "",
            "MAD_K8S_AGENT_PORT": str(AGENT_PORT),
        })
        if model_info.get("multiple_results"):
            env["MAD_K8S_MULTIPLE_RESULTS"] = model_info["multiple_results"]

        labels = {"app": "madengine", "model": self.job_label, "madengine-launcher": "slurm_multi"}
        pod_labels = {"app": "madengine", "job-name": self.job_label, "model": self.job_label}
        gpu_res = {self.gpu_resource_name: str(gpus)}
        extra = {k: str(v) for k, v in (k8s.get("extra_resources") or {}).items()}
        requests = {**gpu_res, **extra}
        for key in ("cpu", "memory"):
            if self._user_k8s(key):
                requests[key] = str(self._user_k8s(key))
        vols, mounts = self._slurm_multi_volumes()

        container = {
            "name": self.main_container_name,
            "image": image,
            "imagePullPolicy": k8s.get("image_pull_policy", "IfNotPresent"),
            "command": ["/bin/bash", "-c"],
            "args": [
                "set -e; mkdir -p /workspace; tar -xzf /mad-k8s-bundle/scripts.tgz -C /workspace; "
                "exec bash /workspace/.mad-k8s/mad_k8s_node.sh"
            ],
            "env": [{"name": k, "value": v} for k, v in sorted(env.items())],
            "ports": [{"name": "mad-k8s", "containerPort": AGENT_PORT}],
            "resources": {"limits": {**gpu_res, **extra}, "requests": requests},
            "volumeMounts": [
                {"name": "mad-k8s-bundle", "mountPath": "/mad-k8s-bundle", "readOnly": True},
                {"name": "results", "mountPath": "/results"},
                {"name": "dshm", "mountPath": "/dev/shm"},
                *mounts,
            ],
        }
        if k8s.get("security_context"):
            container["securityContext"] = k8s["security_context"]

        pod_spec = {
            "restartPolicy": "Never",
            "subdomain": self.service_name,
            # Short hostnames ("<job>-1") are how the script names nodes; resolve them.
            "dnsConfig": {"searches": [f"{self.service_name}.{self.namespace}.svc.cluster.local"]},
            "containers": [container],
            "volumes": [
                {"name": "mad-k8s-bundle", "configMap": {"name": self.configmap_name}},
                {"name": "results", "persistentVolumeClaim": {"claimName": f"{self.job_name}-results"}},
                {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": str(k8s.get("shm_size", "64Gi"))}},
                *vols,
            ],
        }
        if k8s.get("host_network"):
            # The node's NICs (and their RoCE GIDs) instead of the pod network. The
            # card's servers bind fixed ports, so two pods of the run cannot share a
            # node: require one per node rather than let the second fail to bind.
            pod_spec["hostNetwork"] = True
            pod_spec["dnsPolicy"] = "ClusterFirstWithHostNet"
            pod_spec["affinity"] = {"podAntiAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": [
                {"labelSelector": {"matchLabels": {"job-name": self.job_label}},
                 "topologyKey": "kubernetes.io/hostname"}]}}
        if k8s.get("node_selector"):
            pod_spec["nodeSelector"] = k8s["node_selector"]
        if k8s.get("tolerations"):
            pod_spec["tolerations"] = k8s["tolerations"]
        pulls = getattr(self, "_image_pull_secrets_for_pods", None) or [
            {"name": n} for n in (k8s.get("secrets") or {}).get("image_pull_secret_names") or []]
        if pulls:
            pod_spec["imagePullSecrets"] = pulls

        job_spec = {
            "completions": nnodes, "parallelism": nnodes, "completionMode": "Indexed",
            # One attempt: pod 0 is the batch host; rerunning one pod of a multi-node
            # run is not a retry of the run.
            "backoffLimit": 0,
            "template": {"metadata": {"labels": pod_labels, **(
                {"annotations": {str(k): str(v) for k, v in k8s["pod_annotations"].items()}}
                if k8s.get("pod_annotations") else {})}, "spec": pod_spec},
        }
        timeout = resolve_run_timeout(model_info, self.config.cli_timeout)
        if timeout and timeout > 0:
            job_spec["activeDeadlineSeconds"] = int(timeout)
        if k8s.get("ttl_seconds_after_finished") is not None:
            job_spec["ttlSecondsAfterFinished"] = int(k8s["ttl_seconds_after_finished"])

        meta = lambda name: {"name": name, "namespace": self.namespace, "labels": labels}  # noqa: E731
        configmap = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": meta(self.configmap_name),
                     "binaryData": {"scripts.tgz": base64.b64encode(bundle).decode()}}
        service = {"apiVersion": "v1", "kind": "Service", "metadata": meta(self.service_name),
                   "spec": {"clusterIP": "None", "publishNotReadyAddresses": True,
                            "selector": {"job-name": self.job_label},
                            "ports": [{"name": "mad-k8s", "port": AGENT_PORT}]}}
        job = {"apiVersion": "batch/v1", "kind": "Job", "metadata": meta(self.job_name), "spec": job_spec}
        return {"configmap": configmap, "service": service, "job": job, "nnodes": nnodes,
                "gpus": gpus, "image": image}

    def _collect_slurm_multi_results(self, deployment_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Read the perf.csv pod 0 published, the way the SLURM path reads it.

        Rows split into successful/failed runs by their status, and are appended to
        ./perf.csv (single header). No perf.csv is a failed run, never an empty
        success.
        """
        import csv

        # This run's own directory. k8s_results/<job>/ is shared by every run of the
        # card, so searching it found an EARLIER run's perf.csv: a 4-node run was
        # reported with the 2-node run's numbers, and a run that published nothing
        # would have been reported with a previous run's success.
        run_id = getattr(self, "_slurm_multi_run_id", None) or str(int(time.time()))
        results_dir = Path(f"./k8s_results/{deployment_id}/run-{run_id}")
        results_dir.mkdir(parents=True, exist_ok=True)
        self._collect_from_pvc(deployment_id, results_dir, results)
        # Exactly the file pod 0 published for this run (mad_k8s_node.sh).
        published = results_dir / f"{self.job_name}-0" / "perf.csv"
        found = [published] if published.is_file() and published.stat().st_size > 0 else []
        model_name = next(iter((self.manifest.get("built_models") or {}).values()), {}).get("name", deployment_id)
        if not found:
            results["failed_runs"].append({
                "model": model_name, "status": "FAILURE", "performance": "", "metric": "",
                "error": "no perf.csv was published by the card's script (see k8s_results logs)",
            })
            self.console.print("[red]✗ slurm_multi: no perf.csv published by pod 0[/red]")
            return results
        perf = found[0]
        results.setdefault("perf_files", []).append(str(perf))
        with open(perf, newline="") as f:
            reader = csv.DictReader(f)
            rows = [{(k or "").strip(): v for k, v in r.items()} for r in reader]
        for row in rows:
            run = {"model": row.get("model", ""), "status": row.get("status", ""),
                   "performance": row.get("performance", ""), "metric": row.get("metric", ""),
                   "duration": row.get("test_duration", ""), "gpu_arch": row.get("gpu_architecture", ""),
                   "deployment": row.get("deployment_type", ""), "machine": row.get("machine_name", "")}
            (results["successful_runs"] if row.get("status") == "SUCCESS" else results["failed_runs"]).append(run)
        # Append to the run's perf.csv, keeping a single header.
        cwd_perf = Path("perf.csv")
        with open(perf) as src:
            lines = src.readlines()
        has_header = cwd_perf.exists() and cwd_perf.stat().st_size > 0
        with open(cwd_perf, "a") as dst:
            dst.writelines(lines[1:] if has_header else lines)
        self.console.print(
            f"[green]Collected slurm_multi results: {perf} -> {cwd_perf} "
            f"({len(results['successful_runs'])} successful, {len(results['failed_runs'])} failed)[/green]"
        )
        return results
