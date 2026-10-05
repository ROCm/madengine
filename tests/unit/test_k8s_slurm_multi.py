"""The manifests madengine builds to run a slurm_multi card on Kubernetes."""

import base64
import io
import json
import tarfile
from types import SimpleNamespace

import pytest
import yaml

from madengine.deployment import kubernetes as k8s_mod
from madengine.deployment.base import DeploymentConfig
from madengine.deployment.config_loader import ConfigLoader
from madengine.deployment.kubernetes import KubernetesDeployment

CARD = {
    "name": "pyt_sglang_disagg_demo",
    "scripts": "scripts/demo/run_xPyD_models.slurm",
    "n_gpus": "-1",
    "args": "-N 2 -n 2",
    "timeout": 3600,
    "distributed": {"launcher": "slurm_multi", "nnodes": 2},
    "slurm": {"nodes": 2, "gpus_per_node": 8, "time": "24:00:00"},
    "env_vars": {"DOCKER_IMAGE_NAME": "<supply-your-image>", "MODEL_NAME": "Qwen3-32B", "xP": "1"},
}


def _kubeconfig(tmp_path):
    p = tmp_path / "kubeconfig"
    p.write_text("apiVersion: v1\nkind: Config\ncurrent-context: lab\n"
                 "clusters: [{name: lab, cluster: {server: 'https://127.0.0.1:1'}}]\n"
                 "users: [{name: u, user: {token: t}}]\n"
                 "contexts: [{name: lab, context: {cluster: lab, user: u}}]\n")
    return str(p)


def _deployment(tmp_path, monkeypatch, card=None, k8s=None, env_vars=None, image_info=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    for f, text in {"scripts/demo/run_xPyD_models.slurm": "#!/bin/bash\n. ../common/cluster.sh\n",
                    "scripts/demo/models.yaml": "x: 1\n",
                    "scripts/common/cluster.sh": "export SITE=1\n",
                    "scripts/other/big.sh": "unrelated\n"}.items():
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(text)
    card = card or json.loads(json.dumps(CARD))
    manifest = {"built_images": {"img": image_info or {"docker_image": "rocm/sglang:mi300x"}},
                "built_models": {"img": card}, "context": {}}
    (tmp_path / "build_manifest.json").write_text(json.dumps(manifest))
    user = {"gpu_vendor": "AMD", "guest_os": "UBUNTU",
            "k8s": {"kubeconfig": _kubeconfig(tmp_path), "namespace": "lab", **(k8s or {})},
            "distributed": {"nnodes": 2}}
    if env_vars:
        user["env_vars"] = env_vars
    monkeypatch.setattr(k8s_mod.k8s_config, "load_kube_config", lambda **kw: None)
    return KubernetesDeployment(DeploymentConfig(
        target="k8s", manifest_file="build_manifest.json",
        additional_context=ConfigLoader.load_k8s_config(user), cli_timeout=-1))


def _prepared(tmp_path, monkeypatch, **kw):
    d = _deployment(tmp_path, monkeypatch, **kw)
    assert d.prepare() is True
    job = yaml.safe_load(d.job_yaml)
    pod = job["spec"]["template"]["spec"]
    env = {e["name"]: e["value"] for e in pod["containers"][0]["env"]}
    return d, job, pod, env


class TestSlurmMultiJob:
    def test_indexed_job_one_pod_per_card_node(self, tmp_path, monkeypatch):
        d, job, pod, env = _prepared(tmp_path, monkeypatch)
        spec = job["spec"]
        assert (spec["completions"], spec["parallelism"], spec["completionMode"]) == (2, 2, "Indexed")
        assert spec["backoffLimit"] == 0
        assert spec["activeDeadlineSeconds"] == 3600
        assert job["metadata"]["labels"]["app"] == "madengine"
        assert pod["subdomain"] == d.service_name
        assert pod["dnsConfig"]["searches"] == [f"{d.service_name}.lab.svc.cluster.local"]
        assert (env["MAD_K8S_NNODES"], env["MAD_K8S_NODE_PREFIX"]) == ("2", d.job_name)
        assert env["MAD_K8S_SCRIPT"] == "scripts/demo/run_xPyD_models.slurm"
        assert env["MAD_K8S_ARGS"] == "-N 2 -n 2"

    def test_gpus_come_from_the_card_unless_k8s_says(self, tmp_path, monkeypatch):
        _, _, pod, _ = _prepared(tmp_path, monkeypatch)
        assert pod["containers"][0]["resources"]["limits"]["amd.com/gpu"] == "8"
        _, _, pod, _ = _prepared(tmp_path / "b", monkeypatch, k8s={"gpu_count": 4})
        assert pod["containers"][0]["resources"]["limits"]["amd.com/gpu"] == "4"

    def test_image_and_card_env_reach_the_pod_without_preset_env(self, tmp_path, monkeypatch):
        _, _, pod, env = _prepared(tmp_path, monkeypatch, env_vars={"GPUS_PER_NODE": "4"})
        assert pod["containers"][0]["image"] == "rocm/sglang:mi300x"
        assert env["DOCKER_IMAGE_NAME"] == env["MAD_K8S_IMAGE"] == "rocm/sglang:mi300x"
        assert (env["MODEL_NAME"], env["GPUS_PER_NODE"]) == ("Qwen3-32B", "4")
        # The AMD multi-GPU preset sets these for madengine's own launchers; a card
        # that owns its environment must not inherit them.
        assert "NCCL_IB_DISABLE" not in env and "NCCL_SOCKET_IFNAME" not in env
        assert env["LOG_PATH"] == "/results/logs"

    def test_a_preset_key_the_user_sets_is_kept(self, tmp_path, monkeypatch):
        _, _, _, env = _prepared(tmp_path, monkeypatch, env_vars={"NCCL_SOCKET_IFNAME": "ens3"})
        assert env["NCCL_SOCKET_IFNAME"] == "ens3"

    def test_volumes_resources_and_security_context_are_config(self, tmp_path, monkeypatch):
        _, _, pod, _ = _prepared(tmp_path, monkeypatch, k8s={
            "volumes": [{"name": "models", "host_path": "/data/models", "mount_path": "/models", "read_only": True},
                        {"pvc": "shared-logs", "mount_path": "/logs"}],
            "extra_resources": {"rdma/hca": 1}, "security_context": {"capabilities": {"add": ["IPC_LOCK"]}},
            "shm_size": "128Gi", "node_selector": {"node.lab/gpu": "true"}})
        c = pod["containers"][0]
        assert c["resources"]["limits"]["rdma/hca"] == "1"
        assert c["securityContext"] == {"capabilities": {"add": ["IPC_LOCK"]}}
        vols = {v["name"]: v for v in pod["volumes"]}
        assert vols["models"]["hostPath"]["path"] == "/data/models"
        assert vols["mad-vol-1"]["persistentVolumeClaim"]["claimName"] == "shared-logs"
        assert vols["dshm"]["emptyDir"]["sizeLimit"] == "128Gi"
        mounts = {m["mountPath"]: m for m in c["volumeMounts"]}
        assert mounts["/models"]["readOnly"] is True and "/logs" in mounts and "/results" in mounts
        assert pod["nodeSelector"] == {"node.lab/gpu": "true"}

    def test_bundle_carries_the_card_dir_common_and_the_shim(self, tmp_path, monkeypatch):
        d, _, _, _ = _prepared(tmp_path, monkeypatch)
        cm = yaml.safe_load(d.configmap_yaml)
        names = tarfile.open(fileobj=io.BytesIO(base64.b64decode(cm["binaryData"]["scripts.tgz"]))).getnames()
        assert "scripts/demo/run_xPyD_models.slurm" in names and "scripts/common/cluster.sh" in names
        assert ".mad-k8s/mad_k8s.py" in names and ".mad-k8s/mad_k8s_node.sh" in names
        assert not any(n.startswith("scripts/other") for n in names)

    def test_headless_service_publishes_pods_before_ready(self, tmp_path, monkeypatch):
        d, _, _, _ = _prepared(tmp_path, monkeypatch)
        svc = yaml.safe_load(d.service_yaml)
        assert svc["spec"]["clusterIP"] == "None" and svc["spec"]["publishNotReadyAddresses"] is True
        assert svc["spec"]["selector"] == {"job-name": d.job_label}

    def test_a_pushed_image_wins(self, tmp_path, monkeypatch):
        _, _, pod, _ = _prepared(tmp_path, monkeypatch, image_info={
            "docker_image": "ci-local", "registry_image": "registry.lab:5000/sglang:mi300x"})
        assert pod["containers"][0]["image"] == "registry.lab:5000/sglang:mi300x"

    def test_unknown_gpu_count_is_an_error(self, tmp_path, monkeypatch):
        card = json.loads(json.dumps(CARD))
        del card["slurm"]
        d = _deployment(tmp_path, monkeypatch, card=card)
        assert d.prepare() is False


def test_other_launchers_keep_the_template_path(tmp_path, monkeypatch):
    card = json.loads(json.dumps(CARD))
    card["distributed"] = {"launcher": "torchrun", "nnodes": 1}
    card["n_gpus"] = "1"
    d = _deployment(tmp_path, monkeypatch, card=card, image_info={"registry_image": "r/i:t"})
    assert d.prepare() is True
    assert d._slurm_multi is False
    assert "madengine-launcher" not in d.job_yaml


class TestResultsPvcAccessMode:
    def _mode(self, tmp_path, monkeypatch, nnodes, k8s=None):
        d = _deployment(tmp_path, monkeypatch, k8s=k8s)
        d.job_name = "madengine-x"
        seen = {}
        d.core_v1 = SimpleNamespace(create_namespaced_persistent_volume_claim=lambda namespace, body: seen.update(body))
        d._create_results_pvc(nnodes=nnodes)
        return seen["spec"]["accessModes"]

    def test_default_rwx_for_multi_node_rwo_for_one(self, tmp_path, monkeypatch):
        assert self._mode(tmp_path / "a", monkeypatch, 2) == ["ReadWriteMany"]
        assert self._mode(tmp_path / "b", monkeypatch, 1) == ["ReadWriteOnce"]

    def test_override_for_a_cluster_without_rwx(self, tmp_path, monkeypatch):
        assert self._mode(tmp_path, monkeypatch, 2, k8s={"results_access_mode": "ReadWriteOnce"}) == ["ReadWriteOnce"]


class TestPodHostnames:
    def test_dotted_long_card_names_make_valid_hostnames(self, tmp_path, monkeypatch):
        card = json.loads(json.dumps(CARD))
        card["name"] = "sglang_disagg/pyt_sglang_disagg_mooncake_llama-3.1-8b"
        d, job, pod, env = _prepared(tmp_path, monkeypatch, card=card)
        name = job["metadata"]["name"]
        assert "." not in name and len(f"{name}-1") <= 63
        assert env["MAD_K8S_NODE_PREFIX"] == name == pod["subdomain"] == d.service_name
        assert yaml.safe_load(d.configmap_yaml)["metadata"]["name"] == f"{name}-config"
        assert pod["volumes"][1]["persistentVolumeClaim"]["claimName"] == f"{name}-results"

    def test_long_names_stay_distinct(self):
        from madengine.deployment.k8s_slurm_multi import pod_hostname_base
        a = pod_hostname_base("madengine-" + "x" * 60 + "-a", 2)
        b = pod_hostname_base("madengine-" + "x" * 60 + "-b", 2)
        assert a != b and len(a) <= 61 and len(b) <= 61
        assert pod_hostname_base("madengine-qwen3", 2) == "madengine-qwen3"


class TestRdmaNetworking:
    def test_host_network_puts_one_pod_per_node(self, tmp_path, monkeypatch):
        d, _, pod, _ = _prepared(tmp_path, monkeypatch, k8s={"host_network": True})
        assert pod["hostNetwork"] is True and pod["dnsPolicy"] == "ClusterFirstWithHostNet"
        term = pod["affinity"]["podAntiAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][0]
        assert term["labelSelector"]["matchLabels"] == {"job-name": d.job_label}
        assert term["topologyKey"] == "kubernetes.io/hostname"

    def test_pod_annotations_for_a_secondary_rdma_network(self, tmp_path, monkeypatch):
        _, job, _, _ = _prepared(tmp_path, monkeypatch, k8s={
            "pod_annotations": {"k8s.v1.cni.cncf.io/networks": "rdma-net"}})
        assert job["spec"]["template"]["metadata"]["annotations"] == {"k8s.v1.cni.cncf.io/networks": "rdma-net"}

    def test_defaults_are_the_pod_network(self, tmp_path, monkeypatch):
        _, job, pod, _ = _prepared(tmp_path, monkeypatch)
        assert "hostNetwork" not in pod and "affinity" not in pod
        assert "annotations" not in job["spec"]["template"]["metadata"]


class TestCollectOnlyThisRun:
    HEADER = "model,performance,metric,status,test_duration,gpu_architecture,deployment_type,machine_name\n"

    def _collect(self, tmp_path, monkeypatch, publish):
        d, _, _, _ = _prepared(tmp_path, monkeypatch)
        # An earlier run of the same card left its results in the shared job directory.
        old = tmp_path / "k8s_results" / d.job_name / "run-1" / f"{d.job_name}-0"
        old.mkdir(parents=True)
        (old / "perf.csv").write_text(self.HEADER + "M,999.0,tok/s (con=8),SUCCESS,,,,\n")

        def fake_pvc(deployment_id, results_dir, results):
            if publish is not None:
                pod0 = results_dir / f"{d.job_name}-0"
                pod0.mkdir(parents=True)
                (pod0 / "perf.csv").write_text(self.HEADER + publish)

        monkeypatch.setattr(d, "_collect_from_pvc", fake_pvc)
        res = {"successful_runs": [], "failed_runs": [], "logs": [], "artifacts": []}
        return d._collect_slurm_multi_results(d.job_name, res)

    def test_this_runs_rows_not_an_earlier_runs(self, tmp_path, monkeypatch):
        res = self._collect(tmp_path, monkeypatch, "M,123.0,tok/s (con=8),SUCCESS,,,,\n")
        assert [r["performance"] for r in res["successful_runs"]] == ["123.0"]

    def test_nothing_published_is_a_failure_not_an_old_success(self, tmp_path, monkeypatch):
        res = self._collect(tmp_path, monkeypatch, None)
        assert res["successful_runs"] == [] and res["failed_runs"][0]["status"] == "FAILURE"


class TestPresetsDoNotSizeTheCard:
    """A run config that states no gpu_count / distributed block gets madengine's
    single-gpu profile (gpu_count 1, cpu 8, memory 16Gi). A slurm_multi card sizes
    its own pods; only what the run sets overrides it."""

    def _bare(self, tmp_path, monkeypatch, k8s=None):
        tmp_path.mkdir(parents=True, exist_ok=True)
        d = _deployment(tmp_path, monkeypatch, k8s=k8s)
        d.config.additional_context.pop("distributed", None)
        user = {"gpu_vendor": "AMD", "guest_os": "UBUNTU",
                "k8s": {"kubeconfig": _kubeconfig(tmp_path), "namespace": "lab", **(k8s or {})}}
        d.config.additional_context.clear()
        d.config.additional_context.update(ConfigLoader.load_k8s_config(user))
        d.k8s_config = d.config.additional_context["k8s"]
        assert d.k8s_config.get("gpu_count") == (k8s or {}).get("gpu_count", 1)  # the profile's guess
        assert d.prepare() is True
        return yaml.safe_load(d.job_yaml)["spec"]["template"]["spec"]["containers"][0]["resources"]

    def test_card_gpus_win_over_the_profile_guess(self, tmp_path, monkeypatch):
        res = self._bare(tmp_path, monkeypatch)
        assert res["limits"]["amd.com/gpu"] == "8"
        assert "cpu" not in res["requests"] and "memory" not in res["requests"]

    def test_what_the_run_sets_wins(self, tmp_path, monkeypatch):
        res = self._bare(tmp_path, monkeypatch, k8s={"gpu_count": 4, "cpu": "16", "memory": "64Gi"})
        assert res["limits"]["amd.com/gpu"] == "4"
        assert (res["requests"]["cpu"], res["requests"]["memory"]) == ("16", "64Gi")
