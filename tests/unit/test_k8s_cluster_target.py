"""Which cluster a Kubernetes deployment talks to, and what it may delete there.

The Python client and every kubectl call must use one kubeconfig and one
context: on a host that can reach two clusters, they used to be chosen
separately, so a run could deploy to one and copy results from the other.
"""

import json
from types import SimpleNamespace

import pytest

from madengine.deployment import kubernetes as k8s_mod
from madengine.deployment.base import DeploymentConfig
from madengine.deployment.kubernetes import KubernetesDeployment

KUBECONFIG = """
apiVersion: v1
kind: Config
current-context: lab
clusters:
- {name: lab, cluster: {server: "https://127.0.0.1:6550"}}
- {name: prod, cluster: {server: "https://prod.example:6443"}}
users:
- {name: u, user: {token: t}}
contexts:
- {name: lab, context: {cluster: lab, user: u}}
- {name: prod, context: {cluster: prod, user: u}}
"""


def _bare(k8s_config):
    d = object.__new__(KubernetesDeployment)
    d.k8s_config = k8s_config
    d.namespace = "ns"
    return d


@pytest.fixture
def kubeconfig(tmp_path, monkeypatch):
    path = tmp_path / "kubeconfig"
    path.write_text(KUBECONFIG)
    monkeypatch.delenv("KUBECONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # no ~/.kube/config
    return str(path)


class TestResolveTarget:
    def test_explicit_kubeconfig_and_its_current_context(self, kubeconfig):
        assert _bare({"kubeconfig": kubeconfig})._resolve_kube_target() == (kubeconfig, "lab")

    def test_kubeconfig_env_is_honoured(self, kubeconfig, monkeypatch):
        monkeypatch.setenv("KUBECONFIG", kubeconfig)
        assert _bare({})._resolve_kube_target() == (kubeconfig, "lab")

    def test_explicit_kubeconfig_beats_the_env(self, kubeconfig, tmp_path, monkeypatch):
        monkeypatch.setenv("KUBECONFIG", str(tmp_path / "other"))
        assert _bare({"kubeconfig": kubeconfig})._resolve_kube_target()[0] == kubeconfig

    def test_named_context(self, kubeconfig):
        assert _bare({"kubeconfig": kubeconfig, "context": "prod"})._resolve_kube_target() == (kubeconfig, "prod")

    def test_unknown_context_refuses_instead_of_falling_back(self, kubeconfig):
        with pytest.raises(RuntimeError, match="'staging' is not in .*lab, prod.*Refusing"):
            _bare({"kubeconfig": kubeconfig, "context": "staging"})._resolve_kube_target()

    def test_no_kubeconfig_anywhere_means_in_cluster(self, kubeconfig):
        assert _bare({})._resolve_kube_target() == (None, None)


class TestKubectlIsPinned:
    def test_kubectl_names_the_same_file_and_context(self, kubeconfig):
        d = _bare({"kubeconfig": kubeconfig, "context": "prod"})
        d.kubeconfig_path, d.kube_context = d._resolve_kube_target()
        assert d.kubectl_cmd("cp", "a", "b") == [
            "kubectl", "--kubeconfig", kubeconfig, "--context", "prod", "cp", "a", "b"]

    def test_in_cluster_kubectl_is_plain(self):
        d = _bare({})
        d.kubeconfig_path, d.kube_context = None, None
        assert d.kubectl_cmd("exec", "p") == ["kubectl", "exec", "p"]

    def test_client_loads_the_same_target(self, kubeconfig, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(k8s_mod.k8s_config, "load_kube_config",
                            lambda **kw: seen.update(kw))
        monkeypatch.chdir(tmp_path)
        (tmp_path / "build_manifest.json").write_text(json.dumps({"built_images": {}, "built_models": {}, "context": {}}))
        d = KubernetesDeployment(DeploymentConfig(
            target="k8s", manifest_file="build_manifest.json",
            additional_context={"k8s": {"kubeconfig": kubeconfig, "context": "prod"},
                                "gpu_vendor": "AMD", "guest_os": "UBUNTU"}))
        assert seen == {"config_file": kubeconfig, "context": "prod"}
        assert d.kubectl_cmd()[1:] == ["--kubeconfig", kubeconfig, "--context", "prod"]

    def test_every_kubectl_call_goes_through_the_pinned_helper(self):
        import inspect
        from madengine.deployment import k8s_results
        src = inspect.getsource(k8s_results)
        assert '"kubectl",' not in src and '"kubectl", ' not in src


def _obj(labels):
    return SimpleNamespace(metadata=SimpleNamespace(labels=labels))


class TestDeleteOnlyOurs:
    def _run(self, labels, legacy=False):
        deleted = []
        d = _bare({})
        result = d._delete_if_ours("Job", "madengine-m", lambda: _obj(labels),
                                   lambda: deleted.append(1), legacy_unlabelled=legacy)
        return result, deleted

    def test_ours_is_deleted(self):
        assert self._run({"app": "madengine"}) == (True, [1])

    def test_someone_elses_same_name_is_refused(self):
        with pytest.raises(RuntimeError, match="not created by madengine"):
            self._run({"app": "their-app"})

    def test_unlabelled_refused_unless_legacy_kind(self):
        with pytest.raises(RuntimeError):
            self._run({})
        assert self._run({}, legacy=True) == (True, [1])
        with pytest.raises(RuntimeError):
            self._run({"app": "theirs"}, legacy=True)

    def test_absent_is_not_an_error(self):
        d = _bare({})

        def missing():
            raise k8s_mod.ApiException(status=404)
        assert d._delete_if_ours("Job", "x", missing, lambda: 1 / 0) is False
