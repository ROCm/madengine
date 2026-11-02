"""Unit-test isolation from the machine running the tests."""

import pytest


@pytest.fixture(autouse=True)
def _no_host_kubeconfig(tmp_path_factory, monkeypatch):
    """An empty HOME and no $KUBECONFIG for every unit test.

    KubernetesDeployment resolves its cluster from k8s.kubeconfig, else $KUBECONFIG,
    else ~/.kube/config, and reads that file in __init__. Without this, a test's
    outcome depends on the runner's own kubeconfig: one that exists but does not
    parse fails every test that builds a deployment. A test that needs a kubeconfig
    sets k8s.kubeconfig or $KUBECONFIG itself.
    """
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
    monkeypatch.delenv("KUBECONFIG", raising=False)
