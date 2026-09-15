"""
Regression test: madengine's generated llm-d Helm values against real charts.

This locks in a live-cluster verification (see the k8s-llm-d branch) that
standard llm-d's Gateway -> InferencePool -> vLLM path works, and that
madengine's own generated values line up with it: the InferencePool's pod
selector matches the modelservice chart's actual pod labels, the HTTPRoute
that wires the Gateway to the InferencePool gets created at all, and the
Gateway gets the requested GatewayClass. None of this contacts a Kubernetes
cluster -- only ``helm template`` against the chart versions pinned below.

Requires `helm` on PATH and network access to fetch the charts; skipped
otherwise.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from madengine.deployment.llm_d_stack import MODEL_SERVER_LABEL, LlmdStack

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# Chart versions verified together against a live 4-node AMD GPU cluster.
# Upstream schemas are still moving (see llm_d_stack.py's module docstring);
# this test is what catches the next drift, without needing that cluster again.
INFRA_REPO = "https://llm-d-incubation.github.io/llm-d-infra/"
MODELSERVICE_REPO = "https://llm-d-incubation.github.io/llm-d-modelservice/"

CHARTS = {
    "infra": {"ref": "llm-d-infra/llm-d-infra", "version": "v1.4.0"},
    "gaie": {
        "ref": "oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool",
        "version": "v1.5.0",
    },
    "modelservice": {
        "ref": "llm-d-modelservice/llm-d-modelservice",
        "version": "v0.4.16",
    },
}

LLMD_CONFIG = {
    "model": {
        "name": "Qwen2.5-0.5B-Instruct",
        "uri": "hf://Qwen/Qwen2.5-0.5B-Instruct",
        "size": "10Gi",
    },
    "gateway": "istio",
    "prefill": {"replicas": 0},
    "decode": {
        "replicas": 1,
        "tensor_parallel": 1,
        "gpu_count": 1,
        "image": "docker.io/vllm/vllm-openai:latest",
    },
    "charts": CHARTS,
}


@pytest.fixture(scope="module")
def helm_repos(tmp_path_factory, monkeypatch_module):
    """Point helm at an isolated config so this test adds no repos globally."""
    if not shutil.which("helm"):
        pytest.skip("helm not on PATH")

    home = tmp_path_factory.mktemp("helm-home")
    monkeypatch_module.setenv("HELM_REPOSITORY_CONFIG", str(home / "repositories.yaml"))
    monkeypatch_module.setenv("HELM_REPOSITORY_CACHE", str(home / "cache"))
    monkeypatch_module.setenv("HELM_REGISTRY_CONFIG", str(home / "registry.json"))

    for name, url in (
        ("llm-d-infra", INFRA_REPO),
        ("llm-d-modelservice", MODELSERVICE_REPO),
    ):
        result = subprocess.run(
            ["helm", "repo", "add", name, url],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            pytest.skip(f"could not reach chart repo {url}: {result.stderr}")


@pytest.fixture(scope="module")
def monkeypatch_module():
    """A module-scoped MonkeyPatch; pytest's built-in fixture is function-scoped."""
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def _rendered_docs(
    stack: LlmdStack, component: str, tmp_path: Path
) -> List[Dict[str, Any]]:
    """``helm template`` a component and return its parsed YAML documents."""
    values_path = tmp_path / f"{component}-values.yaml"
    values_path.write_text(yaml.safe_dump(stack.values(component)))
    output = stack.template(component, values_path)
    return [doc for doc in yaml.safe_load_all(output) if doc]


def _find(docs: List[Dict[str, Any]], kind: str) -> Dict[str, Any]:
    matches = [d for d in docs if d.get("kind") == kind]
    assert len(matches) == 1, f"expected exactly one {kind}, found {len(matches)}"
    return matches[0]


@pytest.fixture(scope="module")
def stack(helm_repos):
    return LlmdStack(
        llmd_config=LLMD_CONFIG,
        namespace="llm-d-bench",
        release_prefix="madengine-test",
    )


class TestGeneratedValuesAgainstRealCharts:
    def test_gateway_gets_the_requested_class(self, stack, tmp_path):
        docs = _rendered_docs(stack, "infra", tmp_path)
        gateway = _find(docs, "Gateway")
        assert gateway["metadata"]["name"] == stack._gateway_fullname()
        assert gateway["spec"]["gatewayClassName"] == "istio"

    def test_inferencepool_selector_matches_modelservice_pod_labels(
        self, stack, tmp_path
    ):
        pool = _find(_rendered_docs(stack, "gaie", tmp_path), "InferencePool")
        assert pool["spec"]["selector"]["matchLabels"] == MODEL_SERVER_LABEL

        decode = _find(_rendered_docs(stack, "modelservice", tmp_path), "Deployment")
        pod_labels = decode["spec"]["template"]["metadata"]["labels"]
        for key, value in MODEL_SERVER_LABEL.items():
            assert pod_labels.get(key) == value

    def test_httproute_wires_the_gateway_to_the_inferencepool(self, stack, tmp_path):
        route = _find(_rendered_docs(stack, "gaie", tmp_path), "HTTPRoute")
        assert route["spec"]["parentRefs"][0]["name"] == stack._gateway_fullname()
        assert route["spec"]["rules"][0]["backendRefs"][0]["name"] == (
            stack.release_name("gaie")
        )
