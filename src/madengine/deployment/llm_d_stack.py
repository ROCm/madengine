#!/usr/bin/env python3
"""
Helm mechanics for standing an llm-d stack up and taking it back down.

An llm-d stack is three Helm releases, installed in dependency order:

1. ``infra``        — the Gateway and its provider resources (llm-d-infra chart)
2. ``gaie``         — the ``InferencePool`` and Endpoint Picker (GAIE inferencepool chart)
3. ``modelservice`` — the prefill/decode vLLM model servers (llm-d-modelservice chart)

This module owns only the mechanical part: turning a madengine ``llm_d`` config
block into chart values, and shelling out to ``helm``. Readiness polling and
endpoint resolution need a Kubernetes client and live in
:class:`~madengine.deployment.llm_d.LlmdDeployment`.

**On the generated values.** madengine generates the minimum set of values that
expresses its own configuration surface — model, replicas, tensor parallelism,
GPU count, auth secret, gateway class, port. Everything else is left to the
chart's own defaults. The value key paths below track the upstream chart
schemas, which are still moving; they are correct for the chart versions a user
pins, or they are not, and the way to find out without touching a cluster is
``llm_d.dry_run``, which renders the values files and ``helm template`` output
for inspection. ``llm_d.extra_values`` is the escape hatch and is deep-merged
last, so any key madengine gets wrong or omits can be corrected from config
without a madengine change.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from madengine.core.console import Console as ShellConsole

from .config_loader import ConfigLoader

# Install order. Teardown walks this in reverse.
COMPONENTS = ("infra", "gaie", "modelservice")

# Port the model servers listen on, and therefore the port the InferencePool
# targets and the gateway routes to. vLLM's default.
MODEL_SERVER_PORT = 8000

# Label llm-d's charts put on model-server pods; the InferencePool selects on it.
MODEL_SERVER_LABEL = {"llm-d.ai/inferenceServing": "true"}


class LlmdStackError(RuntimeError):
    """A helm operation failed."""


class LlmdStack:
    """Render chart values and drive ``helm`` for one llm-d stack."""

    def __init__(
        self,
        llmd_config: Dict[str, Any],
        namespace: str,
        release_prefix: str,
        gpu_resource_name: str = "amd.com/gpu",
        shell: Optional[ShellConsole] = None,
    ):
        """Initialize.

        Args:
            llmd_config: The resolved ``llm_d`` block from additional_context.
            namespace: Namespace to install into. Must already exist —
                madengine never creates or deletes a namespace.
            release_prefix: Prefix for the three release names.
            gpu_resource_name: Extended resource to request for model servers
                (``amd.com/gpu`` or ``nvidia.com/gpu``).
            shell: Command runner. Defaults to madengine's Console, which
                redacts ``MAD_SECRETS_*`` and known token shapes before printing
                or raising a command.
        """
        self.llmd_config = llmd_config
        self.namespace = namespace
        self.release_prefix = release_prefix
        self.gpu_resource_name = gpu_resource_name
        self.shell = shell or ShellConsole(live_output=True)

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------

    def release_name(self, component: str) -> str:
        """Helm release name for a component.

        The prefix is derived from the sanitized Job name, so re-running the
        same model converges on the same releases rather than colliding with
        them (see ``helm upgrade --install`` in :meth:`install`).
        """
        return f"{self.release_prefix}-{component}"

    @property
    def release_names(self) -> List[str]:
        """All three release names, in install order."""
        return [self.release_name(c) for c in COMPONENTS]

    def chart(self, component: str) -> Dict[str, Any]:
        """Chart ref/version for a component."""
        return (self.llmd_config.get("charts") or {}).get(component) or {}

    # ------------------------------------------------------------------
    # Values
    # ------------------------------------------------------------------

    def values(self, component: str) -> Dict[str, Any]:
        """Chart values for a component, with ``extra_values`` merged last."""
        builders = {
            "infra": self._infra_values,
            "gaie": self._gaie_values,
            "modelservice": self._modelservice_values,
        }
        if component not in builders:
            raise LlmdStackError(f"Unknown llm-d component: {component}")

        base = builders[component]()
        extra = self._extra_values_for(component)
        return ConfigLoader.deep_merge(base, extra) if extra else base

    def _extra_values_for(self, component: str) -> Dict[str, Any]:
        """Resolve ``llm_d.extra_values`` for one component.

        Accepts two shapes, because the common case is overriding the model
        servers and the awkward case is overriding a chart whose schema
        madengine does not model:

        * keyed by component — ``{"gaie": {...}, "modelservice": {...}}``
        * a bare dict — applied to ``modelservice``, the usual target
        """
        extra = self.llmd_config.get("extra_values") or {}
        if not extra:
            return {}
        if any(key in extra for key in COMPONENTS):
            return extra.get(component) or {}
        return extra if component == "modelservice" else {}

    def _infra_values(self) -> Dict[str, Any]:
        """Values for the llm-d-infra chart (the Gateway)."""
        return {
            "gateway": {
                "enabled": True,
                "gatewayClassName": self.llmd_config.get("gateway", "agentgateway"),
            }
        }

    def _gaie_values(self) -> Dict[str, Any]:
        """Values for the GAIE inferencepool chart (InferencePool + EPP)."""
        return {
            "inferencePool": {
                "targetPortNumber": MODEL_SERVER_PORT,
                "modelServers": {"matchLabels": dict(MODEL_SERVER_LABEL)},
            },
            "inferenceExtension": {"replicas": 1},
            "provider": {"name": self.llmd_config.get("gateway", "agentgateway")},
        }

    def _modelservice_values(self) -> Dict[str, Any]:
        """Values for the llm-d-modelservice chart (prefill/decode servers)."""
        model = self.llmd_config.get("model") or {}
        prefill = self.llmd_config.get("prefill") or {}
        decode = self.llmd_config.get("decode") or {}

        artifacts: Dict[str, Any] = {"uri": model.get("uri")}
        # The HF token is referenced by Secret name and read by the model-server
        # pod. It never appears in a values file or on a helm command line.
        if model.get("hf_token_secret"):
            artifacts["authSecretName"] = model["hf_token_secret"]
        if model.get("size"):
            artifacts["size"] = model["size"]

        return {
            "modelArtifacts": artifacts,
            "routing": {
                "modelName": model.get("name"),
                "servicePort": MODEL_SERVER_PORT,
                "inferencePool": {
                    # The GAIE release owns the InferencePool; modelservice
                    # must reference it, not create a second one.
                    "create": False,
                    "name": self.release_name("gaie"),
                },
                "httpRoute": {"create": True},
            },
            "prefill": self._role_values("prefill", prefill),
            "decode": self._role_values("decode", decode),
        }

    def _role_values(self, role: str, role_config: Dict[str, Any]) -> Dict[str, Any]:
        """Values for one prefill/decode role."""
        replicas = int(role_config.get("replicas", 0))
        if replicas <= 0:
            # Aggregated (non-disaggregated) serving: run decode only.
            return {"create": False}

        gpu_count = str(role_config.get("gpu_count", 1))
        tensor_parallel = int(role_config.get("tensor_parallel", 1))

        container: Dict[str, Any] = {
            "name": "vllm",
            "modelCommand": "vllmServe",
            "args": ["--tensor-parallel-size", str(tensor_parallel)],
            "resources": {
                "limits": {self.gpu_resource_name: gpu_count},
                "requests": {self.gpu_resource_name: gpu_count},
            },
        }
        if role_config.get("image"):
            container["image"] = role_config["image"]

        return {"create": True, "replicas": replicas, "containers": [container]}

    def write_values(self, output_dir: Path) -> Dict[str, Path]:
        """Write every component's values file and return the paths."""
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for component in COMPONENTS:
            path = output_dir / f"llm-d-{component}-values.yaml"
            path.write_text(yaml.safe_dump(self.values(component), sort_keys=False))
            paths[component] = path
        return paths

    # ------------------------------------------------------------------
    # helm
    # ------------------------------------------------------------------

    def _chart_args(self, component: str) -> str:
        """``<ref> --version <version>`` for a component."""
        chart = self.chart(component)
        ref = chart.get("ref")
        version = chart.get("version")
        if not ref:
            raise LlmdStackError(
                f"llm_d.charts.{component}.ref is not set; nothing to install."
            )
        if not version:
            # validate() rejects this well before deploy; belt and braces.
            raise LlmdStackError(
                f"llm_d.charts.{component}.version is not pinned; refusing to "
                "install a floating chart version."
            )
        return f"{shlex.quote(ref)} --version {shlex.quote(str(version))}"

    def install(self, component: str, values_path: Path, timeout: int) -> str:
        """Install or upgrade one component. Returns the release name.

        ``helm upgrade --install`` rather than ``helm install`` so a re-run of
        the same model converges instead of failing on an existing release.

        Raises:
            LlmdStackError: If helm fails.
        """
        release = self.release_name(component)
        command = (
            f"helm upgrade --install {shlex.quote(release)} "
            f"{self._chart_args(component)} "
            f"--namespace {shlex.quote(self.namespace)} "
            f"--values {shlex.quote(str(values_path))} "
            f"--wait --timeout {int(timeout)}s"
        )
        try:
            self.shell.sh(command, timeout=int(timeout) + 60)
        except Exception as e:
            raise LlmdStackError(f"helm install of '{release}' failed: {e}") from e
        return release

    def template(self, component: str, values_path: Path) -> str:
        """Render a component with ``helm template``. Contacts no cluster."""
        release = self.release_name(component)
        command = (
            f"helm template {shlex.quote(release)} "
            f"{self._chart_args(component)} "
            f"--namespace {shlex.quote(self.namespace)} "
            f"--values {shlex.quote(str(values_path))}"
        )
        try:
            return self.shell.sh(command, timeout=300)
        except Exception as e:
            raise LlmdStackError(f"helm template of '{release}' failed: {e}") from e

    def uninstall(self, release: str, timeout: int = 600) -> None:
        """Uninstall a release.

        ``--ignore-not-found`` so a teardown that races a manual cleanup, or
        re-runs after a partial unwind, is not an error.

        Raises:
            LlmdStackError: If helm fails for any other reason.
        """
        command = (
            f"helm uninstall {shlex.quote(release)} "
            f"--namespace {shlex.quote(self.namespace)} "
            f"--ignore-not-found --wait --timeout {int(timeout)}s"
        )
        try:
            self.shell.sh(command, timeout=int(timeout) + 60)
        except Exception as e:
            raise LlmdStackError(f"helm uninstall of '{release}' failed: {e}") from e
