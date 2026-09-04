#!/usr/bin/env python3
"""
llm-d deployment target.

llm-d (https://github.com/llm-d/llm-d) is a Kubernetes-native distributed
inference stack: vLLM/SGLang model servers behind a Gateway API Inference
Extension (InferencePool + Endpoint Picker), with prefix-cache-aware routing and
prefill/decode disaggregation.

madengine's role is to benchmark it. The benchmark client is an ordinary
single-pod Kubernetes Job, which is why this class subclasses
``KubernetesDeployment`` rather than ``BaseDeployment``: ConfigMap script
bundling, registry/runtime Secrets, results and data PVCs, live log streaming,
PVC result harvesting and perf.csv writing are all inherited unchanged.

Two modes:

* **attach** — ``llm_d.endpoint_url`` is set. The stack already exists; madengine
  only runs the benchmark against it, and never tears it down.
* **managed** — madengine stands the stack up with ``helm`` and tears it down
  afterwards.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import shlex
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from madengine.core.console import Console as ShellConsole
from madengine.core.errors import ConfigurationError
from madengine.core.image_digest import resolve_pinned_image

from .base import DeploymentResult, DeploymentStatus
from .config_loader import ConfigLoader, apply_deployment_config
from .k8s_names import sanitize_k8s_container_name, sanitize_k8s_object_name
from .kubernetes import KubernetesDeployment
from .llm_d_stack import COMPONENTS, LlmdStack, LlmdStackError

# CRDs that must exist on the cluster before an llm-d stack can be stood up.
# Checked in managed mode only; madengine reports them but never installs them,
# since that is cluster-admin work.
REQUIRED_CRDS = (
    "gateways.gateway.networking.k8s.io",
    "httproutes.gateway.networking.k8s.io",
)

# The InferencePool API is still moving between groups; accept either.
INFERENCEPOOL_CRDS = (
    "inferencepools.inference.networking.k8s.io",
    "inferencepools.inference.networking.x-k8s.io",
)

# Gateway API group/version used to read the Gateway's published address.
GATEWAY_GROUP = "gateway.networking.k8s.io"
GATEWAY_VERSION = "v1"
GATEWAY_PLURAL = "gateways"

# Helm release names must be <= 53 characters. Reserve room for the longest
# component suffix ("-modelservice") so no component can overflow.
_MAX_RELEASE_PREFIX_LEN = 53 - len("-modelservice")

# Seconds between readiness polls. Model loading is measured in minutes, so a
# tight poll only adds API-server traffic.
_READINESS_POLL_INTERVAL = 10


class LlmdDeployment(KubernetesDeployment):
    """Benchmark an llm-d stack, optionally standing it up first."""

    DEPLOYMENT_TYPE = "llm-d"

    def __init__(self, config):
        """Initialize, layering llm-d defaults over the k8s preset stack."""
        # Apply llm-d defaults first, then let KubernetesDeployment.__init__ run
        # its own apply_deployment_config unchanged. load_k8s_config merges the
        # config it is given at highest priority, so that second pass is a no-op
        # over these values and leaves the "llm_d" block untouched — and
        # self.k8s_config ends up pointing at the final dict, not a stale copy.
        apply_deployment_config(config, ConfigLoader.load_llmd_config)
        super().__init__(config)

        self.llmd_config: Dict[str, Any] = config.additional_context.get("llm_d", {})
        self.endpoint_url: Optional[str] = self.llmd_config.get("endpoint_url")

        # --tags selects the model everywhere else (local, k8s, slurm); do the
        # same here before anything reads model.uri or prefill/decode.image.
        self._resolve_model_uri()
        self._resolve_model_images()

        # Releases installed by this run, newest last. Used to unwind a partial
        # standup and to tear down on the way out.
        self._installed_releases: list = []

        # Shell runner for helm. madengine's Console redacts MAD_SECRETS_* and
        # known token shapes from printed and raised commands.
        self.shell = ShellConsole(live_output=True)

        self._stack: Optional[LlmdStack] = None

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------

    @property
    def is_attach_mode(self) -> bool:
        """True when pointed at an already-running llm-d stack."""
        return bool(self.endpoint_url)

    @property
    def is_dry_run(self) -> bool:
        """True when asked to render the stack without touching it."""
        return bool(self.llmd_config.get("dry_run", False))

    @property
    def should_teardown(self) -> bool:
        """Never tear down a stack madengine did not stand up."""
        if self.is_attach_mode:
            return False
        return bool(self.llmd_config.get("teardown", True))

    @property
    def output_dir(self) -> Path:
        """Where rendered values and manifests are written."""
        return Path(self.k8s_config.get("output_dir", "./k8s_manifests"))

    def _release_prefix(self) -> str:
        """Release-name prefix, derived from the model like the Job name is.

        Uses the same sanitizers as ``KubernetesDeployment.prepare()`` so the
        releases and the benchmark Job carry recognisably the same identity, but
        computed here because standup happens before the parent names the Job.
        """
        model_keys = list(self.manifest["built_models"].keys())
        if not model_keys:
            raise ConfigurationError("No models in manifest")
        raw_model_name = self.manifest["built_models"][model_keys[0]]["name"]

        prefix = self.llmd_config.get("release_prefix") or "madengine"
        name = sanitize_k8s_object_name(
            prefix, raw_model_name, max_total_len=_MAX_RELEASE_PREFIX_LEN
        )
        # Helm release names are DNS labels: no dots.
        return sanitize_k8s_container_name(name, max_len=_MAX_RELEASE_PREFIX_LEN)

    def _resolve_model_uri(self) -> None:
        """Default ``model.uri`` from the simpler ``hf_repo`` field.

        With ``hf_repo`` alone, ``model.uri`` becomes ``hf://<hf_repo>``, which
        re-downloads the model on every standup — sugar, not a PVC-caching
        mechanism. With ``hf_repo`` and ``cache_pvc`` both set, ``model.uri``
        instead becomes ``pvc+hf://<cache_pvc>/hf_hub_cache/<hf_repo>``, and
        ``_populate_model_cache`` downloads the repo onto that PVC before
        standup so the chart's ``pvc+hf://`` scheme — which only mounts an
        *already populated* PVC and contains no download logic of its own —
        has something to mount. An explicit ``model.uri`` always wins here, so
        this is opt-in and changes nothing for existing configs.
        """
        model = self.llmd_config.get("model")
        if not model or model.get("uri") or not model.get("hf_repo"):
            return
        if model.get("cache_pvc"):
            model["uri"] = (
                f"pvc+hf://{model['cache_pvc']}/hf_hub_cache/{model['hf_repo']}"
            )
        else:
            model["uri"] = f"hf://{model['hf_repo']}"

    def _resolve_model_images(self) -> None:
        """Default prefill/decode.image to the ``--tags``-selected model's own
        built image, unless the user already named one explicitly.

        Every other target (local, k8s, slurm) runs the image built from the
        selected model's own Dockerfile; llm-d's model-server roles did not,
        because ``prefill.image``/``decode.image`` had no default. This makes
        ``--tags`` determine what actually serves the model here too — the
        chart still needs ``model.uri``/``hf_repo`` to know which weights to
        load into that image.

        Roles that take this default are recorded in
        ``_roles_using_client_image`` so ``_validate_managed_prerequisites`` can
        warn about it: the default is only correct when the image both serves
        and benchmarks, which a purpose-built slim client image does not.
        """
        self._roles_using_client_image: List[str] = []

        model_keys = list(self.manifest.get("built_models", {}).keys())
        if not model_keys:
            return
        image_info = self.manifest.get("built_images", {}).get(model_keys[0], {})
        built_image = image_info.get("registry_image") or image_info.get("docker_image")
        if not built_image:
            return

        resolved_image = resolve_pinned_image(
            built_image,
            image_info.get("image_digest"),
            bool(self.config.additional_context.get("require_pinned_image")),
            model_name=model_keys[0],
        )
        for role in ("prefill", "decode"):
            role_config = self.llmd_config.setdefault(role, {})
            # Same semantics as setdefault: an explicit image always wins.
            if "image" not in role_config:
                role_config["image"] = resolved_image
                self._roles_using_client_image.append(role)

    def _ensure_model_pvc(self) -> None:
        """Create the shared-data PVC an explicit ``model.uri`` names, if it is
        the one madengine itself manages.

        The chart only *mounts* the named PVC by ``claimName`` — it never
        creates or populates one (see ``_resolve_model_uri``). Only
        ``madengine-shared-data`` is madengine's to create — the exact PVC
        :class:`~madengine.deployment.k8s_pvc.KubernetesPVCMixin` already
        creates or reuses for the k8s target's datasets — so this only saves
        the user a manual ``kubectl apply`` before it gets populated, whether
        by ``_populate_model_cache`` or by hand out-of-band. A ``uri`` naming
        any other PVC is assumed to already exist, the same assumption managed
        mode makes about the namespace itself.
        """
        uri = ((self.llmd_config.get("model") or {}).get("uri")) or ""
        if not (uri.startswith("pvc://") or uri.startswith("pvc+hf://")):
            return
        pvc_name = uri.split("://", 1)[1].split("/", 1)[0]
        if pvc_name != "madengine-shared-data":
            return
        self.console.print(f"[dim]Ensuring model PVC '{pvc_name}' exists...[/dim]")
        self._create_or_get_data_pvc()

    def _populate_model_cache(self) -> None:
        """Download ``model.hf_repo`` onto ``model.cache_pvc`` once, ahead of
        standup, so the chart's ``pvc+hf://`` scheme has something to mount.

        Skipped unless ``cache_pvc`` is set — the default remains the
        ``hf://`` re-download-every-run behaviour ``_resolve_model_uri``
        already produces. Idempotency is left to
        ``huggingface_hub.snapshot_download``'s own per-file check: a repeat
        run still spins up the Job, it just re-verifies rather than
        re-downloads. Job names are not unique per run — Job specs are
        immutable, so a stale Job from a previous run is deleted before a
        fresh one is created, the same convergence goal ``helm upgrade
        --install`` serves for the three chart releases.
        """
        model = self.llmd_config.get("model") or {}
        cache_pvc = model.get("cache_pvc")
        hf_repo = model.get("hf_repo")
        if not cache_pvc or not hf_repo:
            return

        job_name = sanitize_k8s_container_name(f"{self._release_prefix()}-hf-cache")
        timeout = int(model.get("cache_timeout", 7200))

        self.console.print(
            f"[blue]Populating '{hf_repo}' onto PVC '{cache_pvc}'...[/blue]"
        )
        self._delete_cache_job(job_name)
        self.batch_v1.create_namespaced_job(
            namespace=self.namespace,
            body=self._cache_job_manifest(job_name, cache_pvc, hf_repo, model, timeout),
        )
        try:
            self._wait_for_cache_job(job_name, timeout)
            self.console.print(
                f"[green]✓ '{hf_repo}' cached on PVC '{cache_pvc}'[/green]"
            )
        finally:
            self._delete_cache_job(job_name)

    def _cache_job_manifest(
        self,
        job_name: str,
        cache_pvc: str,
        hf_repo: str,
        model: Dict[str, Any],
        timeout: int,
    ) -> Dict[str, Any]:
        """Build the one-off download Job as a plain dict, matching
        ``llm_d_stack.py``'s style for this family of files.

        Pip-installs ``huggingface_hub`` on demand rather than requiring a
        custom image, the same convention
        ``scripts/k8s/data/download_aws.sh`` uses for the AWS CLI. The
        resulting cache directory layout (``models--<org>--<repo>/...`` under
        ``cache_dir``) is exactly what the modelservice chart's ``HF_HUB_CACHE``
        expects to find.
        """
        image = model.get("cache_job_image") or "python:3.11-slim"
        download_script = (
            "import os\n"
            "from huggingface_hub import snapshot_download\n"
            f"snapshot_download(repo_id={hf_repo!r}, "
            "cache_dir='/data/hf_hub_cache', "
            "token=os.environ.get('HF_TOKEN'))\n"
        )
        command = [
            "sh",
            "-c",
            "pip install --no-cache-dir -q -U 'huggingface_hub[hf_transfer]' && "
            f"python -c {shlex.quote(download_script)}",
        ]

        env: List[Dict[str, Any]] = [
            {"name": "HF_HUB_ENABLE_HF_TRANSFER", "value": "1"}
        ]
        if model.get("hf_token_secret"):
            # Same Secret and key the modelservice chart itself reads
            # (llm-d-modelservice's _helpers.tpl: authSecretName -> HF_TOKEN).
            env.append(
                {
                    "name": "HF_TOKEN",
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": model["hf_token_secret"],
                            "key": "HF_TOKEN",
                        }
                    },
                }
            )

        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": job_name, "namespace": self.namespace},
            "spec": {
                "backoffLimit": 1,
                "activeDeadlineSeconds": timeout,
                "template": {
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "populate-cache",
                                "image": image,
                                "command": command,
                                "env": env,
                                "volumeMounts": [
                                    {"name": "cache", "mountPath": "/data"}
                                ],
                            }
                        ],
                        "volumes": [
                            {
                                "name": "cache",
                                "persistentVolumeClaim": {"claimName": cache_pvc},
                            }
                        ],
                    }
                },
            },
        }

    def _wait_for_cache_job(self, job_name: str, timeout: int) -> None:
        """Poll the download Job until it succeeds, fails, or times out."""
        from kubernetes.client.rest import ApiException

        deadline = time.monotonic() + timeout
        while True:
            try:
                job = self.batch_v1.read_namespaced_job_status(
                    name=job_name, namespace=self.namespace
                )
            except ApiException as e:
                # Most likely the Job was deleted out from under us (404), or
                # RBAC forbids reading it. Either way the download's outcome is
                # unknowable, and continuing would mount a PVC that may hold
                # nothing — so report it as the configuration problem it is
                # rather than letting a raw ApiException escape.
                raise ConfigurationError(
                    f"Could not read the status of model-cache download Job "
                    f"'{job_name}': {e}. Populate the PVC out-of-band and set "
                    "llm_d.model.uri to it, or clear llm_d.model.cache_pvc to "
                    "download on every standup instead."
                ) from e
            if job.status.succeeded:
                return
            if job.status.failed:
                raise ConfigurationError(
                    f"Model-cache download Job '{job_name}' failed. Inspect it "
                    f"with 'kubectl -n {self.namespace} logs job/{job_name}'."
                )
            if time.monotonic() >= deadline:
                raise ConfigurationError(
                    f"Timed out after {timeout}s waiting for model-cache "
                    f"download Job '{job_name}'. Large models can exceed the "
                    "default; raise llm_d.model.cache_timeout, or inspect with "
                    f"'kubectl -n {self.namespace} logs job/{job_name}'."
                )
            time.sleep(_READINESS_POLL_INTERVAL)

    def _delete_cache_job(self, job_name: str) -> None:
        """Delete the download Job, ignoring not-found. Never raises."""
        from kubernetes.client.rest import ApiException

        try:
            self.batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                propagation_policy="Background",
            )
        except ApiException as e:
            if e.status != 404:
                self.console.print(
                    f"[yellow]⚠ Could not delete model-cache Job '{job_name}': "
                    f"{e}[/yellow]"
                )

    @property
    def stack(self) -> LlmdStack:
        """The helm driver for this run's stack, created on first use."""
        if self._stack is None:
            self._stack = LlmdStack(
                llmd_config=self.llmd_config,
                namespace=self.namespace,
                release_prefix=self._release_prefix(),
                gpu_resource_name=self.gpu_resource_name,
                shell=self.shell,
            )
        return self._stack

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Validate cluster access, config coherence, and managed-mode tooling."""
        if "slurm" in self.config.additional_context:
            self.console.print(
                "[red]✗ Conflicting deployment configuration: both 'llm_d' and "
                "'slurm' present. llm-d is a Kubernetes-native stack; remove the "
                "'slurm' block.[/red]"
            )
            return False

        model_name = (self.llmd_config.get("model") or {}).get("name")
        if not model_name:
            self.console.print(
                "[red]✗ llm_d.model.name is required — it is the model name sent "
                "in inference requests and recorded in perf.csv.[/red]"
            )
            return False

        self._warn_if_manifest_holds_several_models()

        # Attach mode only ever runs the CPU-only client Job from this cluster:
        # the GPUs belong to a stack madengine did not install, which may not
        # even be in this cluster. KubernetesDeployment.validate() rejects a
        # cluster with no node advertising gpu_resource_name, so attach mode
        # checks access without that gate. Managed mode keeps it — there,
        # madengine really is about to schedule GPU pods here.
        if self.is_attach_mode:
            if not self._validate_cluster_access():
                return False
            self.console.print(
                f"[green]✓ Attach mode: benchmarking existing endpoint "
                f"{self.endpoint_url}[/green]"
            )
            return True

        if not super().validate():
            return False

        return self._validate_managed_prerequisites()

    def _validate_cluster_access(self) -> bool:
        """Check cluster connectivity and namespace, without the GPU-node gate.

        Mirrors the first two checks of ``KubernetesDeployment.validate()``
        rather than calling it, because the third one there — a node
        advertising ``gpu_resource_name`` — is a requirement attach mode does
        not have.
        """
        from kubernetes import client
        from kubernetes.client.rest import ApiException

        try:
            version = client.VersionApi().get_code()
            self.console.print(
                f"[green]✓ Connected to K8s cluster "
                f"(v{version.major}.{version.minor})[/green]"
            )
            self.core_v1.read_namespace(self.namespace)
            self.console.print(f"[green]✓ Namespace '{self.namespace}' exists[/green]")
            return True
        except ApiException as e:
            if e.status == 404:
                self.console.print(
                    f"[yellow]⚠ Namespace '{self.namespace}' not found[/yellow]"
                )
            else:
                self.console.print(f"[red]✗ Validation failed: {e}[/red]")
            return False
        except Exception as e:
            self.console.print(f"[red]✗ Validation failed: {e}[/red]")
            return False

    def _warn_if_manifest_holds_several_models(self) -> None:
        """Warn that only the first model is used.

        Inherited from the Kubernetes target, which also benchmarks
        ``model_keys[0]`` alone — but here the choice additionally decides which
        model the *stack* serves and which image serves it, so it is worth
        saying out loud rather than letting ``--tags`` quietly widen.
        """
        model_keys = list(self.manifest.get("built_models", {}).keys())
        if len(model_keys) < 2:
            return
        self.console.print(
            f"[yellow]⚠ {len(model_keys)} models matched --tags; llm-d "
            f"benchmarks only the first ('{model_keys[0]}') and ignores: "
            f"{', '.join(model_keys[1:])}[/yellow]\n"
            "[yellow]  Narrow --tags to one model to be explicit about which "
            "one the stack serves.[/yellow]"
        )

    def _validate_managed_prerequisites(self) -> bool:
        """Check helm, pinned chart versions, and Gateway API CRDs."""
        if not shutil.which("helm"):
            self.console.print(
                "[red]✗ 'helm' not found on PATH.[/red]\n"
                "[yellow]  Managed mode installs the llm-d charts with helm. Either "
                "install helm (https://helm.sh/docs/intro/install/) or set "
                "llm_d.endpoint_url to benchmark an existing stack.[/yellow]"
            )
            return False

        charts = self.llmd_config.get("charts", {})
        unpinned = [
            name
            for name, spec in charts.items()
            if not name.startswith("_") and not (spec or {}).get("version")
        ]
        if unpinned:
            self.console.print(
                f"[red]✗ Unpinned llm-d chart versions: {', '.join(sorted(unpinned))}[/red]\n"
                "[yellow]  Set llm_d.charts.<name>.version for each. Floating chart "
                "versions make benchmark numbers non-reproducible and let an "
                "upstream release change results without a madengine change.[/yellow]"
            )
            return False

        model_uri = (self.llmd_config.get("model") or {}).get("uri")
        if not model_uri:
            self.console.print(
                "[red]✗ llm_d.model.uri is required in managed mode — it is the "
                "artifact the model servers load. Set llm_d.model.hf_repo instead "
                "(e.g. 'Qwen/Qwen3-32B') to have madengine build "
                "'hf://Qwen/Qwen3-32B' for you.[/red]\n"
                "[yellow]  Set llm_d.endpoint_url instead to benchmark an existing "
                "stack.[/yellow]"
            )
            return False

        if not self._validate_crds():
            return False

        self._warn_if_serving_with_the_client_image()

        self.console.print(
            f"[green]✓ Managed mode: releases "
            f"{', '.join(self.stack.release_names)} in namespace "
            f"{self.namespace}[/green]"
        )
        return True

    def _warn_if_serving_with_the_client_image(self) -> None:
        """Warn when the model servers will run the benchmark client's image.

        ``_resolve_model_images`` defaults ``prefill``/``decode.image`` to the
        ``--tags``-selected model's own built image. That is right for a vLLM
        image, which both serves and benchmarks — and wrong for the slim,
        GPU-less client image the docs otherwise recommend, which cannot serve
        anything. The two cases are indistinguishable from here, so warn rather
        than fail: guessing wrong in either direction is worse than saying so.
        """
        if not self._roles_using_client_image:
            return

        roles = ", ".join(self._roles_using_client_image)
        image = (self.llmd_config.get(self._roles_using_client_image[0]) or {}).get(
            "image"
        )
        self.console.print(
            f"[yellow]⚠ llm-d {roles} will serve the model with this run's own "
            f"benchmark-client image:[/yellow]\n"
            f"[yellow]    {image}[/yellow]\n"
            "[yellow]  That works only if the image is itself a serving "
            "container (vLLM/SGLang baked in). A client-only image will fail to "
            "serve. Set llm_d.prefill.image / llm_d.decode.image to name the "
            "serving image explicitly.[/yellow]"
        )

    def _validate_crds(self) -> bool:
        """Report missing Gateway API / InferencePool CRDs; do not install them."""
        try:
            from kubernetes import client

            api = client.ApiextensionsV1Api()
            present = {
                crd.metadata.name for crd in api.list_custom_resource_definition().items
            }
        except Exception as e:
            # A cluster that refuses CRD listing is a permissions matter, not a
            # reason to fail the run outright; helm will report it precisely.
            self.console.print(
                f"[yellow]⚠ Could not list CRDs ({e}); skipping prerequisite check[/yellow]"
            )
            return True

        missing = [crd for crd in REQUIRED_CRDS if crd not in present]
        if not any(crd in present for crd in INFERENCEPOOL_CRDS):
            missing.append("inferencepools.inference.networking.{,x-}k8s.io")

        if missing:
            self.console.print(
                f"[red]✗ Missing CRDs required by llm-d: {', '.join(missing)}[/red]\n"
                "[yellow]  Install the Gateway API and Gateway API Inference "
                "Extension CRDs first (cluster-admin), e.g.:[/yellow]\n"
                "[yellow]  kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/latest/download/standard-install.yaml[/yellow]"
            )
            return False

        self.console.print(
            "[green]✓ Gateway API and InferencePool CRDs present[/green]"
        )
        return True

    # ------------------------------------------------------------------
    # Deploy
    # ------------------------------------------------------------------

    def prepare(self) -> bool:
        """Stand the stack up (managed mode), then render the client Job.

        Standup happens here rather than in :meth:`deploy` because the Job
        manifest carries ``MAD_LLM_D_ENDPOINT``, and the endpoint is read off
        the live Gateway rather than guessed from chart naming conventions. The
        parent renders that manifest in ``prepare()``, so the endpoint has to be
        known by the time ``super().prepare()`` runs.
        """
        if not self.is_attach_mode:
            try:
                self._standup()
            except Exception as e:
                # Unwind before reporting: a half-installed stack holding GPUs
                # is the worst outcome available.
                self.console.print(f"[red]✗ llm-d standup failed: {e}[/red]")
                self._unwind()
                return False

        return super().prepare()

    def _standup(self) -> None:
        """Install the three releases, wait for readiness, resolve the endpoint.

        Raises:
            LlmdStackError: If any helm operation fails.
            ConfigurationError: If the stack comes up but publishes no address.
        """
        self._ensure_model_pvc()
        self._populate_model_cache()
        values_paths = self.stack.write_values(self.output_dir)
        self.console.print(
            f"[dim]Rendered llm-d chart values to {self.output_dir}[/dim]"
        )

        timeout = int(self.llmd_config.get("standup_timeout", 1800))
        for component in COMPONENTS:
            self.console.print(
                f"[blue]Installing llm-d {component} "
                f"({self.stack.release_name(component)})...[/blue]"
            )
            # Record *before* helm runs, not after. A Ctrl-C mid-install raises
            # KeyboardInterrupt, which is a BaseException and so passes straight
            # through prepare()'s `except Exception` — while helm may already
            # have created the release. Teardown only walks this list, so
            # recording after success is how a GPU-holding release gets
            # orphaned. uninstall passes --ignore-not-found, which makes naming
            # a release that was never created free.
            self._installed_releases.append(self.stack.release_name(component))
            release = self.stack.install(component, values_paths[component], timeout)
            self.console.print(f"[green]✓ Installed {release}[/green]")

        self._wait_for_model_servers()
        self.endpoint_url = self._resolve_endpoint()
        self.console.print(
            f"[green]✓ llm-d gateway ready at {self.endpoint_url}[/green]"
        )

    def _unwind(self) -> None:
        """Roll back a partial standup. Honours ``teardown: false``."""
        if not self._installed_releases:
            return
        if not self.should_teardown:
            self.console.print(
                "[yellow]llm_d.teardown is false — leaving the partially "
                f"installed stack for inspection: "
                f"{', '.join(self._installed_releases)}[/yellow]"
            )
            return
        self.console.print("[yellow]Unwinding partially installed llm-d stack[/yellow]")
        self._teardown_stack()

    def _wait_for_model_servers(self) -> None:
        """Second readiness stage: poll the model-server Deployments.

        ``helm --wait`` can return before a model server has finished loading
        weights, so waiting on it alone produces connection errors that look
        like benchmark failures. This polls the Deployments the modelservice
        release owns until every replica reports Ready.

        A stack whose Deployments carry different labels than expected is not
        an error — ``helm --wait`` already gated readiness once, so this stage
        reports that it found nothing to watch and moves on rather than
        blocking a run that would otherwise succeed. The same applies if the
        API call itself fails (e.g. RBAC forbids listing Deployments): this
        stage is best-effort on top of helm's own readiness gate, not a hard
        requirement.
        """
        from kubernetes import client
        from kubernetes.client.rest import ApiException

        selector = (
            f"app.kubernetes.io/instance={self.stack.release_name('modelservice')}"
        )
        apps_v1 = client.AppsV1Api()
        deadline = time.monotonic() + int(
            self.llmd_config.get("readiness_timeout", 1800)
        )

        while True:
            try:
                deployments = apps_v1.list_namespaced_deployment(
                    namespace=self.namespace, label_selector=selector
                ).items
            except ApiException as e:
                self.console.print(
                    f"[yellow]⚠ Could not list model-server Deployments ({e}); "
                    "relying on helm's own readiness gate[/yellow]"
                )
                return

            if not deployments:
                self.console.print(
                    f"[yellow]⚠ No Deployments matched '{selector}'; relying on "
                    "helm's own readiness gate[/yellow]"
                )
                return

            pending = [
                d
                for d in deployments
                if (d.status.ready_replicas or 0) < (d.spec.replicas or 0)
            ]
            if not pending:
                self.console.print(
                    f"[green]✓ {len(deployments)} model-server Deployment(s) "
                    "ready[/green]"
                )
                return

            if time.monotonic() >= deadline:
                names = ", ".join(d.metadata.name for d in pending)
                raise ConfigurationError(
                    f"Timed out after {self.llmd_config.get('readiness_timeout')}s "
                    f"waiting for model-server Deployments to become ready: {names}. "
                    "Large models can exceed the default; raise "
                    "llm_d.readiness_timeout, or inspect the pods with "
                    f"'kubectl -n {self.namespace} get pods -l {selector}'."
                )

            self.console.print(
                f"[dim]Waiting for {len(pending)} model-server Deployment(s)...[/dim]"
            )
            time.sleep(_READINESS_POLL_INTERVAL)

    def _resolve_endpoint(self) -> str:
        """Read the gateway address off the live Gateway resource.

        The Gateway's ``status.addresses`` is where every Gateway API
        implementation publishes its address, so reading it is authoritative
        where guessing a chart-generated Service name would not be.

        Raises:
            ConfigurationError: If no Gateway or no address can be found.
        """
        from kubernetes import client

        api = client.CustomObjectsApi()
        try:
            gateways = api.list_namespaced_custom_object(
                group=GATEWAY_GROUP,
                version=GATEWAY_VERSION,
                namespace=self.namespace,
                plural=GATEWAY_PLURAL,
            ).get("items", [])
        except Exception as e:
            raise ConfigurationError(
                f"Could not list Gateways in namespace '{self.namespace}': {e}. "
                "Set llm_d.endpoint_url to point madengine at the gateway "
                "directly."
            ) from e

        infra_release = self.stack.release_name("infra")
        owned = [
            g
            for g in gateways
            if (g.get("metadata", {}).get("labels") or {}).get(
                "app.kubernetes.io/instance"
            )
            == infra_release
        ]
        if len(owned) > 1:
            names = ", ".join(
                sorted(g.get("metadata", {}).get("name", "<unnamed>") for g in owned)
            )
            raise ConfigurationError(
                f"Found {len(owned)} Gateways in namespace '{self.namespace}' "
                f"labelled for release '{infra_release}' ({names}). madengine "
                "cannot tell which one fronts the stack, and benchmarking the "
                "wrong one would be worse than failing. Set llm_d.endpoint_url "
                "explicitly."
            )
        # Fall back to a lone Gateway in the namespace: some gateway providers
        # label the Gateway after the gateway class rather than the release.
        candidates = owned or (gateways if len(gateways) == 1 else [])

        if not candidates:
            raise ConfigurationError(
                f"Found {len(gateways)} Gateway(s) in namespace "
                f"'{self.namespace}' and none is identifiably owned by release "
                f"'{infra_release}'. Set llm_d.endpoint_url explicitly."
            )

        gateway = candidates[0]
        addresses = (gateway.get("status") or {}).get("addresses") or []
        address = next((a.get("value") for a in addresses if a.get("value")), None)
        if not address:
            raise ConfigurationError(
                f"Gateway '{gateway.get('metadata', {}).get('name')}' has not "
                "published an address. Its controller may still be provisioning, "
                "or no controller is watching its gatewayClassName "
                f"('{self.llmd_config.get('gateway')}'). Check with "
                f"'kubectl -n {self.namespace} get gateway'."
            )

        # Prefer a plain-HTTP listener over whatever happens to be first: a
        # Gateway commonly publishes both http and https, and listener order is
        # not meaningful. The OpenAI-compatible endpoint the client calls is
        # normally the HTTP one.
        listeners = (gateway.get("spec") or {}).get("listeners") or []
        listener = next(
            (
                candidate
                for candidate in listeners
                if str(candidate.get("protocol", "")).upper() == "HTTP"
            ),
            listeners[0] if listeners else None,
        )
        if listener is None:
            return f"http://{address}:80"

        protocol = str(listener.get("protocol", "HTTP")).upper()
        scheme = "https" if protocol in ("HTTPS", "TLS") else "http"
        port = listener.get("port", 443 if scheme == "https" else 80)
        return f"{scheme}://{address}:{port}"

    def deploy(self) -> DeploymentResult:
        """Submit the benchmark client Job.

        In managed mode the stack is already up: :meth:`prepare` stands it up so
        the rendered Job can carry the resolved endpoint.
        """
        if not self.endpoint_url:
            raise ConfigurationError(
                "No llm-d endpoint resolved. Set llm_d.endpoint_url to benchmark "
                "an existing stack, or llm_d.model.uri to have madengine stand "
                "one up."
            )

        return super().deploy()

    # ------------------------------------------------------------------
    # Dry run
    # ------------------------------------------------------------------

    def _dry_run_report(self) -> DeploymentResult:
        """Render values and ``helm template`` output; install nothing.

        Contacts the cluster for nothing, which is also why it runs none of
        ``validate()``'s cluster checks — a dry run is meant to work against a
        cluster that is not yet ready. ``helm template`` still pulls the chart,
        so ``helm`` must be present and the pinned versions reachable.
        """
        if self.is_attach_mode:
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id="",
                message=(
                    "llm_d.dry_run has nothing to render in attach mode: "
                    "endpoint_url is set, so madengine installs no charts."
                ),
            )

        if not shutil.which("helm"):
            # validate() is skipped on this path, so its helm check never ran.
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id="",
                message=(
                    "'helm' not found on PATH. A dry run renders the charts with "
                    "'helm template'; install helm "
                    "(https://helm.sh/docs/intro/install/) to use it."
                ),
            )

        try:
            values_paths = self.stack.write_values(self.output_dir)
            for component in COMPONENTS:
                rendered = self.stack.template(component, values_paths[component])
                out = self.output_dir / f"llm-d-{component}-manifests.yaml"
                out.write_text(rendered)
                self.console.print(f"[green]✓ Rendered {out}[/green]")
        except (LlmdStackError, ConfigurationError) as e:
            return DeploymentResult(
                status=DeploymentStatus.FAILED, deployment_id="", message=str(e)
            )

        self.console.print(
            f"[yellow]Dry run: nothing was installed and no benchmark ran. "
            f"Review {self.output_dir}, then unset llm_d.dry_run.[/yellow]"
        )
        return DeploymentResult(
            status=DeploymentStatus.SUCCESS,
            deployment_id="",
            message=f"llm-d dry run complete; artifacts in {self.output_dir}",
            metrics={"successful_runs": [], "failed_runs": []},
        )

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def execute(self) -> DeploymentResult:
        """Run the deployment, guaranteeing teardown of anything we installed.

        BaseDeployment.execute() calls cleanup() only on failure or
        KeyboardInterrupt, never on success. That is fine for a Job that has
        already terminated, but helm releases hold GPUs indefinitely, so the
        success path needs an explicit teardown.
        """
        if self.is_dry_run:
            # Short-circuits before validate(), so a dry run needs neither GPU
            # nodes nor a reachable cluster beyond a loadable kubeconfig.
            return self._dry_run_report()

        try:
            return super().execute()
        finally:
            self._teardown_stack()

    def _teardown_stack(self) -> None:
        """Uninstall releases this run installed. Never raises."""
        if not self._installed_releases:
            return
        if not self.should_teardown:
            self.console.print(
                "[yellow]llm_d.teardown is false — leaving the stack running. "
                f"Releases: {', '.join(self._installed_releases)}[/yellow]"
            )
            return

        # Uninstall newest first, so dependents go before their dependencies.
        for release in reversed(list(self._installed_releases)):
            try:
                self._uninstall_release(release)
            except Exception as e:
                # A teardown failure must never mask the benchmark result. Print
                # the exact recovery command instead of raising.
                self.console.print(
                    f"[red]✗ Failed to uninstall release '{release}': {e}[/red]\n"
                    f"[yellow]  Run by hand: helm uninstall {release} -n {self.namespace}[/yellow]"
                )
        self._installed_releases = []

    def _uninstall_release(self, release: str) -> None:
        """Uninstall a single helm release."""
        self.console.print(f"[blue]Uninstalling {release}...[/blue]")
        self.stack.uninstall(release)
        self.console.print(f"[green]✓ Uninstalled {release}[/green]")

    # ------------------------------------------------------------------
    # Template context
    # ------------------------------------------------------------------

    def _prepare_env_vars(self, model_info: Dict) -> Dict[str, str]:
        """Add the llm-d endpoint and topology to the benchmark client's env."""
        env_vars = super()._prepare_env_vars(model_info)
        env_vars.update(self._llmd_env_vars())
        return env_vars

    def _llmd_env_vars(self) -> Dict[str, str]:
        """The MAD_LLM_D_* contract the model's run.sh reads."""
        model = self.llmd_config.get("model") or {}
        prefill = self.llmd_config.get("prefill") or {}
        decode = self.llmd_config.get("decode") or {}

        env = {
            "MAD_LLM_D_ENDPOINT": str(self.endpoint_url or ""),
            "MAD_LLM_D_MODEL": str(model.get("name") or ""),
            "MAD_LLM_D_NAMESPACE": str(self.namespace),
            "MAD_LLM_D_PREFILL_REPLICAS": str(prefill.get("replicas", 0)),
            "MAD_LLM_D_DECODE_REPLICAS": str(decode.get("replicas", 0)),
            "MAD_LLM_D_TP": str(decode.get("tensor_parallel", 1)),
        }
        # Only managed mode has releases to name. The prefix, not a full release
        # name, is what lets a client correlate with 'helm list': the three
        # releases are <prefix>-infra, <prefix>-gaie and <prefix>-modelservice.
        if not self.is_attach_mode:
            env["MAD_LLM_D_RELEASE_PREFIX"] = self.stack.release_prefix
        return env
