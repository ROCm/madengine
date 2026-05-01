"""
Kubernetes template context preparation mixin.

Handles building the Jinja2 template context dictionary, environment variable
preparation, data provider configuration, and tools configuration enrichment
for Kubernetes Job manifest rendering.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import configure_multi_node_profiling
from .k8s_names import sanitize_k8s_container_name, sanitize_k8s_label_value
from .k8s_secrets import (
    CONFIGMAP_MAX_BYTES,
    SECRETS_STRATEGY_FROM_LOCAL,
    build_registry_secret_data,
    estimate_configmap_payload_bytes,
    merge_secrets_config,
    resolve_image_pull_secret_refs,
    resolve_runtime_secret_name,
)
from .primus_backend import (
    infer_primus_backend_from_model_name,
    merged_primus_config,
)
from madengine.core.dataprovider import Data
from madengine.core.errors import ConfigurationError
from madengine.utils.gpu_config import resolve_runtime_gpus
from madengine.utils.path_utils import get_madengine_root


class KubernetesTemplateContextMixin:
    """Template context preparation for Kubernetes manifest rendering."""

    def _prepare_template_context(
        self, model_info: Dict, image_info: Dict
    ) -> Dict[str, Any]:
        """
        Prepare context dictionary for Jinja2 template rendering.

        Args:
            model_info: Model configuration from build_manifest.json
            image_info: Image information from build_manifest.json

        Returns:
            Context dictionary with all template variables
        """
        # Use hierarchical GPU resolution: runtime > deployment > model > default
        additional_context = self.config.additional_context.copy()
        additional_context["k8s"] = self.k8s_config
        gpu_count = resolve_runtime_gpus(model_info, additional_context)
        model_name = model_info["name"]

        # Load manifest and credential content for ConfigMap
        with open(self.config.manifest_file, "r") as f:
            manifest_content = f.read()

        credential_content = "{}"
        credential_path = Path("credential.json")
        if credential_path.exists():
            with open(credential_path, "r") as f:
                credential_content = f.read()

        # Load data.json content if exists
        data_json_content = None
        data_path = Path("data.json")
        if data_path.exists():
            with open(data_path, "r") as f:
                data_json_content = f.read()
            self.console.print(f"[dim]Loaded data.json[/dim]")

        # Load model scripts directory content (entire folder, not just one file)
        # This matches local execution which mounts the entire MODEL_DIR/scripts folder
        model_script_path = model_info.get("scripts")  # e.g., "scripts/dummy/run_data_minio.sh"
        model_script_dir = None
        model_script_filename = None
        model_scripts_contents = {}  # Store all scripts in the directory

        if model_script_path:
            script_file = Path(model_script_path)
            # Extract directory and filename
            model_script_dir = str(script_file.parent)  # e.g., "scripts/dummy"
            model_script_filename = script_file.name     # e.g., "run_data_minio.sh"

            # Bundle entire scripts/<model> directory recursively for reliability across
            # different model types (vllm, sglang, etc.) with varying file types and subdirs
            scripts_dir_path = Path(model_script_dir)
            if scripts_dir_path.exists() and scripts_dir_path.is_dir():
                cwd = Path.cwd()
                for f in scripts_dir_path.rglob("*"):
                    if not f.is_file():
                        continue
                    try:
                        content = f.read_text(encoding="utf-8", errors="strict")
                    except (UnicodeDecodeError, OSError):
                        # Skip binary or unreadable files (ConfigMap is text-only)
                        self.console.print(
                            f"[dim]Skipping non-text file: {f.relative_to(scripts_dir_path)}[/dim]"
                        )
                        continue
                    relative_path = (
                        str(f.relative_to(cwd)) if f.is_absolute() else str(f)
                    )
                    model_scripts_contents[relative_path] = content
                self.console.print(
                    f"[dim]Loaded {len(model_scripts_contents)} file(s) from {model_script_dir}[/dim]"
                )
            elif script_file.exists():
                # Fallback: load single file if directory doesn't exist
                with open(script_file, "r") as f:
                    model_scripts_contents[model_script_path] = f.read()
                self.console.print(f"[dim]Loaded single script: {model_script_path}[/dim]")
            else:
                self.console.print(f"[yellow]Warning: Script not found: {model_script_path}[/yellow]")

        # Load K8s tools configuration
        k8s_tools_config = self._load_k8s_tools()

        # Prepare data configuration first
        data_config = self._prepare_data_config(model_info)

        # Store for use in deploy() method
        self._data_config = data_config

        # K8s best practice: Auto-create shared data PVC if needed
        # K8s philosophy: Separate compute (pods) from storage (PVC)
        if data_config and not self.k8s_config.get("data_pvc"):
            # PVC will be auto-created during deployment
            # Use consistent name for reusability across training runs
            self.console.print(
                f"[cyan]📦 Data provider detected: Will auto-create shared data PVC[/cyan]"
            )
            self.console.print(
                f"[dim]   PVC name: madengine-shared-data (reusable across runs)[/dim]"
            )
            self.console.print(
                f"[dim]   Access mode: RWO for single-node, RWX for multi-node (auto-selected)[/dim]"
            )
            self.console.print(
                f"[dim]   To use existing PVC, add 'data_pvc' to your K8s config[/dim]"
            )
            # Set PVC name now so templates are rendered with correct value
            self.k8s_config["data_pvc"] = "madengine-shared-data"

        # Determine data provider script if model needs data
        data_provider_script = None
        data_provider_script_content = None
        if data_config:
            provider_type = data_config.get("provider_type", "local")
            if provider_type in k8s_tools_config.get("data_providers", {}):
                data_provider_script = k8s_tools_config["data_providers"][provider_type]

                # Load K8s data provider script content
                k8s_script_path = get_madengine_root() / data_provider_script["script"]
                if k8s_script_path.exists():
                    with open(k8s_script_path, "r") as f:
                        data_provider_script_content = f.read()
                    self.console.print(f"[dim]Loaded K8s data provider: {data_provider_script['script']}[/dim]")
                else:
                    self.console.print(f"[yellow]Warning: K8s script not found: {k8s_script_path}[/yellow]")

        # Get launcher configuration from manifest's deployment_config or additional_context
        deployment_config = self.manifest.get("deployment_config", {})
        distributed_config = deployment_config.get("distributed", {})
        launcher_config = self.config.additional_context.get("launcher", {})

        # Merge manifest and runtime launcher config (runtime overrides)
        # Use explicit None checking to handle 0 values correctly
        launcher_type = (
            launcher_config.get("type")
            if launcher_config.get("type") is not None
            else distributed_config.get("launcher")
        )

        nnodes = (
            launcher_config.get("nnodes")
            if launcher_config.get("nnodes") is not None
            else distributed_config.get("nnodes", 1)
        )

        # Store for use in deploy() method
        self._nnodes = nnodes

        nproc_per_node = (
            launcher_config.get("nproc_per_node")
            if launcher_config.get("nproc_per_node") is not None
            else distributed_config.get("nproc_per_node")
            if distributed_config.get("nproc_per_node") is not None
            else int(model_info.get("n_gpus", 1))
        )

        master_port = launcher_config.get("master_port", 29500)

        # Validate configuration
        if launcher_type == "torchrun":
            if not isinstance(nnodes, int) or nnodes < 1:
                raise ValueError(f"Invalid nnodes: {nnodes}. Must be positive integer >= 1")
            if not isinstance(nproc_per_node, int) or nproc_per_node < 1:
                raise ValueError(f"Invalid nproc_per_node: {nproc_per_node}. Must be positive integer >= 1")

            self.console.print(f"[cyan]Configuring torchrun: {nnodes} nodes × {nproc_per_node} GPUs/node[/cyan]")

        elif launcher_type == "deepspeed":
            if not isinstance(nnodes, int) or nnodes < 1:
                raise ValueError(f"Invalid nnodes: {nnodes}. Must be positive integer >= 1")
            if not isinstance(nproc_per_node, int) or nproc_per_node < 1:
                raise ValueError(f"Invalid nproc_per_node: {nproc_per_node}. Must be positive integer >= 1")

            self.console.print(f"[cyan]Configuring DeepSpeed: {nnodes} nodes × {nproc_per_node} GPUs/node[/cyan]")

        elif launcher_type == "torchtitan":
            if not isinstance(nnodes, int) or nnodes < 1:
                raise ValueError(f"Invalid nnodes: {nnodes}. Must be positive integer >= 1")
            if not isinstance(nproc_per_node, int) or nproc_per_node < 1:
                raise ValueError(f"Invalid nproc_per_node: {nproc_per_node}. Must be positive integer >= 1")

            self.console.print(f"[cyan]Configuring TorchTitan: {nnodes} nodes × {nproc_per_node} GPUs/node[/cyan]")

        elif launcher_type == "vllm":
            if not isinstance(nnodes, int) or nnodes < 1:
                raise ValueError(f"Invalid nnodes: {nnodes}. Must be positive integer >= 1")
            if not isinstance(nproc_per_node, int) or nproc_per_node < 1:
                raise ValueError(f"Invalid nproc_per_node: {nproc_per_node}. Must be positive integer >= 1")

            self.console.print(f"[cyan]Configuring vLLM: {nnodes} nodes × {nproc_per_node} GPUs/node[/cyan]")

        elif launcher_type == "sglang":
            if not isinstance(nnodes, int) or nnodes < 1:
                raise ValueError(f"Invalid nnodes: {nnodes}. Must be positive integer >= 1")
            if not isinstance(nproc_per_node, int) or nproc_per_node < 1:
                raise ValueError(f"Invalid nproc_per_node: {nproc_per_node}. Must be positive integer >= 1")

            self.console.print(f"[cyan]Configuring SGLang: {nnodes} nodes × {nproc_per_node} GPUs/node[/cyan]")

        elif launcher_type == "megatron":
            if not isinstance(nnodes, int) or nnodes < 1:
                raise ValueError(f"Invalid nnodes: {nnodes}. Must be positive integer >= 1")
            if not isinstance(nproc_per_node, int) or nproc_per_node < 1:
                raise ValueError(f"Invalid nproc_per_node: {nproc_per_node}. Must be positive integer >= 1")

            self.console.print(f"[cyan]Configuring Megatron-LM: {nnodes} nodes × {nproc_per_node} GPUs/node[/cyan]")

        elif launcher_type == "primus":
            if not isinstance(nnodes, int) or nnodes < 1:
                raise ValueError(f"Invalid nnodes: {nnodes}. Must be positive integer >= 1")
            if not isinstance(nproc_per_node, int) or nproc_per_node < 1:
                raise ValueError(f"Invalid nproc_per_node: {nproc_per_node}. Must be positive integer >= 1")

            self.console.print(f"[cyan]Configuring Primus: {nnodes} nodes × {nproc_per_node} GPUs/node[/cyan]")
            self._bundle_primus_k8s_examples_overlay(model_scripts_contents, model_name)

        # Determine if we need multi-node setup
        create_headless_service = False
        launcher_command = None

        if launcher_type == "torchrun":
            if nnodes > 1:
                create_headless_service = True
                self.console.print(f"[dim]Multi-node detected: Creating headless service for pod discovery[/dim]")

            # Generate torchrun launcher command
            launcher_command = self._generate_torchrun_command(
                nnodes=nnodes,
                nproc_per_node=nproc_per_node,
                master_port=master_port,
                model_script=model_info.get("scripts", "run.sh")
            )

        elif launcher_type == "deepspeed":
            if nnodes > 1:
                create_headless_service = True
                self.console.print(f"[dim]Multi-node DeepSpeed: Creating headless service for pod discovery[/dim]")

            model_script = model_info.get("scripts", "run.sh")

            # Check if script is a bash script - if so, execute it directly
            # as it will handle the launcher internally
            if model_script.endswith('.sh'):
                self.console.print(f"[dim]Detected bash script ({model_script}), will execute directly[/dim]")
                launcher_command = self._generate_bash_script_command(
                    nnodes=nnodes,
                    nproc_per_node=nproc_per_node,
                    master_port=master_port,
                    model_script=model_script
                )
            else:
                # Python script - use DeepSpeed launcher
                launcher_command = self._generate_deepspeed_command(
                    nnodes=nnodes,
                    nproc_per_node=nproc_per_node,
                    master_port=master_port,
                    model_script=model_script
                )

        elif launcher_type == "torchtitan":
            if nnodes > 1:
                create_headless_service = True
                self.console.print(f"[dim]Multi-node TorchTitan: Creating headless service for pod discovery[/dim]")

            # Generate TorchTitan launcher command
            launcher_command = self._generate_torchtitan_command(
                nnodes=nnodes,
                nproc_per_node=nproc_per_node,
                master_port=master_port,
                model_script=model_info.get("scripts", "run.sh")
            )

        elif launcher_type == "vllm":
            if nnodes > 1:
                create_headless_service = True
                self.console.print(f"[dim]Multi-node vLLM: Creating headless service for Ray cluster[/dim]")

            # Generate vLLM launcher command (pass model args so run.sh gets --model_repo etc.)
            launcher_command = self._generate_vllm_command(
                nnodes=nnodes,
                nproc_per_node=nproc_per_node,
                master_port=master_port,
                model_script=model_info.get("scripts", "run.sh"),
                model_args=model_info.get("args", ""),
            )

        elif launcher_type == "sglang":
            if nnodes > 1:
                create_headless_service = True
                self.console.print(f"[dim]Multi-node SGLang: Creating headless service for Ray cluster[/dim]")

            # Generate SGLang launcher command (pass model args so run.sh gets CLI args)
            launcher_command = self._generate_sglang_command(
                nnodes=nnodes,
                nproc_per_node=nproc_per_node,
                master_port=master_port,
                model_script=model_info.get("scripts", "run.sh"),
                model_args=model_info.get("args", ""),
            )

        elif launcher_type == "sglang-disagg" or launcher_type == "sglang_disagg":
            if nnodes < 3:
                raise ValueError(
                    f"SGLang Disaggregated requires minimum 3 nodes "
                    f"(1 proxy + 1 prefill + 1 decode), got {nnodes}"
                )

            # Always create headless service for disaggregated architecture
            create_headless_service = True
            self.console.print(f"[dim]SGLang Disaggregated: Creating headless service for {nnodes} pods[/dim]")
            self.console.print(f"[dim]  Architecture: 1 proxy + {max(1, (nnodes-1)*2//5)} prefill + {nnodes-1-max(1, (nnodes-1)*2//5)} decode[/dim]")

            # Generate SGLang Disaggregated launcher command
            launcher_command = self._generate_sglang_disagg_command(
                nnodes=nnodes,
                nproc_per_node=nproc_per_node,
                master_port=master_port,
                model_script=model_info.get("scripts", "run.sh")
            )

        elif launcher_type == "megatron":
            if nnodes > 1:
                create_headless_service = True
                self.console.print(f"[dim]Multi-node Megatron-LM: Creating headless service for pod discovery[/dim]")

            # Generate Megatron-LM launcher command
            launcher_command = self._generate_megatron_command(
                nnodes=nnodes,
                nproc_per_node=nproc_per_node,
                master_port=master_port,
                model_script=model_info.get("scripts", "run.sh")
            )

        elif launcher_type == "primus":
            if nnodes > 1:
                create_headless_service = True
                self.console.print(f"[dim]Multi-node Primus: Creating headless service for pod discovery[/dim]")

            # Generate Primus launcher command (env-only: PRIMUS_CONFIG_PATH, PRIMUS_CLI_EXTRA)
            launcher_command = self._generate_primus_command(
                nnodes=nnodes,
                nproc_per_node=nproc_per_node,
                master_port=master_port,
                model_script=model_info.get("scripts", "run.sh"),
                model_args=model_info.get("args", "") or "",
                model_name=model_info.get("name", "") or "",
            )
            primus_cfg = merged_primus_config(self.manifest, self.config.additional_context)
            backend_hint = (primus_cfg.get("backend") or "").strip().lower()
            inferred_backend = infer_primus_backend_from_model_name(
                model_info.get("name", "") or ""
            )
            config_path_lower = (primus_cfg.get("config_path") or "").lower()
            looks_maxtext = (
                backend_hint == "maxtext"
                or inferred_backend == "MaxText"
                or "maxtext" in config_path_lower
            )
            if looks_maxtext and nnodes > 1:
                self.console.print(
                    "[yellow]Warning: Primus MaxText multi-node may run in-container apt installs "
                    "(InfiniBand-related packages) inside run_pretrain.sh. Ensure your image or "
                    "cluster policy allows this, or use a pre-baked image.[/yellow]"
                )

        # Prepare pre/post scripts (similar to local execution)
        pre_scripts = []
        post_scripts = []

        # Get pre/post scripts from manifest context if available
        if "context" in self.manifest:
            if "pre_scripts" in self.manifest["context"]:
                pre_scripts.extend(self.manifest["context"]["pre_scripts"])
            if "post_scripts" in self.manifest["context"]:
                post_scripts.extend(self.manifest["context"]["post_scripts"])

        # Add system environment collection (rocEnvTool) - same as local execution
        # This is controlled by generate_sys_env_details flag (default: True)
        generate_sys_env_details = self.config.additional_context.get("generate_sys_env_details", True)
        if generate_sys_env_details:
            self.gather_system_env_details(pre_scripts, model_info["name"])

        # Add tool pre/post scripts to the execution lists (like local execution)
        self._add_tool_scripts(pre_scripts, post_scripts)

        # Load pre/post script contents for ConfigMap (since madengine not installed in container)
        pre_post_script_contents = self._load_common_scripts(pre_scripts + post_scripts)

        merged_sec = merge_secrets_config(self.k8s_config)
        strategy = merged_sec.get("strategy", SECRETS_STRATEGY_FROM_LOCAL)
        cred_path = Path("credential.json")
        cred_exists = cred_path.exists()

        created_pull_preview: List[str] = []
        if cred_exists and strategy == SECRETS_STRATEGY_FROM_LOCAL:
            try:
                parsed = json.loads(cred_path.read_text(encoding="utf-8"))
                if build_registry_secret_data(parsed):
                    created_pull_preview.append(f"{self.job_name}-registry-pull")
            except (json.JSONDecodeError, OSError):
                pass

        if strategy == SECRETS_STRATEGY_FROM_LOCAL:
            include_credential_in_configmap = not cred_exists
        else:
            include_credential_in_configmap = False

        created_runtime_name: Optional[str] = (
            f"{self.job_name}-runtime"
            if strategy == SECRETS_STRATEGY_FROM_LOCAL and cred_exists
            else None
        )
        runtime_credentials_secret_name = resolve_runtime_secret_name(
            strategy, merged_sec, created_runtime_name
        )

        image_pull_secrets = resolve_image_pull_secret_refs(
            strategy, merged_sec, created_pull_preview
        )

        ap_prof = self.k8s_config.get("allow_privileged_profiling")
        if ap_prof is None:
            privileged_profiling = bool(self._get_tools_config())
        else:
            privileged_profiling = bool(ap_prof)

        _pytorch_native = frozenset(
            {"torchrun", "deepspeed", "torchtitan", "megatron", "primus"}
        )
        subdomain_val = (
            self.service_name
            if nnodes > 1 and launcher_type in _pytorch_native
            else None
        )

        # Build complete context
        context = {
            # Job metadata
            "job_name": self.job_name,
            "job_label": self.job_label,
            "main_container_name": getattr(
                self, "main_container_name", None
            )
            or sanitize_k8s_container_name(self.job_name),
            "namespace": self.namespace,
            "model_name": model_name,
            "model_label": sanitize_k8s_label_value(model_name),
            # ConfigMap
            "configmap_name": self.configmap_name,
            "manifest_content": manifest_content,
            "credential_content": credential_content,
            "include_credential_in_configmap": include_credential_in_configmap,
            "runtime_credentials_secret_name": runtime_credentials_secret_name,
            "image_pull_secrets": image_pull_secrets,
            "privileged_profiling": privileged_profiling,
            "ttl_seconds_after_finished": self.k8s_config.get(
                "ttl_seconds_after_finished"
            ),
            "data_json_content": data_json_content,
            "model_scripts_contents": model_scripts_contents,  # All scripts in directory
            "model_script_path": model_script_path,
            "model_script_dir": model_script_dir,
            "model_script_filename": model_script_filename,
            # K8s tools
            "data_provider_script": data_provider_script,
            "data_provider_script_content": data_provider_script_content,
            # Image
            "image": image_info["registry_image"],
            "image_pull_policy": self.k8s_config.get("image_pull_policy", "Always"),
            # Resources
            "gpu_resource_name": self.gpu_resource_name,
            "gpu_count": gpu_count,
            "memory": self.k8s_config.get("memory", "128Gi"),
            "memory_limit": self.k8s_config.get("memory_limit", "256Gi"),
            "cpu": self.k8s_config.get("cpu", "32"),
            "cpu_limit": self.k8s_config.get("cpu_limit", "64"),
            # Job spec
            "completions": nnodes,
            "parallelism": nnodes,
            "completion_mode": "Indexed" if nnodes > 1 else None,
            "backoff_limit": self.k8s_config.get("backoff_limit", 3),
            # Pod spec
            "node_selector": self.k8s_config.get("node_selector", {}),
            "tolerations": self.k8s_config.get("tolerations", []),
            "host_ipc": nnodes > 1,  # Enable for multi-node
            "subdomain": subdomain_val,
            # Execution
            "gpu_visibility": ",".join(str(i) for i in range(gpu_count)),  # e.g., "0" for 1 GPU, "0,1" for 2 GPUs
            "gpu_architecture": self.manifest.get("context", {}).get(
                "gpu_architecture", "gfx90a"
            ),
            "model_script": f"{model_info.get('scripts', 'run.sh')} {model_info.get('args', '')}".strip(),
            "launcher_type": launcher_type,
            "launcher_command": launcher_command,
            "nnodes": nnodes,
            "nproc_per_node": nproc_per_node,
            "master_port": master_port,
            "timeout": self.config.timeout,
            # Environment - Merge base env vars with data/tools env vars
            "env_vars": self._prepare_env_vars(model_info),
            # Volumes
            "results_pvc": f"{self.job_name}-results",  # Always create a PVC for results
            "pvc_name": f"{self.job_name}-results",      # PVC name for template
            "data_pvc": self.k8s_config.get("data_pvc"),
            # Multi-node
            "create_headless_service": create_headless_service,
            "service_name": self.service_name,
            "ports": [29500] if create_headless_service else [],
            # Data provider configuration (already prepared above)
            "data_config": data_config,
            # Tools configuration - from manifest.context or additional_context
            "tools_config": self._get_tools_config(),
            # Tool command chains (pre-built for template)
            "launcher_tool_chain": self._build_tool_command_chain(
                self._get_tools_config(), "bash /tmp/run_launcher.sh"
            ) if launcher_command else None,
            "direct_script_tool_chain": self._build_tool_command_chain(
                self._get_tools_config(), f"bash {model_info.get('scripts', 'run.sh')}"
            ),
            # Pre/Post scripts - includes rocEnvTool and any user-defined scripts
            "pre_scripts": pre_scripts,
            "post_scripts": post_scripts,
            # Common script contents for ConfigMap (embedded since madengine not in container)
            "common_script_contents": pre_post_script_contents,
            # Multiple results file (e.g. perf_dummy.csv) - copied to PVC for K8s result collection
            "multiple_results": model_info.get("multiple_results") or "",
        }

        est = estimate_configmap_payload_bytes(context)
        if est > CONFIGMAP_MAX_BYTES:
            raise ConfigurationError(
                f"ConfigMap payload would be ~{est} bytes; Kubernetes limit is ~1 MiB. "
                "Reduce embedded scripts or use a smaller scripts directory."
            )

        return context

    def _get_tools_config(self) -> List[Dict]:
        """
        Get tools configuration from manifest.context or additional_context.

        Prioritizes runtime additional_context, falls back to manifest.context.

        For multi-node runs:
        - Checks rocprofv3 availability (required for MPI profiling)
        - Upgrades "rocprof" to "rocprofv3" for multi-node compatibility
        - Logs warnings if rocprofv3 not available

        Returns:
            List of tool configurations (enriched with cmd from tools.json)
        """
        # Cache the result to avoid repeated expensive checks and duplicate warnings
        if hasattr(self, '_cached_tools_config'):
            return self._cached_tools_config

        # Check runtime additional_context first (allows runtime override)
        tools = self.config.additional_context.get("tools", [])

        # Fall back to manifest.context if no runtime tools
        if not tools and "context" in self.manifest:
            tools = self.manifest["context"].get("tools", [])

        # Apply multi-node profiling logic if applicable
        distributed_config = self.config.additional_context.get("distributed", {})
        nnodes = distributed_config.get("nnodes", 1)

        if nnodes > 1 and tools:
            # Configure multi-node profiling (handles rocprofv3 detection and tool upgrades)
            # Create a simple logger wrapper for configure_multi_node_profiling
            class ConsoleLogger:
                def __init__(self, console):
                    self.console = console

                def info(self, msg):
                    self.console.print(f"[cyan]{msg}[/cyan]")

                def warning(self, msg):
                    self.console.print(f"[yellow]{msg}[/yellow]")

                def debug(self, msg):
                    pass  # Skip debug messages in console

            profiling_config = configure_multi_node_profiling(
                nnodes=nnodes,
                tools_config=tools,
                logger=ConsoleLogger(self.console)
            )

            if profiling_config["enabled"]:
                tools = profiling_config["tools"]
            else:
                # rocprofv3 not available - skip profiling for multi-node
                tools = []

        # Enrich tools with cmd from tools.json for K8s template usage
        result = self._enrich_tools_with_cmd(tools)

        # Cache the result for subsequent calls
        self._cached_tools_config = result
        return result

    def _build_tool_command_chain(self, tools_config: List[Dict], base_command: str) -> str:
        """
        Build a command chain from multiple tools, wrapping the base command.

        Tools are chained from outermost to innermost:
        tool_n wraps tool_2 wraps tool_1 wraps base_command

        Each tool's OUTPUT_FILE env var is set inline to avoid conflicts.

        Args:
            tools_config: List of enriched tool configurations
            base_command: The base command to wrap (e.g., "bash /tmp/run_launcher.sh")

        Returns:
            Complete command chain string
        """
        if not tools_config:
            return base_command

        # Filter tools that have a cmd field
        tools_with_cmd = [t for t in tools_config if t.get("cmd")]

        if not tools_with_cmd:
            return base_command

        # Build command chain from inside out (reverse order)
        cmd_chain = base_command
        for tool in reversed(tools_with_cmd):
            tool_cmd = tool["cmd"].replace("../scripts/common/", "scripts/common/")

            # Set OUTPUT_FILE inline for this specific tool (if defined in tool's env_vars)
            tool_env_vars = tool.get("env_vars", {})
            if "OUTPUT_FILE" in tool_env_vars:
                output_file = tool_env_vars["OUTPUT_FILE"]
                # Prepend OUTPUT_FILE=value to this tool's command only
                cmd_chain = f"OUTPUT_FILE={output_file} {tool_cmd} {cmd_chain}"
            else:
                cmd_chain = f"{tool_cmd} {cmd_chain}"

        return cmd_chain

    def _enrich_tools_with_cmd(self, tools: List[Dict]) -> List[Dict]:
        """
        Enrich tools configuration with cmd field from tools.json.

        This is needed for K8s template to generate the correct encapsulation command.

        Args:
            tools: List of tool configurations (may only have 'name' field)

        Returns:
            Enriched list with 'cmd' field added from tools.json
        """
        if not tools:
            return tools

        # Load tools.json
        tools_json_path = Path(__file__).parent.parent / "scripts" / "common" / "tools.json"
        if not tools_json_path.exists():
            self.console.print(f"[yellow]Warning: tools.json not found at {tools_json_path}[/yellow]")
            return tools

        with open(tools_json_path, "r") as f:
            tools_definitions = json.load(f)

        enriched_tools = []
        for tool in tools:
            tool_name = tool.get("name")
            if not tool_name:
                enriched_tools.append(tool)
                continue

            # Get tool definition from tools.json
            if tool_name not in tools_definitions.get("tools", {}):
                self.console.print(f"[yellow]Warning: Tool '{tool_name}' not found in tools.json[/yellow]")
                enriched_tools.append(tool)
                continue

            tool_def = tools_definitions["tools"][tool_name]

            # Create enriched tool config with cmd
            enriched_tool = tool.copy()
            if "cmd" not in enriched_tool and "cmd" in tool_def:
                enriched_tool["cmd"] = tool_def["cmd"]

            # Also copy env_vars if present
            if "env_vars" not in enriched_tool and "env_vars" in tool_def:
                enriched_tool["env_vars"] = tool_def["env_vars"]

            enriched_tools.append(enriched_tool)

        return enriched_tools

    def _prepare_env_vars(self, model_info: Dict) -> Dict[str, str]:
        """
        Prepare environment variables from multiple sources.

        Merges env vars from:
        1. Base additional_context
        2. Data provider
        3. Tools configuration

        Args:
            model_info: Model configuration

        Returns:
            Merged environment variables dict
        """
        env_vars = {}

        # 1. Base environment variables from additional_context
        base_env = self.config.additional_context.get("env_vars", {})
        env_vars.update(base_env)

        # 1b. Critical ROCm environment variable (if not already set)
        # HSA_NO_SCRATCH_RECLAIM=1 required for AMD MI300X and newer GPUs
        # Prevents performance degradation and NCCL errors
        if "HSA_NO_SCRATCH_RECLAIM" not in env_vars:
            env_vars["HSA_NO_SCRATCH_RECLAIM"] = "1"

        # 2. Data provider environment variables
        data_config = self._prepare_data_config(model_info)
        if data_config:
            if "env_vars" in data_config:
                # Exclude MAD_DATAHOME from data provider's env vars (we set it explicitly below for K8s)
                data_provider_env = {k: v for k, v in data_config["env_vars"].items() if k != "MAD_DATAHOME"}
                env_vars.update(data_provider_env)
            # Always set MAD_DATAHOME for K8s (PVC mount point /data, not /data_dlm_0)
            if "datahome" in data_config:
                env_vars["MAD_DATAHOME"] = data_config["datahome"]

        # 3. Tools configuration environment variables
        # Check both additional_context and manifest.context for tools
        tools_config = self.config.additional_context.get("tools", [])
        if not tools_config and "context" in self.manifest:
            tools_config = self.manifest["context"].get("tools", [])

        for tool in tools_config:
            if "env_vars" in tool:
                # Skip OUTPUT_FILE as it's set inline in command chain to avoid conflicts
                tool_env_vars = {k: v for k, v in tool["env_vars"].items() if k != "OUTPUT_FILE"}
                env_vars.update(tool_env_vars)

        return env_vars

    def _prepare_data_config(self, model_info: Dict) -> Optional[Dict]:
        """
        Prepare data provider configuration for K8s pod.

        Args:
            model_info: Model configuration

        Returns:
            Data configuration dict or None
        """
        if "data" not in model_info or not model_info["data"]:
            return None

        # Initialize data provider if needed
        if not self.data:
            try:
                # Create minimal context for data provider
                # We only need the data.json file to be present
                data_json_file = "data.json"
                if os.path.exists(data_json_file):
                    # Import Context and create minimal instance
                    # Data provider needs this to function
                    self.context_for_data = type('obj', (object,), {
                        'ctx': {},
                        'sh': lambda cmd: os.popen(cmd).read().strip()
                    })()
                    self.data = Data(
                        self.context_for_data,
                        filename=data_json_file,
                        force_mirrorlocal=False
                    )
                else:
                    self.console.print("[yellow]Warning: data.json not found, data provider unavailable[/yellow]")
                    return None
            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not initialize data provider: {e}[/yellow]")
                return None

        try:
            # Get data environment variables
            data_env = self.data.get_env(model_info["data"])

            # Find data provider for this data
            dp = self.data.find_dataprovider(model_info["data"])
            if not dp:
                self.console.print(f"[yellow]Warning: Data provider not found for {model_info['data']}[/yellow]")
                return None

            # Get provider type and source path
            provider_type = dp.provider_type if hasattr(dp, 'provider_type') else "local"
            source_url = dp.config.get("path", "") if hasattr(dp, 'config') else ""

            # K8s best practice: Always use /data (PVC mount point)
            # PVC provides persistent, shared storage across all pods/nodes
            # Separation of storage (PVC) from compute (pods) is K8s standard
            # FORCE datahome to /data for K8s (override data provider's default /data_dlm_0)

            # Filter out MAD_DATAHOME from data provider env vars (will be set explicitly below)
            filtered_data_env = {k: v for k, v in (data_env or {}).items() if k != "MAD_DATAHOME"}
            # Add MAD_DATAHOME with correct K8s value
            filtered_data_env["MAD_DATAHOME"] = "/data"

            return {
                "data_name": model_info["data"],
                "env_vars": filtered_data_env,
                "provider_type": provider_type,
                "source_url": source_url,
                "datahome": "/data",  # Always use PVC mount point for K8s
            }
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not prepare data config: {e}[/yellow]")
            return None
