#!/usr/bin/env python3
"""
Kubernetes Deployment - Container orchestration using Jinja2 templates + Python library.

Uses Jinja2 templates for manifest generation (industry best practice) and
Kubernetes Python client library for applying manifests.
Requires a GPU device plugin matching the configured resource name (AMD or NVIDIA).

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from kubernetes import client
    from kubernetes import config as k8s_config
    from kubernetes.client.rest import ApiException

    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from jinja2 import Template

from .base import BaseDeployment, DeploymentConfig, DeploymentResult, DeploymentStatus, create_jinja_env
from .common import (
    VALID_LAUNCHERS,
    configure_multi_node_profiling,
    is_rocprofv3_available,
    normalize_launcher,
)
from .config_loader import ConfigLoader, apply_deployment_config
from .k8s_secrets import (
    CONFIGMAP_MAX_BYTES,
    SECRETS_STRATEGY_EXISTING,
    SECRETS_STRATEGY_FROM_LOCAL,
    SECRETS_STRATEGY_OMIT,
    create_or_update_secrets_from_credentials,
    delete_job_secrets_if_exist,
    estimate_configmap_payload_bytes,
    merge_secrets_config,
    resolve_image_pull_secret_refs,
    resolve_runtime_secret_name,
    build_registry_secret_data,
)
from madengine.core.dataprovider import Data
from madengine.core.context import Context
from madengine.core.errors import ConfigurationError, create_error_context
from madengine.utils.gpu_config import resolve_runtime_gpus
from madengine.utils.path_utils import get_madengine_root, scripts_base_dir_from
from madengine.utils.run_details import flatten_tags_in_place, get_build_number, get_pipeline

try:
    from madengine.reporting.update_perf_csv import update_perf_csv
    from madengine.reporting.update_perf_super import update_perf_super_json, update_perf_super_csv
    REPORTING_AVAILABLE = True
except ImportError:
    REPORTING_AVAILABLE = False


from .kubernetes_launcher_mixin import KubernetesLauncherMixin


def match_pvc_subdir_to_k8s_pod(
    pvc_subdir: str,
    pod_names: List[str],
    assigned: set,
) -> Optional[str]:
    """
    Map one top-level name under /results/ to a full Kubernetes pod name.

    Matches ``pod == pvc_subdir`` or ``pod.startswith(pvc_subdir + "-")`` among pods
    not yet assigned. Prefer exact equality; if multiple prefix matches, pick the
    first sorted name (deterministic).
    """
    available = sorted(p for p in pod_names if p not in assigned)
    exact = [p for p in available if p == pvc_subdir]
    if exact:
        return exact[0]
    prefixed = [p for p in available if p.startswith(pvc_subdir + "-")]
    if not prefixed:
        return None
    return sorted(prefixed)[0]


def assign_pvc_subdirs_to_pods(pod_dirs: List[str], pod_names: List[str]) -> Dict[str, str]:
    """
    Assign each PVC subdir to at most one pod. Process longest names first so
    short prefixes do not steal pods (e.g. ``foo-0`` before ``foo``).
    """
    cleaned = [d.strip() for d in pod_dirs if d and d.strip()]
    assigned: set = set()
    mapping: Dict[str, str] = {}
    for pvc_subdir in sorted(cleaned, key=lambda x: (-len(x), x)):
        m = match_pvc_subdir_to_k8s_pod(pvc_subdir, pod_names, assigned)
        if m:
            mapping[pvc_subdir] = m
            assigned.add(m)
    return mapping


class KubernetesDeployment(KubernetesLauncherMixin, BaseDeployment):
    """
    Kubernetes cluster deployment using Python client library.

    Uses kubernetes Python API for type-safe, production-ready deployment:
    - client.BatchV1Api(): Job creation and management
    - client.CoreV1Api(): Pod logs and status

    Requires nodes advertising the configured GPU resource (AMD or NVIDIA device plugin).

    **Workflow**:
    1. User has kubeconfig configured (in-cluster or ~/.kube/config)
    2. madengine run --tags model --additional-context '{"deploy": "k8s", ...}'
    3. Creates K8s Job using built Docker image from build phase
    4. Job runs madengine workflow inside container (no docker-in-docker)
    """

    DEPLOYMENT_TYPE = "k8s"
    REQUIRED_TOOLS = []  # No CLI tools needed, uses Python library

    def __init__(self, config: DeploymentConfig):
        """
        Initialize Kubernetes deployment with Jinja2 templates.

        Args:
            config: Deployment configuration

        Raises:
            ImportError: If kubernetes or yaml Python libraries not installed
        """
        if not KUBERNETES_AVAILABLE:
            raise ImportError(
                "Kubernetes Python library not installed.\n"
                "Install with: pip install madengine[kubernetes]\n"
                "Or: pip install kubernetes"
            )

        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML library not installed.\n"
                "Install with: pip install pyyaml"
            )

        apply_deployment_config(config, ConfigLoader.load_k8s_config)
        super().__init__(config)

        # Parse K8s configuration (now with defaults applied)
        self.k8s_config = config.additional_context.get("k8s", {})
        if not self.k8s_config:
            self.k8s_config = config.additional_context.get("kubernetes", {})

        self.namespace = self.k8s_config.get("namespace", "default")
        self.gpu_resource_name = self.k8s_config.get("gpu_resource_name", "amd.com/gpu")

        # Setup Jinja2 template environment
        template_dir = Path(__file__).parent / "templates" / "kubernetes"
        self.jinja_env = create_jinja_env(template_dir)

        # Initialize data provider (will be used if models need data)
        self.data = None
        self.context_for_data = None

        # Load Kubernetes configuration
        kubeconfig_path = self.k8s_config.get("kubeconfig")
        try:
            if kubeconfig_path:
                k8s_config.load_kube_config(config_file=kubeconfig_path)
            else:
                # Try in-cluster first, then default kubeconfig
                try:
                    k8s_config.load_incluster_config()
                except (k8s_config.ConfigException, FileNotFoundError):
                    # Not running in-cluster, try default kubeconfig
                    k8s_config.load_kube_config()
        except Exception as e:
            raise RuntimeError(f"Failed to load Kubernetes config: {e}")

        # Initialize API clients
        self.batch_v1 = client.BatchV1Api()
        self.core_v1 = client.CoreV1Api()

        # Generated resources
        self.job_name = None
        self.configmap_name = None
        self.configmap_yaml = None
        self.job_yaml = None
        self.service_yaml = None

    def validate(self) -> bool:
        """Validate Kubernetes cluster access and configuration."""
        try:
            # Test cluster connectivity
            version = client.VersionApi().get_code()
            self.console.print(
                f"[green]✓ Connected to K8s cluster (v{version.major}.{version.minor})[/green]"
            )

            # Check if namespace exists
            try:
                self.core_v1.read_namespace(self.namespace)
                self.console.print(
                    f"[green]✓ Namespace '{self.namespace}' exists[/green]"
                )
            except ApiException as e:
                if e.status == 404:
                    self.console.print(
                        f"[yellow]⚠ Namespace '{self.namespace}' not found[/yellow]"
                    )
                    return False
                raise

            # Validate GPU device plugin: nodes expose the configured GPU resource
            nodes = self.core_v1.list_node()
            gpu_nodes = [
                n
                for n in nodes.items
                if self.gpu_resource_name in (n.status.allocatable or {})
            ]

            if not gpu_nodes:
                vendor = str(
                    self.config.additional_context.get("gpu_vendor", "AMD")
                ).upper()
                if vendor == "NVIDIA":
                    hint = (
                        "[yellow]  Ensure NVIDIA GPU device plugin is installed, e.g.:[/yellow]\n"
                        "[yellow]  https://github.com/NVIDIA/k8s-device-plugin[/yellow]"
                    )
                else:
                    hint = (
                        "[yellow]  Ensure AMD GPU device plugin is deployed, e.g.:[/yellow]\n"
                        "[yellow]  kubectl create -f https://raw.githubusercontent.com/ROCm/k8s-device-plugin/master/k8s-ds-amdgpu-dp.yaml[/yellow]"
                    )
                self.console.print(
                    f"[yellow]⚠ No nodes with {self.gpu_resource_name} found[/yellow]\n"
                    f"{hint}"
                )
                return False

            self.console.print(
                f"[green]✓ Found {len(gpu_nodes)} node(s) with {self.gpu_resource_name}[/green]"
            )
            return True

        except Exception as e:
            self.console.print(f"[red]✗ Validation failed: {e}[/red]")
            return False

    def prepare(self) -> bool:
        """Generate K8s manifests from Jinja2 templates."""
        try:
            # Get model info
            model_keys = list(self.manifest["built_models"].keys())
            if not model_keys:
                raise ValueError("No models in manifest")

            model_key = model_keys[0]
            model_info = self.manifest["built_models"][model_key]
            image_info = self.manifest["built_images"][model_key]

            # Generate resource names (K8s compatible: lowercase, hyphens)
            model_name = model_info["name"].lower().replace("_", "-")
            self.job_name = f"madengine-{model_name}"
            self.configmap_name = f"{self.job_name}-config"

            # Prepare template context
            context = self._prepare_template_context(model_info, image_info)
            self._image_pull_secrets_for_pods = context.get("image_pull_secrets") or []

            # Render ConfigMap template
            configmap_template = self.jinja_env.get_template("configmap.yaml.j2")
            self.configmap_yaml = configmap_template.render(**context)

            # Render Job template
            job_template = self.jinja_env.get_template("job.yaml.j2")
            self.job_yaml = job_template.render(**context)

            # Optionally render Service template (for multi-node torchrun)
            if context.get("create_headless_service"):
                service_template = self.jinja_env.get_template("service.yaml.j2")
                self.service_yaml = service_template.render(**context)

            # Debug mode: save rendered manifests
            if self.config.additional_context.get("debug", False):
                self._save_debug_manifests()

            self.console.print(
                f"[green]✓ Prepared K8s manifests: {self.job_name}[/green]"
            )
            return True

        except Exception as e:
            self.console.print(f"[red]✗ Failed to prepare manifests: {e}[/red]")
            import traceback

            traceback.print_exc()
            return False

    def gather_system_env_details(
        self, pre_scripts: List[Dict], model_name: str
    ) -> None:
        """
        Gather system environment details by adding rocEnvTool to pre-scripts.
        
        This ensures K8s deployment collects the same system info as local execution.
        
        Args:
            pre_scripts: List of pre-script configurations
            model_name: The model name (used for output file naming)
        """
        # Add rocEnvTool pre-script with model-specific output name
        pre_env_details = {
            "path": "scripts/common/pre_scripts/run_rocenv_tool.sh",
            "args": model_name.replace("/", "_") + "_env"
        }
        pre_scripts.append(pre_env_details)
        self.console.print(f"[dim]Added rocEnvTool to pre-scripts with args: {pre_env_details['args']}[/dim]")
    
    def _add_tool_scripts(self, pre_scripts: List[Dict], post_scripts: List[Dict]) -> None:
        """
        Add tool pre/post scripts to execution lists (similar to local execution).
        
        Extracts pre_scripts and post_scripts from tools.json definitions and adds them
        to the pre_scripts and post_scripts lists for execution in K8s pods.
        
        Args:
            pre_scripts: List to append tool pre-scripts to
            post_scripts: List to append tool post-scripts to
        """
        tools_config = self._get_tools_config()
        if not tools_config:
            return
        
        # Load tools.json to get pre/post script definitions
        tools_json_path = get_madengine_root() / "scripts" / "common" / "tools.json"
        if not tools_json_path.exists():
            return
        
        with open(tools_json_path, "r") as f:
            tools_definitions = json.load(f)
        
        # Add pre/post scripts from each configured tool
        for tool in tools_config:
            tool_name = tool.get("name")
            if not tool_name or tool_name not in tools_definitions.get("tools", {}):
                continue
            
            tool_def = tools_definitions["tools"][tool_name]
            
            # Add pre-scripts (at beginning, like local execution)
            if "pre_scripts" in tool_def:
                pre_scripts[:0] = tool_def["pre_scripts"]
            
            # Add post-scripts (at end, like local execution)
            if "post_scripts" in tool_def:
                post_scripts.extend(tool_def["post_scripts"])
    
    def _load_common_scripts(self, script_list: List[Dict]) -> Dict[str, str]:
        """
        Load common script contents from madengine package for embedding in ConfigMap.
        
        Since madengine is not installed in model Docker images, we need to embed
        the common scripts (pre_scripts, post_scripts, and tool wrapper scripts) in the ConfigMap.
        
        Args:
            script_list: List of script configurations with 'path' field
            
        Returns:
            Dict mapping relative script paths to their contents
        """
        import os
        script_contents = {}
        madengine_root = get_madengine_root()
        
        for script_config in script_list:
            script_path = script_config.get("path", "")
            if not script_path:
                continue
            
            # Convert to absolute path from madengine root
            abs_script_path = madengine_root / script_path
            
            if abs_script_path.exists() and abs_script_path.is_file():
                with open(abs_script_path, "r") as f:
                    script_contents[script_path] = f.read()
                self.console.print(f"[dim]Loaded common script: {script_path}[/dim]")
                
                # If it's run_rocenv_tool.sh, also load the entire rocEnvTool directory
                if "run_rocenv_tool.sh" in script_path:
                    rocenv_dir = abs_script_path.parent / "rocEnvTool"
                    if rocenv_dir.exists() and rocenv_dir.is_dir():
                        # Load all Python files
                        for py_file in rocenv_dir.glob("*.py"):
                            rel_path = f"scripts/common/pre_scripts/rocEnvTool/{py_file.name}"
                            with open(py_file, "r") as f:
                                script_contents[rel_path] = f.read()
                            self.console.print(f"[dim]Loaded rocEnvTool file: {rel_path}[/dim]")
                        
                        # Load all JSON files (e.g., env_tags.json)
                        for json_file in rocenv_dir.glob("*.json"):
                            rel_path = f"scripts/common/pre_scripts/rocEnvTool/{json_file.name}"
                            with open(json_file, "r") as f:
                                script_contents[rel_path] = f.read()
                            self.console.print(f"[dim]Loaded rocEnvTool file: {rel_path}[/dim]")
            else:
                self.console.print(f"[yellow]Warning: Script not found: {script_path} (at {abs_script_path})[/yellow]")
        
        # Load tool wrapper scripts if tools are configured
        tools_config = self._get_tools_config()
        if tools_config:
            self._load_tool_wrapper_scripts(script_contents, tools_config, madengine_root)
        
        return script_contents
    
    def _load_tool_wrapper_scripts(self, script_contents: Dict[str, str], 
                                   tools_config: List[Dict], madengine_root: Path) -> None:
        """
        Load tool wrapper scripts and tools.json for K8s ConfigMap.
        
        This enables profiling tools like rocprof to work in K8s deployments.
        
        Args:
            script_contents: Dict to populate with script contents
            tools_config: List of tool configurations from manifest
            madengine_root: Path to madengine package root
        """
        # Load tools.json first
        tools_json_path = madengine_root / "scripts" / "common" / "tools.json"
        if tools_json_path.exists():
            with open(tools_json_path, "r") as f:
                tools_definitions = json.load(f)
                script_contents["scripts/common/tools.json"] = json.dumps(tools_definitions, indent=2)
            self.console.print(f"[dim]Loaded tools.json[/dim]")
        else:
            self.console.print(f"[yellow]Warning: tools.json not found at {tools_json_path}[/yellow]")
            return
        
        # Extract and load wrapper scripts referenced in tool commands
        for tool in tools_config:
            tool_name = tool.get("name")
            if not tool_name:
                continue
            
            # Get tool definition from tools.json
            if tool_name not in tools_definitions.get("tools", {}):
                self.console.print(f"[yellow]Warning: Tool '{tool_name}' not found in tools.json[/yellow]")
                continue
            
            tool_def = tools_definitions["tools"][tool_name]
            
            # Extract cmd - could be from tool config override or tool definition
            cmd = tool.get("cmd", tool_def.get("cmd", ""))
            
            # Check if cmd references a script in scripts/common/tools/
            if "scripts/common/tools/" in cmd:
                # Parse script path from command (e.g., "bash ../scripts/common/tools/rocprof_wrapper.sh --runtime-trace")
                # or "python3 ../scripts/common/tools/gpu_info_profiler.py"
                # Extract the path portion
                parts = cmd.split()
                for part in parts:
                    if "scripts/common/tools/" in part:
                        # Remove ../ prefix if present
                        script_rel_path = part.replace("../", "")
                        abs_script_path = madengine_root / script_rel_path
                        
                        if abs_script_path.exists() and abs_script_path.is_file():
                            with open(abs_script_path, "r") as f:
                                script_contents[script_rel_path] = f.read()
                            self.console.print(f"[dim]Loaded tool script: {script_rel_path}[/dim]")
                            
                            # If it's a Python script, also load utility modules it might depend on
                            if script_rel_path.endswith('.py'):
                                tools_dir = abs_script_path.parent
                                # Load common utility modules that profiling tools depend on
                                utility_modules = ['amd_smi_utils.py', 'rocm_smi_utils.py', 'pynvml_utils.py']
                                for util_file in utility_modules:
                                    util_path = tools_dir / util_file
                                    if util_path.exists():
                                        util_rel_path = f"scripts/common/tools/{util_file}"
                                        if util_rel_path not in script_contents:
                                            with open(util_path, "r") as f:
                                                script_contents[util_rel_path] = f.read()
                                            self.console.print(f"[dim]Loaded tool utility module: {util_rel_path}[/dim]")
                        else:
                            self.console.print(f"[yellow]Warning: Tool script not found: {script_rel_path} (at {abs_script_path})[/yellow]")
                        break
            
            # Also load any tool-specific pre_scripts and post_scripts
            for script_config in tool_def.get("pre_scripts", []):
                script_path = script_config.get("path", "")
                if script_path and script_path not in script_contents:
                    abs_script_path = madengine_root / script_path
                    if abs_script_path.exists():
                        with open(abs_script_path, "r") as f:
                            script_contents[script_path] = f.read()
                        self.console.print(f"[dim]Loaded tool pre-script: {script_path}[/dim]")
            
            for script_config in tool_def.get("post_scripts", []):
                script_path = script_config.get("path", "")
                if script_path and script_path not in script_contents:
                    abs_script_path = madengine_root / script_path
                    if abs_script_path.exists():
                        with open(abs_script_path, "r") as f:
                            script_contents[script_path] = f.read()
                        self.console.print(f"[dim]Loaded tool post-script: {script_path}[/dim]")
            
            # NEW: Scan pre-scripts for dependencies on scripts/common/tools/ files
            # This handles cases like gpu_info_vram_profiler where the pre-script
            # calls python3 scripts/common/tools/gpu_info_profiler.py but the tool
            # definition has an empty cmd field
            for script_config in tool_def.get("pre_scripts", []):
                script_path = script_config.get("path", "")
                if script_path:
                    abs_script_path = madengine_root / script_path
                    if abs_script_path.exists():
                        # Read the pre-script to find any tool script references
                        with open(abs_script_path, "r") as f:
                            script_content = f.read()
                            # Look for references to scripts/common/tools/ in the pre-script
                            import re
                            # Use non-capturing group (?:...) to avoid capturing just the ../ part
                            tool_refs = re.findall(r'(?:\.\./)?scripts/common/tools/[\w_]+\.py', script_content)
                            for tool_ref in tool_refs:
                                # Clean up the path
                                tool_script_path = tool_ref.strip('"\'').replace("../", "")
                                abs_tool_path = madengine_root / tool_script_path
                                
                                if abs_tool_path.exists() and tool_script_path not in script_contents:
                                    with open(abs_tool_path, "r") as tf:
                                        script_contents[tool_script_path] = tf.read()
                                    self.console.print(f"[dim]Loaded tool dependency: {tool_script_path}[/dim]")
                                    
                                    # Also load utility modules for this Python script
                                    if tool_script_path.endswith('.py'):
                                        tools_dir = abs_tool_path.parent
                                        utility_modules = ['amd_smi_utils.py', 'rocm_smi_utils.py', 'pynvml_utils.py']
                                        for util_file in utility_modules:
                                            util_path = tools_dir / util_file
                                            if util_path.exists():
                                                util_rel_path = f"scripts/common/tools/{util_file}"
                                                if util_rel_path not in script_contents:
                                                    with open(util_path, "r") as uf:
                                                        script_contents[util_rel_path] = uf.read()
                                                    self.console.print(f"[dim]Loaded utility module (from dependency): {util_rel_path}[/dim]")

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
                model_script=model_info.get("scripts", "run.sh")
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
            {"torchrun", "deepspeed", "torchtitan", "megatron"}
        )
        subdomain_val = (
            self.job_name
            if nnodes > 1 and launcher_type in _pytorch_native
            else None
        )

        # Build complete context
        context = {
            # Job metadata
            "job_name": self.job_name,
            "namespace": self.namespace,
            "model_name": model_name,
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
            "service_name": self.job_name,
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

    def _load_k8s_tools(self) -> Dict:
        """
        Load K8s-specific tools configuration.
        
        Returns:
            Dict with K8s tools configuration
        """
        k8s_tools_file = Path(__file__).parent.parent / "scripts" / "k8s" / "tools.json"
        
        if k8s_tools_file.exists():
            try:
                with open(k8s_tools_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.console.print(f"[yellow]Warning: Failed to load K8s tools config: {e}[/yellow]")
                return {}
        else:
            self.console.print(f"[yellow]Warning: K8s tools.json not found at {k8s_tools_file}[/yellow]")
            return {}
    
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
                import os
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

    def _save_debug_manifests(self):
        """Save rendered manifests to disk for debugging."""
        output_dir = Path(self.k8s_config.get("output_dir", "./k8s_manifests"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save ConfigMap
        (output_dir / "configmap.yaml").write_text(self.configmap_yaml)

        # Save Job
        (output_dir / "job.yaml").write_text(self.job_yaml)

        # Save Service if exists
        if self.service_yaml:
            (output_dir / "service.yaml").write_text(self.service_yaml)

        self.console.print(
            f"[yellow]Debug: Manifests saved to {output_dir}[/yellow]"
        )

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
    
    def _cleanup_existing_resources(self):
        """Delete existing Job, ConfigMap, and Service if they exist."""
        # Delete existing Job
        try:
            self.batch_v1.delete_namespaced_job(
                name=self.job_name,
                namespace=self.namespace,
                propagation_policy="Background"
            )
            self.console.print(f"[dim]Deleted existing Job: {self.job_name}[/dim]")
        except ApiException as e:
            if e.status != 404:  # Ignore not found
                pass

        try:
            delete_job_secrets_if_exist(self.core_v1, self.namespace, self.job_name)
            self.console.print(
                f"[dim]Removed job-scoped Secrets for {self.job_name} (if present)[/dim]"
            )
        except ApiException:
            pass
        
        # Delete existing ConfigMap
        try:
            self.core_v1.delete_namespaced_config_map(
                name=self.configmap_name,
                namespace=self.namespace
            )
            self.console.print(f"[dim]Deleted existing ConfigMap: {self.configmap_name}[/dim]")
        except ApiException as e:
            if e.status != 404:
                pass
        
        # Delete existing Service
        if hasattr(self, 'service_yaml') and self.service_yaml:
            try:
                self.core_v1.delete_namespaced_service(
                    name=self.job_name,
                    namespace=self.namespace
                )
                self.console.print(f"[dim]Deleted existing Service: {self.job_name}[/dim]")
            except ApiException as e:
                if e.status != 404:
                    pass
        
        # Delete existing collector pod (must be done before PVC to allow PVC deletion)
        collector_pod_name = f"collector-{self.job_name}"
        try:
            self.core_v1.delete_namespaced_pod(
                name=collector_pod_name,
                namespace=self.namespace,
                grace_period_seconds=0
            )
            self.console.print(f"[dim]Deleted existing collector pod: {collector_pod_name}[/dim]")
            # Wait a moment for pod to release the PVC
            time.sleep(2)
        except ApiException as e:
            if e.status != 404:
                pass
        
        # Delete existing PVC
        pvc_name = f"{self.job_name}-results"
        try:
            self.core_v1.delete_namespaced_persistent_volume_claim(
                name=pvc_name,
                namespace=self.namespace
            )
            self.console.print(f"[dim]Deleted existing PVC: {pvc_name}[/dim]")
            
            # Wait for PVC to be fully deleted (not just marked for deletion)
            max_wait = 90  # Maximum 90 seconds (PV can take time to detach)
            wait_interval = 1  # Check every 1 second
            for i in range(max_wait):
                try:
                    self.core_v1.read_namespaced_persistent_volume_claim(
                        name=pvc_name,
                        namespace=self.namespace
                    )
                    if i > 0 and i % 10 == 0:
                        self.console.print(
                            f"[dim]Waiting for PVC {pvc_name} to be removed... ({i}s)[/dim]"
                        )
                    time.sleep(wait_interval)
                except ApiException as e:
                    if e.status == 404:
                        # PVC is fully deleted
                        break
        except ApiException as e:
            if e.status != 404:
                pass
        
        # Wait a moment for other resources to be deleted
        time.sleep(1)

    def deploy(self) -> DeploymentResult:
        """Apply rendered manifests using kubernetes Python client."""
        try:
            # Clean up any existing resources first
            self._cleanup_existing_resources()
            
            # 1. Create PVC for results storage
            self.console.print("[blue]Creating PVC for results storage...[/blue]")
            nnodes_deploy = getattr(self, "_nnodes", 1)
            pvc_name = self._create_results_pvc(nnodes=nnodes_deploy)
            self.console.print(f"[green]✓ Created PVC: {pvc_name}[/green]")
            
            # 1b. Create or reuse data PVC if data provider is configured and auto-creation was flagged
            if hasattr(self, '_data_config') and self._data_config:
                # Check if we set the PVC name during prepare (auto-creation case)
                data_pvc_name = self.k8s_config.get("data_pvc")
                if data_pvc_name == "madengine-shared-data":
                    # Auto-creation mode: create/reuse the PVC
                    nnodes = getattr(self, '_nnodes', 1)
                    self._create_or_get_data_pvc(nnodes=nnodes)
            
            # 2. Create Secrets from local credential.json (strategy: from_local_credentials)
            merged_sec = merge_secrets_config(self.k8s_config)
            strategy = merged_sec.get("strategy", SECRETS_STRATEGY_FROM_LOCAL)
            cred_path = Path("credential.json")
            if strategy == SECRETS_STRATEGY_FROM_LOCAL and cred_path.exists():
                self.console.print(
                    "[blue]Creating Kubernetes Secrets from credential.json...[/blue]"
                )
                create_or_update_secrets_from_credentials(
                    self.core_v1, self.namespace, self.job_name, cred_path
                )
                self.console.print("[green]✓ Applied registry/runtime Secrets[/green]")

            # 3. Create ConfigMap
            self.console.print("[blue]Creating ConfigMap...[/blue]")
            configmap_dict = yaml.safe_load(self.configmap_yaml)
            self.core_v1.create_namespaced_config_map(
                namespace=self.namespace, body=configmap_dict
            )
            self.console.print(
                f"[green]✓ Created ConfigMap: {self.configmap_name}[/green]"
            )

            # 4. Create Service (if needed for multi-node)
            if self.service_yaml:
                self.console.print("[blue]Creating headless Service...[/blue]")
                service_dict = yaml.safe_load(self.service_yaml)
                self.core_v1.create_namespaced_service(
                    namespace=self.namespace, body=service_dict
                )
                self.console.print(f"[green]✓ Created Service: {self.job_name}[/green]")

            # 5. Create Job
            self.console.print("[blue]Creating Job...[/blue]")
            job_dict = yaml.safe_load(self.job_yaml)
            job = self.batch_v1.create_namespaced_job(
                namespace=self.namespace, body=job_dict
            )

            # Extract image for display
            image = job_dict["spec"]["template"]["spec"]["containers"][0]["image"]

            self.console.print(f"[green]✓ Submitted K8s Job: {self.job_name}[/green]")
            self.console.print(f"  Namespace: {self.namespace}")
            self.console.print(f"  Image: {image}")

            return DeploymentResult(
                status=DeploymentStatus.SUCCESS,
                deployment_id=self.job_name,
                message=f"Job {self.job_name} created successfully",
            )

        except ApiException as e:
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id="",
                message=f"K8s API error: {e.reason} - {e.body}",
            )
        except Exception as e:
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id="",
                message=f"Deployment error: {str(e)}",
            )

    def monitor(self, deployment_id: str) -> DeploymentResult:
        """
        Monitor Job status using Python API.
        
        If live_output is enabled, streams pod logs in real-time.
        Otherwise, polls status periodically.
        """
        # Check if live output is requested
        live_output = self.config.additional_context.get("live_output", False)
        
        if live_output:
            return self._monitor_with_live_logs(deployment_id)
        else:
            return self._monitor_status_only(deployment_id)
    
    def _monitor_status_only(self, deployment_id: str) -> DeploymentResult:
        """Monitor Job status without streaming logs."""
        try:
            job = self.batch_v1.read_namespaced_job_status(
                name=deployment_id, namespace=self.namespace
            )

            # Check job conditions
            if job.status.succeeded:
                return DeploymentResult(
                    status=DeploymentStatus.SUCCESS,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} completed successfully",
                )

            if job.status.failed:
                # Get pod logs to show error
                self._print_pod_logs_on_failure(deployment_id)
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} failed",
                )

            if job.status.active:
                return DeploymentResult(
                    status=DeploymentStatus.RUNNING,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} running ({job.status.active} active pods)",
                )

            return DeploymentResult(
                status=DeploymentStatus.PENDING,
                deployment_id=deployment_id,
                message=f"Job {deployment_id} pending",
            )

        except ApiException as e:
            if e.status == 404:
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} not found",
                )
            raise
    
    def _monitor_with_live_logs(self, deployment_id: str) -> DeploymentResult:
        """Monitor Job and stream logs in real-time."""
        self.console.print(f"\n[cyan]═══ Streaming pod logs (--live-output) ═══[/cyan]\n")
        
        pod_name = None
        log_position = 0
        
        while True:
            try:
                # Check job status
                job = self.batch_v1.read_namespaced_job_status(
                    name=deployment_id, namespace=self.namespace
                )
                
                # Get pod if we don't have it yet
                if not pod_name:
                    pods = self.core_v1.list_namespaced_pod(
                        namespace=self.namespace,
                        label_selector=f"job-name={deployment_id}"
                    )
                    if pods.items:
                        pod_name = pods.items[0].metadata.name
                        self.console.print(f"[dim]Following logs from pod: {pod_name}[/dim]\n")
                
                # Stream logs if we have a pod
                if pod_name:
                    try:
                        # Get logs from current position
                        logs = self.core_v1.read_namespaced_pod_log(
                            name=pod_name,
                            namespace=self.namespace,
                            tail_lines=100 if log_position == 0 else None
                        )
                        
                        # Print new log lines and trigger artifact collection
                        if logs:
                            log_lines = logs.split('\n')
                            if len(log_lines) > log_position:
                                for line in log_lines[log_position:]:
                                    if line.strip():
                                        print(line)
                                log_position = len(log_lines)
                    
                    except ApiException as e:
                        if e.status != 400:  # Ignore "container not ready" errors
                            pass
                
                # Check if job completed
                if job.status.succeeded:
                    self.console.print(f"\n[green]✓ Job {deployment_id} completed successfully[/green]\n")
                    return DeploymentResult(
                        status=DeploymentStatus.SUCCESS,
                        deployment_id=deployment_id,
                        message=f"Job {deployment_id} completed successfully",
                    )
                
                if job.status.failed:
                    self.console.print(f"\n[red]✗ Job {deployment_id} failed[/red]\n")
                    # Print final logs
                    if pod_name:
                        self._print_pod_logs_on_failure(deployment_id)
                    return DeploymentResult(
                        status=DeploymentStatus.FAILED,
                        deployment_id=deployment_id,
                        message=f"Job {deployment_id} failed",
                    )
                
                time.sleep(2)  # Poll every 2 seconds
                
            except ApiException as e:
                if e.status == 404:
                    return DeploymentResult(
                        status=DeploymentStatus.FAILED,
                        deployment_id=deployment_id,
                        message=f"Job {deployment_id} not found",
                    )
                raise
    
    def _print_pod_logs_on_failure(self, deployment_id: str):
        """Print pod logs when job fails (for debugging)."""
        try:
            self.console.print(f"\n[yellow]═══ Pod logs (last 50 lines) ═══[/yellow]\n")
            
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"job-name={deployment_id}"
            )
            
            for pod in pods.items:
                pod_name = pod.metadata.name
                try:
                    logs = self.core_v1.read_namespaced_pod_log(
                        name=pod_name,
                        namespace=self.namespace,
                        tail_lines=50
                    )
                    self.console.print(f"[dim]Pod: {pod_name}[/dim]")
                    print(logs)
                    print()
                except ApiException:
                    pass
        except Exception:
            pass

    def _primary_workload_container_exit_code(self, pod: Any) -> int:
        """
        Exit code of the primary workload container (spec.containers[0]), matched by name
        against container_statuses (ordering-safe if sidecars are added later).
        """
        if not pod.spec or not pod.spec.containers:
            return 0
        primary_name = pod.spec.containers[0].name
        for cs in pod.status.container_statuses or []:
            if cs.name == primary_name and cs.state and cs.state.terminated:
                return cs.state.terminated.exit_code or 0
        # Fallback: first terminated container in spec order
        name_order = [c.name for c in pod.spec.containers]
        for want in name_order:
            for cs in pod.status.container_statuses or []:
                if cs.name == want and cs.state and cs.state.terminated:
                    return cs.state.terminated.exit_code or 0
        return 0

    def _refresh_pod_until_terminal_phase(
        self,
        pod_name: str,
        *,
        timeout_seconds: float = 30.0,
        interval_seconds: float = 0.5,
    ) -> Any:
        """
        Poll read_namespaced_pod until phase is Succeeded or Failed, or timeout.
        Avoids stale list/single-get right after Job completion (phase still Running).
        """
        deadline = time.monotonic() + timeout_seconds
        last: Any = None
        while time.monotonic() < deadline:
            last = self.core_v1.read_namespaced_pod(
                name=pod_name, namespace=self.namespace
            )
            phase = last.status.phase
            if phase in ("Succeeded", "Failed"):
                return last
            time.sleep(interval_seconds)
        return last

    def collect_results(self, deployment_id: str) -> Dict[str, Any]:
        """
        Enhanced results collection from K8s pods following vLLM multi-node best practices.
        
        For Data Parallel deployments (vLLM, SGLang):
        - Each pod runs an independent replica
        - Only pod-0 reports metrics to avoid duplicates
        - Total throughput = pod-0 throughput × num_replicas
        
        Collects:
        1. Pod logs (``k8s_results/<job>/<pod>/pod.log``)
        2. PVC mirror per pod (``.../<pod>/pvc/``), mapped from ``/results/<subdir>/``
        3. File artifacts via kubectl cp when pods are still running (keep-alive path)
        
        Returns:
            Dict with logs, artifacts, and performance results
        """
        results = {
            "job_name": deployment_id,
            "namespace": self.namespace,
            "logs": [],
            "artifacts": [],
            "successful_runs": [],
            "failed_runs": [],
        }

        # Create results directory for this deployment
        results_dir = Path(f"./k8s_results/{deployment_id}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        self.console.print(f"[cyan]📦 Collecting results from K8s job: {deployment_id}[/cyan]")

        try:
            # Get pods for this job
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace, label_selector=f"job-name={deployment_id}"
            )

            # Get model info and build info from manifest
            model_keys = list(self.manifest["built_models"].keys())
            if model_keys:
                model_key = model_keys[0]
                model_info = self.manifest["built_models"][model_key]
            else:
                model_info = {}
            
            # Get build info from built_images
            image_keys = list(self.manifest.get("built_images", {}).keys())
            if image_keys:
                image_key = image_keys[0]
                build_info = self.manifest["built_images"][image_key]
            else:
                build_info = {}

            # Check if this is a multi-node distributed job
            deployment_config = self.manifest.get("deployment_config", {})
            distributed_config = deployment_config.get("distributed", {})
            is_distributed = distributed_config.get("enabled", False)
            nnodes = distributed_config.get("nnodes", 1)
            is_multinode = is_distributed and nnodes > 1
            
            # Determine launcher_type the same way as _prepare_template_context does
            # (deployment_config doesn't store launcher_type directly)
            launcher_config = self.config.additional_context.get("launcher", {})
            launcher_type = (
                launcher_config.get("type") 
                if launcher_config.get("type") is not None 
                else distributed_config.get("launcher")
            )
            
            # Normalize launcher based on deployment type and validity
            launcher_type = normalize_launcher(launcher_type, "kubernetes")
            
            is_ray_launcher = launcher_type in ["vllm", "sglang"]
            
            # Sort pods by name to ensure consistent ordering (pod-0 is master)
            sorted_pods = sorted(pods.items, key=lambda p: p.metadata.name)

            # ========================================================================
            # NEW: Per-Node Collection Strategy
            # Collect logs and artifacts from ALL nodes
            # Parse performance from ALL nodes (each reports node-local metrics)
            # Aggregate metrics based on type (sum for throughput, etc.)
            # ========================================================================
            
            per_node_metrics = []  # Store performance from each node
            results["nodes"] = []  # Store per-node details for display
            
            # Special handling for Ray-based launchers (vLLM, SGLang)
            # These report per-replica metrics, need scaling
            if is_multinode and is_ray_launcher:
                self.console.print(
                    f"[cyan]Multi-node Ray deployment: {nnodes} nodes (Data Parallel mode)[/cyan]"
                )
            
            # Collect from ALL pods
            for pod_index, pod in enumerate(sorted_pods):
                pod_name = pod.metadata.name
                pod_dir = results_dir / pod_name
                pod_dir.mkdir(exist_ok=True)
                
                # Extract node rank from pod name (e.g., madengine-dummy-torchrun-0 -> 0)
                try:
                    node_rank = int(pod_name.rsplit('-', 1)[-1])
                except (ValueError, IndexError):
                    node_rank = pod_index
                
                self.console.print(f"[dim]  Collecting from pod: {pod_name} (node-{node_rank})[/dim]")
                
                try:
                    # 1. Collect pod logs
                    log = self.core_v1.read_namespaced_pod_log(
                        name=pod_name, namespace=self.namespace
                    )
                    log_file = pod_dir / "pod.log"
                    log_file.write_text(log)
                    results["logs"].append({
                        "pod": pod_name,
                        "log": log,
                        "file": str(log_file)
                    })
                    
                    # 2. Parse NODE-LOCAL performance from log
                    perf_data = self._parse_performance_from_log(
                        log, model_info.get("name", "")
                    )
                    
                    # Pod phase/exit can lag right after Job success; poll until terminal or timeout
                    pod = self._refresh_pod_until_terminal_phase(pod_name)
                    pod_status = pod.status.phase if pod else "Unknown"
                    pod_exit_code = (
                        self._primary_workload_container_exit_code(pod) if pod else -1
                    )
                    
                    # Store per-node info for display table
                    node_info = {
                        "node_id": node_rank,
                        "pod_name": pod_name,
                        "status": "SUCCESS" if pod_status == "Succeeded" and pod_exit_code == 0 else "FAILED",
                        "exit_code": pod_exit_code,
                        "performance": perf_data.get("performance") if perf_data else None,
                        "metric": perf_data.get("metric") if perf_data else None,
                        "duration": perf_data.get("duration") if perf_data else None,
                        "log_file": str(log_file)
                    }
                    results["nodes"].append(node_info)
                    
                    if perf_data:
                        # For Ray launchers, this is per-replica metric
                        if is_multinode and is_ray_launcher:
                            perf_data["is_per_replica"] = True
                        per_node_metrics.append(perf_data)
                        self.console.print(
                            f"[green]  ✓ Parsed performance: {perf_data['performance']:.2f} "
                            f"{perf_data['metric']} (node-{node_rank})[/green]"
                        )
                    else:
                        self.console.print(
                            f"[dim]  No performance metric found in node-{node_rank} log[/dim]"
                        )
                        
                except ApiException as e:
                    self.console.print(
                        f"[red]✗ Failed to get logs for pod {pod_name}: {e.reason}[/red]"
                    )
                    results["nodes"].append({
                        "node_id": node_rank,
                        "pod_name": pod_name,
                        "status": "FAILED",
                        "exit_code": -1,
                        "performance": None,
                        "metric": None,
                        "error": f"Failed to get logs: {e.reason}"
                    })
                except Exception as e:
                    self.console.print(
                        f"[red]✗ Error collecting from pod {pod_name}: {e}[/red]"
                    )
                    results["nodes"].append({
                        "node_id": node_rank,
                        "pod_name": pod_name,
                        "status": "FAILED",
                        "exit_code": -1,
                        "performance": None,
                        "metric": None,
                        "error": str(e)
                    })
            
            self.console.print(
                f"[green]✓ Collected logs from {len(results['logs'])} pods[/green]"
            )
            
            # Collect artifacts from PVC before deciding success/failure (needed for multiple_results fallback)
            k8s_pod_names = [p.metadata.name for p in sorted_pods]
            self._collect_from_pvc(deployment_id, results_dir, results, pod_names=k8s_pod_names)
            
            # ========================================================================
            # Aggregate per-node metrics
            # ========================================================================
            if per_node_metrics:
                # Special handling for Ray launchers - multiply by nnodes
                if is_multinode and is_ray_launcher:
                    original_perf = per_node_metrics[0]["performance"]
                    aggregated_perf = original_perf * nnodes
                    self.console.print(
                        f"[green]  Per-replica: {original_perf:.1f} req/s[/green]"
                    )
                    self.console.print(
                        f"[green]  Total capacity: {aggregated_perf:.1f} req/s ({nnodes} nodes)[/green]"
                    )
                    
                    # Create aggregated record manually for Ray
                    aggregated_record = {
                        "model": per_node_metrics[0]["model"],
                        "performance": aggregated_perf,
                        "metric": per_node_metrics[0]["metric"],
                        "status": "SUCCESS",
                        "topology": f"{nnodes}N×{per_node_metrics[0].get('local_gpus', 1)}G",
                        "nnodes": nnodes,
                        "launcher": launcher_type or "N/A",
                        "deployment_type": "kubernetes",
                        "gpu_architecture": per_node_metrics[0].get("gpu_architecture", "N/A"),
                        "duration": per_node_metrics[0].get("duration", "N/A"),
                        "data_name": per_node_metrics[0].get("data_name", "N/A"),
                        "data_provider": per_node_metrics[0].get("data_provider", "N/A"),
                        "aggregation_method": "scaled_by_nnodes",
                        "nodes_contributing": nnodes
                    }
                else:
                    # Use new aggregation logic for other launchers
                    aggregated_record = self._aggregate_node_metrics(
                        per_node_metrics, 
                        nnodes,
                        launcher_type
                    )
                
                if aggregated_record:
                    # Full reporting pipeline: perf_entry at project root, then update_* (same as local/SLURM)
                    self._ensure_perf_csv_exists()
                    run_details_dict = self._build_perf_entry_from_aggregated(
                        aggregated_record, model_info, build_info, deployment_id
                    )
                    perf_entry_path = Path("perf_entry.json")
                    with open(perf_entry_path, "w", encoding="utf-8") as f:
                        json.dump(run_details_dict, f, indent=2)
                    if run_details_dict.get("status") == "SUCCESS":
                        update_perf_csv(perf_csv="perf.csv", single_result=str(perf_entry_path))
                    else:
                        update_perf_csv(perf_csv="perf.csv", exception_result=str(perf_entry_path))
                    scripts_path = model_info.get("scripts", "")
                    scripts_base_dir = scripts_base_dir_from(scripts_path)
                    try:
                        if run_details_dict.get("status") == "SUCCESS":
                            num_entries = update_perf_super_json(
                                single_result=str(perf_entry_path),
                                perf_super_json="perf_super.json",
                                scripts_base_dir=scripts_base_dir,
                            )
                        else:
                            num_entries = update_perf_super_json(
                                exception_result=str(perf_entry_path),
                                perf_super_json="perf_super.json",
                                scripts_base_dir=scripts_base_dir,
                            )
                        update_perf_super_csv(
                            perf_super_json="perf_super.json",
                            perf_super_csv="perf_super.csv",
                            num_entries=num_entries,
                        )
                    except Exception as e:
                        self.console.print(f"[yellow]⚠ Could not update perf_super: {e}[/yellow]")
                    results["successful_runs"].append({
                        "model": model_info.get("name"),
                        "perf_data": aggregated_record,
                        "nodes": results["nodes"],
                        "per_node_metrics": per_node_metrics
                    })
                    self.console.print(
                        f"[green]✓ Aggregated performance from {len(per_node_metrics)} nodes[/green]"
                    )
                    self.console.print(
                        f"[green]✓ Updated perf_entry.json, perf.csv, perf_super.* (Docker-compatible)[/green]"
                    )
            else:
                # No performance from log: try multiple_results CSV (same contract as local Docker)
                # Resolve single CSV path (one pod) or merged CSV path (multi-pod with sum/avg rules)
                resolved_csv_path = self._resolve_multiple_results_csv(
                    results_dir, results, model_info
                )
                if resolved_csv_path and REPORTING_AVAILABLE:
                    # Docker-compatible flow: produce perf.csv, perf_entry.*, perf_super.*
                    gpu_arch = "N/A"
                    if results.get("logs"):
                        import re
                        log_content = results["logs"][0].get("log", "")
                        m = re.search(r"(?:🔹\s*)?Name\s*:\s*(gfx\w+)", log_content)
                        if m:
                            gpu_arch = m.group(1)
                    self._ensure_perf_csv_exists()
                    common_info = self._build_common_info_dict(
                        model_info, build_info, deployment_id, gpu_arch
                    )
                    common_info_path = Path("common_info.json")
                    with open(common_info_path, "w", encoding="utf-8") as f:
                        json.dump(common_info, f, indent=2)
                    update_perf_csv(
                        perf_csv="perf.csv",
                        multiple_results=str(resolved_csv_path),
                        common_info=str(common_info_path),
                        model_name=model_info.get("name", ""),
                    )
                    scripts_path = model_info.get("scripts", "")
                    scripts_base_dir = scripts_base_dir_from(scripts_path)
                    num_entries = update_perf_super_json(
                        perf_super_json="perf_super.json",
                        multiple_results=str(resolved_csv_path),
                        common_info=str(common_info_path),
                        model_name=model_info.get("name", ""),
                        scripts_base_dir=scripts_base_dir,
                    )
                    update_perf_super_csv(
                        perf_super_json="perf_super.json",
                        perf_super_csv="perf_super.csv",
                        num_entries=num_entries,
                    )
                    # Build successful_runs for display (one entry per CSV row)
                    import csv as _csv
                    model_name = model_info.get("name", "")
                    with open(resolved_csv_path, "r", encoding="utf-8", errors="ignore") as f:
                        reader = _csv.DictReader(f)
                        for row in reader:
                            row = {k.strip(): v for k, v in row.items() if k}
                            if row.get("performance") and row.get("metric"):
                                display_model = f"{model_name}_{row.get('model', '')}"
                                record = self._create_multiple_result_row_record(
                                    model_info, build_info, deployment_id,
                                    {
                                        "model": display_model,
                                        "performance": row.get("performance"),
                                        "metric": row.get("metric", ""),
                                        "gpu_architecture": gpu_arch,
                                        "duration": row.get("test_duration", "N/A"),
                                    },
                                )
                                if record:
                                    results["successful_runs"].append({
                                        "model": display_model,
                                        "perf_data": record,
                                        "nodes": [],
                                        "per_node_metrics": [{"model": display_model, "performance": row.get("performance"), "metric": row.get("metric", "")}],
                                    })
                    self.console.print(
                        f"[green]✓ Updated perf.csv, perf_entry.*, perf_super.* (Docker-compatible)[/green]"
                    )
                elif resolved_csv_path and not REPORTING_AVAILABLE:
                    # Fallback when reporting module not available: legacy row-by-row write
                    fallback_metrics = self._parse_multiple_results_from_artifacts(
                        results_dir, results, model_info, build_info
                    )
                    if fallback_metrics:
                        for item in fallback_metrics:
                            record = self._create_multiple_result_row_record(
                                model_info, build_info, deployment_id, item
                            )
                            if record:
                                self._write_to_perf_csv(record)
                                results["successful_runs"].append({
                                    "model": item["model"],
                                    "perf_data": record,
                                    "nodes": [],
                                    "per_node_metrics": [item],
                                })
                        self.console.print(
                            f"[green]✓ Wrote {len(fallback_metrics)} row(s) from multiple_results to perf.csv[/green]"
                        )
                if not resolved_csv_path:
                    # No multiple_results CSV found: record failure
                    error_msg = "No performance metrics found from any node"
                    failure_record = self._create_failure_record(
                        model_info, build_info, deployment_id, error_msg
                    )
                    self._write_to_perf_csv(failure_record)
                    results["failed_runs"].append({
                        "model": model_info.get("name", "Unknown"),
                        "error": error_msg,
                        "nodes": results["nodes"]
                    })
                    self.console.print(
                        f"[yellow]⚠ No performance metrics found, recorded as FAILED[/yellow]"
                    )
                elif resolved_csv_path and not REPORTING_AVAILABLE and not results.get("successful_runs"):
                    # Legacy path ran but produced no valid rows
                    error_msg = "No performance metrics found from any node"
                    failure_record = self._create_failure_record(
                        model_info, build_info, deployment_id, error_msg
                    )
                    self._write_to_perf_csv(failure_record)
                    results["failed_runs"].append({
                        "model": model_info.get("name", "Unknown"),
                        "error": error_msg,
                        "nodes": results["nodes"]
                    })
                    self.console.print(
                        f"[yellow]⚠ No performance metrics found, recorded as FAILED[/yellow]"
                    )
            
            # 4. Generate summary
            self._generate_results_summary(results, results_dir)

        except Exception as e:
            self.console.print(f"[yellow]⚠ Results collection incomplete: {e}[/yellow]")

        return results
    
    def _collect_artifacts_immediately(self, deployment_id: str, pod_name: str) -> None:
        """
        Collect artifacts immediately from a running pod during the sleep period.
        This is called when we detect the "Keeping pod alive" message in logs.
        """
        try:
            # Create results directory
            results_dir = Path("k8s_results") / deployment_id
            results_dir.mkdir(parents=True, exist_ok=True)
            
            pod_dir = results_dir / pod_name
            pod_dir.mkdir(exist_ok=True)
            
            # Collect artifacts
            artifacts = self._collect_pod_artifacts(pod_name, pod_dir)
            
            if artifacts:
                self.console.print(f"[green]✓ Collected {len(artifacts)} artifacts from {pod_name}[/green]")
            else:
                self.console.print(f"[yellow]⚠ No artifacts collected from {pod_name}[/yellow]")
                
        except Exception as e:
            self.console.print(f"[yellow]⚠ Error collecting artifacts: {e}[/yellow]")
    
    def _collect_pod_artifacts(self, pod_name: str, dest_dir: Path) -> List[Dict]:
        """
        Collect file artifacts from pod using kubectl cp.
        
        Collects:
        - perf.csv (performance results)
        - *_env.csv (environment details from rocEnvTool)
        - profiling outputs (rocprof*, results*, *.db)
        - tracing outputs (*_output/ directories)
        - tool-specific outputs
        
        Args:
            pod_name: Name of the Kubernetes pod
            dest_dir: Local directory to save artifacts
            
        Returns:
            List of collected artifact metadata
        """
        artifacts = []
        
        # Define artifact patterns to collect
        artifact_patterns = [
            {"pattern": "perf.csv", "type": "performance"},
            {"pattern": "*_env.csv", "type": "environment"},
            {"pattern": "results*", "type": "profiling"},
            {"pattern": "*.db", "type": "profiling"},
            {"pattern": "trace.*", "type": "tracing"},
            {"pattern": "prof.csv", "type": "profiling"},  # Raw profiler output before post-script renames it
            {"pattern": "gpu_info_*.csv", "type": "profiling"},
            {"pattern": "library_trace.csv", "type": "tracing"},
        ]
        
        for artifact_def in artifact_patterns:
            pattern = artifact_def["pattern"]
            artifact_type = artifact_def["type"]
            
            try:
                # Try direct kubectl cp without exec (works during the sleep period)
                # For patterns with wildcards, try common specific filenames
                if '*' in pattern:
                    # Expand pattern to specific known files
                    if pattern == "*_env.csv":
                        specific_files = ["dummy_prof_env.csv", "dummy_data_minio_env.csv"]
                    elif pattern == "gpu_info_*.csv":
                        specific_files = ["gpu_info_power_profiler_output.csv", "gpu_info_vram_profiler_output.csv"]
                    elif pattern == "results*":
                        specific_files = ["results.csv", "results.txt", "results.json"]
                    elif pattern == "trace.*":
                        specific_files = ["trace.txt", "trace.csv", "trace.json"]
                    else:
                        specific_files = []
                    
                    for filename in specific_files:
                        local_path = dest_dir / filename
                        cp_cmd = [
                            "kubectl", "cp",
                            f"{self.namespace}/{pod_name}:/workspace/{filename}",
                            str(local_path)
                        ]
                        
                        cp_result = subprocess.run(
                            cp_cmd, capture_output=True, text=True, timeout=30
                        )
                        
                        if cp_result.returncode == 0 and local_path.exists():
                            artifacts.append({
                                "pod": pod_name,
                                "type": artifact_type,
                                "source": f"/workspace/{filename}",
                                "local_path": str(local_path),
                                "size": local_path.stat().st_size
                            })
                            self.console.print(
                                f"[dim]    ✓ Collected {artifact_type}: {filename}[/dim]"
                            )
                        elif cp_result.stderr and "No such file" not in cp_result.stderr:
                            # Log unexpected errors (but not "file not found")
                            self.console.print(
                                f"[yellow]    ⚠ Failed to collect {filename}: {cp_result.stderr.strip()}[/yellow]"
                            )
                else:
                    # Direct file - try to copy it
                    local_path = dest_dir / pattern
                    cp_cmd = [
                        "kubectl", "cp",
                        f"{self.namespace}/{pod_name}:/workspace/{pattern}",
                        str(local_path)
                    ]
                    
                    cp_result = subprocess.run(
                        cp_cmd, capture_output=True, text=True, timeout=30
                    )
                    
                    if cp_result.returncode == 0 and local_path.exists():
                        artifacts.append({
                            "pod": pod_name,
                            "type": artifact_type,
                            "source": f"/workspace/{pattern}",
                            "local_path": str(local_path),
                            "size": local_path.stat().st_size
                        })
                        self.console.print(
                            f"[dim]    ✓ Collected {artifact_type}: {pattern}[/dim]"
                        )
                    elif cp_result.stderr and "No such file" not in cp_result.stderr:
                        # Log unexpected errors (but not "file not found")
                        self.console.print(
                            f"[yellow]    ⚠ Failed to collect {pattern}: {cp_result.stderr.strip()}[/yellow]"
                        )
                        
            except subprocess.TimeoutExpired:
                pass  # Timeout - skip this file
            except Exception:
                pass  # File not found or not accessible - this is expected
        
        # Try to collect known output directories using kubectl cp directly (during sleep period)
        output_directories = ["rocprof_output", "rpd_output", "trace_output"]
        for dir_name in output_directories:
            try:
                local_dir = dest_dir / dir_name
                cp_cmd = [
                    "kubectl", "cp",
                    f"{self.namespace}/{pod_name}:/workspace/{dir_name}",
                    str(local_dir)
                ]
                
                cp_result = subprocess.run(
                    cp_cmd, capture_output=True, text=True, timeout=60
                )
                
                if cp_result.returncode == 0 and local_dir.exists():
                    # Count files in directory
                    file_count = sum(1 for _ in local_dir.rglob('*') if _.is_file())
                    if file_count > 0:
                        total_size = sum(f.stat().st_size for f in local_dir.rglob('*') if f.is_file())
                        artifacts.append({
                            "pod": pod_name,
                            "type": "tool_output_directory",
                            "source": f"/workspace/{dir_name}",
                            "local_path": str(local_dir),
                            "file_count": file_count,
                            "size": total_size
                        })
                        self.console.print(
                            f"[dim]    ✓ Collected directory: {dir_name} ({file_count} files, {total_size} bytes)[/dim]"
                        )
            except Exception:
                pass  # Directory not found - this is expected
        
        return artifacts
    
    def _collect_from_pvc(
        self,
        deployment_id: str,
        results_dir: Path,
        results: Dict,
        pod_names: Optional[List[str]] = None,
    ):
        """
        Collect all artifacts from the PVC using a temporary busybox pod.

        This is the best practice for collecting results from completed K8s jobs.
        kubectl cp doesn't work on completed pods, so we use a helper pod.

        When ``pod_names`` is provided, each ``/results/<subdir>/`` is copied to
        ``results_dir/<k8s_pod_name>/pvc/`` by matching subdir to pod name (exact or
        ``pod.startswith(subdir + "-")``). Unmatched subdirs go under
        ``results_dir/pvc_unmapped/<subdir>/``. When ``pod_names`` is omitted, the
        legacy layout ``results_dir/<subdir>/`` is used.

        Args:
            deployment_id: Job deployment ID
            results_dir: Local directory to save results
            results: Results dict to update
            pod_names: Full Kubernetes pod names for this job (ordered)
        """
        pvc_name = f"{deployment_id}-results"
        
        try:
            # Create a temporary pod to access PVC
            collector_pod_name = f"collector-{deployment_id[:15]}"
            
            self.console.print(f"[dim]📦 Collecting artifacts from PVC: {pvc_name}[/dim]")
            
            collector_spec: Dict[str, Any] = {
                "restartPolicy": "Never",
                "containers": [{
                    "name": "collector",
                    "image": "busybox:latest",
                    "command": ["sh", "-c", "sleep 600"],
                    "volumeMounts": [{"name": "results", "mountPath": "/results"}]
                }],
                "volumes": [{"name": "results", "persistentVolumeClaim": {"claimName": pvc_name}}]
            }
            ips = getattr(self, "_image_pull_secrets_for_pods", None) or []
            if ips:
                collector_spec["imagePullSecrets"] = ips

            collector_pod_spec = {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {"name": collector_pod_name, "namespace": self.namespace},
                "spec": collector_spec,
            }

            # Delete existing collector pod if it exists (prevents 409 Conflict)
            try:
                self.core_v1.delete_namespaced_pod(
                    collector_pod_name, self.namespace, grace_period_seconds=0
                )
                time.sleep(2)  # Wait for pod to be deleted
            except ApiException as e:
                if e.status != 404:  # 404 means pod doesn't exist, which is fine
                    pass
            
            # Create collector pod
            self.core_v1.create_namespaced_pod(self.namespace, collector_pod_spec)
            
            # Wait for pod to be ready
            for _ in range(30):  # Wait up to 30 seconds
                try:
                    pod_status = self.core_v1.read_namespaced_pod_status(
                        collector_pod_name, self.namespace
                    )
                    if pod_status.status.phase == "Running":
                        break
                except ApiException as e:
                    # Pod not found yet or not ready - this is expected during startup
                    if e.status != 404:
                        self.console.print(f"[dim]Waiting for collector pod (status: {e.status})...[/dim]")
                time.sleep(1)
            else:
                raise Exception("Collector pod did not start in time")

            # Mount / NFS may need a moment before another pod sees prior job writes.
            time.sleep(2)

            # List pod result directories in PVC (retry: NFS can lag right after Job completion)
            list_cmd = [
                "kubectl",
                "exec",
                collector_pod_name,
                "-n",
                self.namespace,
                "-c",
                "collector",
                "--",
                "ls",
                "-1",
                "/results/",
            ]
            list_result = subprocess.CompletedProcess(
                args=list_cmd, returncode=-1, stdout="", stderr=""
            )
            pod_dirs: List[str] = []
            for attempt in range(45):
                list_result = subprocess.run(
                    list_cmd, capture_output=True, text=True, timeout=30
                )
                if list_result.returncode == 0 and list_result.stdout.strip():
                    pod_dirs = [
                        d
                        for d in list_result.stdout.strip().split("\n")
                        if d and d != "lost+found"
                    ]
                    if pod_dirs:
                        break
                if list_result.stderr.strip():
                    self.console.print(
                        f"[dim]    PVC ls attempt {attempt + 1} (rc={list_result.returncode}): "
                        f"{list_result.stderr.strip()[:300]}[/dim]"
                    )
                time.sleep(1)

            if pod_dirs:
                pvc_map: Dict[str, str] = {}
                if pod_names:
                    pvc_map = assign_pvc_subdirs_to_pods(pod_dirs, pod_names)

                for pod_dir_name in pod_dirs:
                    if not pod_dir_name:
                        continue

                    matched_pod = pvc_map.get(pod_dir_name) if pod_names else None
                    if pod_names:
                        if matched_pod:
                            local_pod_dir = results_dir / matched_pod / "pvc"
                        else:
                            local_pod_dir = results_dir / "pvc_unmapped" / pod_dir_name
                    else:
                        local_pod_dir = results_dir / pod_dir_name

                    local_pod_dir.mkdir(parents=True, exist_ok=True)

                    cp_cmd = [
                        "kubectl",
                        "cp",
                        "-c",
                        "collector",
                        f"{self.namespace}/{collector_pod_name}:/results/{pod_dir_name}",
                        str(local_pod_dir),
                    ]

                    cp_result = subprocess.run(cp_cmd, capture_output=True, text=True, timeout=60)

                    if cp_result.returncode == 0:
                        # Count collected files
                        file_count = sum(1 for _ in local_pod_dir.rglob('*') if _.is_file())
                        if file_count > 0:
                            art: Dict[str, Any] = {
                                "source": f"PVC:{pvc_name}/{pod_dir_name}",
                                "local_path": str(local_pod_dir),
                                "file_count": file_count,
                                "type": "pvc_collection",
                                "pvc_subdir": pod_dir_name,
                            }
                            if pod_names:
                                art["k8s_pod"] = matched_pod
                            results["artifacts"].append(art)
                            if matched_pod:
                                dest_hint = f"{matched_pod}/pvc"
                            elif pod_names:
                                dest_hint = f"pvc_unmapped/{pod_dir_name}"
                            else:
                                dest_hint = pod_dir_name
                            self.console.print(
                                f"[dim]    ✓ Collected {file_count} files from {pod_dir_name} → {dest_hint}[/dim]"
                            )
                
                self.console.print(f"[green]✓ Collected artifacts from PVC[/green]")
            else:
                hint = ""
                if list_result.returncode != 0 or list_result.stderr.strip():
                    hint = (
                        f" (kubectl exec rc={list_result.returncode}"
                        + (
                            f", stderr={list_result.stderr.strip()[:400]!r}"
                            if list_result.stderr.strip()
                            else ""
                        )
                        + ")"
                    )
                self.console.print(
                    f"[yellow]⚠ No results found in PVC after retries{hint}[/yellow]"
                )
            
            # Cleanup collector pod
            self.core_v1.delete_namespaced_pod(
                collector_pod_name, self.namespace, grace_period_seconds=0
            )
            
        except Exception as e:
            self.console.print(f"[yellow]⚠ Could not collect from PVC: {e}[/yellow]")
    
    def _generate_results_summary(self, results: Dict, results_dir: Path):
        """
        Generate a summary JSON of all collected artifacts.
        
        Args:
            results: Results dict with logs and artifacts
            results_dir: Directory where results are saved
        """
        summary = {
            "job_name": results["job_name"],
            "namespace": results["namespace"],
            "collected_at": datetime.now().isoformat(),
            "k8s_results_layout": (
                "Per pod: <job>/<pod_name>/pod.log (API log) and "
                "<job>/<pod_name>/pvc/ (mirror of /results/<subdir>/). "
                "Unmatched PVC subdirs: <job>/pvc_unmapped/<subdir>/."
            ),
            "layout_version": 2,
            "pods": len(results["logs"]),
            "total_artifacts": len(results["artifacts"]),
            "artifacts_by_type": {},
            "artifacts": results["artifacts"],
            "successful_runs": len(results["successful_runs"]),
            "failed_runs": len(results["failed_runs"]),
        }
        
        # Group artifacts by type
        for artifact in results["artifacts"]:
            artifact_type = artifact.get("type", "unknown")
            summary["artifacts_by_type"][artifact_type] = summary["artifacts_by_type"].get(artifact_type, 0) + 1
        
        summary_file = results_dir / "results_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        
        self.console.print(f"[green]✓ Results summary: {summary_file}[/green]")
        
        # Print summary table if artifacts were collected
        if summary["artifacts_by_type"]:
            from rich.table import Table
            table = Table(title="Collected Artifacts")
            table.add_column("Type", style="cyan")
            table.add_column("Count", justify="right", style="green")
            
            for artifact_type, count in sorted(summary["artifacts_by_type"].items()):
                table.add_row(artifact_type, str(count))
            
            self.console.print(table)
    
    def _create_failure_record(self, model_info: Dict, build_info: Dict, pod_name: str, error_msg: str) -> Dict:
        """
        Create a failure record for perf.csv when performance metrics are missing.
        
        Args:
            model_info: Model information from manifest
            build_info: Build information from manifest
            pod_name: Kubernetes pod name
            error_msg: Error message describing the failure
            
        Returns:
            Dict with all perf.csv fields marked as FAILED
        """
        import os
        
        # Get topology information for failure record
        deployment_config = self.manifest.get("deployment_config", {})
        distributed_config = deployment_config.get("distributed", {})
        nnodes = distributed_config.get("nnodes", 1)
        nproc_per_node = distributed_config.get("nproc_per_node")
        if nproc_per_node is None:
            nproc_per_node = int(model_info.get("n_gpus", 1))
        # Launcher: use distributed.launcher when set, otherwise "native" for k8s
        launcher = normalize_launcher(distributed_config.get("launcher"), "kubernetes")
        
        # Create a record with the same structure as successful runs
        # but with performance=0, metric="", and status="FAILED"
        result = {
            # Core identification
            "model": model_info.get("name", ""),
            "n_gpus": str(nnodes * nproc_per_node),
            "nnodes": str(nnodes),
            "gpus_per_node": str(nproc_per_node),
            
            # Model configuration
            "training_precision": model_info.get("training_precision", ""),
            "pipeline": get_pipeline(),
            "args": model_info.get("args", ""),
            "tags": model_info.get("tags", ""),

            # Build information
            "docker_file": build_info.get("dockerfile", ""),
            "base_docker": build_info.get("base_docker", ""),
            "docker_sha": build_info.get("docker_sha", ""),
            "docker_image": build_info.get("docker_image", ""),

            # Runtime information
            "git_commit": "",
            "machine_name": pod_name,
            "deployment_type": "kubernetes",
            "launcher": launcher,
            "gpu_architecture": "",

            # Performance metrics - FAILED
            "performance": "0",
            "metric": error_msg,  # Store error message in metric field
            "relative_change": "",
            "status": "FAILURE",  # Use "FAILURE" to match CSV schema

            # Timing
            "build_duration": build_info.get("build_duration", ""),
            "test_duration": "",

            # Data information
            "dataname": model_info.get("data", ""),
            "data_provider_type": "",
            "data_size": "",
            "data_download_duration": "",

            # Build tracking
            "build_number": get_build_number(),
            "additional_docker_run_options": model_info.get("additional_docker_run_options", ""),
        }
        flatten_tags_in_place(result)
        return result

    # Standard perf.csv header (must match container_runner.ensure_perf_csv_exists)
    _PERF_CSV_HEADER = (
        "model,n_gpus,nnodes,gpus_per_node,training_precision,pipeline,args,tags,"
        "docker_file,base_docker,docker_sha,docker_image,git_commit,machine_name,"
        "deployment_type,launcher,gpu_architecture,performance,metric,relative_change,"
        "status,build_duration,test_duration,dataname,data_provider_type,data_size,"
        "data_download_duration,build_number,additional_docker_run_options"
    )

    def _ensure_perf_csv_exists(self) -> None:
        """Ensure perf.csv exists with standard header (same as Docker container_runner)."""
        perf_csv_path = Path("perf.csv")
        if not perf_csv_path.exists():
            perf_csv_path.write_text(self._PERF_CSV_HEADER + "\n", encoding="utf-8")
            self.console.print("[dim]Created perf.csv with standard header[/dim]")

    def _build_perf_entry_from_aggregated(
        self,
        aggregated_record: Dict[str, Any],
        model_info: Dict[str, Any],
        build_info: Dict[str, Any],
        deployment_id: str,
    ) -> Dict[str, Any]:
        """Build full run_details dict from aggregated record for perf_entry and update_* pipeline."""
        from madengine.utils.config_parser import ConfigParser

        deployment_config = self.manifest.get("deployment_config", {})
        distributed_config = deployment_config.get("distributed", {})
        nnodes = distributed_config.get("nnodes", 1)
        nproc_per_node = distributed_config.get("nproc_per_node")
        if nproc_per_node is None:
            nproc_per_node = int(model_info.get("n_gpus", 1))
        launcher = normalize_launcher(distributed_config.get("launcher"), "kubernetes")
        test_duration = aggregated_record.get("test_duration") or aggregated_record.get("duration", "")
        run_details = {
            "model": model_info.get("name", aggregated_record.get("model", "")),
            "n_gpus": str(aggregated_record.get("n_gpus", nnodes * nproc_per_node)),
            "nnodes": str(aggregated_record.get("nnodes", nnodes)),
            "gpus_per_node": str(aggregated_record.get("gpus_per_node", nproc_per_node)),
            "training_precision": model_info.get("training_precision", ""),
            "pipeline": get_pipeline(),
            "args": model_info.get("args", ""),
            "tags": model_info.get("tags", ""),
            "docker_file": build_info.get("dockerfile", ""),
            "base_docker": build_info.get("base_docker", ""),
            "docker_sha": build_info.get("docker_sha", ""),
            "docker_image": build_info.get("docker_image", ""),
            "git_commit": "",
            "machine_name": deployment_id,
            "deployment_type": "kubernetes",
            "launcher": launcher,
            "gpu_architecture": aggregated_record.get("gpu_architecture", ""),
            "performance": str(aggregated_record.get("performance", "")),
            "metric": aggregated_record.get("metric", ""),
            "relative_change": "",
            "status": aggregated_record.get("status", "SUCCESS"),
            "build_duration": build_info.get("build_duration", ""),
            "test_duration": test_duration,
            "dataname": aggregated_record.get("data_name", model_info.get("data", "")),
            "data_provider_type": aggregated_record.get("data_provider", ""),
            "data_size": "",
            "data_download_duration": "",
            "build_number": get_build_number(),
            "additional_docker_run_options": model_info.get("additional_docker_run_options", ""),
        }
        flatten_tags_in_place(run_details)
        try:
            scripts_path = model_info.get("scripts", "")
            scripts_base_dir = scripts_base_dir_from(scripts_path)
            config_parser = ConfigParser(scripts_base_dir=scripts_base_dir)
            run_details["configs"] = config_parser.parse_and_load(
                model_info.get("args", ""), scripts_path
            )
        except Exception:
            run_details["configs"] = None
        return run_details

    def _build_common_info_dict(
        self,
        model_info: Dict,
        build_info: Dict,
        deployment_id: str,
        gpu_architecture: str = "",
    ) -> Dict:
        """
        Build common_info dict for update_perf_csv / update_perf_super (Docker-compatible).
        Same shape as container_runner create_run_details_dict; model/performance/metric
        are omitted so they are filled from the multiple_results CSV.
        """
        deployment_config = self.manifest.get("deployment_config", {})
        distributed_config = deployment_config.get("distributed", {})
        nnodes = distributed_config.get("nnodes", 1)
        nproc_per_node = distributed_config.get("nproc_per_node")
        if nproc_per_node is None:
            nproc_per_node = int(model_info.get("n_gpus", 1))
        total_gpus = nnodes * nproc_per_node
        gpus_per_node = str(nproc_per_node)
        nnodes_str = str(nnodes)
        # Launcher: use distributed.launcher when set, otherwise "native" for k8s
        launcher = normalize_launcher(distributed_config.get("launcher"), "kubernetes")
        result = {
            "n_gpus": str(total_gpus),
            "nnodes": nnodes_str,
            "gpus_per_node": gpus_per_node,
            "training_precision": model_info.get("training_precision", ""),
            "pipeline": get_pipeline(),
            "args": model_info.get("args", ""),
            "tags": model_info.get("tags", ""),
            "docker_file": build_info.get("dockerfile", ""),
            "base_docker": build_info.get("base_docker", ""),
            "docker_sha": build_info.get("docker_sha", ""),
            "docker_image": build_info.get("docker_image", ""),
            "git_commit": "",
            "machine_name": deployment_id,
            "deployment_type": "kubernetes",
            "launcher": launcher,
            "gpu_architecture": gpu_architecture,
            "relative_change": "",
            "build_duration": build_info.get("build_duration", ""),
            "test_duration": "",
            "dataname": model_info.get("data", ""),
            "data_provider_type": "",
            "data_size": "",
            "data_download_duration": "",
            "build_number": get_build_number(),
            "additional_docker_run_options": model_info.get("additional_docker_run_options", ""),
        }
        flatten_tags_in_place(result)
        return result

    def _create_multiple_result_row_record(
        self,
        model_info: Dict,
        build_info: Dict,
        deployment_id: str,
        item: Dict,
    ) -> Dict:
        """
        Build one perf.csv row for a single row from a multiple_results CSV.
        Same shape as _create_failure_record but with SUCCESS and item's performance/metric/model.
        """
        import os
        
        deployment_config = self.manifest.get("deployment_config", {})
        distributed_config = deployment_config.get("distributed", {})
        nnodes = distributed_config.get("nnodes", 1)
        nproc_per_node = distributed_config.get("nproc_per_node")
        if nproc_per_node is None:
            nproc_per_node = int(model_info.get("n_gpus", 1))
        
        # Launcher: use distributed.launcher when set, otherwise "native" for k8s
        launcher = normalize_launcher(distributed_config.get("launcher"), "kubernetes")
        result = {
            "model": item.get("model", model_info.get("name", "")),
            "n_gpus": str(nnodes * nproc_per_node),
            "nnodes": str(nnodes),
            "gpus_per_node": str(nproc_per_node),
            "training_precision": model_info.get("training_precision", ""),
            "pipeline": get_pipeline(),
            "args": model_info.get("args", ""),
            "tags": model_info.get("tags", ""),
            "docker_file": build_info.get("dockerfile", ""),
            "base_docker": build_info.get("base_docker", ""),
            "docker_sha": build_info.get("docker_sha", ""),
            "docker_image": build_info.get("docker_image", ""),
            "git_commit": "",
            "machine_name": deployment_id,
            "deployment_type": "kubernetes",
            "launcher": launcher,
            "gpu_architecture": item.get("gpu_architecture", ""),
            "performance": str(item.get("performance", "")),
            "metric": item.get("metric", ""),
            "relative_change": "",
            "status": "SUCCESS",
            "build_duration": build_info.get("build_duration", ""),
            "test_duration": item.get("duration", ""),
            "dataname": model_info.get("data", ""),
            "data_provider_type": "",
            "data_size": "",
            "data_download_duration": "",
            "build_number": get_build_number(),
            "additional_docker_run_options": model_info.get("additional_docker_run_options", ""),
        }
        flatten_tags_in_place(result)
        return result
    
    def _parse_multiple_results_from_artifacts(
        self,
        results_dir: Path,
        results: Dict,
        model_info: Dict,
        build_info: Dict,
    ) -> List[Dict]:
        """
        Parse performance from a multiple_results CSV (e.g. perf_dummy.csv) collected from PVC.
        Used when the model only writes CSV and does not print 'performance: X Y' to the log
        (same contract as local container_runner multiple_results handling).
        
        Returns:
            List of perf_data dicts (same shape as _parse_node_performance), or empty list.
        """
        import csv as csv_module
        multiple_results_file = model_info.get("multiple_results")
        filename = Path(multiple_results_file).name if multiple_results_file else None
        # Try to get gpu_architecture from first pod log
        gpu_arch = "N/A"
        if results.get("logs"):
            import re
            log_content = results["logs"][0].get("log", "")
            gpu_arch_match = re.search(r"(?:🔹\s*)?Name\s*:\s*(gfx\w+)", log_content)
            if gpu_arch_match:
                gpu_arch = gpu_arch_match.group(1)
        parsed_list = []
        for art in results.get("artifacts", []):
            if art.get("type") != "pvc_collection":
                continue
            local_path = Path(art.get("local_path", ""))
            if not local_path.is_dir():
                continue
            # Prefer exact filename (same as Docker multiple_results); fallback to any perf_*.csv
            csv_path = (local_path / filename) if filename else None
            if not csv_path or not csv_path.is_file():
                perf_csvs = sorted(local_path.glob("perf_*.csv"))
                csv_path = perf_csvs[0] if perf_csvs else None
            if not csv_path or not csv_path.is_file():
                continue
            try:
                with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv_module.DictReader(f)
                    if not reader.fieldnames or "performance" not in reader.fieldnames or "metric" not in reader.fieldnames:
                        continue
                    for row_idx, row in enumerate(reader):
                        perf_val = row.get("performance", "").strip()
                        metric_val = row.get("metric", "").strip()
                        if not perf_val or not metric_val:
                            continue
                        try:
                            perf_float = float(perf_val)
                        except (ValueError, TypeError):
                            continue
                        # Same model naming as local handle_multiple_results: model_name + "_" + str(model)
                        row_model = row.get("model", row_idx)
                        display_model = f"{model_info.get('name')}_{row_model}"
                        parsed_list.append({
                            "model": display_model,
                            "performance": perf_float,
                            "metric": metric_val,
                            "node_id": row_idx,
                            "local_gpus": 1,
                            "duration": "N/A",
                            "gpu_architecture": gpu_arch,
                            "data_name": "N/A",
                            "data_provider": "N/A",
                        })
                if parsed_list:
                    self.console.print(
                        f"[green]  ✓ Parsed performance from {csv_path.name} ({len(parsed_list)} row(s))[/green]"
                    )
                    return parsed_list
            except Exception as e:
                self.console.print(
                    f"[dim]  Could not parse {csv_path.name} from PVC: {e}[/dim]"
                )
        return []

    def _aggregation_for_extra_column(self, column_name: str) -> str:
        """
        Return how to aggregate an extra CSV column when merging multi-node results.
        Best practice: throughput/counts -> sum; latencies/utilization -> average;
        duration/capacity -> max; identifiers -> first.
        """
        col = column_name.lower().strip()
        # Sum: counts, totals, throughput-like
        if any(k in col for k in [
            "count", "total", "samples", "tokens", "throughput",
            "requests", "images", "bandwidth", "ops"
        ]):
            return "sum"
        # Average: rates per unit, utilization, ratios
        if any(k in col for k in [
            "utilization", "usage", "percent", "ratio", "latency",
            "time_ms", "ttft", "tpot", "accuracy", "loss"
        ]):
            return "average"
        # Max: duration (slowest node), memory, capacity
        if any(k in col for k in [
            "duration", "time", "seconds", "memory", "bytes", "mb", "gb"
        ]):
            return "max"
        return "first"

    def _merge_multi_node_multiple_results_csv(
        self, csv_paths: List[Path], output_path: Path
    ) -> bool:
        """
        Merge multiple pod multiple_results CSVs into one with sum/average rules.
        Rows are aligned by index (row 0 from each pod -> one merged row 0).
        - performance: aggregated by _determine_aggregation_method(metric) (sum or average).
        - Other numeric columns: by _aggregation_for_extra_column (sum/average/max).
        - model, metric: taken from first CSV.
        """
        import csv as csv_module
        import statistics

        required = ["model", "performance", "metric"]
        rows_by_index: Dict[int, List[Dict]] = {}

        for path in csv_paths:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv_module.DictReader(f)
                    fieldnames = [c.strip() for c in (reader.fieldnames or [])]
                    if not all(h in fieldnames for h in required):
                        continue
                    for idx, row in enumerate(reader):
                        row = {k.strip(): v for k, v in row.items() if k}
                        if not row.get("performance") or not row.get("metric"):
                            continue
                        try:
                            float(str(row["performance"]).strip())
                        except (ValueError, TypeError):
                            continue
                        if idx not in rows_by_index:
                            rows_by_index[idx] = []
                        rows_by_index[idx].append(row)
            except Exception as e:
                self.console.print(f"[dim]  Could not read {path.name}: {e}[/dim]")
                continue

        if not rows_by_index:
            return False

        # Build union of columns (required first, then rest)
        extra_cols = set()
        for group in rows_by_index.values():
            for row in group:
                extra_cols.update(k for k in row if k not in required)
        all_columns = list(required) + sorted(extra_cols)
        merged_rows = []
        for idx in sorted(rows_by_index.keys()):
            group = rows_by_index[idx]
            first = group[0]
            metric_name = (first.get("metric") or "").strip()
            perf_agg = self._determine_aggregation_method(metric_name)
            perf_values = []
            for r in group:
                try:
                    perf_values.append(float(str(r.get("performance", "")).strip()))
                except (ValueError, TypeError):
                    pass
            if not perf_values:
                continue
            if perf_agg == "sum":
                performance = sum(perf_values)
            elif perf_agg == "average":
                performance = statistics.mean(perf_values)
            elif perf_agg == "max":
                performance = max(perf_values)
            else:
                performance = sum(perf_values)
            merged = {
                "model": first.get("model", ""),
                "performance": performance,
                "metric": first.get("metric", ""),
            }
            for col in all_columns:
                if col in merged:
                    continue
                values = [r.get(col) for r in group]
                try:
                    nums = [float(str(v).strip()) for v in values if v is not None and str(v).strip()]
                except (ValueError, TypeError):
                    nums = []
                if nums:
                    extra_agg = self._aggregation_for_extra_column(col)
                    if extra_agg == "sum":
                        merged[col] = sum(nums)
                    elif extra_agg == "average":
                        merged[col] = statistics.mean(nums)
                    elif extra_agg == "max":
                        merged[col] = max(nums)
                    else:
                        merged[col] = first.get(col, "")
                else:
                    merged[col] = first.get(col, "")
            merged_rows.append(merged)

        if not merged_rows:
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv_module.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(merged_rows)
        self.console.print(
            f"[green]  ✓ Merged {len(csv_paths)} pod CSV(s) into {len(merged_rows)} row(s) → {output_path.name}[/green]"
        )
        return True

    def _resolve_multiple_results_csv(
        self, results_dir: Path, results: Dict, model_info: Dict
    ) -> Optional[Path]:
        """
        Resolve path to a single multiple_results CSV for update_perf_csv.
        Single pod: return that CSV path. Multi-pod: merge all pod CSVs with
        sum/average rules and return path to merged file.
        """
        multiple_results_file = model_info.get("multiple_results")
        filename = Path(multiple_results_file).name if multiple_results_file else None
        csv_paths: List[Path] = []
        for art in results.get("artifacts", []):
            if art.get("type") != "pvc_collection":
                continue
            local_path = Path(art.get("local_path", ""))
            if not local_path.is_dir():
                continue
            csv_path = (local_path / filename) if filename else None
            if not csv_path or not csv_path.is_file():
                perf_csvs = sorted(local_path.glob("perf_*.csv"))
                csv_path = perf_csvs[0] if perf_csvs else None
            if csv_path and csv_path.is_file():
                csv_paths.append(csv_path)
        if not csv_paths:
            return None
        if len(csv_paths) == 1:
            return csv_paths[0]
        merged_path = results_dir / "multiple_results_merged.csv"
        if self._merge_multi_node_multiple_results_csv(csv_paths, merged_path):
            return merged_path
        return csv_paths[0]

    def cleanup(self, deployment_id: str) -> bool:
        """Delete Job, ConfigMap, Service and associated pods."""
        success = True

        try:
            # Delete Job (propagates to pods)
            self.batch_v1.delete_namespaced_job(
                name=deployment_id,
                namespace=self.namespace,
                propagation_policy="Background",
            )
            self.console.print(f"[yellow]Deleted K8s Job: {deployment_id}[/yellow]")
        except ApiException as e:
            if e.status != 404:
                self.console.print(f"[yellow]⚠ Job cleanup warning: {e.reason}[/yellow]")
                success = False
        except Exception as e:
            self.console.print(f"[yellow]⚠ Job cleanup error: {e}[/yellow]")
            success = False

        # Delete ConfigMap
        try:
            configmap_name = f"{deployment_id}-config"
            self.core_v1.delete_namespaced_config_map(
                name=configmap_name, namespace=self.namespace
            )
            self.console.print(
                f"[yellow]Deleted ConfigMap: {configmap_name}[/yellow]"
            )
        except ApiException as e:
            if e.status != 404:
                self.console.print(
                    f"[yellow]⚠ ConfigMap cleanup warning: {e.reason}[/yellow]"
                )
        except Exception:
            pass

        # Delete Service (if exists)
        try:
            self.core_v1.delete_namespaced_service(
                name=deployment_id, namespace=self.namespace
            )
            self.console.print(f"[yellow]Deleted Service: {deployment_id}[/yellow]")
        except ApiException as e:
            if e.status != 404:
                pass  # Service may not exist for single-node jobs
        except Exception:
            pass

        return success

