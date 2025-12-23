#!/usr/bin/env python3
"""
Kubernetes Deployment - Container orchestration using Jinja2 templates + Python library.

Uses Jinja2 templates for manifest generation (industry best practice) and
Kubernetes Python client library for applying manifests.
Requires AMD GPU Device Plugin: https://github.com/ROCm/k8s-device-plugin

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

from jinja2 import Environment, FileSystemLoader, Template

from .base import BaseDeployment, DeploymentConfig, DeploymentResult, DeploymentStatus
from .config_loader import ConfigLoader
from madengine.core.dataprovider import Data
from madengine.core.context import Context
from madengine.core.errors import ConfigurationError, create_error_context
from madengine.utils.gpu_config import resolve_runtime_gpus


class KubernetesDeployment(BaseDeployment):
    """
    Kubernetes cluster deployment using Python client library.

    Uses kubernetes Python API for type-safe, production-ready deployment:
    - client.BatchV1Api(): Job creation and management
    - client.CoreV1Api(): Pod logs and status

    Requires AMD GPU Device Plugin: https://github.com/ROCm/k8s-device-plugin

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

        # Apply intelligent defaults using ConfigLoader
        # This merges built-in presets with user configuration
        full_config = ConfigLoader.load_k8s_config(config.additional_context)
        config.additional_context = full_config

        super().__init__(config)

        # Parse K8s configuration (now with defaults applied)
        self.k8s_config = config.additional_context.get("k8s", {})
        if not self.k8s_config:
            self.k8s_config = config.additional_context.get("kubernetes", {})

        self.namespace = self.k8s_config.get("namespace", "default")
        self.gpu_resource_name = self.k8s_config.get("gpu_resource_name", "amd.com/gpu")

        # Setup Jinja2 template environment
        template_dir = Path(__file__).parent / "templates" / "kubernetes"
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        
        # Register custom Jinja2 filters
        self.jinja_env.filters['dirname'] = lambda path: str(Path(path).parent)
        
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

            # Validate AMD GPU Device Plugin is deployed
            nodes = self.core_v1.list_node()
            amd_gpu_nodes = [
                n
                for n in nodes.items
                if self.gpu_resource_name in n.status.allocatable
            ]

            if not amd_gpu_nodes:
                self.console.print(
                    f"[yellow]⚠ No nodes with {self.gpu_resource_name} found[/yellow]\n"
                    f"[yellow]  Ensure AMD GPU Device Plugin is deployed:[/yellow]\n"
                    f"[yellow]  kubectl create -f https://raw.githubusercontent.com/ROCm/k8s-device-plugin/master/k8s-ds-amdgpu-dp.yaml[/yellow]"
                )
                return False

            self.console.print(f"[green]✓ Found {len(amd_gpu_nodes)} AMD GPU nodes[/green]")
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
        tools_json_path = Path(__file__).parent.parent / "scripts" / "common" / "tools.json"
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
        madengine_root = Path(__file__).parent.parent  # Go up to madengine/ directory
        
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
            
            # Load ALL scripts from the model's scripts directory
            # This is critical for models that have multiple helper scripts
            scripts_dir_path = Path(model_script_dir)
            if scripts_dir_path.exists() and scripts_dir_path.is_dir():
                for script in scripts_dir_path.glob("*.sh"):
                    with open(script, "r") as f:
                        # Use the path directly if relative, otherwise convert to relative
                        if script.is_absolute():
                            relative_path = str(script.relative_to(Path.cwd()))
                        else:
                            relative_path = str(script)
                        model_scripts_contents[relative_path] = f.read()
                
                # Also check for Python scripts
                for script in scripts_dir_path.glob("*.py"):
                    with open(script, "r") as f:
                        # Use the path directly if relative, otherwise convert to relative
                        if script.is_absolute():
                            relative_path = str(script.relative_to(Path.cwd()))
                        else:
                            relative_path = str(script)
                        model_scripts_contents[relative_path] = f.read()
                
                # Also check for JSON config files (e.g., DeepSpeed configs)
                for script in scripts_dir_path.glob("*.json"):
                    with open(script, "r") as f:
                        # Use the path directly if relative, otherwise convert to relative
                        if script.is_absolute():
                            relative_path = str(script.relative_to(Path.cwd()))
                        else:
                            relative_path = str(script)
                        model_scripts_contents[relative_path] = f.read()
                
                self.console.print(f"[dim]Loaded {len(model_scripts_contents)} script(s) from {model_script_dir}[/dim]")
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
                k8s_script_path = Path(__file__).parent.parent / data_provider_script["script"]
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
            
            # Generate vLLM launcher command
            launcher_command = self._generate_vllm_command(
                nnodes=nnodes,
                nproc_per_node=nproc_per_node,
                master_port=master_port,
                model_script=model_info.get("scripts", "run.sh")
            )

        elif launcher_type == "sglang":
            if nnodes > 1:
                create_headless_service = True
                self.console.print(f"[dim]Multi-node SGLang: Creating headless service for Ray cluster[/dim]")
            
            # Generate SGLang launcher command
            launcher_command = self._generate_sglang_command(
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
            "subdomain": self.job_name if (launcher_type == "torchrun" and nnodes > 1) else None,
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
        }

        return context
    
    def _get_tools_config(self) -> List[Dict]:
        """
        Get tools configuration from manifest.context or additional_context.
        
        Prioritizes runtime additional_context, falls back to manifest.context.
        
        Returns:
            List of tool configurations (enriched with cmd from tools.json)
        """
        # Check runtime additional_context first (allows runtime override)
        tools = self.config.additional_context.get("tools", [])
        
        # Fall back to manifest.context if no runtime tools
        if not tools and "context" in self.manifest:
            tools = self.manifest["context"].get("tools", [])
        
        # Enrich tools with cmd from tools.json for K8s template usage
        return self._enrich_tools_with_cmd(tools)
    
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
    
    def _generate_torchrun_command(
        self, nnodes: int, nproc_per_node: int, master_port: int, model_script: str
    ) -> str:
        """
        Generate torchrun launcher command for K8s Indexed Jobs.
        
        For single-node (nnodes=1), generates standalone torchrun command.
        For multi-node (nnodes>1), generates distributed torchrun with headless
        service DNS for coordination.
        
        Uses K8s environment variables for distributed coordination:
        - JOB_COMPLETION_INDEX: Pod index (0, 1, 2, ...)
        - Headless service DNS for MASTER_ADDR
        
        Args:
            nnodes: Number of nodes (pods). Must be >= 1.
            nproc_per_node: GPUs per node. Must be >= 1.
            master_port: Master communication port. Must be 1-65535.
            model_script: Path to model's run script. Cannot be empty.
        
        Returns:
            Complete torchrun command string
        
        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate inputs (defensive programming)
        if not isinstance(nnodes, int) or nnodes < 1:
            raise ValueError(f"nnodes must be integer >= 1, got {nnodes}")
        if not isinstance(nproc_per_node, int) or nproc_per_node < 1:
            raise ValueError(f"nproc_per_node must be integer >= 1, got {nproc_per_node}")
        if not isinstance(master_port, int) or not (1 <= master_port <= 65535):
            raise ValueError(f"master_port must be 1-65535, got {master_port}")
        if not model_script or not isinstance(model_script, str):
            raise ValueError(f"model_script must be non-empty string, got {model_script}")
        
        # Check if model_script is a bash script
        # If so, execute it directly as it handles torchrun internally
        if model_script.endswith('.sh'):
            # For bash scripts, set environment variables and execute script
            # The script itself will invoke torchrun with the appropriate Python file
            if nnodes == 1:
                return f"""export MAD_MULTI_NODE_RUNNER="torchrun --standalone --nproc_per_node={nproc_per_node}"
export MAD_RUNTIME_NGPUS={nproc_per_node}
bash {model_script}"""
            else:
                return f"""# Multi-node torchrun setup (Kubernetes Indexed Job)
export MASTER_ADDR="{self.job_name}-0.{self.job_name}.{self.namespace}.svc.cluster.local"
export MASTER_PORT={master_port}
export MAD_MULTI_NODE_RUNNER="torchrun --nnodes={nnodes} --nproc_per_node={nproc_per_node} --node_rank=${{JOB_COMPLETION_INDEX}} --master_addr=${{MASTER_ADDR}} --master_port={master_port}"
export MAD_RUNTIME_NGPUS={nproc_per_node}
bash {model_script}"""
        
        # For Python scripts, invoke torchrun directly
        # For single-node, simpler standalone command
        if nnodes == 1:
            return f"""torchrun \\
    --standalone \\
    --nnodes=1 \\
    --nproc_per_node={nproc_per_node} \\
    {model_script}"""
        
        # Multi-node: Use headless service DNS and JOB_COMPLETION_INDEX
        return f"""# Multi-node torchrun setup (Kubernetes Indexed Job)
export MASTER_ADDR="{self.job_name}-0.{self.job_name}.{self.namespace}.svc.cluster.local"
export MASTER_PORT={master_port}
export RANK=${{JOB_COMPLETION_INDEX}}
export WORLD_SIZE={nnodes}
export LOCAL_RANK=0
export NNODES={nnodes}
export NPROC_PER_NODE={nproc_per_node}

echo "Torchrun Configuration:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  RANK: $RANK"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"

torchrun \\
    --nnodes={nnodes} \\
    --nproc_per_node={nproc_per_node} \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
    --rdzv_id={self.job_name} \\
    --role=worker \\
    --tee=3 \\
    {model_script}"""
    
    def _generate_deepspeed_command(
        self, nnodes: int, nproc_per_node: int, master_port: int, model_script: str
    ) -> str:
        """
        Generate DeepSpeed launcher command for K8s Indexed Jobs.
        
        DeepSpeed has its own launcher that handles:
        - ZeRO optimization stages (ZeRO-1, ZeRO-2, ZeRO-3)
        - Gradient accumulation
        - Mixed precision training
        - Pipeline parallelism
        - Hostfile management (handled by K8s in our case)
        
        For single-node (nnodes=1), uses localhost setup.
        For multi-node (nnodes>1), uses headless service DNS for coordination.
        
        Args:
            nnodes: Number of nodes (pods). Must be >= 1.
            nproc_per_node: GPUs per node. Must be >= 1.
            master_port: Master communication port. Must be 1-65535.
            model_script: Path to model's run script. Cannot be empty.
        
        Returns:
            Complete DeepSpeed launcher command string
        
        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate inputs
        if not isinstance(nnodes, int) or nnodes < 1:
            raise ValueError(f"nnodes must be integer >= 1, got {nnodes}")
        if not isinstance(nproc_per_node, int) or nproc_per_node < 1:
            raise ValueError(f"nproc_per_node must be integer >= 1, got {nproc_per_node}")
        if not isinstance(master_port, int) or not (1 <= master_port <= 65535):
            raise ValueError(f"master_port must be 1-65535, got {master_port}")
        if not model_script or not isinstance(model_script, str):
            raise ValueError(f"model_script must be non-empty string, got {model_script}")
        
        # For single-node
        if nnodes == 1:
            return f"""# DeepSpeed Single-Node Setup
export MASTER_ADDR=localhost
export MASTER_PORT={master_port}
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE={nproc_per_node}

echo "DeepSpeed Configuration:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  NUM_GPUS: {nproc_per_node}"

# DeepSpeed launcher (single-node)
deepspeed --num_gpus={nproc_per_node} \\
    --master_port={master_port} \\
    {model_script}"""
        
        # Multi-node: Use K8s headless service for coordination
        return f"""# Multi-node DeepSpeed setup (Kubernetes Indexed Job)
export MASTER_ADDR="{self.job_name}-0.{self.job_name}.{self.namespace}.svc.cluster.local"
export MASTER_PORT={master_port}
export RANK=${{JOB_COMPLETION_INDEX}}
export LOCAL_RANK=0
export WORLD_SIZE={nnodes * nproc_per_node}
export NNODES={nnodes}
export NPROC_PER_NODE={nproc_per_node}

echo "DeepSpeed Multi-Node Configuration:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  RANK (Node Rank): $RANK"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  NNODES: $NNODES"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"

# Create hostfile for DeepSpeed (K8s Indexed Job aware)
cat > /tmp/hostfile << EOF
{self.job_name}-0.{self.job_name}.{self.namespace}.svc.cluster.local slots={nproc_per_node}
EOF

# Add all nodes to hostfile
for i in $(seq 1 $((NNODES - 1))); do
    echo "{self.job_name}-$i.{self.job_name}.{self.namespace}.svc.cluster.local slots={nproc_per_node}" >> /tmp/hostfile
done

echo ""
echo "Generated hostfile:"
cat /tmp/hostfile
echo ""

# DeepSpeed launcher (multi-node with hostfile)
deepspeed --hostfile=/tmp/hostfile \\
    --master_addr=$MASTER_ADDR \\
    --master_port=$MASTER_PORT \\
    --num_nodes={nnodes} \\
    --num_gpus={nproc_per_node} \\
    {model_script}"""
    
    def _generate_bash_script_command(
        self, nnodes: int, nproc_per_node: int, master_port: int, model_script: str
    ) -> str:
        """
        Generate command to execute a bash script directly.
        
        This is used when the model script is a .sh file that handles
        launcher invocation internally (e.g., using torchrun inside the script).
        
        Sets up environment variables for distributed training that the bash
        script can use.
        
        Args:
            nnodes: Number of nodes (pods)
            nproc_per_node: GPUs per node
            master_port: Master communication port
            model_script: Path to the bash script
        
        Returns:
            Command to execute the bash script with environment setup
        """
        # For single-node
        if nnodes == 1:
            return f"""# Bash Script Execution (Single-Node)
# Setting up environment for script to use
export MASTER_ADDR=localhost
export MASTER_PORT={master_port}
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE={nproc_per_node}
export NNODES=1
export NPROC_PER_NODE={nproc_per_node}

echo "Bash Script Configuration:"
echo "  Script: {model_script}"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  NNODES: $NNODES"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"
echo ""

# Execute the bash script directly
bash {model_script}"""
        
        # Multi-node: Use K8s headless service for coordination
        return f"""# Bash Script Execution (Multi-Node)
# Setting up environment for script to use
export MASTER_ADDR="{self.job_name}-0.{self.job_name}.{self.namespace}.svc.cluster.local"
export MASTER_PORT={master_port}
export RANK=${{JOB_COMPLETION_INDEX}}
export LOCAL_RANK=0
export WORLD_SIZE={nnodes * nproc_per_node}
export NNODES={nnodes}
export NPROC_PER_NODE={nproc_per_node}

echo "Bash Script Multi-Node Configuration:"
echo "  Script: {model_script}"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  RANK (Node Rank): $RANK"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  NNODES: $NNODES"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"
echo ""

# Execute the bash script directly
bash {model_script}"""
    
    def _generate_torchtitan_command(
        self, nnodes: int, nproc_per_node: int, master_port: int, model_script: str
    ) -> str:
        """
        Generate TorchTitan launcher command for K8s Indexed Jobs.
        
        TorchTitan is a PyTorch native platform for large-scale LLM pre-training
        that supports multi-dimensional parallelism:
        - FSDP2 (Fully Sharded Data Parallel v2)
        - Tensor Parallel (TP)
        - Pipeline Parallel (PP)
        - Context Parallel (CP)
        
        TorchTitan uses torchrun as its underlying distributed launcher but
        requires additional configuration for its parallelism strategies.
        
        For single-node (nnodes=1): Uses standalone torchrun with TP
        For multi-node (nnodes>1): Uses distributed torchrun with TP+PP+FSDP2
        
        Uses K8s environment variables for distributed coordination:
        - JOB_COMPLETION_INDEX: Pod index (0, 1, 2, ...)
        - Headless service DNS for MASTER_ADDR
        
        Args:
            nnodes: Number of nodes (pods). Must be >= 1.
            nproc_per_node: GPUs per node. Must be >= 1.
            master_port: Master communication port. Must be 1-65535.
            model_script: Path to model's run script. Cannot be empty.
        
        Returns:
            Complete torchtitan launch command string with environment setup
        
        Raises:
            ValueError: If any parameter is invalid
        
        Example single-node output:
            export TORCHTITAN_TENSOR_PARALLEL_SIZE=8
            export TORCHTITAN_PIPELINE_PARALLEL_SIZE=1
            torchrun --standalone --nproc_per_node=8 train.py --config llama3_8b.toml
        
        Example multi-node output:
            export MASTER_ADDR="job-0.job.namespace.svc.cluster.local"
            export TORCHTITAN_TENSOR_PARALLEL_SIZE=8
            export TORCHTITAN_PIPELINE_PARALLEL_SIZE=4
            export TORCHTITAN_FSDP_ENABLED=1
            torchrun --nnodes=4 --nproc_per_node=8 ... train.py --config llama3_405b.toml
        """
        # Validate inputs
        if not isinstance(nnodes, int) or nnodes < 1:
            raise ValueError(f"nnodes must be integer >= 1, got {nnodes}")
        if not isinstance(nproc_per_node, int) or nproc_per_node < 1:
            raise ValueError(f"nproc_per_node must be integer >= 1, got {nproc_per_node}")
        if not isinstance(master_port, int) or not (1 <= master_port <= 65535):
            raise ValueError(f"master_port must be 1-65535, got {master_port}")
        if not model_script or not isinstance(model_script, str):
            raise ValueError(f"model_script must be non-empty string, got {model_script}")
        
        # For single-node, use standalone mode with Tensor Parallelism only
        if nnodes == 1:
            return f"""# TorchTitan single-node setup (Tensor Parallelism)
export TORCHTITAN_TENSOR_PARALLEL_SIZE={nproc_per_node}
export TORCHTITAN_PIPELINE_PARALLEL_SIZE=1
export TORCHTITAN_FSDP_ENABLED=0
export TORCHTITAN_CONTEXT_PARALLEL_SIZE=1

echo "TorchTitan Configuration (Single Node):"
echo "  Tensor Parallel Size: {nproc_per_node}"
echo "  Pipeline Parallel Size: 1"
echo "  Total GPUs: {nproc_per_node}"

torchrun \\
    --standalone \\
    --nnodes=1 \\
    --nproc_per_node={nproc_per_node} \\
    {model_script}"""
        
        # Multi-node: Use headless service DNS and enable all parallelism strategies
        return f"""# TorchTitan multi-node setup (K8s Indexed Job)
export MASTER_ADDR="{self.job_name}-0.{self.job_name}.{self.namespace}.svc.cluster.local"
export MASTER_PORT={master_port}
export RANK=${{JOB_COMPLETION_INDEX}}
export WORLD_SIZE={nnodes}
export LOCAL_RANK=0
export NNODES={nnodes}
export NPROC_PER_NODE={nproc_per_node}

# TorchTitan multi-dimensional parallelism configuration
# These can be overridden by TOML config file in model script
export TORCHTITAN_TENSOR_PARALLEL_SIZE={nproc_per_node}
export TORCHTITAN_PIPELINE_PARALLEL_SIZE={nnodes}
export TORCHTITAN_FSDP_ENABLED=1
export TORCHTITAN_CONTEXT_PARALLEL_SIZE=1

echo "TorchTitan Configuration (Multi-Node):"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  RANK: $RANK"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  Tensor Parallel Size: {nproc_per_node}"
echo "  Pipeline Parallel Size: {nnodes}"
echo "  FSDP: Enabled"
echo "  Total GPUs: {nnodes * nproc_per_node}"

torchrun \\
    --nnodes={nnodes} \\
    --nproc_per_node={nproc_per_node} \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
    --rdzv_id={self.job_name} \\
    --role=worker \\
    --tee=3 \\
    {model_script}"""
    
    def _generate_vllm_command(
        self, nnodes: int, nproc_per_node: int, master_port: int, model_script: str
    ) -> str:
        """
        Generate vLLM launcher command for K8s Indexed Jobs.
        
        vLLM is an inference engine with its own process management via Ray.
        Unlike training frameworks, vLLM doesn't use torchrun.
        
        Architecture:
        - Single-node: Tensor Parallelism (TP) across GPUs, no Ray needed
        - Multi-node: Data Parallelism where each node runs independent vLLM replica
          * Each replica uses TP across its local GPUs
          * Ray coordinates resources on each node independently
          * Benefits: Simpler, more robust, better for inference serving
        
        For K8s multi-node:
        - Each pod runs its own independent vLLM instance
        - Uses Ray for local GPU coordination
        - NO shared Ray cluster across pods (Data Parallelism mode)
        
        Args:
            nnodes: Number of nodes (pods). Must be >= 1.
            nproc_per_node: GPUs per node. Must be >= 1.
            master_port: Master communication port (for Ray). Must be 1-65535.
            model_script: Path to model's run script. Cannot be empty.
        
        Returns:
            Complete vLLM launch setup with environment configuration
        
        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate inputs
        if not isinstance(nnodes, int) or nnodes < 1:
            raise ValueError(f"nnodes must be integer >= 1, got {nnodes}")
        if not isinstance(nproc_per_node, int) or nproc_per_node < 1:
            raise ValueError(f"nproc_per_node must be integer >= 1, got {nproc_per_node}")
        if not isinstance(master_port, int) or not (1 <= master_port <= 65535):
            raise ValueError(f"master_port must be 1-65535, got {master_port}")
        if not model_script or not isinstance(model_script, str):
            raise ValueError(f"model_script must be non-empty string, got {model_script}")
        
        # For single-node, simple TP setup (no Ray needed)
        if nnodes == 1:
            return f"""# vLLM single-node setup (Tensor Parallelism)
export VLLM_TENSOR_PARALLEL_SIZE={nproc_per_node}
export VLLM_PIPELINE_PARALLEL_SIZE=1
export VLLM_DISTRIBUTED_BACKEND="auto"
export NNODES=1
export NPROC_PER_NODE={nproc_per_node}
export NODE_RANK=0

echo "vLLM Configuration (Single Node):"
echo "  Tensor Parallel Size: {nproc_per_node}"
echo "  Pipeline Parallel Size: 1"
echo "  Distributed Backend: auto (no Ray)"
echo "  Total GPUs: {nproc_per_node}"

# vLLM handles process management - just run the script
{model_script}"""
        
        # Multi-node: Data Parallelism with independent Ray clusters per pod
        return f"""# vLLM multi-node setup (K8s Data Parallelism Mode)
export MASTER_ADDR="{self.job_name}-0.{self.job_name}.{self.namespace}.svc.cluster.local"
export MASTER_PORT={master_port}
export NODE_RANK=${{JOB_COMPLETION_INDEX}}
export NNODES={nnodes}
export NPROC_PER_NODE={nproc_per_node}

# vLLM Data Parallelism configuration
# Each pod runs INDEPENDENT vLLM replica (no shared Ray cluster)
export VLLM_TENSOR_PARALLEL_SIZE={nproc_per_node}
export VLLM_PIPELINE_PARALLEL_SIZE=1
export VLLM_DISTRIBUTED_BACKEND="ray"

# Get current pod IP for Ray
POD_IP=$(hostname -i | awk '{{print $1}}')
export VLLM_HOST_IP="$POD_IP"

echo "vLLM Configuration (Multi-Node Data Parallelism):"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  NODE_RANK: $NODE_RANK (Pod Index)"
echo "  NNODES: $NNODES"
echo "  Tensor Parallel Size: {nproc_per_node} (per pod)"
echo "  Data Parallel Size: {nnodes} (independent replicas)"
echo "  Pod IP: $POD_IP"
echo "  Total GPUs: {nnodes * nproc_per_node}"
echo ""
echo "Mode: Each pod runs independent vLLM replica with local Ray"

# Clean any existing Ray processes
ray stop --force 2>/dev/null || true
pkill -9 -f "ray::" 2>/dev/null || true
sleep 2

# Start independent Ray cluster on THIS pod only
echo "Starting Ray cluster on Pod $NODE_RANK..."
ray start --head --port=6379 --node-ip-address="$POD_IP" --num-gpus={nproc_per_node}
sleep 3

echo "Ray cluster ready:"
ray status

# Run vLLM inference script
{model_script}

# Cleanup Ray on exit
trap "ray stop --force 2>/dev/null || true" EXIT"""

    def _generate_sglang_command(
        self, nnodes: int, nproc_per_node: int, master_port: int, model_script: str
    ) -> str:
        """
        Generate SGLang launcher command for K8s Indexed Jobs.
        
        SGLang is an inference engine with native launcher (sglang.launch_server).
        Similar to vLLM, it manages its own process spawning via Ray.
        
        Architecture:
        - Single-node: Tensor Parallelism (TP) across GPUs
        - Multi-node: Uses SGLang's native multi-node launcher with Ray
          * TP across GPUs within each node
          * Ray for distributed coordination
        
        For K8s:
        - Uses headless service for node discovery (similar to torchrun)
        - Each pod knows its rank via JOB_COMPLETION_INDEX
        - SGLang native launcher handles Ray cluster setup
        
        Args:
            nnodes: Number of nodes (pods). Must be >= 1.
            nproc_per_node: GPUs per node. Must be >= 1.
            master_port: Master communication port (for NCCL/Ray). Must be 1-65535.
            model_script: Path to model's run script. Cannot be empty.
        
        Returns:
            Complete SGLang launch setup with environment configuration
        
        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate inputs
        if not isinstance(nnodes, int) or nnodes < 1:
            raise ValueError(f"nnodes must be integer >= 1, got {nnodes}")
        if not isinstance(nproc_per_node, int) or nproc_per_node < 1:
            raise ValueError(f"nproc_per_node must be integer >= 1, got {nproc_per_node}")
        if not isinstance(master_port, int) or not (1 <= master_port <= 65535):
            raise ValueError(f"master_port must be 1-65535, got {master_port}")
        if not model_script or not isinstance(model_script, str):
            raise ValueError(f"model_script must be non-empty string, got {model_script}")
        
        # For single-node, simple TP setup
        if nnodes == 1:
            return f"""# SGLang single-node setup (Tensor Parallelism)
export SGLANG_TENSOR_PARALLEL_SIZE={nproc_per_node}
export SGLANG_PIPELINE_PARALLEL_SIZE=1
export NNODES=1
export NPROC_PER_NODE={nproc_per_node}
export NODE_RANK=0

echo "SGLang Configuration (Single Node):"
echo "  Tensor Parallel Size: {nproc_per_node}"
echo "  Total GPUs: {nproc_per_node}"

# SGLang native launcher handles everything
{model_script}"""
        
        # Multi-node: Use SGLang's native multi-node support
        return f"""# SGLang multi-node setup (K8s Indexed Job)
export MASTER_ADDR="{self.job_name}-0.{self.job_name}.{self.namespace}.svc.cluster.local"
export MASTER_PORT={master_port}
export NODE_RANK=${{JOB_COMPLETION_INDEX}}
export NNODES={nnodes}
export NPROC_PER_NODE={nproc_per_node}

# SGLang parallelism configuration
export SGLANG_TENSOR_PARALLEL_SIZE={nproc_per_node}
export SGLANG_PIPELINE_PARALLEL_SIZE=1

# Get current pod IP
POD_IP=$(hostname -i | awk '{{print $1}}')
export SGLANG_HOST_IP="$POD_IP"

echo "SGLang Configuration (Multi-Node):"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  NODE_RANK: $NODE_RANK (Pod Index)"
echo "  NNODES: $NNODES"
echo "  Tensor Parallel Size: {nproc_per_node}"
echo "  Pod IP: $POD_IP"
echo "  Total GPUs: {nnodes * nproc_per_node}"

# Clean any existing Ray processes
ray stop --force 2>/dev/null || true
pkill -9 -f "ray::" 2>/dev/null || true
sleep 2

# SGLang native launcher will handle Ray cluster coordination
# Pass NCCL init address for multi-node setup
export NCCL_INIT_ADDR="${{MASTER_ADDR}}:${{MASTER_PORT}}"

echo "Starting SGLang with native multi-node launcher..."
{model_script}

# Cleanup Ray on exit
trap "ray stop --force 2>/dev/null || true" EXIT"""

    def _generate_megatron_command(
        self, nnodes: int, nproc_per_node: int, master_port: int, model_script: str
    ) -> str:
        """
        Generate Megatron-LM launcher command for K8s Indexed Jobs.
        
        Megatron-LM is a training framework for large transformers with tensor and pipeline parallelism.
        It uses torchrun as the underlying launcher but with Megatron-specific environment variables.
        
        Architecture:
        - Single-node: Tensor Parallelism (TP) across GPUs
        - Multi-node: Tensor + Pipeline Parallelism
          * TP across GPUs within each node
          * PP across nodes
        
        For K8s:
        - Uses headless service for node discovery (like torchrun/deepspeed)
        - Each pod knows its rank via JOB_COMPLETION_INDEX
        - Sets TENSOR_MODEL_PARALLEL_SIZE and PIPELINE_MODEL_PARALLEL_SIZE (Megatron-Core standard)
        
        Args:
            nnodes: Number of nodes (pods). Must be >= 1.
            nproc_per_node: GPUs per node. Must be >= 1.
            master_port: Master communication port (for NCCL). Must be 1-65535.
            model_script: Path to model's run script. Cannot be empty.
        
        Returns:
            Complete Megatron-LM launch setup with environment configuration
        
        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate inputs
        if not isinstance(nnodes, int) or nnodes < 1:
            raise ValueError(f"nnodes must be integer >= 1, got {nnodes}")
        if not isinstance(nproc_per_node, int) or nproc_per_node < 1:
            raise ValueError(f"nproc_per_node must be integer >= 1, got {nproc_per_node}")
        if not isinstance(master_port, int) or not (1 <= master_port <= 65535):
            raise ValueError(f"master_port must be 1-65535, got {master_port}")
        if not model_script or not isinstance(model_script, str):
            raise ValueError(f"model_script must be non-empty string, got {model_script}")
        
        # For single-node, use TP only
        if nnodes == 1:
            return f"""# Megatron-LM single-node setup (Tensor Parallelism)
export TENSOR_MODEL_PARALLEL_SIZE={min(nproc_per_node, 8)}
export PIPELINE_MODEL_PARALLEL_SIZE=1
export CONTEXT_PARALLEL_SIZE=1
export NNODES=1
export NPROC_PER_NODE={nproc_per_node}
export MASTER_ADDR=localhost
export MASTER_PORT={master_port}
export NODE_RANK=0

echo "Megatron-LM Configuration (Single-Node):"
echo "  Tensor Model Parallel Size: {min(nproc_per_node, 8)}"
echo "  Pipeline Model Parallel Size: 1"
echo "  Total GPUs: {nproc_per_node}"

# Launch using torchrun with Megatron configuration
torchrun \\
    --standalone \\
    --nproc_per_node={nproc_per_node} \\
    {model_script}"""
        
        # Multi-node: TP + PP
        else:
            # Use headless service for node discovery (set by template)
            return f"""# Megatron-LM multi-node setup (Tensor + Pipeline Parallelism)
export TENSOR_MODEL_PARALLEL_SIZE={nproc_per_node}
export PIPELINE_MODEL_PARALLEL_SIZE={nnodes}
export CONTEXT_PARALLEL_SIZE=1
export NNODES={nnodes}
export NPROC_PER_NODE={nproc_per_node}
export NODE_RANK=${{JOB_COMPLETION_INDEX}}
export MASTER_ADDR=${{MASTER_ADDR}}
export MASTER_PORT={master_port}

echo "Megatron-LM Configuration (Multi-Node):"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  NODE_RANK: $NODE_RANK (Pod Index)"
echo "  NNODES: $NNODES"
echo "  Tensor Model Parallel Size: {nproc_per_node}"
echo "  Pipeline Model Parallel Size: {nnodes}"
echo "  Total GPUs: {nnodes * nproc_per_node}"

# Wait for all pods to be ready (K8s Indexed Job coordination)
echo "Waiting for all {nnodes} pods to be ready..."
sleep 5

# Launch using torchrun with Megatron multi-node configuration
torchrun \\
    --nnodes={nnodes} \\
    --nproc_per_node={nproc_per_node} \\
    --node_rank=${{NODE_RANK}} \\
    --master_addr=${{MASTER_ADDR}} \\
    --master_port={master_port} \\
    {model_script}"""
    
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

    def _create_results_pvc(self) -> str:
        """
        Create a PersistentVolumeClaim for results storage.
        
        Returns:
            Name of the created PVC
        """
        pvc_name = f"{self.job_name}-results"
        
        # Render PVC template
        template_dir = Path(__file__).parent / "templates" / "kubernetes"
        pvc_template = template_dir / "pvc.yaml.j2"
        
        with open(pvc_template, "r") as f:
            pvc_template_str = f.read()
        
        template = Template(pvc_template_str)
        pvc_yaml = template.render(
            pvc_name=pvc_name,
            namespace=self.namespace,
            storage_size=self.k8s_config.get("results_storage_size", "10Gi"),
            storage_class=self.k8s_config.get("storage_class")
        )
        
        # Create PVC
        pvc_dict = yaml.safe_load(pvc_yaml)
        self.core_v1.create_namespaced_persistent_volume_claim(
            namespace=self.namespace, body=pvc_dict
        )
        
        return pvc_name
    
    def _create_or_get_data_pvc(self, nnodes: int = 1) -> str:
        """
        Create or reuse a shared PersistentVolumeClaim for data storage.
        
        K8s best practice: Use shared PVC for data (separate from compute pods).
        This PVC is reusable across multiple training runs.
        
        Args:
            nnodes: Number of nodes (determines access mode requirements)
        
        Returns:
            Name of the PVC (existing or newly created)
        """
        # Use a consistent name for reusability (not job-specific)
        pvc_name = "madengine-shared-data"
        
        # Check if PVC already exists (idempotent)
        try:
            existing_pvc = self.core_v1.read_namespaced_persistent_volume_claim(
                name=pvc_name,
                namespace=self.namespace
            )
            self.console.print(f"[dim]✓ Using existing data PVC: {pvc_name}[/dim]")
            
            # Verify access mode for multi-node
            if nnodes > 1:
                access_modes = existing_pvc.spec.access_modes
                if "ReadWriteMany" not in access_modes:
                    self.console.print(
                        f"[yellow]⚠️  Warning: PVC {pvc_name} doesn't support ReadWriteMany[/yellow]"
                    )
                    self.console.print(
                        f"[yellow]   Multi-node deployment may fail. Current modes: {access_modes}[/yellow]"
                    )
            
            return pvc_name
            
        except ApiException as e:
            if e.status != 404:
                raise  # Unexpected error
            
            # PVC doesn't exist, create it
            # Determine access mode based on deployment topology
            # RWO (ReadWriteOnce): Single-node - works with most storage classes (local-path, EBS, etc.)
            # RWX (ReadWriteMany): Multi-node - requires shared storage (NFS, CephFS, etc.)
            access_mode = "ReadWriteMany" if nnodes > 1 else "ReadWriteOnce"
            
            self.console.print(f"[blue]Creating shared data PVC: {pvc_name}...[/blue]")
            self.console.print(f"[dim]  Access mode: {access_mode} ({'multi-node' if nnodes > 1 else 'single-node'})[/dim]")
            
            # Render data PVC template
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
                storage_class=self.k8s_config.get("storage_class")
            )
            
            # Create PVC
            pvc_dict = yaml.safe_load(pvc_yaml)
            self.core_v1.create_namespaced_persistent_volume_claim(
                namespace=self.namespace, body=pvc_dict
            )
            
            # Wait for PVC to be bound (important!)
            self.console.print(f"[dim]Waiting for PVC to be bound...[/dim]")
            for _ in range(30):  # Wait up to 30 seconds
                try:
                    pvc = self.core_v1.read_namespaced_persistent_volume_claim(
                        name=pvc_name, namespace=self.namespace
                    )
                    if pvc.status.phase == "Bound":
                        self.console.print(f"[green]✓ PVC bound successfully[/green]")
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
        
        # Delete existing PVC
        pvc_name = f"{self.job_name}-results"
        try:
            self.core_v1.delete_namespaced_persistent_volume_claim(
                name=pvc_name,
                namespace=self.namespace
            )
            self.console.print(f"[dim]Deleted existing PVC: {pvc_name}[/dim]")
        except ApiException as e:
            if e.status != 404:
                pass
        
        # Wait a moment for resources to be deleted
        import time
        time.sleep(2)  # Increased to allow PVC deletion

    def deploy(self) -> DeploymentResult:
        """Apply rendered manifests using kubernetes Python client."""
        try:
            # Clean up any existing resources first
            self._cleanup_existing_resources()
            
            # 1. Create PVC for results storage
            self.console.print("[blue]Creating PVC for results storage...[/blue]")
            pvc_name = self._create_results_pvc()
            self.console.print(f"[green]✓ Created PVC: {pvc_name}[/green]")
            
            # 1b. Create or reuse data PVC if data provider is configured and auto-creation was flagged
            if hasattr(self, '_data_config') and self._data_config:
                # Check if we set the PVC name during prepare (auto-creation case)
                data_pvc_name = self.k8s_config.get("data_pvc")
                if data_pvc_name == "madengine-shared-data":
                    # Auto-creation mode: create/reuse the PVC
                    nnodes = getattr(self, '_nnodes', 1)
                    self._create_or_get_data_pvc(nnodes=nnodes)
            
            # 2. Create ConfigMap
            self.console.print("[blue]Creating ConfigMap...[/blue]")
            configmap_dict = yaml.safe_load(self.configmap_yaml)
            self.core_v1.create_namespaced_config_map(
                namespace=self.namespace, body=configmap_dict
            )
            self.console.print(
                f"[green]✓ Created ConfigMap: {self.configmap_name}[/green]"
            )

            # 3. Create Service (if needed for multi-node)
            if self.service_yaml:
                self.console.print("[blue]Creating headless Service...[/blue]")
                service_dict = yaml.safe_load(self.service_yaml)
                self.core_v1.create_namespaced_service(
                    namespace=self.namespace, body=service_dict
                )
                self.console.print(f"[green]✓ Created Service: {self.job_name}[/green]")

            # 4. Create Job
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
        import time
        
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

    def collect_results(self, deployment_id: str) -> Dict[str, Any]:
        """
        Enhanced results collection from K8s pods following vLLM multi-node best practices.
        
        For Data Parallel deployments (vLLM, SGLang):
        - Each pod runs an independent replica
        - Only pod-0 reports metrics to avoid duplicates
        - Total throughput = pod-0 throughput × num_replicas
        
        Collects:
        1. Pod logs
        2. File artifacts via kubectl cp (profiling, tracing, env details)
        3. Results from shared PVC (if configured)
        
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
            is_ray_launcher = launcher_type in ["vllm", "sglang"]
            
            # Sort pods by name to ensure consistent ordering (pod-0 is master)
            sorted_pods = sorted(pods.items, key=lambda p: p.metadata.name)

            # For multi-node Ray-based launchers (vLLM, SGLang), only collect from pod-0
            # Worker pods run independent replicas and don't output metrics
            if is_multinode and is_ray_launcher:
                self.console.print(
                    f"[cyan]Multi-node Ray deployment: {nnodes} nodes (Data Parallel mode)[/cyan]"
                )
                self.console.print(
                    f"[dim]  Collecting from master pod only (pod-0)[/dim]"
                )
                pods_to_process = [sorted_pods[0]] if sorted_pods else []
                num_skipped = len(sorted_pods) - len(pods_to_process)
            else:
                pods_to_process = sorted_pods
                num_skipped = 0

            # Collect from each pod
            for pod_index, pod in enumerate(pods_to_process):
                pod_name = pod.metadata.name
                pod_dir = results_dir / pod_name
                pod_dir.mkdir(exist_ok=True)
                
                self.console.print(f"[dim]  Collecting from pod: {pod_name}[/dim]")
                
                try:
                    # 1. Collect pod logs
                    log = self.core_v1.read_namespaced_pod_log(
                        name=pod_name, namespace=self.namespace
                    )
                    log_file = pod_dir / f"{pod_name}.log"
                    log_file.write_text(log)
                    results["logs"].append({
                        "pod": pod_name,
                        "log": log,
                        "file": str(log_file)
                    })
                    
                    # 2. Parse performance from log
                    perf_data = self._parse_performance_from_log(
                        log, model_info, build_info, pod_name
                    )
                    
                    if perf_data:
                        # For multi-node Ray deployments, multiply by nnodes
                        # This gives total throughput (Data Parallel mode)
                        if is_multinode and is_ray_launcher:
                            original_perf = perf_data.get("performance", 0.0)
                            perf_data["performance"] = original_perf * nnodes
                            perf_data["performance_per_replica"] = original_perf
                            perf_data["topology_note"] = (
                                f"Data Parallel: {nnodes} independent replicas"
                            )
                            
                            self.console.print(
                                f"[green]  Per-replica: {original_perf:.1f} req/s[/green]"
                            )
                            self.console.print(
                                f"[green]  Total capacity: {perf_data['performance']:.1f} req/s "
                                f"({nnodes} nodes)[/green]"
                            )
                        
                        results["successful_runs"].append(perf_data)
                        # Write to local perf.csv
                        self._write_to_perf_csv(perf_data)
                    else:
                        # Only mark as FAILED if we expected metrics from this pod
                        error_msg = "Failed to parse performance metrics from logs"
                        failure_record = self._create_failure_record(
                            model_info, build_info, pod_name, error_msg
                        )
                        results["failed_runs"].append({
                            "model": model_info.get("name", "Unknown"),
                            "pod": pod_name,
                            "error": error_msg,
                            "perf_data": failure_record
                        })
                        # Write failure to perf.csv
                        self._write_to_perf_csv(failure_record)
                        self.console.print(
                            f"[yellow]⚠ No performance metrics found for pod {pod_name}, "
                            f"recorded as FAILED[/yellow]"
                        )
                        
                except ApiException as e:
                    # Only create failure record if we expected metrics from this pod
                    error_msg = f"Failed to get logs: {e.reason}"
                    failure_record = self._create_failure_record(
                        model_info, build_info, pod_name, error_msg
                    )
                    results["failed_runs"].append({
                        "model": model_info.get("name", "Unknown"),
                        "pod": pod_name,
                        "error": error_msg,
                        "perf_data": failure_record
                    })
                    # Write failure to perf.csv
                    self._write_to_perf_csv(failure_record)
                    self.console.print(
                        f"[red]✗ Failed to get logs for pod {pod_name}: {e.reason}[/red]"
                    )
                except Exception as e:
                    error_msg = str(e)
                    failure_record = self._create_failure_record(
                        model_info, build_info, pod_name, error_msg
                    )
                    results["failed_runs"].append({
                        "model": model_info.get("name", "Unknown"),
                        "pod": pod_name,
                        "error": error_msg,
                        "perf_data": failure_record
                    })
                    # Write failure to perf.csv
                    self._write_to_perf_csv(failure_record)
                    self.console.print(
                        f"[red]✗ Error collecting results from pod {pod_name}: {e}[/red]"
                    )
            
            # Report what we skipped for multi-node
            if num_skipped > 0:
                self.console.print(
                    f"[dim]  Skipped {num_skipped} worker pod(s) "
                    f"(no metrics expected in Data Parallel mode)[/dim]"
                )

            self.console.print(
                f"[green]✓ Collected logs from {len(results['logs'])} pods[/green]"
            )
            
            if results["successful_runs"]:
                self.console.print(
                    f"[green]✓ Parsed {len(results['successful_runs'])} performance results[/green]"
                )
                self.console.print(
                    f"[green]✓ Updated local perf.csv[/green]"
                )
            
            # 4. Collect all artifacts from PVC
            self._collect_from_pvc(deployment_id, results_dir, results)
            
            # 5. Generate summary
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
    
    def _collect_from_pvc(self, deployment_id: str, results_dir: Path, results: Dict):
        """
        Collect all artifacts from the PVC using a temporary busybox pod.
        
        This is the best practice for collecting results from completed K8s jobs.
        kubectl cp doesn't work on completed pods, so we use a helper pod.
        
        Args:
            deployment_id: Job deployment ID
            results_dir: Local directory to save results
            results: Results dict to update
        """
        pvc_name = f"{deployment_id}-results"
        
        try:
            # Create a temporary pod to access PVC
            collector_pod_name = f"collector-{deployment_id[:15]}"
            
            self.console.print(f"[dim]📦 Collecting artifacts from PVC: {pvc_name}[/dim]")
            
            collector_pod_spec = {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {"name": collector_pod_name, "namespace": self.namespace},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [{
                        "name": "collector",
                        "image": "busybox:latest",
                        "command": ["sh", "-c", "sleep 600"],
                        "volumeMounts": [{"name": "results", "mountPath": "/results"}]
                    }],
                    "volumes": [{"name": "results", "persistentVolumeClaim": {"claimName": pvc_name}}]
                }
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
            
            # List pod result directories in PVC
            list_cmd = [
                "kubectl", "exec", collector_pod_name, "-n", self.namespace, "--",
                "ls", "-1", "/results/"
            ]
            list_result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=10)
            
            if list_result.returncode == 0 and list_result.stdout.strip():
                pod_dirs = list_result.stdout.strip().split('\n')
                
                for pod_dir_name in pod_dirs:
                    if not pod_dir_name:
                        continue
                    
                    # Copy entire pod directory
                    local_pod_dir = results_dir / pod_dir_name
                    local_pod_dir.mkdir(exist_ok=True)
                    
                    cp_cmd = [
                        "kubectl", "cp",
                        f"{self.namespace}/{collector_pod_name}:/results/{pod_dir_name}",
                        str(local_pod_dir)
                    ]
                    
                    cp_result = subprocess.run(cp_cmd, capture_output=True, text=True, timeout=60)
                    
                    if cp_result.returncode == 0:
                        # Count collected files
                        file_count = sum(1 for _ in local_pod_dir.rglob('*') if _.is_file())
                        if file_count > 0:
                            results["artifacts"].append({
                                "source": f"PVC:{pvc_name}/{pod_dir_name}",
                                "local_path": str(local_pod_dir),
                                "file_count": file_count,
                                "type": "pvc_collection"
                            })
                            self.console.print(f"[dim]    ✓ Collected {file_count} files from {pod_dir_name}[/dim]")
                
                self.console.print(f"[green]✓ Collected artifacts from PVC[/green]")
            else:
                self.console.print(f"[yellow]⚠ No results found in PVC[/yellow]")
            
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
    
    def _parse_performance_from_log(self, log: str, model_info: Dict, build_info: Dict, pod_name: str) -> Optional[Dict]:
        """
        Parse pod log to extract performance metrics.
        
        Creates a result dict matching the local execution CSV format for consistency.
        
        Args:
            log: Pod log content
            model_info: Model information from manifest
            build_info: Build information from manifest  
            pod_name: Kubernetes pod name
            
        Returns:
            Dict with all perf.csv fields, or None if parsing failed
        """
        import re
        import os
        from datetime import datetime
        
        # Look for performance line: "performance: 12345 metric_name"
        perf_pattern = r'performance:\s+([0-9,.]+)\s+([a-zA-Z_/]+)'
        match = re.search(perf_pattern, log)
        
        if not match:
            return None
        
        performance = match.group(1).replace(',', '')  # Remove commas
        metric = match.group(2)
        
        # NEW: Extract topology information from log
        # Format: "topology: 2 nodes 2 gpus_per_node 4 total_gpus"
        topology_pattern = r'topology:\s+(\d+)\s+nodes\s+(\d+)\s+gpus_per_node\s+(\d+)\s+total_gpus'
        topology_match = re.search(topology_pattern, log)
        
        if topology_match:
            nnodes = topology_match.group(1)
            gpus_per_node = topology_match.group(2)
            total_gpus = topology_match.group(3)
        else:
            # Fallback: Try to get from manifest distributed config
            distributed_config = self.manifest.get("deployment_config", {}).get("distributed", {})
            nnodes = str(distributed_config.get("nnodes", 1))
            gpus_per_node = str(distributed_config.get("nproc_per_node", 1))
            total_gpus = str(model_info.get("n_gpus", 1))
        
        # Extract GPU architecture from device ID in log
        gpu_architecture = ""
        gpu_match = re.search(r'0x([0-9a-fA-F]+)', log)
        if gpu_match:
            device_id = gpu_match.group(1)
            # Map device IDs to architecture names (same as MAD_SYSTEM_GPU_ARCHITECTURE)
            gpu_map = {
                '74a1': 'gfx90a',  # MI250X
                '740c': 'gfx90a',  # MI210
                '740f': 'gfx90a',  # MI210  
                '7408': 'gfx908',  # MI100
                '73a1': 'gfx942',  # MI300X
                '740f': 'gfx940',  # MI300A
            }
            gpu_architecture = gpu_map.get(device_id, "")
        
        # Extract test duration from logs if available
        test_duration = ""
        # Look for "test_duration: 1.234s" format
        duration_match = re.search(r'test_duration:\s+([0-9.]+)s?', log, re.IGNORECASE)
        if duration_match:
            test_duration = duration_match.group(1)
        
        # Extract data provider metrics from logs if available
        # These are printed by the data provider scripts via "✓ Data metrics: ..."
        dataname = model_info.get("data", "")  # Get from model info
        data_provider_type = ""
        data_size = ""
        data_download_duration = ""
        
        # Look for "=== Data Provider: <type> ===" line
        provider_match = re.search(r'===\s+Data Provider:\s+(\w+)\s+===', log)
        if provider_match:
            data_provider_type = provider_match.group(1)
        
        # Look for data metrics line: "✓ Data metrics: Duration=18s, Size=1.3G"
        metrics_match = re.search(r'Duration=([0-9]+)s,\s+Size=([0-9.]+[KMGT]?)', log)
        if metrics_match:
            data_download_duration = metrics_match.group(1)
            data_size = metrics_match.group(2)
        
        # Alternative: Look for individual Duration and Size lines
        if not data_download_duration:
            duration_data_match = re.search(r'Duration:\s+([0-9]+)s', log)
            if duration_data_match:
                data_download_duration = duration_data_match.group(1)
        
        if not data_size:
            size_match = re.search(r'Size:\s+([0-9.]+[KMGT]?)', log)
            if size_match:
                data_size = size_match.group(1)
        
        # Build performance result dict matching local execution format EXACTLY
        # This ensures compatibility with existing perf.csv analysis tools
        result = {
            # Core identification
            "model": model_info.get("name", ""),
            "n_gpus": total_gpus,  # Use parsed total_gpus
            "nnodes": nnodes,  # NEW: Number of nodes
            "gpus_per_node": gpus_per_node,  # NEW: GPUs per node
            
            # Model configuration
            "training_precision": model_info.get("training_precision", ""),
            "pipeline": os.environ.get("pipeline", ""),
            "args": model_info.get("args", ""),
            "tags": model_info.get("tags", ""),
            
            # Build information
            "docker_file": build_info.get("dockerfile", ""),
            "base_docker": build_info.get("base_docker", ""),
            "docker_sha": build_info.get("docker_sha", ""),
            "docker_image": build_info.get("docker_image", ""),
            
        # Runtime information
        "git_commit": "",  # Not available in K8s pod
        "machine_name": pod_name,  # Use pod name as machine identifier
        "deployment_type": "kubernetes",  # Deployment environment
        "launcher": distributed_config.get("launcher", "native"),  # Execution launcher (native, torchrun, megatron, etc.)
        "gpu_architecture": gpu_architecture,
            
            # Performance metrics
            "performance": performance,
            "metric": metric,
            "relative_change": "",
            "status": "SUCCESS",
            
            # Timing
            "build_duration": build_info.get("build_duration", ""),
            "test_duration": test_duration,
            
            # Data information
            "dataname": dataname,
            "data_provider_type": data_provider_type,
            "data_size": data_size,
            "data_download_duration": data_download_duration,
            
            # Build tracking
            "build_number": os.environ.get("BUILD_NUMBER", "0"),
            "additional_docker_run_options": model_info.get("additional_docker_run_options", ""),
        }
        
        # Flatten tags if they are in list format (same as local execution)
        if isinstance(result["tags"], list):
            result["tags"] = ",".join(str(item) for item in result["tags"])
        
        return result
    
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
            "pipeline": os.environ.get("pipeline", ""),
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
            "build_number": os.environ.get("BUILD_NUMBER", "0"),
            "additional_docker_run_options": model_info.get("additional_docker_run_options", ""),
        }
        
        # Flatten tags if they are in list format
        if isinstance(result["tags"], list):
            result["tags"] = ",".join(str(item) for item in result["tags"])
        
        return result
    
    def _write_to_perf_csv(self, perf_data: Dict):
        """
        Write performance data to local perf.csv file.
        
        Uses the same format as local execution for consistency.
        Matches the schema from container_runner.py's create_run_details_dict().
        """
        import csv
        from pathlib import Path
        
        perf_csv_path = Path("perf.csv")
        
        # Check if file exists to determine if we need headers
        file_exists = perf_csv_path.exists()
        
        # CSV headers matching local execution format EXACTLY
        # This is the same order as in container_runner.py line 69
        # Enhanced with topology fields for multi-node tracking
        headers = [
            "model",
            "n_gpus",
            "nnodes",              # NEW: Number of nodes
            "gpus_per_node",       # NEW: GPUs per node
            "training_precision",
            "pipeline",
            "args",
            "tags",
            "docker_file",
            "base_docker",
            "docker_sha",
            "docker_image",
            "git_commit",
            "machine_name",
            "deployment_type",
            "launcher",            # Execution launcher (native, docker, torchrun, etc.)
            "gpu_architecture",
            "performance",
            "metric",
            "relative_change",
            "status",
            "build_duration",
            "test_duration",
            "dataname",
            "data_provider_type",
            "data_size",
            "data_download_duration",
            "build_number",
            "additional_docker_run_options",
        ]
        
        # Write to CSV
        with open(perf_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
            
            # Write headers if new file
            if not file_exists:
                writer.writeheader()
            
            # Write data row (only fields in headers will be written)
            writer.writerow(perf_data)

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

