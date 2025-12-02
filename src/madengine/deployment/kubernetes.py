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
import time
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

from jinja2 import Environment, FileSystemLoader

from .base import BaseDeployment, DeploymentConfig, DeploymentResult, DeploymentStatus
from madengine.core.dataprovider import Data
from madengine.core.context import Context


class KubernetesDeployment(BaseDeployment):
    """
    Kubernetes cluster deployment using Python client library.

    Uses kubernetes Python API for type-safe, production-ready deployment:
    - client.BatchV1Api(): Job creation and management
    - client.CoreV1Api(): Pod logs and status

    Requires AMD GPU Device Plugin: https://github.com/ROCm/k8s-device-plugin

    **Workflow**:
    1. User has kubeconfig configured (in-cluster or ~/.kube/config)
    2. madengine-cli run --tags model --additional-context '{"deploy": "k8s", ...}'
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

        super().__init__(config)

        # Parse K8s configuration
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
                except:
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
    
    def _load_common_scripts(self, script_list: List[Dict]) -> Dict[str, str]:
        """
        Load common script contents from madengine package for embedding in ConfigMap.
        
        Since madengine is not installed in model Docker images, we need to embed
        the common scripts (pre_scripts, post_scripts) in the ConfigMap.
        
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
        
        return script_contents

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
        gpu_count = int(model_info.get("n_gpus", 1))
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

        # Load model run script content
        run_script_content = None
        model_script_path = model_info.get("scripts")  # e.g., "scripts/dummy/run_data_minio.sh"
        model_script_dir = None
        model_script_filename = None
        
        if model_script_path:
            script_file = Path(model_script_path)
            # Extract directory and filename
            model_script_dir = str(script_file.parent)  # e.g., "scripts/dummy"
            model_script_filename = script_file.name     # e.g., "run_data_minio.sh"
            
            if script_file.exists():
                with open(script_file, "r") as f:
                    run_script_content = f.read()
                self.console.print(f"[dim]Loaded script: {model_script_path}[/dim]")
            else:
                self.console.print(f"[yellow]Warning: Script not found: {model_script_path}[/yellow]")
        
        # Load K8s tools configuration
        k8s_tools_config = self._load_k8s_tools()
        
        # Prepare data configuration first
        data_config = self._prepare_data_config(model_info)
        
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
        
        # Get launcher configuration if present
        launcher_config = self.config.additional_context.get("launcher")
        launcher_type = launcher_config.get("type") if launcher_config else None
        launcher_command = None

        # Determine if we need multi-node setup
        nnodes = 1
        create_headless_service = False
        subdomain = None

        if launcher_type == "torchrun":
            nnodes = launcher_config.get("nnodes", 1)
            if nnodes > 1:
                create_headless_service = True
                subdomain = self.job_name

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
            "run_script_content": run_script_content,
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
            "subdomain": subdomain,
            # Execution
            "gpu_visibility": "0",
            "gpu_architecture": self.manifest.get("context", {}).get(
                "gpu_architecture", "gfx90a"
            ),
            "model_script": model_info.get("scripts", "run.sh"),
            "launcher_type": launcher_type,
            "launcher_command": launcher_command,
            "timeout": self.config.timeout,
            # Environment - Merge base env vars with data/tools env vars
            "env_vars": self._prepare_env_vars(model_info),
            # Volumes
            "results_pvc": self.k8s_config.get("results_pvc"),
            "data_pvc": self.k8s_config.get("data_pvc"),
            # Multi-node
            "create_headless_service": create_headless_service,
            "service_name": self.job_name,
            "ports": [29500] if create_headless_service else [],
            # Data provider configuration (already prepared above)
            "data_config": data_config,
            # Tools configuration - from manifest.context or additional_context
            "tools_config": self._get_tools_config(),
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
            List of tool configurations
        """
        # Check runtime additional_context first (allows runtime override)
        tools = self.config.additional_context.get("tools", [])
        
        # Fall back to manifest.context if no runtime tools
        if not tools and "context" in self.manifest:
            tools = self.manifest["context"].get("tools", [])
        
        return tools
    
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
        if data_config and "env_vars" in data_config:
            env_vars.update(data_config["env_vars"])
        
        # 3. Tools configuration environment variables
        # Check both additional_context and manifest.context for tools
        tools_config = self.config.additional_context.get("tools", [])
        if not tools_config and "context" in self.manifest:
            tools_config = self.manifest["context"].get("tools", [])
        
        for tool in tools_config:
            if "env_vars" in tool:
                env_vars.update(tool["env_vars"])
        
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
            
            return {
                "data_name": model_info["data"],
                "env_vars": data_env or {},
                "provider_type": provider_type,
                "source_url": source_url,
                "datahome": data_env.get("MAD_DATAHOME", "/data_dlm_0") if data_env else "/data_dlm_0",
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
        
        # Wait a moment for resources to be deleted
        import time
        time.sleep(1)

    def deploy(self) -> DeploymentResult:
        """Apply rendered manifests using kubernetes Python client."""
        try:
            # Clean up any existing resources first
            self._cleanup_existing_resources()
            
            # 1. Create ConfigMap
            self.console.print("[blue]Creating ConfigMap...[/blue]")
            configmap_dict = yaml.safe_load(self.configmap_yaml)
            self.core_v1.create_namespaced_config_map(
                namespace=self.namespace, body=configmap_dict
            )
            self.console.print(
                f"[green]✓ Created ConfigMap: {self.configmap_name}[/green]"
            )

            # 2. Create Service (if needed for multi-node)
            if self.service_yaml:
                self.console.print("[blue]Creating headless Service...[/blue]")
                service_dict = yaml.safe_load(self.service_yaml)
                self.core_v1.create_namespaced_service(
                    namespace=self.namespace, body=service_dict
                )
                self.console.print(f"[green]✓ Created Service: {self.job_name}[/green]")

            # 3. Create Job
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
                        
                        # Print new log lines
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
        Collect Job results and logs.
        
        Parses pod logs to extract performance metrics and creates
        local perf.csv entries compatible with madengine format.
        """
        results = {
            "job_name": deployment_id,
            "namespace": self.namespace,
            "logs": [],
            "successful_runs": [],
            "failed_runs": [],
        }

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

            # Collect logs from each pod
            for pod in pods.items:
                pod_name = pod.metadata.name
                try:
                    log = self.core_v1.read_namespaced_pod_log(
                        name=pod_name, namespace=self.namespace
                    )
                    results["logs"].append({"pod": pod_name, "log": log})
                    
                    # Parse log to extract performance metrics
                    perf_data = self._parse_performance_from_log(log, model_info, build_info, pod_name)
                    if perf_data:
                        results["successful_runs"].append(perf_data)
                        # Write to local perf.csv
                        self._write_to_perf_csv(perf_data)
                    else:
                        results["failed_runs"].append({
                            "pod": pod_name,
                            "error": "Failed to parse performance metrics from logs"
                        })
                        
                except ApiException as e:
                    results["failed_runs"].append({
                        "pod": pod_name,
                        "error": f"Failed to get logs: {e.reason}"
                    })

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

        except Exception as e:
            self.console.print(f"[yellow]⚠ Results collection incomplete: {e}[/yellow]")

        return results
    
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
        
        # Extract duration from logs if available
        test_duration = ""
        duration_match = re.search(r'duration:\s+([0-9.]+)', log, re.IGNORECASE)
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
            "n_gpus": str(model_info.get("n_gpus", "1")),
            
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
        headers = [
            "model",
            "n_gpus",
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

