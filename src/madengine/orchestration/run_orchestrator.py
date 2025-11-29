#!/usr/bin/env python3
"""
Run Orchestrator - Coordinates model execution workflow.

Supports:
1. Run-only (with manifest): Run pre-built images
2. Full workflow (with tags): Build + Run
3. Local execution: Direct container execution
4. Distributed deployment: SLURM or Kubernetes

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console as RichConsole

from madengine.core.console import Console
from madengine.core.context import Context
from madengine.core.dataprovider import Data
from madengine.core.errors import (
    ConfigurationError,
    RuntimeError as MADRuntimeError,
    create_error_context,
    handle_error,
)


class RunOrchestrator:
    """
    Orchestrates the run workflow.

    Responsibilities:
    - Load manifest or trigger build
    - Determine execution target (local vs distributed)
    - Delegate to appropriate executor (container_runner or deployment)
    - Collect and aggregate results
    """

    def __init__(self, args, additional_context: Optional[Dict] = None):
        """
        Initialize run orchestrator.

        Args:
            args: CLI arguments namespace
            additional_context: Dict from --additional-context
        """
        self.args = args
        self.console = Console(live_output=getattr(args, "live_output", True))
        self.rich_console = RichConsole()

        # Merge additional_context from args and parameter
        merged_context = {}
        if hasattr(args, "additional_context") and args.additional_context:
            try:
                if isinstance(args.additional_context, str):
                    merged_context = json.loads(args.additional_context)
                elif isinstance(args.additional_context, dict):
                    merged_context = args.additional_context
            except json.JSONDecodeError:
                pass

        if additional_context:
            merged_context.update(additional_context)

        self.additional_context = merged_context

        # Initialize context in runtime mode (with GPU detection for local)
        # This will be lazy-initialized only when needed
        self.context = None
        self.data = None

    def _init_runtime_context(self):
        """Initialize runtime context (with GPU detection)."""
        if self.context is not None:
            return

        # Context expects additional_context as a string, not dict
        context_string = json.dumps(self.additional_context) if self.additional_context else None
        self.context = Context(
            additional_context=context_string,
            build_only_mode=False,
        )

        # Initialize data provider if data config exists
        data_json_file = getattr(self.args, "data_config_file_name", "data.json")
        if os.path.exists(data_json_file):
            self.data = Data(
                self.context,
                filename=data_json_file,
                force_mirrorlocal=getattr(self.args, "force_mirror_local", False),
            )

    def execute(
        self,
        manifest_file: Optional[str] = None,
        tags: Optional[list] = None,
        registry: Optional[str] = None,
        timeout: int = 3600,
    ) -> Dict:
        """
        Execute run workflow.

        Supports two modes:
        1. Run-only: If manifest_file provided
        2. Full workflow: If tags provided (build + run)

        Args:
            manifest_file: Path to build_manifest.json
            tags: Model tags to build (triggers build phase if no manifest)
            registry: Optional registry override
            timeout: Execution timeout in seconds

        Returns:
            Execution results dict

        Raises:
            ConfigurationError: If neither manifest nor tags provided
            MADRuntimeError: If execution fails
        """
        self.rich_console.print(f"\n[dim]{'=' * 60}[/dim]")
        self.rich_console.print("[bold blue]🚀 RUN PHASE[/bold blue]")
        self.rich_console.print(f"[dim]{'=' * 60}[/dim]\n")

        try:
            # Step 1: Ensure we have a manifest (build if needed)
            if not manifest_file or not os.path.exists(manifest_file):
                if not tags:
                    raise ConfigurationError(
                        "Either --manifest-file or --tags required",
                        context=create_error_context(
                            operation="run_phase",
                            component="RunOrchestrator",
                        ),
                        suggestions=[
                            "Provide --manifest-file path to run pre-built images",
                            "Provide --tags to build and run models",
                        ],
                    )

                self.rich_console.print("[cyan]No manifest found, building first...[/cyan]\n")
                manifest_file = self._build_phase(tags, registry)

            # Step 2: Load manifest and merge with runtime context
            manifest_file = self._load_and_merge_manifest(manifest_file)

            # Step 3: Determine execution target
            target = self.additional_context.get("deploy", "local")

            self.rich_console.print(f"[bold cyan]Deployment target: {target}[/bold cyan]\n")

            # Step 4: Execute based on target
            if target == "local":
                return self._execute_local(manifest_file, timeout)
            else:
                return self._execute_distributed(target, manifest_file)

        except (ConfigurationError, MADRuntimeError):
            raise
        except Exception as e:
            context = create_error_context(
                operation="run_phase",
                component="RunOrchestrator",
            )
            raise MADRuntimeError(
                f"Run phase failed: {e}",
                context=context,
                suggestions=[
                    "Check manifest file exists and is valid",
                    "Verify Docker daemon is running",
                    "Check network connectivity",
                ],
            ) from e

    def _build_phase(self, tags: list, registry: Optional[str] = None) -> str:
        """Trigger build phase if needed."""
        from .build_orchestrator import BuildOrchestrator

        # Update args with tags
        self.args.tags = tags

        build_orch = BuildOrchestrator(self.args, self.additional_context)
        manifest_file = build_orch.execute(
            registry=registry,
            clean_cache=getattr(self.args, "clean_docker_cache", False),
        )

        return manifest_file

    def _load_and_merge_manifest(self, manifest_file: str) -> str:
        """Load manifest and merge with runtime --additional-context."""
        if not os.path.exists(manifest_file):
            raise FileNotFoundError(f"Build manifest not found: {manifest_file}")

        with open(manifest_file, "r") as f:
            manifest = json.load(f)

        print(f"Loaded manifest with {len(manifest.get('built_images', {}))} images")

        # Merge deployment configs (runtime overrides build-time)
        if "deployment_config" in manifest and self.additional_context:
            stored_config = manifest["deployment_config"]

            # Runtime --additional-context overrides stored config
            for key in ["deploy", "slurm", "k8s", "kubernetes", "distributed", "vllm", "env_vars"]:
                if key in self.additional_context:
                    stored_config[key] = self.additional_context[key]

            manifest["deployment_config"] = stored_config

            # Write back merged config
            with open(manifest_file, "w") as f:
                json.dump(manifest, f, indent=2)

            print("Merged runtime deployment config with manifest")

        return manifest_file

    def _execute_local(self, manifest_file: str, timeout: int) -> Dict:
        """Execute locally using container_runner."""
        self.rich_console.print("[cyan]Executing locally...[/cyan]\n")

        # Initialize runtime context (with GPU detection)
        self._init_runtime_context()

        # Show node ROCm info
        self._show_node_info()

        # Import from execution layer
        from madengine.execution.container_runner import ContainerRunner

        # Load credentials
        credentials = self._load_credentials()

        # Load manifest to restore context
        with open(manifest_file, "r") as f:
            manifest = json.load(f)

        # Restore context from manifest if present
        if "context" in manifest:
            manifest_context = manifest["context"]
            if "tools" in manifest_context:
                self.context.ctx["tools"] = manifest_context["tools"]
            if "pre_scripts" in manifest_context:
                self.context.ctx["pre_scripts"] = manifest_context["pre_scripts"]
            if "post_scripts" in manifest_context:
                self.context.ctx["post_scripts"] = manifest_context["post_scripts"]
            if "encapsulate_script" in manifest_context:
                self.context.ctx["encapsulate_script"] = manifest_context["encapsulate_script"]

        # Filter images by GPU architecture
        try:
            runtime_gpu_arch = self.context.get_system_gpu_architecture()
            print(f"Runtime GPU architecture detected: {runtime_gpu_arch}")

            compatible_images = self._filter_images_by_gpu_architecture(
                manifest["built_images"], runtime_gpu_arch
            )

            if not compatible_images:
                raise MADRuntimeError(
                    f"No compatible images for GPU architecture '{runtime_gpu_arch}'",
                    context=create_error_context(
                        operation="filter_images",
                        component="RunOrchestrator",
                    ),
                    suggestions=[
                        f"Build images for {runtime_gpu_arch} using --target-archs",
                        "Check manifest contains images for your GPU",
                    ],
                )

            manifest["built_images"] = compatible_images
            print(f"Filtered to {len(compatible_images)} compatible images\n")

        except Exception as e:
            self.rich_console.print(f"[yellow]Warning: GPU filtering failed: {e}[/yellow]")
            self.rich_console.print("[yellow]Proceeding with all images[/yellow]\n")

        # Copy scripts
        self._copy_scripts()

        # Initialize runner
        runner = ContainerRunner(
            self.context,
            self.data,
            self.console,
            live_output=getattr(self.args, "live_output", False),
        )
        runner.set_credentials(credentials)

        if hasattr(self.args, "output") and self.args.output:
            runner.set_perf_csv_path(self.args.output)

        # Determine phase suffix
        phase_suffix = (
            ".run"
            if hasattr(self.args, "_separate_phases") and self.args._separate_phases
            else ""
        )

        # Run models
        results = runner.run_models_from_manifest(
            manifest_file=manifest_file,
            registry=getattr(self.args, "registry", None),
            timeout=timeout,
            keep_alive=getattr(self.args, "keep_alive", False),
            phase_suffix=phase_suffix,
        )

        self.rich_console.print(f"\n[green]✓ Local execution complete[/green]")
        self.rich_console.print(f"[dim]{'=' * 60}[/dim]\n")

        return results

    def _execute_distributed(self, target: str, manifest_file: str) -> Dict:
        """Execute on distributed infrastructure."""
        self.rich_console.print(f"[cyan]Deploying to {target}...[/cyan]\n")

        # Import from deployment layer
        from madengine.deployment.factory import DeploymentFactory
        from madengine.deployment.base import DeploymentConfig

        # Create deployment configuration
        deployment_config = DeploymentConfig(
            target=target,
            manifest_file=manifest_file,
            additional_context=self.additional_context,
            timeout=getattr(self.args, "timeout", 3600),
            monitor=self.additional_context.get("monitor", True),
            cleanup_on_failure=self.additional_context.get("cleanup_on_failure", True),
        )

        # Create and execute deployment
        deployment = DeploymentFactory.create(deployment_config)
        result = deployment.execute()

        if result.is_success:
            self.rich_console.print(f"[green]✓ Deployment to {target} complete[/green]")
            self.rich_console.print(f"  Deployment ID: {result.deployment_id}")
            if result.logs_path:
                self.rich_console.print(f"  Logs: {result.logs_path}")
        else:
            self.rich_console.print(f"[red]✗ Deployment to {target} failed[/red]")
            self.rich_console.print(f"  Error: {result.message}")

        self.rich_console.print(f"[dim]{'=' * 60}[/dim]\n")

        return result.metrics or {}

    def _show_node_info(self):
        """Show node ROCm information."""
        self.console.sh("echo 'MAD Run Models'")

        host_os = self.context.ctx.get("host_os", "")
        if "HOST_UBUNTU" in host_os:
            print(self.console.sh("apt show rocm-libs -a", canFail=True))
        elif "HOST_CENTOS" in host_os:
            print(self.console.sh("yum info rocm-libs", canFail=True))
        elif "HOST_SLES" in host_os:
            print(self.console.sh("zypper info rocm-libs", canFail=True))
        elif "HOST_AZURE" in host_os:
            print(self.console.sh("tdnf info rocm-libs", canFail=True))
        else:
            self.rich_console.print("[yellow]Warning: Unable to detect host OS[/yellow]")

    def _copy_scripts(self):
        """Copy common scripts to model directories.
        
        Handles two scenarios:
        1. MAD Project: scripts/common already exists with pre/post scripts
        2. madengine Testing: Need to copy from src/madengine/scripts/common
        """
        import shutil

        # Step 1: Check if MODEL_DIR is set and copy if needed
        model_dir_env = os.environ.get("MODEL_DIR")
        if model_dir_env and os.path.exists(model_dir_env) and model_dir_env != ".":
            self.rich_console.print(f"[yellow]📁 MODEL_DIR detected: {model_dir_env}[/yellow]")
            self.rich_console.print("[yellow]Copying MODEL_DIR contents for run phase...[/yellow]")
            
            # Copy docker/ and scripts/ from MODEL_DIR
            for subdir in ["docker", "scripts"]:
                src_path = Path(model_dir_env) / subdir
                if src_path.exists():
                    dest_path = Path(subdir)
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    shutil.copytree(src_path, dest_path)
            
            self.rich_console.print("[green]✓ MODEL_DIR structure copied (docker/, scripts/)[/green]")

        # Step 2: Copy madengine's common scripts (pre_scripts, post_scripts, tools)
        # This provides the execution framework scripts
        madengine_common = Path("src/madengine/scripts/common")
        if madengine_common.exists():
            print(f"Copying madengine common scripts from {madengine_common} to scripts/common")
            
            dest_common = Path("scripts/common")
            
            # Copy pre_scripts, post_scripts, tools if they exist
            for item in ["pre_scripts", "post_scripts", "tools", "tools.json", "test_echo.sh"]:
                src_item = madengine_common / item
                if src_item.exists():
                    dest_item = dest_common / item
                    if dest_item.exists():
                        if dest_item.is_dir():
                            shutil.rmtree(dest_item)
                        else:
                            dest_item.unlink()
                    
                    if src_item.is_dir():
                        shutil.copytree(src_item, dest_item)
                    else:
                        dest_common.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_item, dest_item)
                    print(f"  Copied {item}")

        # Step 3: Distribute scripts/common to each model directory
        common_scripts = Path("scripts/common")
        if not common_scripts.exists():
            self.rich_console.print("[yellow]⚠️  No scripts/common directory found after copy, skipping distribution[/yellow]")
            return

        print(f"Distributing common scripts to model directories")

        for model_script_dir in Path("scripts").iterdir():
            if model_script_dir.is_dir() and model_script_dir.name != "common":
                dest = model_script_dir / "common"
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(common_scripts, dest)
                print(f"  Copied to {dest}")

    def _load_credentials(self) -> Optional[Dict]:
        """Load credentials from credential.json and environment."""
        credentials = None

        credential_file = "credential.json"
        if os.path.exists(credential_file):
            try:
                with open(credential_file) as f:
                    credentials = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load credentials: {e}")

        # Override with environment variables
        docker_hub_user = os.environ.get("MAD_DOCKERHUB_USER")
        docker_hub_password = os.environ.get("MAD_DOCKERHUB_PASSWORD")
        docker_hub_repo = os.environ.get("MAD_DOCKERHUB_REPO")

        if docker_hub_user and docker_hub_password:
            if credentials is None:
                credentials = {}
            credentials["dockerhub"] = {
                "username": docker_hub_user,
                "password": docker_hub_password,
            }
            if docker_hub_repo:
                credentials["dockerhub"]["repository"] = docker_hub_repo

        return credentials

    def _filter_images_by_gpu_architecture(
        self, built_images: Dict, runtime_gpu_arch: str
    ) -> Dict:
        """Filter images compatible with runtime GPU architecture."""
        compatible_images = {}

        for model_name, image_info in built_images.items():
            image_arch = image_info.get("gpu_architecture", "")

            # Legacy images without architecture - treat as compatible
            if not image_arch:
                compatible_images[model_name] = image_info
                continue

            # Check if architectures match (exact match only for now)
            # Future: support compatibility groups (gfx908/gfx90a are NOT compatible)
            if image_arch == runtime_gpu_arch:
                compatible_images[model_name] = image_info

        return compatible_images

