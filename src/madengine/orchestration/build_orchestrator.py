#!/usr/bin/env python3
"""
Build Orchestrator - Coordinates Docker image building workflow.

Extracted from distributed_orchestrator.py build_phase() method.
Manages the discovery, building, and manifest generation for Docker images.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console as RichConsole

from madengine.core.console import Console
from madengine.core.context import Context
from madengine.core.errors import (
    BuildError,
    ConfigurationError,
    DiscoveryError,
    create_error_context,
    handle_error,
)
from madengine.utils.discover_models import DiscoverModels
from madengine.execution.docker_builder import DockerBuilder


class BuildOrchestrator:
    """
    Orchestrates the build workflow.

    Responsibilities:
    - Discover models by tags
    - Build Docker images
    - Push to registry (optional)
    - Generate build_manifest.json
    - Save deployment_config from --additional-context
    """

    def __init__(self, args, additional_context: Optional[Dict] = None):
        """
        Initialize build orchestrator.

        Args:
            args: CLI arguments namespace
            additional_context: Dict from --additional-context (merged with args if present)
        """
        self.args = args
        self.console = Console(live_output=getattr(args, "live_output", True))
        self.rich_console = RichConsole()

        # Merge additional_context from args and parameter
        merged_context = {}
        
        # Load from file first if provided
        if hasattr(args, "additional_context_file") and args.additional_context_file:
            try:
                with open(args.additional_context_file, "r") as f:
                    merged_context = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load additional_context_file: {e}")
        
        # Then merge string additional_context (overrides file)
        if hasattr(args, "additional_context") and args.additional_context:
            try:
                if isinstance(args.additional_context, str):
                    # Use ast.literal_eval for Python dict syntax (single quotes)
                    # This matches what Context class expects
                    import ast
                    context_from_string = ast.literal_eval(args.additional_context)
                    merged_context.update(context_from_string)
                elif isinstance(args.additional_context, dict):
                    merged_context.update(args.additional_context)
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse additional_context: {e}")
                pass

        # Finally merge parameter additional_context (overrides all)
        if additional_context:
            merged_context.update(additional_context)

        self.additional_context = merged_context
        
        # Apply ConfigLoader to infer deploy type, validate, and apply defaults
        if self.additional_context:
            try:
                from madengine.deployment.config_loader import ConfigLoader
                # This will:
                # 1. Infer deploy type from k8s/slurm presence
                # 2. Validate for conflicts (e.g., both k8s and slurm)
                # 3. Apply appropriate defaults
                # 4. Add 'deploy' field for internal use
                self.additional_context = ConfigLoader.load_config(self.additional_context)
            except ValueError as e:
                # Configuration validation error - fail fast
                self.rich_console.print(f"[red]Configuration Error: {e}[/red]")
                raise SystemExit(1)
            except Exception as e:
                # Other errors during config loading - warn but continue
                self.rich_console.print(f"[yellow]Warning: Could not apply config defaults: {e}[/yellow]")

        # Initialize context in build-only mode (no GPU detection)
        # Context expects additional_context as a string representation of Python dict
        # Use repr() instead of json.dumps() because Context uses ast.literal_eval()
        context_string = repr(merged_context) if merged_context else None
        self.context = Context(
            additional_context=context_string,
            build_only_mode=True,
        )

        # Load credentials if available
        self.credentials = self._load_credentials()

    def _load_credentials(self) -> Optional[Dict]:
        """Load credentials from credential.json and environment variables."""
        credentials = None

        # Try loading from file
        credential_file = "credential.json"
        if os.path.exists(credential_file):
            try:
                with open(credential_file) as f:
                    credentials = json.load(f)
                print(f"Loaded credentials from {credential_file}: {list(credentials.keys())}")
            except Exception as e:
                context = create_error_context(
                    operation="load_credentials",
                    component="BuildOrchestrator",
                    file_path=credential_file,
                )
                handle_error(
                    ConfigurationError(
                        f"Could not load credentials: {e}",
                        context=context,
                        suggestions=[
                            "Check if credential.json exists and has valid JSON format"
                        ],
                    )
                )

        # Override with environment variables if present
        docker_hub_user = os.environ.get("MAD_DOCKERHUB_USER")
        docker_hub_password = os.environ.get("MAD_DOCKERHUB_PASSWORD")
        docker_hub_repo = os.environ.get("MAD_DOCKERHUB_REPO")

        if docker_hub_user and docker_hub_password:
            print("Found Docker Hub credentials in environment variables")
            if credentials is None:
                credentials = {}

            credentials["dockerhub"] = {
                "username": docker_hub_user,
                "password": docker_hub_password,
            }
            if docker_hub_repo:
                credentials["dockerhub"]["repository"] = docker_hub_repo

        return credentials

    def _copy_scripts(self):
        """[DEPRECATED] Copy common scripts to model directories.
        
        This method is no longer called during build phase as it's not needed.
        Build phase only creates Docker images - script execution happens in run phase.
        Scripts are copied by run_orchestrator._copy_scripts() for local execution.
        K8s and Slurm deployments have their own script management mechanisms.
        """
        # No-op: This method is deprecated and should not be called
        pass

    def execute(
        self,
        registry: Optional[str] = None,
        clean_cache: bool = False,
        manifest_output: str = "build_manifest.json",
        batch_build_metadata: Optional[Dict] = None,
        use_image: Optional[str] = None,
        build_on_compute: bool = False,
    ) -> str:
        """
        Execute build workflow.

        Args:
            registry: Optional registry to push images to
            clean_cache: Whether to use --no-cache for Docker builds
            manifest_output: Output file for build manifest
            batch_build_metadata: Optional batch build metadata
            use_image: Pre-built Docker image to use (skip Docker build)
            build_on_compute: Build on SLURM compute node instead of login node

        Returns:
            Path to generated build_manifest.json

        Raises:
            DiscoveryError: If model discovery fails
            BuildError: If Docker build fails
        """
        # Handle pre-built image mode
        if use_image:
            return self._execute_with_prebuilt_image(
                use_image=use_image,
                manifest_output=manifest_output,
            )
        
        # Handle build-on-compute mode
        if build_on_compute:
            return self._execute_build_on_compute(
                registry=registry,
                clean_cache=clean_cache,
                manifest_output=manifest_output,
                batch_build_metadata=batch_build_metadata,
            )
        self.rich_console.print(f"\n[dim]{'=' * 60}[/dim]")
        self.rich_console.print("[bold blue]🔨 BUILD PHASE[/bold blue]")
        self.rich_console.print("[yellow](Build-only mode - no GPU detection)[/yellow]")
        self.rich_console.print(f"[dim]{'=' * 60}[/dim]\n")

        try:
            # Step 1: Discover models
            self.rich_console.print("[bold cyan]🔍 Discovering models...[/bold cyan]")
            discover_models = DiscoverModels(args=self.args)
            models = discover_models.run()

            if not models:
                raise DiscoveryError(
                    "No models discovered",
                    context=create_error_context(
                        operation="discover_models",
                        component="BuildOrchestrator",
                    ),
                    suggestions=[
                        "Check if models.json exists",
                        "Verify --tags parameter is correct",
                        "Ensure model definitions have matching tags",
                    ],
                )

            self.rich_console.print(f"[green]✓ Found {len(models)} models[/green]\n")

            # Step 2: Validate build context (scripts not needed for build phase)
            # Build phase only creates Docker images - script execution happens in run phase
            # Note: K8s and Slurm have their own script management mechanisms
            if "MAD_SYSTEM_GPU_ARCHITECTURE" not in self.context.ctx["docker_build_arg"]:
                self.rich_console.print(
                    "[yellow]⚠️  Warning: MAD_SYSTEM_GPU_ARCHITECTURE not provided[/yellow]"
                )
                self.rich_console.print(
                    "[dim]  Provide GPU architecture via --additional-context:[/dim]"
                )
                self.rich_console.print(
                    '[dim]  --additional-context \'{"docker_build_arg": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx90a"}}\'[/dim]\n'
                )

            # Step 3: Build Docker images
            self.rich_console.print("[bold cyan]🏗️  Building Docker images...[/bold cyan]")
            builder = DockerBuilder(
                self.context,
                self.console,
                live_output=getattr(self.args, "live_output", False),
            )

            # Determine phase suffix for log files
            # Build phase always uses .build suffix to avoid conflicts with run logs
            phase_suffix = ".build"

            # Get target architectures from args if provided
            target_archs = getattr(self.args, "target_archs", [])
            if target_archs:
                processed_archs = []
                for arch_arg in target_archs:
                    # Split comma-separated values
                    processed_archs.extend(
                        [arch.strip() for arch in arch_arg.split(",") if arch.strip()]
                    )
                target_archs = processed_archs

            # Build all models (resilient to individual failures)
            build_summary = builder.build_all_models(
                models,
                self.credentials,
                clean_cache,
                registry,
                phase_suffix,
                batch_build_metadata=batch_build_metadata,
                target_archs=target_archs,
            )

            # Extract results
            failed_builds = build_summary.get("failed_builds", [])
            successful_builds = build_summary.get("successful_builds", [])

            # Report build results
            if len(successful_builds) > 0:
                self.rich_console.print(
                    f"\n[green]✓ Built {len(successful_builds)} images[/green]"
                )

            if len(failed_builds) > 0:
                self.rich_console.print(
                    f"[yellow]⚠️  {len(failed_builds)} model(s) failed to build:[/yellow]"
                )
                for failed in failed_builds:
                    model_name = failed.get("model", "unknown")
                    error_msg = failed.get("error", "unknown error")
                    self.rich_console.print(f"  [red]• {model_name}: {error_msg}[/red]")

            # Step 4: ALWAYS generate manifest (even with partial failures)
            self.rich_console.print("\n[bold cyan]📄 Generating build manifest...[/bold cyan]")
            builder.export_build_manifest(manifest_output, registry, batch_build_metadata)

            # Step 5: Save build summary to manifest
            self._save_build_summary(manifest_output, build_summary)

            # Step 6: Save deployment_config to manifest
            self._save_deployment_config(manifest_output)

            self.rich_console.print(f"[green]✓ Build complete: {manifest_output}[/green]")
            self.rich_console.print(f"[dim]{'=' * 60}[/dim]\n")

            # Step 7: Check if we should fail (only if ALL builds failed)
            if len(failed_builds) > 0:
                if len(successful_builds) == 0:
                    # All builds failed - this is critical
                    raise BuildError(
                        "All builds failed - no images available",
                        context=create_error_context(
                            operation="build_images",
                            component="BuildOrchestrator",
                        ),
                        suggestions=[
                            "Check Docker build logs in *.build.live.log files",
                            "Verify Dockerfile syntax",
                            "Ensure base images are accessible",
                        ],
                    )
                else:
                    # Partial success - log warning but don't raise
                    self.rich_console.print(
                        f"[yellow]⚠️  Warning: Partial build - "
                        f"{len(successful_builds)} succeeded, {len(failed_builds)} failed[/yellow]"
                    )

            return manifest_output

        except (DiscoveryError, BuildError):
            raise
        except Exception as e:
            context = create_error_context(
                operation="build_phase",
                component="BuildOrchestrator",
            )
            raise BuildError(
                f"Build phase failed: {e}",
                context=context,
                suggestions=[
                    "Check Docker daemon is running",
                    "Verify network connectivity for image pulls",
                    "Check disk space for Docker builds",
                ],
            ) from e

    def _save_build_summary(self, manifest_file: str, build_summary: Dict):
        """Save build summary to manifest for display purposes."""
        try:
            with open(manifest_file, "r") as f:
                manifest = json.load(f)

            # Add summary to manifest
            manifest["summary"] = build_summary

            with open(manifest_file, "w") as f:
                json.dump(manifest, f, indent=2)

        except Exception as e:
            self.rich_console.print(f"[yellow]Warning: Could not save build summary: {e}[/yellow]")

    def _save_deployment_config(self, manifest_file: str):
        """Save deployment_config from --additional-context to manifest."""
        if not self.additional_context:
            self.rich_console.print("[dim]No additional_context provided, skipping deployment config[/dim]")
            return

        try:
            with open(manifest_file, "r") as f:
                manifest = json.load(f)

            # Extract deployment configuration
            # Auto-detect target from config presence if not explicitly set
            target = self.additional_context.get("deploy")
            if not target:
                # Auto-detect based on config presence
                if self.additional_context.get("slurm"):
                    target = "slurm"
                elif self.additional_context.get("k8s") or self.additional_context.get("kubernetes"):
                    target = "k8s"
                else:
                    target = "local"
            
            # Get env_vars and filter out MIOPEN_USER_DB_PATH
            # This variable must be set per-process in multi-GPU training to avoid database conflicts
            env_vars = self.additional_context.get("env_vars", {}).copy()
            if "MIOPEN_USER_DB_PATH" in env_vars:
                del env_vars["MIOPEN_USER_DB_PATH"]
                print("ℹ️  Filtered MIOPEN_USER_DB_PATH from env_vars (will be set per-process in training)")
            
            deployment_config = {
                "target": target,
                "slurm": self.additional_context.get("slurm"),
                "k8s": self.additional_context.get("k8s"),
                "kubernetes": self.additional_context.get("kubernetes"),
                "distributed": self.additional_context.get("distributed"),
                "vllm": self.additional_context.get("vllm"),
                "env_vars": env_vars,
                "debug": self.additional_context.get("debug", False),
            }

            # Remove None values
            deployment_config = {
                k: v for k, v in deployment_config.items() if v is not None
            }

            if deployment_config and deployment_config != {"target": "local", "env_vars": {}}:
                manifest["deployment_config"] = deployment_config

                with open(manifest_file, "w") as f:
                    json.dump(manifest, f, indent=2)

                self.rich_console.print(f"[green]✓ Saved deployment config to {manifest_file}[/green]")
            else:
                self.rich_console.print("[dim]No deployment config to save (local execution)[/dim]")

        except Exception as e:
            # Non-fatal - just warn
            self.rich_console.print(f"[yellow]Warning: Could not save deployment config: {e}[/yellow]")

    def _execute_with_prebuilt_image(
        self,
        use_image: str,
        manifest_output: str = "build_manifest.json",
    ) -> str:
        """
        Generate manifest for a pre-built Docker image (skip Docker build).
        
        This is useful when using external images like:
        - lmsysorg/sglang:v0.5.2rc1-rocm700-mi30x
        - nvcr.io/nvidia/pytorch:24.01-py3
        
        Args:
            use_image: Pre-built Docker image name
            manifest_output: Output file for build manifest
            
        Returns:
            Path to generated build_manifest.json
        """
        self.rich_console.print(f"\n[dim]{'=' * 60}[/dim]")
        self.rich_console.print("[bold blue]🔨 BUILD PHASE (Pre-built Image Mode)[/bold blue]")
        self.rich_console.print(f"[cyan]Using pre-built image: {use_image}[/cyan]")
        self.rich_console.print(f"[dim]{'=' * 60}[/dim]\n")

        try:
            # Step 1: Discover models
            self.rich_console.print("[bold cyan]🔍 Discovering models...[/bold cyan]")
            discover_models = DiscoverModels(args=self.args)
            models = discover_models.run()

            if not models:
                raise DiscoveryError(
                    "No models discovered",
                    context=create_error_context(
                        operation="discover_models",
                        component="BuildOrchestrator",
                    ),
                    suggestions=[
                        "Check if models.json exists",
                        "Verify --tags parameter is correct",
                    ],
                )

            self.rich_console.print(f"[green]✓ Found {len(models)} models[/green]\n")

            # Step 2: Generate manifest with pre-built image
            self.rich_console.print("[bold cyan]📄 Generating manifest for pre-built image...[/bold cyan]")
            
            manifest = {
                "built_images": {
                    use_image: {
                        "image_name": use_image,
                        "docker_image": use_image,
                        "dockerfile": "",
                        "build_time": 0,
                        "prebuilt": True,
                    }
                },
                "built_models": {},
                "context": self.context.ctx if hasattr(self.context, 'ctx') else {},
                "credentials_required": [],
                "summary": {
                    "successful_builds": [],
                    "failed_builds": [],
                    "total_build_time": 0,
                    "successful_pushes": [],
                    "failed_pushes": [],
                },
            }

            # Add each discovered model with the pre-built image
            # Use the image name as the key (matches how madengine build does it)
            for model in models:
                model_name = model.get("name", "unknown")
                model_distributed = model.get("distributed", {})
                
                # Use image name as key so slurm.py can find docker_image
                manifest["built_models"][use_image] = {
                    "name": model_name,
                    "image": use_image,
                    "docker_image": use_image,
                    "dockerfile": model.get("dockerfile", ""),
                    "scripts": model.get("scripts", ""),
                    "data": model.get("data", ""),
                    "n_gpus": model.get("n_gpus", "8"),
                    "owner": model.get("owner", ""),
                    "training_precision": model.get("training_precision", ""),
                    "multiple_results": model.get("multiple_results", ""),
                    "tags": model.get("tags", []),
                    "timeout": model.get("timeout", -1),
                    "args": model.get("args", ""),
                    "slurm": model.get("slurm", {}),
                    "distributed": model_distributed,
                    "env_vars": model.get("env_vars", {}),
                    "prebuilt": True,
                }
                manifest["summary"]["successful_builds"].append(model_name)

            # Save manifest
            with open(manifest_output, "w") as f:
                json.dump(manifest, f, indent=2)

            # Save deployment config
            self._save_deployment_config(manifest_output)
            
            # Merge model's distributed config (especially launcher) into deployment_config
            # This ensures sglang-disagg launcher is in deployment_config even if not in additional-context
            if models and models[0].get("distributed"):
                with open(manifest_output, "r") as f:
                    saved_manifest = json.load(f)
                
                model_distributed = models[0].get("distributed", {})
                if "deployment_config" not in saved_manifest:
                    saved_manifest["deployment_config"] = {}
                
                # Merge model's distributed into deployment_config.distributed
                if "distributed" not in saved_manifest["deployment_config"]:
                    saved_manifest["deployment_config"]["distributed"] = {}
                
                # Copy launcher and other critical fields from model config
                for key in ["launcher", "nnodes", "nproc_per_node", "backend", "port", "sglang_disagg"]:
                    if key in model_distributed and key not in saved_manifest["deployment_config"]["distributed"]:
                        saved_manifest["deployment_config"]["distributed"][key] = model_distributed[key]
                
                with open(manifest_output, "w") as f:
                    json.dump(saved_manifest, f, indent=2)

            self.rich_console.print(f"[green]✓ Generated manifest: {manifest_output}[/green]")
            self.rich_console.print(f"  Pre-built image: {use_image}")
            self.rich_console.print(f"  Models: {len(models)}")
            self.rich_console.print(f"[dim]{'=' * 60}[/dim]\n")

            return manifest_output

        except (DiscoveryError, BuildError):
            raise
        except Exception as e:
            raise BuildError(
                f"Failed to generate manifest for pre-built image: {e}",
                context=create_error_context(
                    operation="prebuilt_manifest",
                    component="BuildOrchestrator",
                ),
            ) from e

    def _execute_build_on_compute(
        self,
        registry: Optional[str] = None,
        clean_cache: bool = False,
        manifest_output: str = "build_manifest.json",
        batch_build_metadata: Optional[Dict] = None,
    ) -> str:
        """
        Execute Docker build on a SLURM compute node instead of login node.
        
        This submits a SLURM job that runs the Docker build on a compute node,
        which is useful when:
        - Login node has limited disk space
        - Login node shouldn't run heavy workloads
        - Compute nodes have faster storage/network
        
        Args:
            registry: Optional registry to push images to
            clean_cache: Whether to use --no-cache for Docker builds
            manifest_output: Output file for build manifest
            batch_build_metadata: Optional batch build metadata
            
        Returns:
            Path to generated build_manifest.json
        """
        import subprocess
        import os
        
        self.rich_console.print(f"\n[dim]{'=' * 60}[/dim]")
        self.rich_console.print("[bold blue]🔨 BUILD PHASE (Compute Node Mode)[/bold blue]")
        self.rich_console.print("[cyan]Building on SLURM compute node...[/cyan]")
        self.rich_console.print(f"[dim]{'=' * 60}[/dim]\n")

        # Check if we're inside an existing allocation
        inside_allocation = os.environ.get("SLURM_JOB_ID") is not None
        existing_job_id = os.environ.get("SLURM_JOB_ID", "")

        # Get SLURM config from additional_context
        slurm_config = self.additional_context.get("slurm", {})
        partition = slurm_config.get("partition", "gpu")
        reservation = slurm_config.get("reservation", "")
        time_limit = slurm_config.get("time", "02:00:00")
        # Get number of nodes - build on ALL nodes so image is available everywhere
        nodes = slurm_config.get("nodes", 1)

        # Build the madengine build command (without --build-on-compute to avoid recursion)
        tags = getattr(self.args, 'tags', [])
        tags_str = " ".join([f"-t {tag}" for tag in tags]) if tags else ""
        
        # Write additional context to a file to avoid shell quoting issues
        context_file_path = None
        additional_context_str = ""
        if self.additional_context:
            import json
            context_file_path = Path("madengine_build_context.json")
            with open(context_file_path, 'w') as f:
                json.dump(self.additional_context, f)
            self.rich_console.print(f"  Context file: {context_file_path}")

        # Base build command
        build_cmd_parts = ["madengine", "build"]
        if tags_str:
            build_cmd_parts.extend(tags_str.split())
        if context_file_path:
            build_cmd_parts.extend(["--additional-context-file", str(context_file_path)])
        build_cmd_parts.extend(["--manifest-output", manifest_output])
        if registry:
            build_cmd_parts.extend(["--registry", registry])
        if clean_cache:
            build_cmd_parts.append("--clean-docker-cache")
        
        build_cmd = " ".join(build_cmd_parts)

        if inside_allocation:
            # Run build on compute node via srun
            self.rich_console.print(f"[cyan]Running build via srun (inside allocation {existing_job_id})...[/cyan]")
            cmd = ["srun", "-N1", "--ntasks=1", "bash", "-c", build_cmd]
        else:
            # Generate and submit build script
            self.rich_console.print("[cyan]Submitting build job via sbatch...[/cyan]")
            
            # Get absolute path for context file
            abs_context_file = str(context_file_path.absolute()) if context_file_path else ""
            abs_manifest_output = str(Path(manifest_output).absolute())
            
            # Rebuild command with absolute paths for sbatch
            build_cmd_abs = f"madengine build {tags_str}"
            if abs_context_file:
                build_cmd_abs += f" --additional-context-file {abs_context_file}"
            build_cmd_abs += f" --manifest-output {abs_manifest_output}"
            if registry:
                build_cmd_abs += f" --registry {registry}"
            if clean_cache:
                build_cmd_abs += " --clean-docker-cache"
            
            # Discover models to get Dockerfile path
            discover_models = DiscoverModels(args=self.args)
            models = discover_models.run()
            dockerfile_path = ""
            dockerfile_name = ""
            if models:
                dockerfile = models[0].get("dockerfile", "")
                # Find the actual Dockerfile
                import glob
                dockerfile_patterns = [
                    f"{dockerfile}.ubuntu.amd.Dockerfile",
                    f"{dockerfile}.Dockerfile",
                    f"{dockerfile}",
                ]
                for pattern in dockerfile_patterns:
                    matches = glob.glob(pattern)
                    if matches:
                        dockerfile_path = matches[0]
                        dockerfile_name = Path(dockerfile_path).name
                        break
            
            self.rich_console.print(f"  Nodes: {nodes} (building on all nodes)")
            if dockerfile_path:
                self.rich_console.print(f"  Dockerfile: {dockerfile_path}")
            
            build_script_content = f"""#!/bin/bash
#SBATCH --job-name=madengine-build
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={nodes}
#SBATCH --time={time_limit}
{f'#SBATCH --reservation={reservation}' if reservation else ''}
#SBATCH --output=madengine_build_%j.out
#SBATCH --error=madengine_build_%j.err

echo "=== Building on compute nodes ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Node list: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo ""

# Change to submission directory
cd {Path.cwd().absolute()}

# Activate virtual environment if available
if [ -f "{Path('/shared_inference/ravgupta/madenginev2_slurm/venv/bin/activate').absolute()}" ]; then
    source {Path('/shared_inference/ravgupta/madenginev2_slurm/venv/bin/activate').absolute()}
    echo "Activated virtual environment"
fi# Step 1: Build Docker image on ALL nodes in parallel
echo ""
echo "=== Building Docker image on all $SLURM_NNODES nodes ==="
DOCKERFILE="{dockerfile_path}"
if [ -n "$DOCKERFILE" ] && [ -f "$DOCKERFILE" ]; then
    # Get the image name - must match exactly what madengine generates
    # Format: ci-<model_name>_<dockerfile_basename_without_.Dockerfile>
    IMAGE_NAME=$(basename $DOCKERFILE .Dockerfile)
    FULL_IMAGE_NAME="ci-{models[0].get('name', 'model') if models else 'model'}_$IMAGE_NAME"
    
    echo "Dockerfile: $DOCKERFILE"
    echo "Image name: $FULL_IMAGE_NAME"
    
    # Build on all nodes in parallel using srun
    srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES bash -c "
        echo \\\"[\\$(hostname)] Building Docker image...\\\"
        cd {Path.cwd().absolute()}
        docker build --network=host -t $FULL_IMAGE_NAME --pull -f $DOCKERFILE ./docker
        BUILD_RC=\\$?
        if [ \\$BUILD_RC -eq 0 ]; then
            echo \\\"[\\$(hostname)] Docker build SUCCESS\\\"
        else
            echo \\\"[\\$(hostname)] Docker build FAILED with exit code \\$BUILD_RC\\\"
        fi
        exit \\$BUILD_RC
    "
    DOCKER_BUILD_EXIT=$?
    
    if [ $DOCKER_BUILD_EXIT -ne 0 ]; then
        echo "Docker build failed on one or more nodes"
        exit $DOCKER_BUILD_EXIT
    fi
    echo ""
    echo "=== Docker image built on all nodes ==="
fi

# Step 2: Run madengine build on rank 0 to generate manifest
echo ""
echo "=== Generating build manifest ==="
echo "Build command: {build_cmd_abs}"
echo ""

{build_cmd_abs}
BUILD_EXIT=$?

echo ""
echo "=== Build completed with exit code: $BUILD_EXIT ==="
exit $BUILD_EXIT
"""
            build_script_path = Path("madengine_build_job.sh")
            build_script_path.write_text(build_script_content)
            build_script_path.chmod(0o755)
            
            self.rich_console.print(f"  Build script: {build_script_path}")
            cmd = ["sbatch", "--wait", str(build_script_path)]

        # Execute the build
        self.rich_console.print(f"  Command: {' '.join(cmd)}")
        self.rich_console.print("")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=False,  # Let output flow to console
                text=True,
            )
            
            if result.returncode != 0:
                raise BuildError(
                    f"Build on compute node failed with exit code {result.returncode}",
                    context=create_error_context(
                        operation="build_on_compute",
                        component="BuildOrchestrator",
                    ),
                    suggestions=[
                        "Check the build log files (madengine_build_*.out/err)",
                        "Verify SLURM partition and reservation settings",
                        "Ensure Docker is available on compute nodes",
                    ],
                )
            
            self.rich_console.print(f"[green]✓ Build completed on compute node[/green]")
            self.rich_console.print(f"[green]✓ Manifest: {manifest_output}[/green]")
            return manifest_output
            
        except subprocess.TimeoutExpired:
            raise BuildError(
                "Build on compute node timed out",
                context=create_error_context(
                    operation="build_on_compute",
                    component="BuildOrchestrator",
                ),
            )
        except Exception as e:
            raise BuildError(
                f"Failed to build on compute node: {e}",
                context=create_error_context(
                    operation="build_on_compute",
                    component="BuildOrchestrator",
                ),
            ) from e