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
            # If use_image is "auto", resolve from model card
            if use_image == "auto":
                use_image = self._resolve_image_from_model_card()
            
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
        
        # For normal build: check if slurm_multi launcher requires registry
        # Discover models first to check launcher
        discover_models = DiscoverModels(args=self.args)
        discovered_models = discover_models.run()
        
        if discovered_models:
            for model in discovered_models:
                launcher = model.get("distributed", {}).get("launcher", "")
                if launcher in ["slurm_multi", "slurm-multi"] and not registry:
                    model_name = model.get("name", "unknown")
                    raise ConfigurationError(
                        f"slurm_multi launcher requires --registry or --use-image",
                        context=create_error_context(
                            operation="build",
                            component="BuildOrchestrator",
                            model=model_name,
                            launcher=launcher,
                        ),
                        suggestions=[
                            "Use --registry docker.io/myorg to push image (nodes will pull in parallel)",
                            "Use --use-image to use a pre-built image from registry",
                            "Use --build-on-compute --registry to build on compute and push",
                            "For subsequent runs with same image, use: --use-image",
                        ],
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
                
                # Merge DOCKER_IMAGE_NAME into env_vars for parallel pull in run phase
                model_env_vars = model.get("env_vars", {}).copy()
                model_env_vars["DOCKER_IMAGE_NAME"] = use_image
                
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
                    "env_vars": model_env_vars,
                    "prebuilt": True,
                }
                manifest["summary"]["successful_builds"].append(model_name)

            # Save manifest
            with open(manifest_output, "w") as f:
                json.dump(manifest, f, indent=2)

            # Save deployment config
            self._save_deployment_config(manifest_output)
            
            # Merge model's distributed and slurm config into deployment_config
            # This ensures launcher and slurm settings are in deployment_config even if not in additional-context
            if models:
                with open(manifest_output, "r") as f:
                    saved_manifest = json.load(f)
                
                if "deployment_config" not in saved_manifest:
                    saved_manifest["deployment_config"] = {}
                
                # Merge model's distributed config
                model_distributed = models[0].get("distributed", {})
                if model_distributed:
                    if "distributed" not in saved_manifest["deployment_config"]:
                        saved_manifest["deployment_config"]["distributed"] = {}
                    
                    # Copy launcher and other critical fields from model config
                    for key in ["launcher", "nnodes", "nproc_per_node", "backend", "port", "sglang_disagg", "vllm_disagg"]:
                        if key in model_distributed and key not in saved_manifest["deployment_config"]["distributed"]:
                            saved_manifest["deployment_config"]["distributed"][key] = model_distributed[key]
                
                # Merge model's slurm config into deployment_config.slurm
                # This enables run phase to auto-detect SLURM deployment without --additional-context
                model_slurm = models[0].get("slurm", {})
                if model_slurm:
                    if "slurm" not in saved_manifest["deployment_config"]:
                        saved_manifest["deployment_config"]["slurm"] = {}
                    
                    # Copy slurm settings from model config
                    for key in ["partition", "nodes", "gpus_per_node", "time", "exclusive", "reservation", "output_dir", "nodelist"]:
                        if key in model_slurm and key not in saved_manifest["deployment_config"]["slurm"]:
                            saved_manifest["deployment_config"]["slurm"][key] = model_slurm[key]
                
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

    def _resolve_image_from_model_card(self) -> str:
        """
        Resolve Docker image name from model card's DOCKER_IMAGE_NAME env var.
        
        This method discovers models and extracts the DOCKER_IMAGE_NAME from
        env_vars. If multiple models have different images, uses the first
        and prints a warning.
        
        Returns:
            Docker image name from model card
            
        Raises:
            ConfigurationError: If no DOCKER_IMAGE_NAME found in any model
        """
        self.rich_console.print("[bold cyan]🔍 Auto-detecting image from model card...[/bold cyan]")
        
        # Discover models to get their env_vars
        discover_models = DiscoverModels(args=self.args)
        models = discover_models.run()
        
        if not models:
            raise ConfigurationError(
                "No models discovered for image auto-detection",
                context=create_error_context(
                    operation="resolve_image",
                    component="BuildOrchestrator",
                ),
                suggestions=[
                    "Specify image name explicitly with --use-image <image>",
                    "Check if models.json exists",
                    "Verify --tags parameter is correct",
                ],
            )
        
        # Collect DOCKER_IMAGE_NAME from all models
        images_found = {}
        for model in models:
            model_name = model.get("name", "unknown")
            env_vars = model.get("env_vars", {})
            docker_image = env_vars.get("DOCKER_IMAGE_NAME")
            
            if docker_image:
                images_found[model_name] = docker_image
        
        if not images_found:
            model_names = [m.get("name", "unknown") for m in models]
            raise ConfigurationError(
                "No DOCKER_IMAGE_NAME found in model card env_vars",
                context=create_error_context(
                    operation="resolve_image",
                    component="BuildOrchestrator",
                    model_names=model_names,
                ),
                suggestions=[
                    "Add DOCKER_IMAGE_NAME to model's env_vars in models.json",
                    "Specify image name explicitly with --use-image <image>",
                    'Example: "env_vars": {"DOCKER_IMAGE_NAME": "myimage:tag"}',
                ],
            )
        
        # Use first model's image
        first_model = list(images_found.keys())[0]
        resolved_image = images_found[first_model]
        
        # Warn if multiple models have different images
        unique_images = set(images_found.values())
        if len(unique_images) > 1:
            self.rich_console.print(
                f"[yellow]⚠️  Warning: Multiple models have different DOCKER_IMAGE_NAME values:[/yellow]"
            )
            for model_name, image in images_found.items():
                self.rich_console.print(f"   - {model_name}: {image}")
            self.rich_console.print(
                f"[yellow]   Using image from '{first_model}': {resolved_image}[/yellow]\n"
            )
        else:
            self.rich_console.print(f"[green]✓ Auto-detected image: {resolved_image}[/green]\n")
        
        return resolved_image

    def _execute_build_on_compute(
        self,
        registry: Optional[str] = None,
        clean_cache: bool = False,
        manifest_output: str = "build_manifest.json",
        batch_build_metadata: Optional[Dict] = None,
    ) -> str:
        """
        Execute Docker build on a SLURM compute node and push to registry.
        
        Build workflow:
        1. Build on 1 compute node only
        2. Push image to registry
        3. Store registry image name in manifest
        4. Run phase will pull image in parallel on all nodes
        
        Args:
            registry: Registry to push images to (REQUIRED)
            clean_cache: Whether to use --no-cache for Docker builds
            manifest_output: Output file for build manifest
            batch_build_metadata: Optional batch build metadata
            
        Returns:
            Path to generated build_manifest.json
        """
        import subprocess
        import os
        import glob
        
        self.rich_console.print(f"\n[dim]{'=' * 60}[/dim]")
        self.rich_console.print("[bold blue]🔨 BUILD PHASE (Compute Node Mode)[/bold blue]")
        self.rich_console.print("[cyan]Building on 1 compute node, pushing to registry...[/cyan]")
        self.rich_console.print(f"[dim]{'=' * 60}[/dim]\n")

        # Discover models first to get SLURM config from model card
        self.rich_console.print("[bold cyan]🔍 Discovering models...[/bold cyan]")
        discover_models = DiscoverModels(args=self.args)
        models = discover_models.run()
        
        if not models:
            raise DiscoveryError(
                "No models discovered for build-on-compute",
                context=create_error_context(
                    operation="build_on_compute",
                    component="BuildOrchestrator",
                ),
                suggestions=[
                    "Check if models.json exists",
                    "Verify --tags parameter is correct",
                ],
            )
        
        model = models[0]
        model_name = model.get("name", "unknown")
        self.rich_console.print(f"[green]✓ Found model: {model_name}[/green]\n")
        
        # Merge SLURM config: model card (base) + additional-context (override)
        model_slurm_config = model.get("slurm", {})
        context_slurm_config = self.additional_context.get("slurm", {})
        
        # Start with model card config, then override with command-line context
        slurm_config = {**model_slurm_config, **context_slurm_config}
        
        self.rich_console.print("[bold cyan]📋 SLURM Configuration (merged):[/bold cyan]")
        if model_slurm_config:
            self.rich_console.print(f"  [dim]From model card:[/dim] {list(model_slurm_config.keys())}")
        if context_slurm_config:
            self.rich_console.print(f"  [dim]From --additional-context (overrides):[/dim] {list(context_slurm_config.keys())}")
        
        # Validate required fields
        partition = slurm_config.get("partition")
        if not partition:
            raise ConfigurationError(
                "Missing required SLURM field: partition",
                context=create_error_context(
                    operation="build_on_compute",
                    component="BuildOrchestrator",
                ),
                suggestions=[
                    'Add "partition" to model card\'s slurm section',
                    'Or specify via --additional-context \'{"slurm": {"partition": "gpu"}}\'',
                ],
            )
        
        reservation = slurm_config.get("reservation", "")
        time_limit = slurm_config.get("time", "02:00:00")
        
        self.rich_console.print(f"  Partition: {partition}")
        self.rich_console.print(f"  Time limit: {time_limit}")
        if reservation:
            self.rich_console.print(f"  Reservation: {reservation}")
        self.rich_console.print("")
        
        # Validate registry credentials
        self.rich_console.print("[bold cyan]🔐 Registry Configuration:[/bold cyan]")
        self.rich_console.print(f"  Registry: {registry}")
        
        # Check for credentials - either from environment or credential.json
        dockerhub_user = os.environ.get("MAD_DOCKERHUB_USER", "")
        dockerhub_password = os.environ.get("MAD_DOCKERHUB_PASSWORD", "")
        
        # Try to load from credential.json if env vars not set
        credential_file = Path("credential.json")
        if not dockerhub_user and credential_file.exists():
            try:
                with open(credential_file) as f:
                    creds = json.load(f)
                    dockerhub_creds = creds.get("dockerhub", {})
                    dockerhub_user = dockerhub_creds.get("username", "")
                    dockerhub_password = dockerhub_creds.get("password", "")
                    if dockerhub_user:
                        self.rich_console.print(f"  Credentials: Found in credential.json")
            except (json.JSONDecodeError, IOError) as e:
                self.rich_console.print(f"  [yellow]Warning: Could not read credential.json: {e}[/yellow]")
        elif dockerhub_user:
            self.rich_console.print(f"  Credentials: Found in environment (MAD_DOCKERHUB_USER)")
        
        # Determine if registry requires authentication
        requires_auth = True
        public_registries = ["docker.io", "ghcr.io", "gcr.io", "quay.io", "nvcr.io"]
        registry_lower = registry.lower() if registry else ""
        
        # For docker.io pushes, authentication is always required
        if any(pub_reg in registry_lower for pub_reg in public_registries):
            if not dockerhub_user or not dockerhub_password:
                raise ConfigurationError(
                    f"Registry credentials required for pushing to {registry}",
                    context=create_error_context(
                        operation="build_on_compute",
                        component="BuildOrchestrator",
                        registry=registry,
                    ),
                    suggestions=[
                        "Set environment variables: MAD_DOCKERHUB_USER and MAD_DOCKERHUB_PASSWORD",
                        'Or create credential.json: {"dockerhub": {"username": "...", "password": "..."}}',
                        "For Docker Hub, use a Personal Access Token (PAT) as password",
                        f"Example: export MAD_DOCKERHUB_USER=myuser",
                        f"Example: export MAD_DOCKERHUB_PASSWORD=dckr_pat_xxxxx",
                    ],
                )
            self.rich_console.print(f"  Auth: Will login to registry before push")
        else:
            # Private/internal registry - may not need auth
            self.rich_console.print(f"  Auth: Private registry (auth may not be required)")
            requires_auth = dockerhub_user and dockerhub_password
        
        self.rich_console.print("")
        
        # Check if we're inside an existing allocation
        inside_allocation = os.environ.get("SLURM_JOB_ID") is not None
        existing_job_id = os.environ.get("SLURM_JOB_ID", "")
        
        # Find Dockerfile
        dockerfile = model.get("dockerfile", "")
        dockerfile_path = ""
        dockerfile_patterns = [
            f"{dockerfile}.ubuntu.amd.Dockerfile",
            f"{dockerfile}.Dockerfile",
            f"{dockerfile}",
        ]
        for pattern in dockerfile_patterns:
            matches = glob.glob(pattern)
            if matches:
                dockerfile_path = matches[0]
                break
        
        if not dockerfile_path:
            raise ConfigurationError(
                f"Dockerfile not found for model {model_name}",
                context=create_error_context(
                    operation="build_on_compute",
                    component="BuildOrchestrator",
                    dockerfile=dockerfile,
                ),
                suggestions=[
                    f"Check if {dockerfile}.ubuntu.amd.Dockerfile exists",
                    "Verify the dockerfile path in models.json",
                ],
            )
        
        # Generate image name for registry
        dockerfile_basename = Path(dockerfile_path).name.replace(".Dockerfile", "").replace(".ubuntu.amd", "")
        local_image_name = f"ci-{model_name}_{dockerfile_basename}"
        
        # Determine registry image name based on registry format
        # docker.io/namespace/repo -> use model name as tag: docker.io/namespace/repo:model_name
        # docker.io/namespace -> use model name as repo: docker.io/namespace/model_name:latest
        registry_parts = registry.replace("docker.io/", "").split("/")
        if len(registry_parts) >= 2:
            # Registry already includes repo name (e.g., rocm/pytorch-private)
            # Use model name as tag
            registry_image_name = f"{registry}:{model_name}"
            self.rich_console.print(f"  [dim]Registry format: namespace/repo -> using model name as tag[/dim]")
        else:
            # Registry is just namespace (e.g., myuser)
            # Use model name as repo
            registry_image_name = f"{registry}/{model_name}:latest"
            self.rich_console.print(f"  [dim]Registry format: namespace -> using model name as repo[/dim]")
        
        self.rich_console.print("[bold cyan]🐳 Docker Configuration:[/bold cyan]")
        self.rich_console.print(f"  Dockerfile: {dockerfile_path}")
        self.rich_console.print(f"  Local image: {local_image_name}")
        self.rich_console.print(f"  Registry image: {registry_image_name}")
        self.rich_console.print("")
        
        # Determine registry host for docker login
        registry_host = registry.split("/")[0] if "/" in registry else registry
        
        # Build script content - builds on 1 node, pushes to registry
        build_script_content = f"""#!/bin/bash
#SBATCH --job-name=madengine-build
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time={time_limit}
{f'#SBATCH --reservation={reservation}' if reservation else ''}
#SBATCH --output=madengine_build_%j.out
#SBATCH --error=madengine_build_%j.err

echo "============================================================"
echo "=== MADENGINE BUILD ON COMPUTE NODE ==="
echo "============================================================"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Build Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Registry: {registry}"
echo ""

# Change to submission directory
cd {Path.cwd().absolute()}

# Step 0: Docker login for registry push
echo "=== Step 0: Docker Registry Authentication ==="
DOCKER_USER="${{MAD_DOCKERHUB_USER:-}}"
DOCKER_PASS="${{MAD_DOCKERHUB_PASSWORD:-}}"

# Try credential.json if env vars not set
if [ -z "$DOCKER_USER" ] && [ -f "credential.json" ]; then
    echo "Reading credentials from credential.json..."
    DOCKER_USER=$(python3 -c "import json; print(json.load(open('credential.json')).get('dockerhub', {{}}).get('username', ''))" 2>/dev/null || echo "")
    DOCKER_PASS=$(python3 -c "import json; print(json.load(open('credential.json')).get('dockerhub', {{}}).get('password', ''))" 2>/dev/null || echo "")
fi

if [ -n "$DOCKER_USER" ] && [ -n "$DOCKER_PASS" ]; then
    echo "Logging in to registry as $DOCKER_USER..."
    echo "$DOCKER_PASS" | docker login {registry_host} -u "$DOCKER_USER" --password-stdin
    LOGIN_RC=$?
    if [ $LOGIN_RC -ne 0 ]; then
        echo ""
        echo "❌ Docker login FAILED with exit code $LOGIN_RC"
        echo ""
        echo "Troubleshooting:"
        echo "  - Verify MAD_DOCKERHUB_USER and MAD_DOCKERHUB_PASSWORD are correct"
        echo "  - For Docker Hub, use a Personal Access Token (PAT) not your password"
        echo "  - Check if the registry URL is correct: {registry_host}"
        exit $LOGIN_RC
    fi
    echo "✅ Docker login SUCCESS"
else
    echo "No credentials found - assuming public registry or pre-authenticated"
fi
echo ""

# Step 1: Build Docker image
echo ""
echo "=== Step 1: Building Docker image ==="
echo "Dockerfile: {dockerfile_path}"
echo "Local image name: {local_image_name}"
echo ""

docker build --network=host -t {local_image_name} {"--no-cache" if clean_cache else ""} --pull -f {dockerfile_path} ./docker
BUILD_RC=$?

if [ $BUILD_RC -ne 0 ]; then
    echo ""
    echo "❌ Docker build FAILED on $(hostname) with exit code $BUILD_RC"
    exit $BUILD_RC
fi

echo ""
echo "✅ Docker build SUCCESS on $(hostname)"
echo ""

# Step 2: Tag and push to registry
echo "=== Step 2: Pushing to registry ==="
echo "Tagging: {local_image_name} -> {registry_image_name}"
docker tag {local_image_name} {registry_image_name}

echo "Pushing: {registry_image_name}"
docker push {registry_image_name}
PUSH_RC=$?

if [ $PUSH_RC -ne 0 ]; then
    echo ""
    echo "❌ Docker push FAILED with exit code $PUSH_RC"
    echo ""
    echo "Troubleshooting:"
    echo "  - Check if you have push access to {registry}"
    echo "  - Verify credentials are correct (MAD_DOCKERHUB_USER, MAD_DOCKERHUB_PASSWORD)"
    echo "  - For Docker Hub, ensure the repository exists or you have create permissions"
    exit $PUSH_RC
fi

echo ""
echo "============================================================"
echo "✅ BUILD AND PUSH COMPLETE"
echo "============================================================"
echo ""
echo "Build Node: $(hostname)"
echo "Registry Image: {registry_image_name}"
echo ""
echo "Run phase will pull this image in parallel on all nodes."
echo "============================================================"

exit 0
"""
        
        build_script_path = Path("madengine_build_job.sh")
        build_script_path.write_text(build_script_content)
        build_script_path.chmod(0o755)
        
        if inside_allocation:
            self.rich_console.print(f"[cyan]Running build via srun (inside allocation {existing_job_id})...[/cyan]")
            cmd = ["srun", "-N1", "--ntasks=1", "bash", str(build_script_path)]
        else:
            self.rich_console.print("[cyan]Submitting build job via sbatch...[/cyan]")
            cmd = ["sbatch", "--wait", str(build_script_path)]
        
        self.rich_console.print(f"  Build script: {build_script_path}")
        self.rich_console.print(f"  Command: {' '.join(cmd)}")
        self.rich_console.print("")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=False,
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
                        "Verify registry credentials are configured",
                    ],
                )
            
            # Generate manifest with registry image name
            self.rich_console.print(f"\n[bold cyan]📄 Generating manifest...[/bold cyan]")
            
            manifest = {
                "built_images": {
                    registry_image_name: {
                        "image_name": registry_image_name,
                        "docker_image": registry_image_name,
                        "local_image": local_image_name,
                        "dockerfile": dockerfile_path,
                        "build_time": 0,
                        "built_on_compute": True,
                        "registry": registry,
                    }
                },
                "built_models": {
                    registry_image_name: {
                        "name": model_name,
                        "image": registry_image_name,
                        "docker_image": registry_image_name,
                        "dockerfile": dockerfile_path,
                        "scripts": model.get("scripts", ""),
                        "data": model.get("data", ""),
                        "n_gpus": model.get("n_gpus", "8"),
                        "tags": model.get("tags", []),
                        "slurm": slurm_config,
                        "distributed": model.get("distributed", {}),
                        "env_vars": {**model.get("env_vars", {}), "DOCKER_IMAGE_NAME": registry_image_name},
                        "built_on_compute": True,
                    }
                },
                "context": self.context.ctx if hasattr(self.context, 'ctx') else {},
                "deployment_config": {
                    "slurm": slurm_config,
                    "distributed": model.get("distributed", {}),
                },
                "credentials_required": [],
                "summary": {
                    "successful_builds": [model_name],
                    "failed_builds": [],
                    "total_build_time": 0,
                    "successful_pushes": [registry_image_name],
                    "failed_pushes": [],
                },
            }
            
            with open(manifest_output, "w") as f:
                json.dump(manifest, f, indent=2)
            
            self.rich_console.print(f"[green]✓ Build completed on compute node[/green]")
            self.rich_console.print(f"[green]✓ Image pushed: {registry_image_name}[/green]")
            self.rich_console.print(f"[green]✓ Manifest: {manifest_output}[/green]")
            self.rich_console.print(f"[dim]{'=' * 60}[/dim]\n")
            
            return manifest_output
            
        except subprocess.TimeoutExpired:
            raise BuildError(
                "Build on compute node timed out",
                context=create_error_context(
                    operation="build_on_compute",
                    component="BuildOrchestrator",
                ),
            )
        except (DiscoveryError, ConfigurationError, BuildError):
            raise
        except Exception as e:
            raise BuildError(
                f"Failed to build on compute node: {e}",
                context=create_error_context(
                    operation="build_on_compute",
                    component="BuildOrchestrator",
                ),
            ) from e