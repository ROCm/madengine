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
        """Copy common scripts to model directories."""
        common_scripts = Path("scripts/common")
        if not common_scripts.exists():
            return

        print(f"Copying common scripts from {common_scripts}")

        for model_script_dir in Path("scripts").iterdir():
            if model_script_dir.is_dir() and model_script_dir.name != "common":
                dest = model_script_dir / "common"
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(common_scripts, dest)
                print(f"  Copied to {dest}")

    def execute(
        self,
        registry: Optional[str] = None,
        clean_cache: bool = False,
        manifest_output: str = "build_manifest.json",
        batch_build_metadata: Optional[Dict] = None,
    ) -> str:
        """
        Execute build workflow.

        Args:
            registry: Optional registry to push images to
            clean_cache: Whether to use --no-cache for Docker builds
            manifest_output: Output file for build manifest
            batch_build_metadata: Optional batch build metadata

        Returns:
            Path to generated build_manifest.json

        Raises:
            DiscoveryError: If model discovery fails
            BuildError: If Docker build fails
        """
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

            # Step 2: Copy common scripts
            self.rich_console.print("[bold cyan]📋 Copying scripts...[/bold cyan]")
            self._copy_scripts()
            self.rich_console.print("[green]✓ Scripts copied[/green]\n")

            # Step 3: Validate build context
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

            # Step 4: Build Docker images
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

            # Step 5: ALWAYS generate manifest (even with partial failures)
            self.rich_console.print("\n[bold cyan]📄 Generating build manifest...[/bold cyan]")
            builder.export_build_manifest(manifest_output, registry, batch_build_metadata)

            # Step 6: Save build summary to manifest
            self._save_build_summary(manifest_output, build_summary)

            # Step 7: Save deployment_config to manifest
            self._save_deployment_config(manifest_output)

            self.rich_console.print(f"[green]✓ Build complete: {manifest_output}[/green]")
            self.rich_console.print(f"[dim]{'=' * 60}[/dim]\n")

            # Step 8: Check if we should fail (only if ALL builds failed)
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
            
            deployment_config = {
                "target": target,
                "slurm": self.additional_context.get("slurm"),
                "k8s": self.additional_context.get("k8s"),
                "kubernetes": self.additional_context.get("kubernetes"),
                "distributed": self.additional_context.get("distributed"),
                "vllm": self.additional_context.get("vllm"),
                "env_vars": self.additional_context.get("env_vars", {}),
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

