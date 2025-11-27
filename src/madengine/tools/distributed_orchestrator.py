#!/usr/bin/env python3
"""
Distributed Runner Orchestrator for MADEngine

This module provides orchestration capabilities for distributed execution
scenarios like Ansible or Kubernetes, where Docker image building and
container execution are separated across different nodes.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import json
import typing
from rich.console import Console as RichConsole
from madengine.core.console import Console
from madengine.core.context import Context
from madengine.core.dataprovider import Data
from madengine.core.errors import (
    handle_error, create_error_context, ConfigurationError, 
    BuildError, DiscoveryError, RuntimeError as MADRuntimeError
)
from madengine.tools.discover_models import DiscoverModels
from madengine.tools.docker_builder import DockerBuilder
from madengine.tools.container_runner import ContainerRunner


class DistributedOrchestrator:
    """Orchestrator for distributed MADEngine workflows."""

    def __init__(self, args, build_only_mode: bool = False):
        """Initialize the distributed orchestrator.

        Args:
            args: Command-line arguments
            build_only_mode: Whether running in build-only mode (no GPU detection)
        """
        self.args = args
        self.console = Console(live_output=getattr(args, "live_output", True))
        self.rich_console = RichConsole()

        # Initialize context with appropriate mode
        self.context = Context(
            additional_context=getattr(args, "additional_context", None),
            additional_context_file=getattr(args, "additional_context_file", None),
            build_only_mode=build_only_mode,
        )

        # Initialize data provider if data config exists
        data_json_file = getattr(args, "data_config_file_name", "data.json")
        if os.path.exists(data_json_file):
            self.data = Data(
                self.context,
                filename=data_json_file,
                force_mirrorlocal=getattr(args, "force_mirror_local", False),
            )
        else:
            self.data = None

        # Load credentials
        self.credentials = None
        try:
            credential_file = "credential.json"
            if os.path.exists(credential_file):
                with open(credential_file) as f:
                    self.credentials = json.load(f)
                print(f"Credentials: {list(self.credentials.keys())}")
        except Exception as e:
            context = create_error_context(
                operation="load_credentials", 
                component="DistributedOrchestrator",
                file_path=credential_file
            )
            handle_error(
                ConfigurationError(
                    f"Could not load credentials: {e}",
                    context=context,
                    suggestions=["Check if credential.json exists and has valid JSON format"]
                )
            )

        # Check for Docker Hub environment variables and override credentials
        docker_hub_user = None
        docker_hub_password = None
        docker_hub_repo = None

        if "MAD_DOCKERHUB_USER" in os.environ:
            docker_hub_user = os.environ["MAD_DOCKERHUB_USER"]
        if "MAD_DOCKERHUB_PASSWORD" in os.environ:
            docker_hub_password = os.environ["MAD_DOCKERHUB_PASSWORD"]
        if "MAD_DOCKERHUB_REPO" in os.environ:
            docker_hub_repo = os.environ["MAD_DOCKERHUB_REPO"]

        if docker_hub_user and docker_hub_password:
            print("Found Docker Hub credentials in environment variables")
            if self.credentials is None:
                self.credentials = {}

            # Override or add Docker Hub credentials
            self.credentials["dockerhub"] = {
                "repository": docker_hub_repo,
                "username": docker_hub_user,
                "password": docker_hub_password,
            }
            print("Docker Hub credentials updated from environment variables")
            print(f"Docker Hub credentials: {self.credentials['dockerhub']}")

    def build_phase(
        self,
        registry: str = None,
        clean_cache: bool = False,
        manifest_output: str = "build_manifest.json",
        batch_build_metadata: typing.Optional[dict] = None,
    ) -> typing.Dict:
        """Execute the build phase - build all Docker images.

        This method supports both build-only mode (for dedicated build nodes)
        and full workflow mode. In build-only mode, GPU detection is skipped
        and docker build args should be provided via --additional-context.

        Args:
            registry: Optional registry to push images to
            clean_cache: Whether to use --no-cache for builds
            manifest_output: Output file for build manifest
            batch_build_metadata: Optional batch build metadata for batch builds

        Returns:
            dict: Build summary
        """
        self.rich_console.print(f"\n[dim]{'=' * 60}[/dim]")
        self.rich_console.print("[bold blue]🔨 STARTING BUILD PHASE[/bold blue]")
        if self.context._build_only_mode:
            self.rich_console.print("[yellow](Build-only mode - no GPU detection)[/yellow]")
        self.rich_console.print(f"\n[dim]{'=' * 60}[/dim]")

        # Print the arguments as a dictionary for better readability
        print(
            f"Building models with args: {vars(self.args) if hasattr(self.args, '__dict__') else self.args}"
        )

        # Discover models
        self.rich_console.print(f"\n[dim]{'=' * 60}[/dim]")
        self.rich_console.print("[bold cyan]🔍 DISCOVERING MODELS[/bold cyan]")
        discover_models = DiscoverModels(args=self.args)
        models = discover_models.run()

        print(f"Discovered {len(models)} models to build")

        # Copy scripts for building
        self.rich_console.print(f"\n[dim]{'=' * 60}[/dim]")
        self.rich_console.print("[bold cyan]📋 COPYING SCRIPTS[/bold cyan]")
        self._copy_scripts()

        # Validate build context for build-only mode
        if self.context._build_only_mode:
            if (
                "MAD_SYSTEM_GPU_ARCHITECTURE"
                not in self.context.ctx["docker_build_arg"]
            ):
                self.rich_console.print(
                    "[yellow]⚠️  Warning: MAD_SYSTEM_GPU_ARCHITECTURE not provided in build context.[/yellow]"
                )
                print(
                    "For build-only nodes, please provide GPU architecture via --additional-context:"
                )
                print(
                    '  --additional-context \'{"docker_build_arg": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx908"}}\''
                )

        # Initialize builder
        builder = DockerBuilder(
            self.context,
            self.console,
            live_output=getattr(self.args, "live_output", False),
        )

        # Determine phase suffix for log files
        phase_suffix = (
            ".build"
            if hasattr(self.args, "_separate_phases") and self.args._separate_phases
            else ""
        )

        # Get target architectures from args if provided
        target_archs = getattr(self.args, "target_archs", [])
        
        # Handle comma-separated architectures in a single string
        if target_archs:
            processed_archs = []
            for arch_arg in target_archs:
                # Split comma-separated values and add to list
                processed_archs.extend([arch.strip() for arch in arch_arg.split(',') if arch.strip()])
            target_archs = processed_archs

        # If batch_build_metadata is provided, use it to set per-model registry/registry_image
        build_summary = builder.build_all_models(
            models,
            self.credentials,
            clean_cache,
            registry,
            phase_suffix,
            batch_build_metadata=batch_build_metadata,
            target_archs=target_archs,
        )

        # Export build manifest with registry information
        builder.export_build_manifest(manifest_output, registry, batch_build_metadata)

        self.rich_console.print(f"\n[dim]{'=' * 60}[/dim]")
        self.rich_console.print("[bold green]✅ BUILD PHASE COMPLETED[/bold green]")
        self.rich_console.print(f"  [green]Successful builds: {len(build_summary['successful_builds'])}[/green]")
        self.rich_console.print(f"  [red]Failed builds: {len(build_summary['failed_builds'])}[/red]")
        self.rich_console.print(f"  [blue]Total build time: {build_summary['total_build_time']:.2f} seconds[/blue]")
        print(f"  Manifest saved to: {manifest_output}")
        self.rich_console.print(f"\n[dim]{'=' * 60}[/dim]")

        # Cleanup scripts
        self.cleanup()

        return build_summary

    def generate_local_image_manifest(
        self,
        container_image: str,
        manifest_output: str = "build_manifest.json",
    ) -> typing.Dict:
        """Generate a build manifest for a local container image.

        This method creates a build manifest that references a local container image,
        skipping the build phase entirely. This is useful for legacy compatibility
        when using MAD_CONTAINER_IMAGE.

        Args:
            container_image: The local container image tag (e.g., 'model:tag')
            manifest_output: Output file for build manifest

        Returns:
            dict: Build summary compatible with regular build phase
        """
        self.rich_console.print(f"\n[dim]{'=' * 60}[/dim]")
        self.rich_console.print("[bold blue]🏠 GENERATING LOCAL IMAGE MANIFEST[/bold blue]")
        self.rich_console.print(f"Container Image: [yellow]{container_image}[/yellow]")
        self.rich_console.print(f"\n[dim]{'=' * 60}[/dim]")

        # Ensure runtime context is initialized for local image mode
        self.context.ensure_runtime_context()
        
        # Discover models to get the model information
        self.rich_console.print(f"\n[dim]{'=' * 60}[/dim]")
        self.rich_console.print("[bold cyan]🔍 DISCOVERING MODELS[/bold cyan]")
        discover_models = DiscoverModels(args=self.args)
        models = discover_models.run()

        print(f"Discovered {len(models)} models for local image")

        # Copy scripts for running (even though we're skipping build)
        self.rich_console.print(f"\n[dim]{'=' * 60}[/dim]")
        self.rich_console.print("[bold cyan]📋 COPYING SCRIPTS[/bold cyan]")
        self._copy_scripts()

        # Create manifest entries for all discovered models using the local image
        built_images = {}
        built_models = {}
        successful_builds = []

        for model in models:
            model_name = model["name"]
            # Generate a pseudo-image name for compatibility
            image_name = f"ci-{model_name.replace('/', '_').lower()}_local"
            
            # Create build info entry for the local image
            built_images[image_name] = {
                "model_name": model_name,
                "docker_image": container_image,  # Use the provided local image
                "dockerfile": model.get("dockerfile", ""),
                "build_time": 0.0,  # No build time for local image
                "registry": None,  # Local image, no registry
                "local_image_mode": True,  # Flag to indicate this is a local image
            }

            # Create model info entry - use image_name as key for proper mapping
            built_models[image_name] = {
                "docker_image": container_image,
                "image_name": image_name,
                **model  # Include all original model information
            }

            successful_builds.append(model_name)

        # Extract credentials from models
        credentials_required = list(
            set(
                [
                    model.get("cred", "")
                    for model in models
                    if model.get("cred", "") != ""
                ]
            )
        )

        # Create the manifest structure compatible with regular build phase
        manifest = {
            "built_images": built_images,
            "built_models": built_models,
            "context": {
                "docker_env_vars": self.context.ctx.get("docker_env_vars", {}),
                "docker_mounts": self.context.ctx.get("docker_mounts", {}),
                "docker_build_arg": self.context.ctx.get("docker_build_arg", {}),
                "gpu_vendor": self.context.ctx.get("gpu_vendor", ""),
                "docker_gpus": self.context.ctx.get("docker_gpus", ""),
                "MAD_CONTAINER_IMAGE": container_image,  # Include the local image reference
            },
            "credentials_required": credentials_required,
            "local_image_mode": True,
            "local_container_image": container_image,
        }

        # Add multi-node args to context if present
        if "build_multi_node_args" in self.context.ctx:
            manifest["context"]["multi_node_args"] = self.context.ctx[
                "build_multi_node_args"
            ]

        # Write the manifest file
        with open(manifest_output, "w") as f:
            json.dump(manifest, f, indent=2)

        # Create build summary compatible with regular build phase
        build_summary = {
            "successful_builds": successful_builds,
            "failed_builds": [],
            "total_build_time": 0.0,
            "manifest_file": manifest_output,
            "local_image_mode": True,
            "container_image": container_image,
        }

        self.rich_console.print(f"\n[dim]{'=' * 60}[/dim]")
        self.rich_console.print("[bold green]✅ LOCAL IMAGE MANIFEST GENERATED[/bold green]")
        self.rich_console.print(f"  [green]Models configured: {len(successful_builds)}[/green]")
        self.rich_console.print(f"  [blue]Container Image: {container_image}[/blue]")
        self.rich_console.print(f"  [blue]Manifest saved to: {manifest_output}[/blue]")
        self.rich_console.print(f"\n[dim]{'=' * 60}[/dim]")

        # Cleanup scripts (optional for local image mode)
        self.cleanup()

        return build_summary

    def run_phase(
        self,
        manifest_file: str = "build_manifest.json",
        registry: str = None,
        timeout: int = 7200,
        keep_alive: bool = False,
    ) -> typing.Dict:
        """Execute the run phase - run containers with models.

        This method requires GPU context and will initialize runtime context
        if not already done. Should only be called on GPU nodes.

        Args:
            manifest_file: Build manifest file from build phase
            registry: Registry to pull images from (if different from build)
            timeout: Execution timeout per model
            keep_alive: Whether to keep containers alive after execution

        Returns:
            dict: Execution summary
        """
        self.rich_console.print(f"\n[dim]{'=' * 60}[/dim]")
        self.rich_console.print("[bold blue]🏃 STARTING RUN PHASE[/bold blue]")
        self.rich_console.print(f"\n[dim]{'=' * 60}[/dim]")

        # Ensure runtime context is initialized (GPU detection, env vars, etc.)
        self.context.ensure_runtime_context()

        print(f"Running models with args {self.args}")

        self.console.sh("echo 'MAD Run Models'")

        # show node rocm info
        host_os = self.context.ctx.get("host_os", "")
        if host_os.find("HOST_UBUNTU") != -1:
            print(self.console.sh("apt show rocm-libs -a", canFail=True))
        elif host_os.find("HOST_CENTOS") != -1:
            print(self.console.sh("yum info rocm-libs", canFail=True))
        elif host_os.find("HOST_SLES") != -1:
            print(self.console.sh("zypper info rocm-libs", canFail=True))
        elif host_os.find("HOST_AZURE") != -1:
            print(self.console.sh("tdnf info rocm-libs", canFail=True))
        else:
            self.rich_console.print("[red]❌ ERROR: Unable to detect host OS.[/red]")

        # Load build manifest
        if not os.path.exists(manifest_file):
            raise FileNotFoundError(f"Build manifest not found: {manifest_file}")

        with open(manifest_file, "r") as f:
            manifest = json.load(f)

        print(f"Loaded manifest with {len(manifest['built_images'])} images")
        
        # Restore context from manifest if present (for tools, pre/post scripts, etc.)
        if "context" in manifest:
            manifest_context = manifest["context"]
            
            # Restore tools configuration if present in manifest
            if "tools" in manifest_context:
                self.context.ctx["tools"] = manifest_context["tools"]
                print(f"Restored tools configuration from manifest: {manifest_context['tools']}")
            
            # Restore pre/post scripts if present in manifest
            if "pre_scripts" in manifest_context:
                self.context.ctx["pre_scripts"] = manifest_context["pre_scripts"]
                print(f"Restored pre_scripts from manifest")
            if "post_scripts" in manifest_context:
                self.context.ctx["post_scripts"] = manifest_context["post_scripts"]
                print(f"Restored post_scripts from manifest")
            if "encapsulate_script" in manifest_context:
                self.context.ctx["encapsulate_script"] = manifest_context["encapsulate_script"]
                print(f"Restored encapsulate_script from manifest")

        # Filter images by GPU architecture compatibility
        try:
            runtime_gpu_arch = self.context.get_system_gpu_architecture()
            print(f"Runtime GPU architecture detected: {runtime_gpu_arch}")
            
            # Filter manifest images by GPU architecture compatibility
            compatible_images = self._filter_images_by_gpu_architecture(
                manifest["built_images"], runtime_gpu_arch
            )
            
            if not compatible_images:
                available_archs = list(set(
                    img.get('gpu_architecture', 'unknown') 
                    for img in manifest['built_images'].values()
                ))
                available_archs = [arch for arch in available_archs if arch != 'unknown']
                
                if available_archs:
                    error_msg = (
                        f"No compatible Docker images found for runtime GPU architecture '{runtime_gpu_arch}'. "
                        f"Available image architectures: {available_archs}. "
                        f"Please build images for the target architecture using: "
                        f"--target-archs {runtime_gpu_arch}"
                    )
                else:
                    error_msg = (
                        f"No compatible Docker images found for runtime GPU architecture '{runtime_gpu_arch}'. "
                        f"The manifest contains legacy images without architecture information. "
                        f"These will be treated as compatible for backward compatibility."
                    )
                
                raise RuntimeError(error_msg)
            
            # Update manifest to only include compatible images
            manifest["built_images"] = compatible_images
            print(f"Filtered to {len(compatible_images)} compatible images for GPU architecture '{runtime_gpu_arch}'")
            
        except Exception as e:
            # If GPU architecture detection fails, proceed with all images for backward compatibility
            self.rich_console.print(
                f"[yellow]Warning: GPU architecture filtering failed: {e}[/yellow]"
            )
            self.rich_console.print(
                "[yellow]Proceeding with all available images (backward compatibility mode)[/yellow]"
            )

        # Registry is now per-image; CLI registry is fallback
        if registry:
            print(f"Using registry from CLI: {registry}")
        else:
            self.rich_console.print(
                "[yellow]No registry specified, will use per-image registry or local images only[/yellow]"
            )

        # Copy scripts for running
        self._copy_scripts()

        # Initialize runner
        runner = ContainerRunner(
            self.context,
            self.data,
            self.console,
            live_output=getattr(self.args, "live_output", False),
        )
        runner.set_credentials(self.credentials)

        # Set perf.csv output path if specified in args
        if hasattr(self.args, "output") and self.args.output:
            runner.set_perf_csv_path(self.args.output)

        # Determine phase suffix for log files
        phase_suffix = (
            ".run"
            if hasattr(self.args, "_separate_phases") and self.args._separate_phases
            else ""
        )

        # Use built models from manifest if available, otherwise discover models
        if "built_models" in manifest and manifest["built_models"]:
            self.rich_console.print("[cyan]Using model information from build manifest[/cyan]")
            models = list(manifest["built_models"].values())
        else:
            self.rich_console.print(
                "[yellow]No model information in manifest, discovering models from current configuration[/yellow]"
            )
            # Discover models (to get execution parameters)
            discover_models = DiscoverModels(args=self.args)
            models = discover_models.run()

        # Create execution summary
        execution_summary = {
            "successful_runs": [],
            "failed_runs": [],
            "total_execution_time": 0,
        }

        # Map models to their built images
        if "built_models" in manifest and manifest["built_models"]:
            # Direct mapping from manifest - built_models maps image_name -> model_info
            print("Using direct model-to-image mapping from manifest")
            for image_name, build_info in manifest["built_images"].items():
                if image_name in manifest["built_models"]:
                    model_info = manifest["built_models"][image_name]
                    try:
                        print(
                            f"\nRunning model {model_info['name']} with image {image_name}"
                        )
                        
                        # Check if MAD_CONTAINER_IMAGE is set in context (for local image mode)
                        if "MAD_CONTAINER_IMAGE" in self.context.ctx:
                            actual_image = self.context.ctx["MAD_CONTAINER_IMAGE"]
                            print(f"Using MAD_CONTAINER_IMAGE override: {actual_image}")
                            print("Warning: User override MAD_CONTAINER_IMAGE. Model support on image not guaranteed.")
                        else:
                            # Use per-image registry if present, else CLI registry
                            effective_registry = build_info.get("registry", registry)
                            registry_image = build_info.get("registry_image")
                            docker_image = build_info.get("docker_image")
                            if registry_image:
                                if effective_registry:
                                    print(f"Pulling image from registry: {registry_image}")
                                    try:
                                        registry_image_str = (
                                            str(registry_image) if registry_image else ""
                                        )
                                        docker_image_str = (
                                            str(docker_image) if docker_image else ""
                                        )
                                        effective_registry_str = (
                                            str(effective_registry)
                                            if effective_registry
                                            else ""
                                        )
                                        runner.pull_image(
                                            registry_image_str,
                                            docker_image_str,
                                            effective_registry_str,
                                            self.credentials,
                                        )
                                        actual_image = docker_image_str
                                        print(
                                            f"Successfully pulled and tagged as: {docker_image_str}"
                                        )
                                    except Exception as e:
                                        print(
                                            f"Failed to pull from registry, falling back to local image: {e}"
                                        )
                                        actual_image = docker_image
                                else:
                                    print(
                                        f"Attempting to pull registry image as-is: {registry_image}"
                                    )
                                    try:
                                        registry_image_str = (
                                            str(registry_image) if registry_image else ""
                                        )
                                        docker_image_str = (
                                            str(docker_image) if docker_image else ""
                                        )
                                        runner.pull_image(
                                            registry_image_str, docker_image_str
                                        )
                                        actual_image = docker_image_str
                                        print(
                                            f"Successfully pulled and tagged as: {docker_image_str}"
                                        )
                                    except Exception as e:
                                        print(
                                            f"Failed to pull from registry, falling back to local image: {e}"
                                        )
                                        actual_image = docker_image
                            else:
                                # No registry_image key - run container directly using docker_image
                                actual_image = build_info["docker_image"]
                                print(
                                    f"No registry image specified, using local image: {actual_image}"
                                )

                        # Run the container
                        run_results = runner.run_container(
                            model_info,
                            actual_image,
                            build_info,
                            keep_alive=keep_alive,
                            timeout=timeout,
                            phase_suffix=phase_suffix,
                            generate_sys_env_details=getattr(
                                self.args, "generate_sys_env_details", True
                            ),
                        )

                        # Add to appropriate list based on actual status
                        if run_results.get("status") == "SUCCESS":
                            execution_summary["successful_runs"].append(run_results)
                            self.rich_console.print(
                                f"[green]✅ Successfully completed: {model_info['name']} -> {run_results['status']}[/green]"
                            )
                        else:
                            execution_summary["failed_runs"].append(run_results)
                            self.rich_console.print(
                                f"[red]❌ Failed to complete: {model_info['name']} -> {run_results['status']}[/red]"
                            )

                        execution_summary["total_execution_time"] += run_results.get(
                            "test_duration", 0
                        )

                    except Exception as e:
                        self.rich_console.print(
                            f"[red]❌ Failed to run {model_info['name']} with image {image_name}: {e}[/red]"
                        )
                        execution_summary["failed_runs"].append(
                            {
                                "model": model_info["name"],
                                "image": image_name,
                                "error": str(e),
                            }
                        )
                else:
                    self.rich_console.print(f"[yellow]⚠️  Warning: No model info found for built image: {image_name}[/yellow]")
        else:
            # Fallback to name-based matching for backward compatibility
            self.rich_console.print("[yellow]Using name-based matching (fallback mode)[/yellow]")
            for model_info in models:
                model_name = model_info["name"]

                # Find matching built images for this model
                matching_images = []
                for image_name, build_info in manifest["built_images"].items():
                    if model_name.replace("/", "_").lower() in image_name:
                        matching_images.append((image_name, build_info))

                if not matching_images:
                    self.rich_console.print(f"[red]❌ No built images found for model: {model_name}[/red]")
                    execution_summary["failed_runs"].append(
                        {"model": model_name, "error": "No built images found"}
                    )
                    continue

                # Run each matching image
                for image_name, build_info in matching_images:
                    try:
                        print(f"\nRunning model {model_name} with image {image_name}")

                        # Handle registry image pulling and tagging according to manifest
                        if "registry_image" in build_info:
                            # Registry image exists - pull it and tag as docker_image, then run with docker_image
                            registry_image = build_info["registry_image"]
                            docker_image = build_info["docker_image"]

                            # Extract registry from the registry_image format
                            effective_registry = registry
                            if not effective_registry and registry_image:
                                registry_parts = registry_image.split("/")
                                if len(registry_parts) > 1 and "." in registry_parts[0]:
                                    effective_registry = registry_parts[0]
                                elif (
                                    registry_image.startswith("docker.io/")
                                    or "/" in registry_image
                                ):
                                    effective_registry = "docker.io"

                            if effective_registry:
                                print(f"Pulling image from registry: {registry_image}")
                                try:
                                    # Ensure all parameters are strings and credentials is properly formatted
                                    registry_image_str = (
                                        str(registry_image) if registry_image else ""
                                    )
                                    docker_image_str = (
                                        str(docker_image) if docker_image else ""
                                    )
                                    effective_registry_str = (
                                        str(effective_registry)
                                        if effective_registry
                                        else ""
                                    )

                                    # Pull registry image and tag it as docker_image
                                    runner.pull_image(
                                        registry_image_str,
                                        docker_image_str,
                                        effective_registry_str,
                                        self.credentials,
                                    )
                                    actual_image = docker_image_str
                                    print(
                                        f"Successfully pulled and tagged as: {docker_image_str}"
                                    )
                                except Exception as e:
                                    print(
                                        f"Failed to pull from registry, falling back to local image: {e}"
                                    )
                                    actual_image = docker_image
                            else:
                                # Registry image exists but no valid registry found, try to pull as-is and tag
                                print(
                                    f"Attempting to pull registry image as-is: {registry_image}"
                                )
                                try:
                                    registry_image_str = (
                                        str(registry_image) if registry_image else ""
                                    )
                                    docker_image_str = (
                                        str(docker_image) if docker_image else ""
                                    )
                                    runner.pull_image(
                                        registry_image_str, docker_image_str
                                    )
                                    actual_image = docker_image_str
                                    print(
                                        f"Successfully pulled and tagged as: {docker_image_str}"
                                    )
                                except Exception as e:
                                    print(
                                        f"Failed to pull from registry, falling back to local image: {e}"
                                    )
                                    actual_image = docker_image
                        else:
                            # No registry_image key - run container directly using docker_image
                            actual_image = build_info["docker_image"]
                            print(
                                f"No registry image specified, using local image: {actual_image}"
                            )

                        # Run the container
                        run_results = runner.run_container(
                            model_info,
                            actual_image,
                            build_info,
                            keep_alive=keep_alive,
                            timeout=timeout,
                            phase_suffix=phase_suffix,
                            generate_sys_env_details=getattr(
                                self.args, "generate_sys_env_details", True
                            ),
                        )

                        # Add to appropriate list based on actual status
                        if run_results.get("status") == "SUCCESS":
                            execution_summary["successful_runs"].append(run_results)
                            self.rich_console.print(
                                f"[green]✅ Successfully completed: {model_name} -> {run_results['status']}[/green]"
                            )
                        else:
                            execution_summary["failed_runs"].append(run_results)
                            self.rich_console.print(
                                f"[red]❌ Failed to complete: {model_name} -> {run_results['status']}[/red]"
                            )

                        execution_summary["total_execution_time"] += run_results.get(
                            "test_duration", 0
                        )

                    except Exception as e:
                        self.rich_console.print(
                            f"[red]❌ Failed to run {model_name} with image {image_name}: {e}[/red]"
                        )
                        execution_summary["failed_runs"].append(
                            {"model": model_name, "image": image_name, "error": str(e)}
                        )

        self.rich_console.print(f"\n[dim]{'=' * 60}[/dim]")
        self.rich_console.print("[bold green]✅ RUN PHASE COMPLETED[/bold green]")
        self.rich_console.print(f"  [green]Successful runs: {len(execution_summary['successful_runs'])}[/green]")
        self.rich_console.print(f"  [red]Failed runs: {len(execution_summary['failed_runs'])}[/red]")
        self.rich_console.print(
            f"  [blue]Total execution time: {execution_summary['total_execution_time']:.2f} seconds[/blue]"
        )
        self.rich_console.print(f"\n[dim]{'=' * 60}[/dim]")

        # Convert output CSV to HTML like run_models.py does
        try:
            from madengine.tools.csv_to_html import convert_csv_to_html

            perf_csv_path = getattr(self.args, "output", "perf.csv")
            if os.path.exists(perf_csv_path):
                print("Converting output csv to html...")
                convert_csv_to_html(file_path=perf_csv_path)
        except Exception as e:
            self.rich_console.print(f"[yellow]⚠️  Warning: Could not convert CSV to HTML: {e}[/yellow]")

        # Cleanup scripts
        self.cleanup()

        return execution_summary

    def full_workflow(
        self,
        registry: str = None,
        clean_cache: bool = False,
        timeout: int = 7200,
        keep_alive: bool = False,
    ) -> typing.Dict:
        """Execute the complete workflow: build then run.

        Args:
            registry: Optional registry for image distribution
            clean_cache: Whether to use --no-cache for builds
            timeout: Execution timeout per model
            keep_alive: Whether to keep containers alive after execution

        Returns:
            dict: Complete workflow summary
        """
        self.rich_console.print(f"\n[dim]{'=' * 80}[/dim]")
        self.rich_console.print("[bold magenta]🚀 STARTING COMPLETE DISTRIBUTED WORKFLOW[/bold magenta]")
        self.rich_console.print(f"\n[dim]{'=' * 80}[/dim]")

        # Build phase
        build_summary = self.build_phase(registry, clean_cache)

        # Run phase
        execution_summary = self.run_phase(timeout=timeout, keep_alive=keep_alive)

        # Combine summaries
        workflow_summary = {
            "build_phase": build_summary,
            "run_phase": execution_summary,
            "overall_success": (
                len(build_summary["failed_builds"]) == 0
                and len(execution_summary["failed_runs"]) == 0
            ),
        }

        self.rich_console.print(f"\n[dim]{'=' * 80}[/dim]")
        if workflow_summary['overall_success']:
            self.rich_console.print("[bold green]🎉 COMPLETE WORKFLOW FINISHED SUCCESSFULLY[/bold green]")
            self.rich_console.print(f"  [green]Overall success: {workflow_summary['overall_success']}[/green]")
        else:
            self.rich_console.print("[bold red]❌ COMPLETE WORKFLOW FINISHED WITH ERRORS[/bold red]")
            self.rich_console.print(f"  [red]Overall success: {workflow_summary['overall_success']}[/red]")
        self.rich_console.print(f"\n[dim]{'=' * 80}[/dim]")

        return workflow_summary

    def _copy_scripts(self) -> None:
        """Copy scripts to the current directory."""
        scripts_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "scripts"
        )
        print(f"Package path: {scripts_path}")
        # copy the scripts to the model directory
        self.console.sh(f"cp -vLR --preserve=all {scripts_path} .")
        print(f"Scripts copied to {os.getcwd()}/scripts")

    def _filter_images_by_gpu_architecture(self, built_images: typing.Dict, runtime_arch: str) -> typing.Dict:
        """Filter built images by GPU architecture compatibility.
        
        Args:
            built_images: Dictionary of built images from manifest
            runtime_arch: Runtime GPU architecture (e.g., 'gfx908')
            
        Returns:
            dict: Filtered dictionary containing only compatible images
        """
        compatible = {}
        
        self.rich_console.print(f"[cyan]Filtering images for runtime GPU architecture: {runtime_arch}[/cyan]")
        
        for image_name, image_info in built_images.items():
            image_arch = image_info.get("gpu_architecture")
            
            if not image_arch:
                # Legacy images without architecture info - assume compatible for backward compatibility
                self.rich_console.print(
                    f"[yellow]  Warning: Image {image_name} has no architecture info, assuming compatible (legacy mode)[/yellow]"
                )
                compatible[image_name] = image_info
            elif image_arch == runtime_arch:
                # Exact architecture match
                self.rich_console.print(
                    f"[green]  ✓ Compatible: {image_name} (architecture: {image_arch})[/green]"
                )
                compatible[image_name] = image_info
            else:
                # Architecture mismatch
                self.rich_console.print(
                    f"[red]  ✗ Incompatible: {image_name} (architecture: {image_arch}, runtime: {runtime_arch})[/red]"
                )
        
        if not compatible:
            self.rich_console.print(f"[red]No compatible images found for runtime architecture: {runtime_arch}[/red]")
        else:
            self.rich_console.print(f"[green]Found {len(compatible)} compatible image(s)[/green]")
        
        return compatible

    def cleanup(self) -> None:
        """Cleanup the scripts/common directory."""
        # check the directory exists
        if os.path.exists("scripts/common"):
            # List of directories/files to clean up
            cleanup_targets = [
                "scripts/common/tools",
                "scripts/common/test_echo.sh",
                "scripts/common/pre_scripts",
                "scripts/common/post_scripts",
            ]
            
            for target in cleanup_targets:
                if os.path.exists(target):
                    try:
                        # Try normal removal first
                        self.console.sh(f"rm -rf {target}", canFail=True)
                    except Exception:
                        # If that fails, try to fix permissions and remove
                        try:
                            # Fix permissions recursively (ignore errors)
                            self.console.sh(f"chmod -R u+w {target} 2>/dev/null || true", canFail=True)
                            # Try removal again (allow failure)
                            self.console.sh(f"rm -rf {target} 2>/dev/null || true", canFail=True)
                            
                            # If directory still exists (e.g., __pycache__ with root-owned files),
                            # just warn the user instead of failing
                            if os.path.exists(target):
                                self.rich_console.print(
                                    f"[yellow]⚠️  Warning: Could not fully remove {target} (permission denied for some files)[/yellow]"
                                )
                                self.rich_console.print(
                                    f"[dim]You may need to manually remove it with: sudo rm -rf {target}[/dim]"
                                )
                        except Exception as e:
                            # Even permission fixing failed, just warn
                            self.rich_console.print(
                                f"[yellow]⚠️  Warning: Could not clean up {target}: {e}[/yellow]"
                            )
            
            print(f"scripts/common directory cleanup attempted.")
