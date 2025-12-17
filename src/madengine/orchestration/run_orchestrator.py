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
from madengine.utils.session_tracker import SessionTracker


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
                    # Use ast.literal_eval for Python dict syntax (single quotes)
                    # This matches what Context class expects
                    import ast
                    parsed = ast.literal_eval(args.additional_context)
                    print(f"📝 RunOrchestrator: Parsed additional_context keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'not a dict'}")
                    merged_context = parsed
                elif isinstance(args.additional_context, dict):
                    merged_context = args.additional_context
                    print(f"📝 RunOrchestrator: Got dict additional_context keys: {list(merged_context.keys())}")
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse additional_context: {e}")
                print(f"Raw additional_context: {args.additional_context[:200] if args.additional_context else 'None'}")
                pass

        if additional_context:
            merged_context.update(additional_context)

        self.additional_context = merged_context
        print(f"📝 RunOrchestrator: Final additional_context keys: {list(self.additional_context.keys()) if self.additional_context else 'None'}")

        # Track if we copied MODEL_DIR contents (for cleanup)
        self._copied_from_model_dir = False
        
        # Track if we ran build phase in this workflow (for log combination)
        self._did_build_phase = False
        
        # Initialize session tracker for filtering current run results
        perf_csv_path = getattr(args, "output", "perf.csv")
        self.session_tracker = SessionTracker(perf_csv_path)
        
        # Initialize context in runtime mode (with GPU detection for local)
        # This will be lazy-initialized only when needed
        self.context = None
        self.data = None

    def _init_runtime_context(self):
        """Initialize runtime context (with GPU detection)."""
        # Always reinitialize context in runtime mode for run phase
        # This ensures GPU detection and proper runtime context even after build phase
        
        # Context expects additional_context as a string representation of Python dict
        # Use repr() instead of json.dumps() because Context uses ast.literal_eval()
        if self.additional_context:
            context_string = repr(self.additional_context)
        else:
            context_string = None
            
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

        # Track session start for filtering current run results
        # The marker file is automatically saved in same directory as perf.csv
        session_start_row = self.session_tracker.start_session()

        try:
            # Check for MAD_CONTAINER_IMAGE (local image mode)
            # This must be checked before normal build/manifest flow
            mad_container_image = None
            if self.additional_context:
                mad_container_image = self.additional_context.get("MAD_CONTAINER_IMAGE")
            
            if mad_container_image:
                # Local image mode: Skip build, create synthetic manifest
                if not tags:
                    raise ConfigurationError(
                        "Tags required for MAD_CONTAINER_IMAGE mode",
                        context=create_error_context(
                            operation="local_image_mode",
                            component="RunOrchestrator",
                        ),
                        suggestions=[
                            "Provide --tags to specify which models to run",
                            "Example: --tags model_name --additional-context \"{'MAD_CONTAINER_IMAGE': 'rocm/tensorflow:latest'}\"",
                        ],
                    )
                
                # Generate synthetic manifest using the provided image
                manifest_file = self._create_manifest_from_local_image(
                    image_name=mad_container_image,
                    tags=tags,
                    manifest_output=getattr(self.args, "manifest_output", "build_manifest.json"),
                )
            
            # Step 1: Ensure we have a manifest (build if needed)
            elif not manifest_file or not os.path.exists(manifest_file):
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
                self._did_build_phase = True  # Mark that we built in this workflow

            # Step 2: Load manifest and merge with runtime context
            manifest_file = self._load_and_merge_manifest(manifest_file)

            # Step 3: Determine execution target from manifest's deployment_config
            # (with optional runtime override)
            with open(manifest_file) as f:
                manifest = json.load(f)
            
            deployment_config = manifest.get("deployment_config", {})
            
            # Update additional_context with deployment_config for deployment layer
            if not self.additional_context:
                self.additional_context = {}
            
            # Merge deployment_config into additional_context (for deployment layer to use)
            for key in ["slurm", "k8s", "kubernetes", "distributed", "vllm", "env_vars", "debug"]:
                if key in deployment_config and key not in self.additional_context:
                    self.additional_context[key] = deployment_config[key]
            
            # Infer deployment target from config structure (Convention over Configuration)
            # No explicit "deploy" field needed - presence of k8s/slurm indicates deployment type
            target = self._infer_deployment_target(self.additional_context)
            
            # Legacy support: check manifest for explicit target
            if not target or target == "local":
                target = deployment_config.get("target", "local")
            
            self.rich_console.print(f"[bold cyan]Deployment target: {target}[/bold cyan]\n")

            # Step 4: Execute based on target
            try:
                if target == "local" or target == "docker":
                    results = self._execute_local(manifest_file, timeout)
                else:
                    results = self._execute_distributed(target, manifest_file)
                
                # Combine build and run logs for full workflow
                if self._did_build_phase and (target == "local" or target == "docker"):
                    self._combine_build_and_run_logs()
                
                # Add session information to results for filtering
                results["session_start_row"] = session_start_row
                results["session_row_count"] = self.session_tracker.get_session_row_count()
                
                # Always cleanup madengine package files after execution
                self.rich_console.print("\n[dim]🧹 Cleaning up madengine package files...[/dim]")
                self._cleanup_model_dir_copies()
                
                # NOTE: Do NOT cleanup session marker here!
                # It's needed by display functions in CLI layer
                # Cleanup happens in CLI after display (via perf_csv_path)
                
                return results
                
            except Exception as e:
                # Always cleanup madengine package files even on error
                self.rich_console.print("\n[dim]🧹 Cleaning up madengine package files...[/dim]")
                self._cleanup_model_dir_copies()
                raise

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

    def _create_manifest_from_local_image(
        self, 
        image_name: str, 
        tags: list, 
        manifest_output: str = "build_manifest.json"
    ) -> str:
        """
        Create a synthetic manifest for a user-provided local image.
        
        This enables MAD_CONTAINER_IMAGE functionality where users can skip
        the build phase and directly run models using a pre-existing Docker image.
        
        Args:
            image_name: Docker image name/tag (e.g., 'rocm/tensorflow:latest')
            tags: Model tags to discover
            manifest_output: Output path for the manifest file
            
        Returns:
            Path to the generated manifest file
            
        Raises:
            DiscoveryError: If no models are found
            RuntimeError: If image validation fails
        """
        from madengine.utils.discover_models import DiscoverModels
        from madengine.core.errors import DiscoveryError
        
        self.rich_console.print(f"[yellow]🏠 Local Image Mode: Using {image_name}[/yellow]")
        self.rich_console.print(f"[dim]Skipping build phase, creating synthetic manifest...[/dim]\n")
        
        # Validate that the image exists locally or can be pulled
        try:
            self.console.sh(f"docker image inspect {image_name} > /dev/null 2>&1")
            self.rich_console.print(f"[green]✓ Image {image_name} found locally[/green]")
        except (subprocess.CalledProcessError, RuntimeError) as e:
            self.rich_console.print(f"[yellow]⚠️  Image {image_name} not found locally, attempting to pull...[/yellow]")
            try:
                self.console.sh(f"docker pull {image_name}")
                self.rich_console.print(f"[green]✓ Successfully pulled {image_name}[/green]")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to find or pull image {image_name}. "
                    f"Ensure the image exists locally or can be pulled from a registry. "
                    f"Error: {e}"
                )
        
        # Discover models by tags (without building)
        self.args.tags = tags
        discover_models = DiscoverModels(args=self.args)
        models = discover_models.run()
        
        if not models:
            raise DiscoveryError(
                "No models discovered for local image mode",
                context=create_error_context(
                    operation="create_local_image_manifest",
                    component="RunOrchestrator",
                ),
                suggestions=[
                    "Check if models.json exists",
                    "Verify --tags parameter is correct",
                    "Ensure model definitions have matching tags",
                ],
            )
        
        self.rich_console.print(f"[green]✓ Discovered {len(models)} model(s) for tags: {tags}[/green]\n")
        
        # Initialize build-only context for manifest generation
        # (we need context structure, but skip GPU detection since we're not building)
        context_string = repr(self.additional_context) if self.additional_context else None
        build_context = Context(
            additional_context=context_string,
            build_only_mode=True,
        )
        
        # Create manifest structure
        manifest = {
            "built_images": {},
            "built_models": {},
            "context": build_context.ctx,
            "local_image_mode": True,
            "local_image_name": image_name,
            "deployment_config": self.additional_context.get("deployment_config", {}),
        }
        
        # For each model, create a synthetic entry using the provided image
        for model in models:
            model_name = model["name"]
            # Create a synthetic image identifier (not an actual built image)
            synthetic_image_id = f"local-{model_name.replace('/', '_')}"
            
            manifest["built_images"][synthetic_image_id] = {
                "docker_image": image_name,  # Use user-provided image
                "dockerfile": "N/A (local image mode)",
                "build_status": "SKIPPED",
                "build_time": 0,
                "local_image": True,
                "registry_image": None,
            }
            
            # Convert data list to comma-separated string (required by dataprovider)
            data_field = model.get("data", [])
            if isinstance(data_field, list):
                data_str = ",".join(data_field) if data_field else ""
            else:
                data_str = data_field if data_field else ""
            
            # Build model info dict with all fields that ContainerRunner expects
            # Use exact field names from models.json format
            manifest["built_models"][synthetic_image_id] = {
                "name": model_name,
                "tags": model.get("tags", []),
                "dockerfile": "N/A (local image mode)",
                "scripts": model.get("scripts", ""),  # models.json uses "scripts" (plural)
                "n_gpus": model.get("n_gpus", "1"),  # models.json uses "n_gpus" (string format)
                "owner": model.get("owner", ""),
                "training_precision": model.get("training_precision", ""),
                "args": model.get("args", ""),  # Required field for docker run
                "timeout": model.get("timeout", None),  # Optional timeout override
                "data": data_str,
                "cred": model.get("cred", ""),
                "deprecated": model.get("deprecated", False),
                "skip_gpu_arch": model.get("skip_gpu_arch", []),
                "additional_docker_run_options": model.get("additional_docker_run_options", ""),
            }
        
        # Write manifest to file
        with open(manifest_output, "w") as f:
            json.dump(manifest, f, indent=2)
        
        self.rich_console.print(f"[green]✓ Generated synthetic manifest: {manifest_output}[/green]")
        self.rich_console.print(f"[yellow]⚠️  Warning: User-provided image {image_name}. Model support not guaranteed.[/yellow]\n")
        
        return manifest_output

    def _load_and_merge_manifest(self, manifest_file: str) -> str:
        """Load manifest and merge with runtime --additional-context."""
        if not os.path.exists(manifest_file):
            raise FileNotFoundError(f"Build manifest not found: {manifest_file}")

        with open(manifest_file, "r") as f:
            manifest = json.load(f)

        print(f"Loaded manifest with {len(manifest.get('built_images', {}))} images")

        # Merge deployment configs and context (runtime overrides build-time)
        if self.additional_context:
            # Merge deployment_config
            if "deployment_config" in manifest:
                stored_config = manifest["deployment_config"]
                # Runtime --additional-context overrides stored config
                for key in ["deploy", "slurm", "k8s", "kubernetes", "distributed", "vllm", "env_vars", "debug"]:
                    if key in self.additional_context:
                        stored_config[key] = self.additional_context[key]
                manifest["deployment_config"] = stored_config
            
            # Merge context (tools, pre_scripts, post_scripts, encapsulate_script)
            if "context" not in manifest:
                manifest["context"] = {}
            
            merge_keys = ["tools", "pre_scripts", "post_scripts", "encapsulate_script"]
            context_updated = False
            for key in merge_keys:
                if key in self.additional_context:
                    manifest["context"][key] = self.additional_context[key]
                    context_updated = True
            
            if context_updated or "deployment_config" in manifest:
                # Write back merged config
                with open(manifest_file, "w") as f:
                    json.dump(manifest, f, indent=2)
                print("Merged runtime context and deployment config with manifest")

        return manifest_file

    def _execute_local(self, manifest_file: str, timeout: int) -> Dict:
        """Execute locally using container_runner."""
        self.rich_console.print("[cyan]Executing locally...[/cyan]\n")

        # Load manifest first to check if we have Docker images
        with open(manifest_file, "r") as f:
            manifest = json.load(f)
        
        has_docker_images = bool(manifest.get("built_images", {}))
        
        if has_docker_images:
            # Using Docker containers - containers have GPU support built-in
            self.rich_console.print("[dim cyan]Using Docker containers with built-in GPU support[/dim cyan]\n")
        
        # Initialize runtime context (runs full GPU detection on compute nodes)
        self._init_runtime_context()
        
        # Show node info
        self._show_node_info()

        # Import from execution layer
        from madengine.execution.container_runner import ContainerRunner

        # Load credentials
        credentials = self._load_credentials()

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
        
        # Merge runtime additional_context (takes precedence over manifest)
        # This allows users to override tools/scripts at runtime
        if self.additional_context:
            if "tools" in self.additional_context:
                self.context.ctx["tools"] = self.additional_context["tools"]
                self.rich_console.print(
                    f"[dim]  Using tools from runtime --additional-context[/dim]"
                )
            if "pre_scripts" in self.additional_context:
                self.context.ctx["pre_scripts"] = self.additional_context["pre_scripts"]
            if "post_scripts" in self.additional_context:
                self.context.ctx["post_scripts"] = self.additional_context["post_scripts"]
            if "encapsulate_script" in self.additional_context:
                self.context.ctx["encapsulate_script"] = self.additional_context["encapsulate_script"]

        # Filter images by GPU vendor and architecture
        # Filter images by GPU compatibility
        try:
            if has_docker_images:
                # Docker images are pre-built for specific GPUs, skip runtime filtering
                self.rich_console.print("[dim cyan]Using all Docker images (already GPU-specific from build)[/dim cyan]\n")
                compatible_images = manifest["built_images"]
            else:
                # Bare-metal execution: filter by runtime GPU
                runtime_gpu_vendor = self.context.get_gpu_vendor()
                runtime_gpu_arch = self.context.get_system_gpu_architecture()
                print(f"Runtime GPU vendor: {runtime_gpu_vendor}")
                print(f"Runtime GPU architecture detected: {runtime_gpu_arch}")

                compatible_images = self._filter_images_by_gpu_compatibility(
                    manifest["built_images"], runtime_gpu_vendor, runtime_gpu_arch
                )

                if not compatible_images:
                    raise MADRuntimeError(
                        f"No compatible images for GPU vendor '{runtime_gpu_vendor}' and architecture '{runtime_gpu_arch}'",
                        context=create_error_context(
                            operation="filter_images",
                            component="RunOrchestrator",
                        ),
                        suggestions=[
                            f"Build images for {runtime_gpu_vendor} GPU",
                            f"Build images for {runtime_gpu_arch} using --target-archs",
                            "Check manifest contains images for your GPU",
                        ],
                    )

                manifest["built_images"] = compatible_images
                print(f"Filtered to {len(compatible_images)} compatible images\n")
                
                # Filter by skip_gpu_arch from model definitions
                if "built_models" in manifest and compatible_images:
                    self.rich_console.print("[cyan]Checking skip_gpu_arch model restrictions...[/cyan]")
                    compatible_images = self._filter_images_by_skip_gpu_arch(
                        compatible_images, manifest["built_models"], runtime_gpu_arch
                    )
                manifest["built_images"] = compatible_images
                print(f"After skip_gpu_arch filtering: {len(compatible_images)} images to run\n")
            
            # NOTE: Dockerfile context filtering is already done during build phase
            # Re-filtering during run phase causes issues because:
            # 1. The build phase already filtered dockerfiles based on build-time context
            # 2. All built images should be runnable on the runtime node
            # 3. Legacy behavior: filtering happens once (either build or run, not both)
            
            # Write filtered manifest back to file so runner sees the filtered list
            with open(manifest_file, "w") as f:
                json.dump(manifest, f, indent=2)

        except Exception as e:
            import traceback
            self.rich_console.print(f"[yellow]Warning: GPU/Context filtering failed: {e}[/yellow]")
            self.rich_console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
            self.rich_console.print("[yellow]Proceeding with all images[/yellow]\n")

        # Copy scripts
        self._copy_scripts()

        # Initialize runner
        runner = ContainerRunner(
            self.context,
            self.data,
            self.console,
            live_output=getattr(self.args, "live_output", False),
            additional_context=self.additional_context,
        )
        runner.set_credentials(credentials)

        if hasattr(self.args, "output") and self.args.output:
            runner.set_perf_csv_path(self.args.output)

        # Run phase always uses .run suffix for consistency
        phase_suffix = ".run"

        # Run models
        results = runner.run_models_from_manifest(
            manifest_file=manifest_file,
            registry=getattr(self.args, "registry", None),
            timeout=timeout,
            keep_alive=getattr(self.args, "keep_alive", False),
            keep_model_dir=getattr(self.args, "keep_model_dir", False),
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

        # Add runtime flags to additional_context for deployment layer
        if "live_output" not in self.additional_context:
            self.additional_context["live_output"] = getattr(self.args, "live_output", False)
        
        # Pass session_start_row for result filtering in collect_results
        session_start_row = self.session_tracker.session_start_row
        if "session_start_row" not in self.additional_context:
            self.additional_context["session_start_row"] = session_start_row

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

        # Return metrics in the format expected by display_results_table
        # Extract successful_runs and failed_runs from metrics if available
        if result.metrics:
            return {
                "successful_runs": result.metrics.get("successful_runs", []),
                "failed_runs": result.metrics.get("failed_runs", []),
            }
        else:
            return {"successful_runs": [], "failed_runs": []}

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

    def _cleanup_model_dir_copies(self):
        """Clean up only madengine package files from scripts/common directory.
        
        This cleanup removes ONLY the files that were copied from madengine package:
        - scripts/common/tools.json
        - scripts/common/test_echo.sh
        - scripts/common/pre_scripts/
        - scripts/common/post_scripts/
        - scripts/common/tools/
        
        This preserves the user's actual scripts/ and docker/ directories in MAD project.
        """
        import shutil
        import subprocess
        
        # Only clean up scripts/common/ subdirectories that came from madengine package
        common_dir = Path("scripts/common")
        if not common_dir.exists():
            return
        
        # List of items to clean up (from madengine package)
        items_to_cleanup = [
            "tools.json",
            "test_echo.sh",
            "pre_scripts",
            "post_scripts",
            "tools"
        ]
        
        for item_name in items_to_cleanup:
            item_path = common_dir / item_name
            if item_path.exists():
                try:
                    if item_path.is_dir():
                        # Fix permissions first for directories
                        try:
                            subprocess.run(
                                ["chmod", "-R", "+w", str(item_path)],
                                capture_output=True,
                                timeout=10
                            )
                        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as e:
                            print(f"Warning: chmod failed for {item_path}: {e}")
                        shutil.rmtree(item_path)
                    else:
                        item_path.unlink()
                    self.rich_console.print(f"[dim]  Cleaned up: scripts/common/{item_name}[/dim]")
                except Exception as e:
                    # Try with sudo for permission issues
                    try:
                        subprocess.run(
                            ["sudo", "rm", "-rf", str(item_path)],
                            check=True,
                            capture_output=True,
                            timeout=10
                        )
                        self.rich_console.print(f"[dim]  Cleaned up: scripts/common/{item_name} (elevated)[/dim]")
                    except Exception as e2:
                        self.rich_console.print(
                            f"[yellow]⚠️  Warning: Could not clean up {item_path}: {e2}[/yellow]"
                        )

    def _combine_build_and_run_logs(self):
        """Combine build.live.log and run.live.log into live.log for full workflow.
        
        For full workflow (build + run), this creates a unified log file by:
        1. Finding all *.build.live.log and corresponding *.run.live.log files
        2. Concatenating them into *.live.log
        3. Keeping the original build and run logs for reference
        """
        import glob
        
        build_logs = glob.glob("*.build.live.log")
        if not build_logs:
            return  # No build logs to combine
        
        self.rich_console.print("\n[dim]📝 Combining build and run logs...[/dim]")
        combined_count = 0
        
        for build_log in build_logs:
            # Derive the base name and corresponding run log
            base_name = build_log.replace(".build.live.log", "")
            run_log = f"{base_name}.run.live.log"
            combined_log = f"{base_name}.live.log"
            
            # Check if run log exists
            if not os.path.exists(run_log):
                continue  # Skip if run log doesn't exist
            
            try:
                # Combine build and run logs
                with open(combined_log, 'w') as outfile:
                    # Add build log
                    with open(build_log, 'r') as infile:
                        outfile.write(infile.read())
                    
                    # Add separator
                    outfile.write("\n" + "=" * 80 + "\n")
                    outfile.write("RUN PHASE LOG\n")
                    outfile.write("=" * 80 + "\n\n")
                    
                    # Add run log
                    with open(run_log, 'r') as infile:
                        outfile.write(infile.read())
                
                combined_count += 1
                self.rich_console.print(f"[dim]  Combined: {combined_log}[/dim]")
                
            except Exception as e:
                self.rich_console.print(
                    f"[yellow]⚠️  Warning: Could not combine logs for {base_name}: {e}[/yellow]"
                )
        
        if combined_count > 0:
            self.rich_console.print(f"[dim]✓ Combined {combined_count} log file(s)[/dim]")

    def _copy_scripts(self):
        """Copy common scripts to model directories.
        
        Handles scenarios:
        1. MAD Project: scripts/ already exists in current directory - just add madengine common files
        2. External MODEL_DIR: Copy from external path to current directory
        3. madengine Testing: Copy from src/madengine/scripts/common
        
        NOTE: Does NOT delete existing scripts/ or docker/ directories in current working directory.
        """
        import shutil

        # Define ignore function for cache files (used for all copy operations)
        def ignore_cache_files(directory, files):
            """Ignore Python cache files and directories."""
            return [f for f in files if f.endswith('.pyc') or f == '__pycache__' or f.endswith('.pyo')]
        
        # Step 1: Check if MODEL_DIR points to external directory and copy if needed
        # MODEL_DIR default is "." (current directory), so only copy if it's different
        model_dir_env = os.environ.get("MODEL_DIR", ".")
        model_dir_abs = os.path.abspath(model_dir_env)
        current_dir_abs = os.path.abspath(".")
        
        # Only copy if MODEL_DIR points to a different directory (not current dir)
        if model_dir_abs != current_dir_abs and os.path.exists(model_dir_env):
            self.rich_console.print(f"[yellow]📁 External MODEL_DIR detected: {model_dir_env}[/yellow]")
            self.rich_console.print("[yellow]Copying MODEL_DIR contents for run phase...[/yellow]")
            
            # Copy docker/ and scripts/ from MODEL_DIR (without deleting existing ones first)
            for subdir in ["docker", "scripts"]:
                src_path = Path(model_dir_env) / subdir
                if src_path.exists():
                    dest_path = Path(subdir)
                    # Use copytree with dirs_exist_ok=True to merge instead of replace
                    if dest_path.exists():
                        # Only warn, don't delete existing directories
                        self.rich_console.print(f"[dim]  Note: Merging {subdir}/ from MODEL_DIR with existing directory[/dim]")
                    shutil.copytree(src_path, dest_path, dirs_exist_ok=True, ignore=ignore_cache_files)
            
            self.rich_console.print("[green]✓ MODEL_DIR structure copied (docker/, scripts/)[/green]")
        elif not os.path.exists(model_dir_env):
            self.rich_console.print(f"[yellow]⚠️  Warning: MODEL_DIR '{model_dir_env}' does not exist, using current directory[/yellow]")

        # Step 2: Copy madengine's common scripts (pre_scripts, post_scripts, tools)
        # This provides the execution framework scripts
        # Find madengine installation path (works for both development and installed package)
        madengine_common = None
        
        # Option 1: Development mode - check if running from source
        dev_path = Path("src/madengine/scripts/common")
        if dev_path.exists():
            madengine_common = dev_path
            print(f"Found madengine scripts in development mode: {madengine_common}")
        else:
            # Option 2: Installed package - find via module location
            try:
                import madengine
                madengine_module_path = Path(madengine.__file__).parent
                installed_path = madengine_module_path / "scripts" / "common"
                if installed_path.exists():
                    madengine_common = installed_path
                    print(f"Found madengine scripts in installed package: {madengine_common}")
            except Exception as e:
                print(f"Could not locate madengine scripts: {e}")
        
        if madengine_common and madengine_common.exists():
            print(f"Copying madengine common scripts from {madengine_common} to scripts/common")
            
            dest_common = Path("scripts/common")
            # Ensure the destination directory exists before copying
            dest_common.mkdir(parents=True, exist_ok=True)
            
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
                        shutil.copytree(src_item, dest_item, ignore=ignore_cache_files)
                    else:
                        shutil.copy2(src_item, dest_item)
                    print(f"  Copied {item}")
        else:
            self.rich_console.print("[yellow]⚠️  Could not find madengine scripts directory[/yellow]")

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
                shutil.copytree(common_scripts, dest, ignore=ignore_cache_files)
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

    def _filter_images_by_gpu_compatibility(
        self, built_images: Dict, runtime_gpu_vendor: str, runtime_gpu_arch: str
    ) -> Dict:
        """Filter images compatible with runtime GPU vendor and architecture.
        
        Args:
            built_images: Dictionary of built images from manifest
            runtime_gpu_vendor: Runtime GPU vendor (AMD, NVIDIA, NONE)
            runtime_gpu_arch: Runtime GPU architecture (gfx90a, sm_90, etc.)
            
        Returns:
            Dictionary of compatible images
        """
        compatible_images = {}

        for model_name, image_info in built_images.items():
            image_gpu_vendor = image_info.get("gpu_vendor", "")
            image_arch = image_info.get("gpu_architecture", "")

            # Legacy images without vendor info - treat as compatible for backward compatibility
            if not image_gpu_vendor:
                self.rich_console.print(
                    f"[yellow]  Warning: {model_name} has no gpu_vendor, treating as compatible (legacy)[/yellow]"
                )
                compatible_images[model_name] = image_info
                continue

            # Check GPU vendor compatibility first (most important)
            if runtime_gpu_vendor == "NONE" or image_gpu_vendor == runtime_gpu_vendor:
                # Vendor matches or CPU-only, check architecture if specified
                if image_arch:
                    # Architecture specified, must match
                    if image_arch == runtime_gpu_arch:
                        compatible_images[model_name] = image_info
                    else:
                        self.rich_console.print(
                            f"[dim]  Skipping {model_name}: architecture mismatch "
                            f"({image_arch} != {runtime_gpu_arch})[/dim]"
                        )
                else:
                    # No architecture specified, vendor match is enough
                    compatible_images[model_name] = image_info
            else:
                # Vendor mismatch
                self.rich_console.print(
                    f"[dim]  Skipping {model_name}: GPU vendor mismatch "
                    f"({image_gpu_vendor} != {runtime_gpu_vendor})[/dim]"
                )

        return compatible_images
    
    def _filter_images_by_gpu_architecture(
        self, built_images: Dict, runtime_gpu_arch: str
    ) -> Dict:
        """Legacy method for backward compatibility."""
        # Get runtime GPU vendor
        runtime_gpu_vendor = self.context.get_gpu_vendor() if self.context else "NONE"
        return self._filter_images_by_gpu_compatibility(
            built_images, runtime_gpu_vendor, runtime_gpu_arch
        )

    def _filter_images_by_skip_gpu_arch(
        self, built_images: Dict, built_models: Dict, runtime_gpu_arch: str
    ) -> Dict:
        """Filter out models that should skip the current GPU architecture.
        
        This implements the skip_gpu_arch logic from model definitions,
        where models can specify GPU architectures they don't support.
        
        Args:
            built_images: Dictionary of built images from manifest
            built_models: Dictionary of model metadata from manifest
            runtime_gpu_arch: Runtime GPU architecture (gfx90a, A100, etc.)
            
        Returns:
            Dictionary of images that should run (not skipped)
        """
        if getattr(self.args, 'disable_skip_gpu_arch', False):
            # User disabled skip logic, run all models
            self.rich_console.print("[dim]  --disable-skip-gpu-arch flag set, skipping GPU architecture checks[/dim]")
            return built_images
        
        compatible_images = {}
        
        for model_name, image_info in built_images.items():
            # Get model metadata to check skip_gpu_arch field
            model_info = built_models.get(model_name, {})
            skip_gpu_arch_str = model_info.get("skip_gpu_arch", "")
            
            if skip_gpu_arch_str:
                # Parse comma-separated list of architectures to skip
                skip_list = [arch.strip() for arch in skip_gpu_arch_str.split(",")]
                
                # Normalize architecture comparison (handle "NVIDIA A100" -> "A100")
                sys_gpu_arch = runtime_gpu_arch
                if sys_gpu_arch and "NVIDIA" in sys_gpu_arch:
                    sys_gpu_arch = sys_gpu_arch.split()[1]
                
                if sys_gpu_arch in skip_list:
                    self.rich_console.print(
                        f"[yellow]  Skipping model {model_name} as it is not supported on {runtime_gpu_arch} architecture.[/yellow]"
                    )
                    
                    # Write SKIPPED status to perf CSV
                    self._write_skipped_status(model_name, image_info, runtime_gpu_arch)
                    continue
            
            compatible_images[model_name] = image_info
        
        return compatible_images

    def _write_skipped_status(self, model_name: str, image_info: Dict, gpu_arch: str) -> None:
        """Write SKIPPED status to perf CSV for models that were skipped.
        
        Args:
            model_name: Name of the model that was skipped
            image_info: Image information dictionary
            gpu_arch: GPU architecture that caused the skip
        """
        try:
            from madengine.reporting.update_perf_csv import update_perf_csv
            import json
            import tempfile
            
            # Create a perf entry for the skipped model
            perf_entry = {
                "model": model_name,
                "status": "SKIPPED",
                "reason": f"Model not supported on {gpu_arch} architecture",
                "gpu_architecture": gpu_arch,
            }
            
            # Write to temporary JSON file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(perf_entry, f)
                temp_file = f.name
            
            # Get output CSV path from args
            output_csv = getattr(self.args, 'output', 'perf.csv')
            
            # Update perf CSV with skipped entry
            update_perf_csv(exception_result=temp_file, perf_csv=output_csv)
            
            # Clean up temp file
            import os
            os.unlink(temp_file)
            
        except Exception as e:
            self.rich_console.print(f"[dim]  Warning: Could not write SKIPPED status to CSV: {e}[/dim]")

    def _infer_deployment_target(self, config: Dict) -> str:
        """
        Infer deployment target from configuration structure.
        
        Convention over Configuration:
        - Presence of "k8s" or "kubernetes" field → k8s deployment
        - Presence of "slurm" field → slurm deployment
        - Neither present → local execution
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Deployment target: "k8s", "slurm", or "local"
        """
        if "k8s" in config or "kubernetes" in config:
            return "k8s"
        elif "slurm" in config:
            return "slurm"
        else:
            return "local"
    
    def _filter_images_by_dockerfile_context(self, built_images: Dict) -> Dict:
        """Filter images by dockerfile context matching runtime context.
        
        This implements the legacy behavior where dockerfiles are filtered
        at runtime based on their CONTEXT header matching the current runtime context.
        
        Args:
            built_images: Dictionary of built images from manifest
            
        Returns:
            Dictionary of images that match the runtime context
        """
        if not self.context:
            return built_images
        
        compatible_images = {}
        
        for image_name, image_info in built_images.items():
            dockerfile = image_info.get("dockerfile", "")
            
            if not dockerfile:
                # No dockerfile info, include by default (legacy compatibility)
                compatible_images[image_name] = image_info
                continue
            
            # Check if dockerfile exists
            if not os.path.exists(dockerfile):
                self.rich_console.print(
                    f"[dim]  Warning: Dockerfile {dockerfile} not found. Including by default.[/dim]"
                )
                compatible_images[image_name] = image_info
                continue
            
            # Read dockerfile context header
            try:
                dockerfile_context_str = self.console.sh(
                    f"head -n5 {dockerfile} | grep '# CONTEXT ' | sed 's/# CONTEXT //g'"
                ).strip()
                
                if not dockerfile_context_str:
                    # No context header, include by default
                    compatible_images[image_name] = image_info
                    continue
                
                # Create a dict with this dockerfile and its context
                dockerfile_dict = {dockerfile: dockerfile_context_str}
                
                # Use context.filter() to check if this dockerfile matches runtime context
                filtered = self.context.filter(dockerfile_dict)
                
                if filtered:
                    # Dockerfile matches runtime context
                    compatible_images[image_name] = image_info
                else:
                    self.rich_console.print(
                        f"[dim]  Skipping {image_name}: dockerfile context doesn't match runtime context[/dim]"
                    )
                    
            except Exception as e:
                # If we can't read the dockerfile, include it by default
                self.rich_console.print(
                    f"[dim]  Warning: Could not read context for {dockerfile}: {e}. Including by default.[/dim]"
                )
                compatible_images[image_name] = image_info
        
        return compatible_images

