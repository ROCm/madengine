#!/usr/bin/env python3
"""
Distributed Runner Orchestrator for MADEngine

This module provides orchestration capabilities for distributed execution
scenarios like Ansible or Kubernetes, where Docker image building and
container execution are separated across different nodes.
"""

import os
import json
import typing
from madengine.core.console import Console
from madengine.core.context import Context
from madengine.core.dataprovider import Data
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
        self.console = Console(live_output=getattr(args, 'live_output', True))
        
        # Initialize context with appropriate mode
        self.context = Context(
            additional_context=getattr(args, 'additional_context', None),
            additional_context_file=getattr(args, 'additional_context_file', None),
            build_only_mode=build_only_mode
        )
        
        # Initialize data provider if data config exists
        # No longer support top-level registry; use per-image registry if present
        if registry:
            print(f"Using registry from CLI: {registry}")
        else:
            print("No registry specified via CLI; will use per-image registry from built_images if present.")
        
        # Load credentials
        self.credentials = None
        try:
            credential_file = "credential.json"
            if os.path.exists(credential_file):
                with open(credential_file) as f:
                    self.credentials = json.load(f)
                print(f"Credentials: {list(self.credentials.keys())}")
        except Exception as e:
            print(f"Warning: Could not load credentials: {e}")
        
        # Check for Docker Hub environment variables and override credentials
        docker_hub_user = None
        docker_hub_password = None
        docker_hub_repo = None

        if 'MAD_DOCKERHUB_USER' in os.environ:
            docker_hub_user = os.environ['MAD_DOCKERHUB_USER']
        if 'MAD_DOCKERHUB_PASSWORD' in os.environ:
            docker_hub_password = os.environ['MAD_DOCKERHUB_PASSWORD']
        if 'MAD_DOCKERHUB_REPO' in os.environ:
            docker_hub_repo = os.environ['MAD_DOCKERHUB_REPO']
        
        if docker_hub_user and docker_hub_password:
            print("Found Docker Hub credentials in environment variables")
            if self.credentials is None:
                self.credentials = {}
            
            # Override or add Docker Hub credentials
            self.credentials['dockerhub'] = {
                'repository': docker_hub_repo,
                'username': docker_hub_user,
                'password': docker_hub_password
            }
            print("Docker Hub credentials updated from environment variables")
            print(f"Docker Hub credentials: {self.credentials['dockerhub']}")
    
    def build_phase(self, registry: str = None, clean_cache: bool = False, 
                   manifest_output: str = "build_manifest.json", batch_build_metadata: typing.Optional[dict] = None) -> typing.Dict:
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
        print("=" * 60)
        print("STARTING BUILD PHASE")
        if self.context._build_only_mode:
            print("(Build-only mode - no GPU detection)")
        print("=" * 60)
        
        print(f"Building models with args {self.args}")
        
        # Discover models
        discover_models = DiscoverModels(args=self.args)
        models = discover_models.run()
        
        print(f"Discovered {len(models)} models to build")
        
        # Copy scripts for building
        self._copy_scripts()
        
        # Validate build context for build-only mode
        if self.context._build_only_mode:
            if "MAD_SYSTEM_GPU_ARCHITECTURE" not in self.context.ctx["docker_build_arg"]:
                print("Warning: MAD_SYSTEM_GPU_ARCHITECTURE not provided in build context.")
                print("For build-only nodes, please provide GPU architecture via --additional-context:")
                print('  --additional-context \'{"docker_build_arg": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx908"}}\'')
        
        # Initialize builder
        builder = DockerBuilder(self.context, self.console, live_output=getattr(self.args, 'live_output', False))
        
        # Determine phase suffix for log files
        phase_suffix = ".build" if hasattr(self.args, '_separate_phases') and self.args._separate_phases else ""
        
        # If batch_build_metadata is provided, use it to set per-model registry/registry_image
        build_summary = builder.build_all_models(
            models,
            self.credentials,
            clean_cache,
            registry,
            phase_suffix,
            batch_build_metadata=batch_build_metadata
        )
        
        # Export build manifest with registry information
        builder.export_build_manifest(manifest_output, registry)
        
        print("=" * 60)
        print("BUILD PHASE COMPLETED")
        print(f"  Successful builds: {len(build_summary['successful_builds'])}")
        print(f"  Failed builds: {len(build_summary['failed_builds'])}")
        print(f"  Total build time: {build_summary['total_build_time']:.2f} seconds")
        print(f"  Manifest saved to: {manifest_output}")
        print("=" * 60)
        
        # Cleanup scripts
        self.cleanup()
        
        return build_summary
    
    def run_phase(self, manifest_file: str = "build_manifest.json", 
                 registry: str = None, timeout: int = 7200,
                 keep_alive: bool = False) -> typing.Dict:
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
        print("=" * 60)
        print("STARTING RUN PHASE")
        print("=" * 60)

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
            print("ERROR: Unable to detect host OS.")
        
        # Load build manifest
        if not os.path.exists(manifest_file):
            raise FileNotFoundError(f"Build manifest not found: {manifest_file}")
        
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        print(f"Loaded manifest with {len(manifest['built_images'])} images")
        
        # Auto-detect registry from manifest if not provided via CLI
        if not registry and "registry" in manifest:
            manifest_registry = manifest["registry"]
            if manifest_registry and manifest_registry.strip():  # Check for non-empty string
                registry = manifest_registry
                print(f"Auto-detected registry from manifest: {registry}")
            else:
                print("Manifest registry is empty, will use local images only")
        elif registry:
            print(f"Using registry from CLI: {registry}")
        else:
            print("No registry specified, will use local images only")
        
        # Copy scripts for running
        self._copy_scripts()
        
        # Initialize runner
        runner = ContainerRunner(self.context, self.data, self.console, live_output=getattr(self.args, 'live_output', False))
        runner.set_credentials(self.credentials)
        
        # Set perf.csv output path if specified in args
        if hasattr(self.args, 'output') and self.args.output:
            runner.set_perf_csv_path(self.args.output)
        
        # Determine phase suffix for log files
        phase_suffix = ".run" if hasattr(self.args, '_separate_phases') and self.args._separate_phases else ""
        
        # Use built models from manifest if available, otherwise discover models
        if "built_models" in manifest and manifest["built_models"]:
            print("Using model information from build manifest")
            models = list(manifest["built_models"].values())
        else:
            print("No model information in manifest, discovering models from current configuration")
            # Discover models (to get execution parameters)
            discover_models = DiscoverModels(args=self.args)
            models = discover_models.run()
        
        # Create execution summary
        execution_summary = {
            "successful_runs": [],
            "failed_runs": [],
            "total_execution_time": 0
        }
        
        # Map models to their built images
        if "built_models" in manifest and manifest["built_models"]:
            # Direct mapping from manifest - built_models maps image_name -> model_info
            print("Using direct model-to-image mapping from manifest")
            for image_name, build_info in manifest["built_images"].items():
                if image_name in manifest["built_models"]:
                    model_info = manifest["built_models"][image_name]
                    try:
                        print(f"\nRunning model {model_info['name']} with image {image_name}")
                        
                        # Handle registry image pulling and tagging according to manifest
                        if "registry_image" in build_info:
                            # Registry image exists - pull it and tag as docker_image, then run with docker_image
                            registry_image = build_info["registry_image"]
                            docker_image = build_info["docker_image"]
                            
                            # Extract registry from the registry_image format
                            effective_registry = registry
                            if not effective_registry and registry_image:
                                registry_parts = registry_image.split('/')
                                if len(registry_parts) > 1 and '.' in registry_parts[0]:
                                    effective_registry = registry_parts[0]
                                elif registry_image.startswith('docker.io/') or '/' in registry_image:
                                    effective_registry = "docker.io"
                            
                            if effective_registry:
                                print(f"Pulling image from registry: {registry_image}")
                                try:
                                    # Ensure all parameters are strings and credentials is properly formatted
                                    registry_image_str = str(registry_image) if registry_image else ""
                                    docker_image_str = str(docker_image) if docker_image else ""
                                    effective_registry_str = str(effective_registry) if effective_registry else ""
                                    
                                    # Pull registry image and tag it as docker_image
                                    runner.pull_image(registry_image_str, docker_image_str, effective_registry_str, self.credentials)
                                    actual_image = docker_image_str
                                    print(f"Successfully pulled and tagged as: {docker_image_str}")
                                except Exception as e:
                                    print(f"Failed to pull from registry, falling back to local image: {e}")
                                    actual_image = docker_image
                            else:
                                # Registry image exists but no valid registry found, try to pull as-is and tag
                                print(f"Attempting to pull registry image as-is: {registry_image}")
                                try:
                                    registry_image_str = str(registry_image) if registry_image else ""
                                    docker_image_str = str(docker_image) if docker_image else ""
                                    runner.pull_image(registry_image_str, docker_image_str)
                                    actual_image = docker_image_str
                                    print(f"Successfully pulled and tagged as: {docker_image_str}")
                                except Exception as e:
                                    print(f"Failed to pull from registry, falling back to local image: {e}")
                                    actual_image = docker_image
                        else:
                            # No registry_image key - run container directly using docker_image
                            actual_image = build_info["docker_image"]
                            print(f"No registry image specified, using local image: {actual_image}")
                        
                        # Run the container
                        run_results = runner.run_container(
                            model_info, actual_image, build_info, 
                            keep_alive=keep_alive, timeout=timeout, phase_suffix=phase_suffix,
                            generate_sys_env_details=getattr(self.args, 'generate_sys_env_details', True)
                        )
                        
                        # Add to appropriate list based on actual status
                        if run_results.get("status") == "SUCCESS":
                            execution_summary["successful_runs"].append(run_results)
                            print(f"Successfully completed: {model_info['name']} -> {run_results['status']}")
                        else:
                            execution_summary["failed_runs"].append(run_results)
                            print(f"Failed to complete: {model_info['name']} -> {run_results['status']}")
                        
                        execution_summary["total_execution_time"] += run_results.get("test_duration", 0)
                        
                    except Exception as e:
                        print(f"Failed to run {model_info['name']} with image {image_name}: {e}")
                        execution_summary["failed_runs"].append({
                            "model": model_info['name'],
                            "image": image_name,
                            "error": str(e)
                        })
                else:
                    print(f"Warning: No model info found for built image: {image_name}")
        else:
            # Fallback to name-based matching for backward compatibility
            print("Using name-based matching (fallback mode)")
            for model_info in models:
                model_name = model_info["name"]
                
                # Find matching built images for this model
                matching_images = []
                for image_name, build_info in manifest["built_images"].items():
                    if model_name.replace("/", "_").lower() in image_name:
                        matching_images.append((image_name, build_info))
                
                if not matching_images:
                    print(f"No built images found for model: {model_name}")
                    execution_summary["failed_runs"].append({
                        "model": model_name,
                        "error": "No built images found"
                    })
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
                                registry_parts = registry_image.split('/')
                                if len(registry_parts) > 1 and '.' in registry_parts[0]:
                                    effective_registry = registry_parts[0]
                                elif registry_image.startswith('docker.io/') or '/' in registry_image:
                                    effective_registry = "docker.io"
                            
                            if effective_registry:
                                print(f"Pulling image from registry: {registry_image}")
                                try:
                                    # Ensure all parameters are strings and credentials is properly formatted
                                    registry_image_str = str(registry_image) if registry_image else ""
                                    docker_image_str = str(docker_image) if docker_image else ""
                                    effective_registry_str = str(effective_registry) if effective_registry else ""
                                    
                                    # Pull registry image and tag it as docker_image
                                    runner.pull_image(registry_image_str, docker_image_str, effective_registry_str, self.credentials)
                                    actual_image = docker_image_str
                                    print(f"Successfully pulled and tagged as: {docker_image_str}")
                                except Exception as e:
                                    print(f"Failed to pull from registry, falling back to local image: {e}")
                                    actual_image = docker_image
                            else:
                                # Registry image exists but no valid registry found, try to pull as-is and tag
                                print(f"Attempting to pull registry image as-is: {registry_image}")
                                try:
                                    registry_image_str = str(registry_image) if registry_image else ""
                                    docker_image_str = str(docker_image) if docker_image else ""
                                    runner.pull_image(registry_image_str, docker_image_str)
                                    actual_image = docker_image_str
                                    print(f"Successfully pulled and tagged as: {docker_image_str}")
                                except Exception as e:
                                    print(f"Failed to pull from registry, falling back to local image: {e}")
                                    actual_image = docker_image
                        else:
                            # No registry_image key - run container directly using docker_image
                            actual_image = build_info["docker_image"]
                            print(f"No registry image specified, using local image: {actual_image}")
                        
                        # Run the container
                        run_results = runner.run_container(
                            model_info, actual_image, build_info, 
                            keep_alive=keep_alive, timeout=timeout, phase_suffix=phase_suffix,
                            generate_sys_env_details=getattr(self.args, 'generate_sys_env_details', True)
                        )
                        
                        # Add to appropriate list based on actual status
                        if run_results.get("status") == "SUCCESS":
                            execution_summary["successful_runs"].append(run_results)
                            print(f"Successfully completed: {model_name} -> {run_results['status']}")
                        else:
                            execution_summary["failed_runs"].append(run_results)
                            print(f"Failed to complete: {model_name} -> {run_results['status']}")
                        
                        execution_summary["total_execution_time"] += run_results.get("test_duration", 0)
                        
                    except Exception as e:
                        print(f"Failed to run {model_name} with image {image_name}: {e}")
                        execution_summary["failed_runs"].append({
                            "model": model_name,
                            "image": image_name,
                            "error": str(e)
                        })
        
        print("=" * 60)
        print("RUN PHASE COMPLETED")
        print(f"  Successful runs: {len(execution_summary['successful_runs'])}")
        print(f"  Failed runs: {len(execution_summary['failed_runs'])}")
        print(f"  Total execution time: {execution_summary['total_execution_time']:.2f} seconds")
        print("=" * 60)
        
        # Convert output CSV to HTML like run_models.py does
        try:
            from madengine.tools.csv_to_html import convert_csv_to_html
            perf_csv_path = getattr(self.args, 'output', 'perf.csv')
            if os.path.exists(perf_csv_path):
                print("Converting output csv to html...")
                convert_csv_to_html(file_path=perf_csv_path)
        except Exception as e:
            print(f"Warning: Could not convert CSV to HTML: {e}")
        
        # Cleanup scripts
        self.cleanup()
        
        return execution_summary
    
    def full_workflow(self, registry: str = None, clean_cache: bool = False,
                     timeout: int = 7200, keep_alive: bool = False) -> typing.Dict:
        """Execute the complete workflow: build then run.
        
        Args:
            registry: Optional registry for image distribution
            clean_cache: Whether to use --no-cache for builds
            timeout: Execution timeout per model
            keep_alive: Whether to keep containers alive after execution
            
        Returns:
            dict: Complete workflow summary
        """
        print("=" * 80)
        print("STARTING COMPLETE DISTRIBUTED WORKFLOW")
        print("=" * 80)
        
        # Build phase
        build_summary = self.build_phase(registry, clean_cache)
        
        # Run phase
        execution_summary = self.run_phase(timeout=timeout, keep_alive=keep_alive)
        
        # Combine summaries
        workflow_summary = {
            "build_phase": build_summary,
            "run_phase": execution_summary,
            "overall_success": (
                len(build_summary["failed_builds"]) == 0 and
                len(execution_summary["failed_runs"]) == 0
            )
        }
        
        print("=" * 80)
        print("COMPLETE WORKFLOW FINISHED")
        print(f"  Overall success: {workflow_summary['overall_success']}")
        print("=" * 80)
        
        return workflow_summary
    
    def _copy_scripts(self) -> None:
        """Copy scripts to the current directory."""
        scripts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
        print(f"Package path: {scripts_path}")
        # copy the scripts to the model directory
        self.console.sh(f"cp -vLR --preserve=all {scripts_path} .")
        print(f"Scripts copied to {os.getcwd()}/scripts")
    
    def cleanup(self) -> None:
        """Cleanup the scripts/common directory."""
        # check the directory exists
        if os.path.exists("scripts/common"):
            # check tools.json exists in scripts/common directory
            if os.path.exists("scripts/common/tools.json"):
                # remove the scripts/common/tools.json file
                # Use force removal and handle permission errors gracefully
                try:
                    self.console.sh("rm -rf scripts/common/tools")
                except RuntimeError:
                    # If normal removal fails due to permissions, try with force
                    self.console.sh("chmod -R u+w scripts/common/tools 2>/dev/null || true")
                    self.console.sh("rm -rf scripts/common/tools || true")
            # check test_echo.sh exists in scripts/common directory
            if os.path.exists("scripts/common/test_echo.sh"):
                # remove the scripts/common/test_echo.sh file
                self.console.sh("rm -rf scripts/common/test_echo.sh")
            # check folder pre_scripts exists in scripts/common directory
            if os.path.exists("scripts/common/pre_scripts"):
                # remove the scripts/common/pre_scripts directory
                self.console.sh("rm -rf scripts/common/pre_scripts")
            # check folder post_scripts exists in scripts/common directory
            if os.path.exists("scripts/common/post_scripts"):
                # remove the scripts/common/post_scripts directory
                self.console.sh("rm -rf scripts/common/post_scripts")
            if os.path.exists("scripts/common/tools"):
                # remove the scripts/common/tools directory
                self.console.sh("rm -rf scripts/common/tools")
            print(f"scripts/common directory has been cleaned up.")


