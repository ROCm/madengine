#!/usr/bin/env python3
"""
Docker Image Builder Module for MADEngine

This module handles the Docker image building phase separately from execution,
enabling distributed workflows where images are built on a central host
and then distributed to remote nodes for execution.
"""

import os
import time
import json
import re
import typing
from contextlib import redirect_stdout, redirect_stderr
from rich.console import Console as RichConsole
from madengine.core.console import Console
from madengine.core.context import Context
from madengine.utils.ops import PythonicTee


class DockerBuilder:
    """Class responsible for building Docker images for models."""

    # GPU architecture variables used in MAD/DLM Dockerfiles
    GPU_ARCH_VARIABLES = [
        "MAD_SYSTEM_GPU_ARCHITECTURE",
        "PYTORCH_ROCM_ARCH", 
        "GPU_TARGETS",
        "GFX_COMPILATION_ARCH",
        "GPU_ARCHS"
    ]

    def __init__(
        self, context: Context, console: Console = None, live_output: bool = False
    ):
        """Initialize the Docker Builder.

        Args:
            context: The MADEngine context
            console: Optional console instance
            live_output: Whether to show live output
        """
        self.context = context
        self.console = console or Console(live_output=live_output)
        self.live_output = live_output
        self.rich_console = RichConsole()
        self.built_images = {}  # Track built images
        self.built_models = {}  # Track built models

    def get_context_path(self, info: typing.Dict) -> str:
        """Get the context path for Docker build.

        Args:
            info: The model info dict.

        Returns:
            str: The context path.
        """
        if "dockercontext" in info and info["dockercontext"] != "":
            return info["dockercontext"]
        else:
            return "./docker"

    def get_build_arg(self, run_build_arg: typing.Dict = {}) -> str:
        """Get the build arguments.

        Args:
            run_build_arg: The run build arguments.

        Returns:
            str: The build arguments.
        """
        if not run_build_arg and "docker_build_arg" not in self.context.ctx:
            return ""

        build_args = ""
        for build_arg in self.context.ctx["docker_build_arg"].keys():
            build_args += (
                "--build-arg "
                + build_arg
                + "='"
                + self.context.ctx["docker_build_arg"][build_arg]
                + "' "
            )

        if run_build_arg:
            for key, value in run_build_arg.items():
                build_args += "--build-arg " + key + "='" + value + "' "

        return build_args

    def build_image(
        self,
        model_info: typing.Dict,
        dockerfile: str,
        credentials: typing.Dict = None,
        clean_cache: bool = False,
        phase_suffix: str = "",
        additional_build_args: typing.Dict[str, str] = None,
        override_image_name: str = None,
    ) -> typing.Dict:
        """Build a Docker image for the given model.

        Args:
            model_info: The model information dictionary
            dockerfile: Path to the Dockerfile
            credentials: Optional credentials dictionary
            clean_cache: Whether to use --no-cache
            phase_suffix: Suffix for log file name (e.g., ".build" or "")
            additional_build_args: Additional build arguments to pass to Docker
            override_image_name: Override the generated image name

        Returns:
            dict: Build information including image name, build duration, etc.
        """
        # Generate image name first
        if override_image_name:
            docker_image = override_image_name
        else:
            image_docker_name = (
                model_info["name"].replace("/", "_").lower()
                + "_"
                + os.path.basename(dockerfile).replace(".Dockerfile", "")
            )
            docker_image = "ci-" + image_docker_name

        # Create log file for this build
        cur_docker_file_basename = os.path.basename(dockerfile).replace(
            ".Dockerfile", ""
        )
        log_file_path = (
            model_info["name"].replace("/", "_")
            + "_"
            + cur_docker_file_basename
            + phase_suffix
            + ".live.log"
        )
        # Replace / with _ in log file path (already done above, but keeping for safety)
        log_file_path = log_file_path.replace("/", "_")

        self.rich_console.print(f"\n[bold green]🔨 Starting Docker build for model:[/bold green] [bold cyan]{model_info['name']}[/bold cyan]")
        print(f"📁 Dockerfile: {dockerfile}")
        print(f"🏷️  Target image: {docker_image}")
        print(f"📝 Build log: {log_file_path}")
        self.rich_console.print(f"[dim]{'='*80}[/dim]")

        # Get docker context
        docker_context = self.get_context_path(model_info)

        # Prepare build args
        run_build_arg = {}
        if "cred" in model_info and model_info["cred"] != "" and credentials:
            if model_info["cred"] not in credentials:
                raise RuntimeError(
                    f"Credentials({model_info['cred']}) not found for model {model_info['name']}"
                )
            # Add cred to build args
            for key_cred, value_cred in credentials[model_info["cred"]].items():
                run_build_arg[model_info["cred"] + "_" + key_cred.upper()] = value_cred

        # Add additional build args if provided (for multi-architecture builds)
        if additional_build_args:
            run_build_arg.update(additional_build_args)

        build_args = self.get_build_arg(run_build_arg)

        use_cache_str = "--no-cache" if clean_cache else ""

        # Build the image with logging
        build_start_time = time.time()

        build_command = (
            f"docker build {use_cache_str} --network=host "
            f"-t {docker_image} --pull -f {dockerfile} "
            f"{build_args} {docker_context}"
        )

        # Execute build with log redirection
        with open(log_file_path, mode="w", buffering=1) as outlog:
            with redirect_stdout(
                PythonicTee(outlog, self.live_output)
            ), redirect_stderr(PythonicTee(outlog, self.live_output)):
                print(f"🔨 Executing build command...")
                self.console.sh(build_command, timeout=None)

                build_duration = time.time() - build_start_time

                print(f"⏱️  Build Duration: {build_duration:.2f} seconds")
                print(f"🏷️  MAD_CONTAINER_IMAGE is {docker_image}")
                self.rich_console.print(f"[bold green]✅ Docker build completed successfully[/bold green]")
                self.rich_console.print(f"[dim]{'='*80}[/dim]")

                # Get base docker info
                base_docker = ""
                if (
                    "docker_build_arg" in self.context.ctx
                    and "BASE_DOCKER" in self.context.ctx["docker_build_arg"]
                ):
                    base_docker = self.context.ctx["docker_build_arg"]["BASE_DOCKER"]
                else:
                    base_docker = self.console.sh(
                        f"grep '^ARG BASE_DOCKER=' {dockerfile} | sed -E 's/ARG BASE_DOCKER=//g'"
                    )

                print(f"BASE DOCKER is {base_docker}")

                # Get docker SHA
                docker_sha = ""
                try:
                    docker_sha = self.console.sh(
                        f'docker manifest inspect {base_docker} | grep digest | head -n 1 | cut -d \\" -f 4'
                    )
                    print(f"BASE DOCKER SHA is {docker_sha}")
                except Exception as e:
                    self.rich_console.print(f"[yellow]Warning: Could not get docker SHA: {e}[/yellow]")

        # Infer GPU vendor from dockerfile path
        gpu_vendor = self._infer_gpu_vendor_from_dockerfile(dockerfile)
        
        build_info = {
            "model": model_info["name"],
            "docker_image": docker_image,
            "dockerfile": dockerfile,
            "base_docker": base_docker,
            "docker_sha": docker_sha,
            "build_duration": build_duration,
            "build_command": build_command,
            "log_file": log_file_path,
            "gpu_vendor": gpu_vendor,  # Add GPU vendor for filtering
        }

        # Store built image info
        self.built_images[docker_image] = build_info

        # Store model info linked to the built image
        self.built_models[docker_image] = model_info

        self.rich_console.print(f"[bold green]Successfully built image:[/bold green] [cyan]{docker_image}[/cyan]")

        return build_info

    def login_to_registry(self, registry: str, credentials: typing.Dict = None) -> None:
        """Login to a Docker registry.

        Args:
            registry: Registry URL (e.g., "localhost:5000", "docker.io", or empty for DockerHub)
            credentials: Optional credentials dictionary containing username/password
        """
        if not credentials:
            print("No credentials provided for registry login")
            return

        # Check if registry credentials are available
        registry_key = registry if registry else "dockerhub"

        # Handle docker.io as dockerhub
        if registry and registry.lower() == "docker.io":
            registry_key = "dockerhub"

        if registry_key not in credentials:
            error_msg = f"No credentials found for registry: {registry_key}"
            if registry_key == "dockerhub":
                error_msg += f"\nPlease add dockerhub credentials to credential.json:\n"
                error_msg += "{\n"
                error_msg += '  "dockerhub": {\n'
                error_msg += '    "repository": "your-repository",\n'
                error_msg += '    "username": "your-dockerhub-username",\n'
                error_msg += '    "password": "your-dockerhub-password-or-token"\n'
                error_msg += "  }\n"
                error_msg += "}"
            else:
                error_msg += (
                    f"\nPlease add {registry_key} credentials to credential.json:\n"
                )
                error_msg += "{\n"
                error_msg += f'  "{registry_key}": {{\n'
                error_msg += f'    "repository": "your-repository",\n'
                error_msg += f'    "username": "your-{registry_key}-username",\n'
                error_msg += f'    "password": "your-{registry_key}-password"\n'
                error_msg += "  }\n"
                error_msg += "}"
            self.rich_console.print(f"[red]{error_msg}[/red]")
            raise RuntimeError(error_msg)

        creds = credentials[registry_key]

        if "username" not in creds or "password" not in creds:
            error_msg = f"Invalid credentials format for registry: {registry_key}"
            error_msg += f"\nCredentials must contain 'username' and 'password' fields"
            self.rich_console.print(f"[red]{error_msg}[/red]")
            raise RuntimeError(error_msg)

        # Ensure credential values are strings
        username = str(creds["username"])
        password = str(creds["password"])

        # Perform docker login
        login_command = f"echo '{password}' | docker login"

        if registry and registry.lower() not in ["docker.io", "dockerhub"]:
            login_command += f" {registry}"

        login_command += f" --username {username} --password-stdin"

        try:
            self.console.sh(login_command, secret=True)
            self.rich_console.print(f"[green]✅ Successfully logged in to registry: {registry or 'DockerHub'}[/green]")
        except Exception as e:
            self.rich_console.print(f"[red]❌ Failed to login to registry {registry}: {e}[/red]")
            raise

    def push_image(
        self,
        docker_image: str,
        registry: str = None,
        credentials: typing.Dict = None,
        explicit_registry_image: str = None,
    ) -> str:
        """Push the built image to a registry.

        Args:
            docker_image: The local docker image name
            registry: Optional registry URL (e.g., "localhost:5000", "docker.io", or empty for DockerHub)
            credentials: Optional credentials dictionary for registry authentication

        Returns:
            str: The full registry image name
        """
        if not registry:
            print(f"No registry specified, image remains local: {docker_image}")
            return docker_image

        # Login to registry if credentials are provided
        if credentials:
            self.login_to_registry(registry, credentials)

        # Determine registry image name (this should match what was already determined)
        if explicit_registry_image:
            registry_image = explicit_registry_image
        else:
            registry_image = self._determine_registry_image_name(
                docker_image, registry, credentials
            )

        try:
            # Tag the image if different from local name
            if registry_image != docker_image:
                print(f"Tagging image: docker tag {docker_image} {registry_image}")
                tag_command = f"docker tag {docker_image} {registry_image}"
                self.console.sh(tag_command)
            else:
                print(
                    f"No tag needed, docker_image and registry_image are the same: {docker_image}"
                )

            # Push the image
            push_command = f"docker push {registry_image}"
            self.rich_console.print(f"\n[bold blue]🚀 Starting docker push to registry...[/bold blue]")
            print(f"📤 Registry: {registry}")
            print(f"🏷️  Image: {registry_image}")
            self.console.sh(push_command)

            self.rich_console.print(f"[bold green]✅ Successfully pushed image to registry:[/bold green] [cyan]{registry_image}[/cyan]")
            self.rich_console.print(f"[dim]{'='*80}[/dim]")
            return registry_image

        except Exception as e:
            self.rich_console.print(f"[red]❌ Failed to push image {docker_image} to registry {registry}: {e}[/red]")
            raise

    def export_build_manifest(
        self,
        output_file: str = "build_manifest.json",
        registry: str = None,
        batch_build_metadata: typing.Optional[dict] = None,
    ) -> None:
        """Export enhanced build information to a manifest file.

        This creates a comprehensive build manifest that includes all necessary
        information for deployment, reducing the need for separate execution configs.

        Args:
            output_file: Path to output manifest file
            registry: Registry used for building (added to each image entry)
            batch_build_metadata: Optional metadata for batch builds
        """
        # Extract credentials from models
        credentials_required = list(
            set(
                [
                    model.get("cred", "")
                    for model in self.built_models.values()
                    if model.get("cred", "") != ""
                ]
            )
        )

        # Set registry for each built image
        for image_name, build_info in self.built_images.items():
            # If registry is not set in build_info, set it from argument
            if registry:
                build_info["registry"] = registry

            # If registry is set in batch_build_metadata, override it
            docker_file = build_info.get("dockerfile", "")
            truncated_docker_file = docker_file.split("/")[-1].split(".Dockerfile")[0]
            model_name = (
                image_name.split("ci-")[1].split(truncated_docker_file)[0].rstrip("_")
            )
            if batch_build_metadata and model_name in batch_build_metadata:
                self.rich_console.print(
                    f"[yellow]Overriding registry for {model_name} from batch_build_metadata[/yellow]"
                )
                build_info["registry"] = batch_build_metadata[model_name].get(
                    "registry"
                )

        manifest = {
            "built_images": self.built_images,
            "built_models": self.built_models,
            "context": {
                "docker_env_vars": self.context.ctx.get("docker_env_vars", {}),
                "docker_mounts": self.context.ctx.get("docker_mounts", {}),
                "docker_build_arg": self.context.ctx.get("docker_build_arg", {}),
                "gpu_vendor": self.context.ctx.get("gpu_vendor", ""),
                "docker_gpus": self.context.ctx.get("docker_gpus", ""),
            },
            "credentials_required": credentials_required,
        }
        
        # Preserve tools configuration if present in context
        if "tools" in self.context.ctx:
            manifest["context"]["tools"] = self.context.ctx["tools"]
        
        # Preserve pre/post scripts if present in context
        if "pre_scripts" in self.context.ctx:
            manifest["context"]["pre_scripts"] = self.context.ctx["pre_scripts"]
        if "post_scripts" in self.context.ctx:
            manifest["context"]["post_scripts"] = self.context.ctx["post_scripts"]
        if "encapsulate_script" in self.context.ctx:
            manifest["context"]["encapsulate_script"] = self.context.ctx["encapsulate_script"]

        # Add multi-node args to context if present
        if "build_multi_node_args" in self.context.ctx:
            manifest["context"]["multi_node_args"] = self.context.ctx[
                "build_multi_node_args"
            ]

        # Add push failure summary if any pushes failed
        push_failures = []
        for image_name, build_info in self.built_images.items():
            if "push_failed" in build_info and build_info["push_failed"]:
                push_failures.append(
                    {
                        "image": image_name,
                        "intended_registry_image": build_info.get("registry_image"),
                        "error": build_info.get("push_error"),
                    }
                )

        if push_failures:
            manifest["push_failures"] = push_failures

        with open(output_file, "w") as f:
            json.dump(manifest, f, indent=2)

        self.rich_console.print(f"[green]Build manifest exported to:[/green] {output_file}")
        if push_failures:
            self.rich_console.print(f"[yellow]Warning: {len(push_failures)} image(s) failed to push to registry[/yellow]")
            for failure in push_failures:
                self.rich_console.print(
                    f"[red]  - {failure['image']} -> {failure['intended_registry_image']}: {failure['error']}[/red]"
                )

    def build_all_models(
        self,
        models: typing.List[typing.Dict],
        credentials: typing.Dict = None,
        clean_cache: bool = False,
        registry: str = None,
        phase_suffix: str = "",
        batch_build_metadata: typing.Optional[dict] = None,
        target_archs: typing.List[str] = None,  # New parameter
    ) -> typing.Dict:
        """Build images for all models, with optional multi-architecture support.

        Args:
            models: List of model information dictionaries
            credentials: Optional credentials dictionary
            clean_cache: Whether to use --no-cache
            registry: Optional registry to push images to
            phase_suffix: Suffix for log file name (e.g., ".build" or "")
            batch_build_metadata: Optional batch build metadata
            target_archs: Optional list of target GPU architectures for multi-arch builds

        Returns:
            dict: Summary of all built images
        """
        self.rich_console.print(f"[bold blue]Building Docker images for {len(models)} models...[/bold blue]")
        
        if target_archs:
            self.rich_console.print(f"[bold cyan]Multi-architecture build mode enabled for: {', '.join(target_archs)}[/bold cyan]")
        else:
            self.rich_console.print(f"[bold cyan]Single architecture build mode[/bold cyan]")

        build_summary = {
            "successful_builds": [],
            "failed_builds": [],
            "total_build_time": 0,
            "successful_pushes": [],
            "failed_pushes": [],
        }
        
        for model_info in models:
            # Check if MAD_SYSTEM_GPU_ARCHITECTURE is provided in additional_context
            # This overrides --target-archs and uses default flow
            if ("docker_build_arg" in self.context.ctx and 
                "MAD_SYSTEM_GPU_ARCHITECTURE" in self.context.ctx["docker_build_arg"]):
                self.rich_console.print(f"[yellow]Info: MAD_SYSTEM_GPU_ARCHITECTURE provided in additional_context, "
                      f"disabling --target-archs and using default flow for model {model_info['name']}[/yellow]")
                # Use single architecture build mode regardless of target_archs
                try:
                    single_build_info = self._build_model_single_arch(
                        model_info, credentials, clean_cache, 
                        registry, phase_suffix, batch_build_metadata
                    )
                    build_summary["successful_builds"].extend(single_build_info)
                    build_summary["total_build_time"] += sum(
                        info.get("build_duration", 0) for info in single_build_info
                    )
                except Exception as e:
                    build_summary["failed_builds"].append({
                        "model": model_info["name"],
                        "error": str(e)
                    })
            elif target_archs:
                # Multi-architecture build mode - always use architecture suffix
                for arch in target_archs:
                    try:
                        # Always build with architecture suffix when --target-archs is used
                        arch_build_info = self._build_model_for_arch(
                            model_info, arch, credentials, clean_cache, 
                            registry, phase_suffix, batch_build_metadata
                        )
                        
                        build_summary["successful_builds"].extend(arch_build_info)
                        build_summary["total_build_time"] += sum(
                            info.get("build_duration", 0) for info in arch_build_info
                        )
                    except Exception as e:
                        build_summary["failed_builds"].append({
                            "model": model_info["name"],
                            "architecture": arch,
                            "error": str(e)
                        })
            else:
                # Single architecture build mode (existing behavior - no validation needed)
                try:
                    single_build_info = self._build_model_single_arch(
                        model_info, credentials, clean_cache, 
                        registry, phase_suffix, batch_build_metadata
                    )
                    build_summary["successful_builds"].extend(single_build_info)
                    build_summary["total_build_time"] += sum(
                        info.get("build_duration", 0) for info in single_build_info
                    )
                except Exception as e:
                    build_summary["failed_builds"].append({
                        "model": model_info["name"],
                        "error": str(e)
                    })
        
        return build_summary

    def _check_dockerfile_has_gpu_variables(self, model_info: typing.Dict) -> typing.Tuple[bool, str]:
        """
        Check if model's Dockerfile contains GPU architecture variables.
        Returns (has_gpu_vars, dockerfile_path)
        """
        try:
            # Find dockerfiles for this model
            dockerfiles = self._get_dockerfiles_for_model(model_info)
            
            for dockerfile_path in dockerfiles:
                with open(dockerfile_path, 'r') as f:
                    dockerfile_content = f.read()
                
                # Parse GPU architecture variables from Dockerfile
                dockerfile_gpu_vars = self._parse_dockerfile_gpu_variables(dockerfile_content)
                
                if dockerfile_gpu_vars:
                    return True, dockerfile_path
                else:
                    return False, dockerfile_path
            
            # No dockerfiles found
            return False, "No Dockerfile found"
            
        except Exception as e:
            self.rich_console.print(f"[yellow]Warning: Error checking GPU variables for model {model_info['name']}: {e}[/yellow]")
            return False, "Error reading Dockerfile"

    def _get_dockerfiles_for_model(self, model_info: typing.Dict) -> typing.List[str]:
        """Get dockerfiles for a model."""
        try:
            all_dockerfiles = self.console.sh(
                f"ls {model_info['dockerfile']}.*"
            ).split("\n")

            dockerfiles = {}
            for cur_docker_file in all_dockerfiles:
                # Get context of dockerfile
                dockerfiles[cur_docker_file] = self.console.sh(
                    f"head -n5 {cur_docker_file} | grep '# CONTEXT ' | sed 's/# CONTEXT //g'"
                )

            # Filter dockerfiles based on context
            dockerfiles = self.context.filter(dockerfiles)
            
            return list(dockerfiles.keys())
            
        except Exception as e:
            self.rich_console.print(f"[yellow]Warning: Error finding dockerfiles for model {model_info['name']}: {e}[/yellow]")
            return []

    def _validate_target_arch_against_dockerfile(self, model_info: typing.Dict, target_arch: str) -> bool:
        """
        Validate that target architecture is compatible with model's Dockerfile GPU variables.
        Called during build phase when --target-archs is provided.
        """
        try:
            # Find dockerfiles for this model
            dockerfiles = self._get_dockerfiles_for_model(model_info)
            
            for dockerfile_path in dockerfiles:
                with open(dockerfile_path, 'r') as f:
                    dockerfile_content = f.read()
                
                # Parse GPU architecture variables from Dockerfile
                dockerfile_gpu_vars = self._parse_dockerfile_gpu_variables(dockerfile_content)
                
                if not dockerfile_gpu_vars:
                    # No GPU variables found - target arch is acceptable
                    self.rich_console.print(f"[cyan]Info: No GPU architecture variables found in {dockerfile_path}, "
                          f"target architecture '{target_arch}' is acceptable[/cyan]")
                    continue
                
                # Validate target architecture against each GPU variable
                for var_name, var_values in dockerfile_gpu_vars.items():
                    if not self._is_target_arch_compatible_with_variable(
                        var_name, var_values, target_arch
                    ):
                        self.rich_console.print(f"[red]Error: Target architecture '{target_arch}' is not compatible "
                              f"with {var_name}={var_values} in {dockerfile_path}[/red]")
                        return False
                
                self.rich_console.print(f"[cyan]Info: Target architecture '{target_arch}' validated successfully "
                      f"against {dockerfile_path}[/cyan]")
            
            return True
            
        except FileNotFoundError as e:
            self.rich_console.print(f"[yellow]Warning: Dockerfile not found for model {model_info['name']}: {e}[/yellow]")
            return True  # Assume compatible if Dockerfile not found
        except Exception as e:
            self.rich_console.print(f"[yellow]Warning: Error validating target architecture for model {model_info['name']}: {e}[/yellow]")
            return True  # Assume compatible on parsing errors

    def _parse_dockerfile_gpu_variables(self, dockerfile_content: str) -> typing.Dict[str, typing.List[str]]:
        """Parse GPU architecture variables from Dockerfile content."""
        gpu_variables = {}
        
        for var_name in self.GPU_ARCH_VARIABLES:
            # Look for ARG declarations
            arg_pattern = rf"ARG\s+{var_name}=([^\s\n]+)"
            arg_matches = re.findall(arg_pattern, dockerfile_content, re.IGNORECASE)
            
            # Look for ENV declarations  
            env_pattern = rf"ENV\s+{var_name}[=\s]+([^\s\n]+)"
            env_matches = re.findall(env_pattern, dockerfile_content, re.IGNORECASE)
            
            # Process found values
            all_matches = arg_matches + env_matches
            if all_matches:
                # Take the last defined value (in case of multiple definitions)
                raw_value = all_matches[-1].strip('"\'')
                parsed_values = self._parse_gpu_variable_value(var_name, raw_value)
                if parsed_values:
                    gpu_variables[var_name] = parsed_values
        
        return gpu_variables

    def _parse_gpu_variable_value(self, var_name: str, raw_value: str) -> typing.List[str]:
        """Parse GPU variable value based on variable type and format."""
        architectures = []
        
        # Handle different variable formats
        if var_name in ["GPU_TARGETS", "GPU_ARCHS", "PYTORCH_ROCM_ARCH"]:
            # These often contain multiple architectures separated by semicolons or commas
            if ";" in raw_value:
                architectures = [arch.strip() for arch in raw_value.split(";") if arch.strip()]
            elif "," in raw_value:
                architectures = [arch.strip() for arch in raw_value.split(",") if arch.strip()]
            else:
                architectures = [raw_value.strip()]
        else:
            # Single architecture value (MAD_SYSTEM_GPU_ARCHITECTURE, GFX_COMPILATION_ARCH)
            architectures = [raw_value.strip()]
        
        # Normalize architecture names
        normalized_archs = []
        for arch in architectures:
            normalized = self._normalize_architecture_name(arch)
            if normalized:
                normalized_archs.append(normalized)
        
        return normalized_archs

    def _normalize_architecture_name(self, arch: str) -> str:
        """Normalize architecture name to standard format."""
        arch = arch.lower().strip()
        
        # Handle common variations and aliases
        if arch.startswith("gfx"):
            return arch
        elif arch in ["mi100", "mi-100"]:
            return "gfx908"
        elif arch in ["mi200", "mi-200", "mi210", "mi250"]:
            return "gfx90a"
        elif arch in ["mi300", "mi-300", "mi300a"]:
            return "gfx940"
        elif arch in ["mi300x", "mi-300x"]:
            return "gfx942"
        elif arch.startswith("mi"):
            # Unknown MI series - return as is for potential future support
            return arch
        
        return arch if arch else None

    def _is_target_arch_compatible_with_variable(
        self, 
        var_name: str, 
        var_values: typing.List[str], 
        target_arch: str
    ) -> bool:
        """
        Validate that target architecture is compatible with a specific GPU variable.
        Used during build phase validation.
        """
        if var_name == "MAD_SYSTEM_GPU_ARCHITECTURE":
            # MAD_SYSTEM_GPU_ARCHITECTURE will be overridden by target_arch, so always compatible
            return True
        
        elif var_name in ["PYTORCH_ROCM_ARCH", "GPU_TARGETS", "GPU_ARCHS"]:
            # Multi-architecture variables - target arch must be in the list
            return target_arch in var_values
        
        elif var_name == "GFX_COMPILATION_ARCH":
            # Compilation architecture should be compatible with target arch
            return len(var_values) == 1 and (
                var_values[0] == target_arch or
                self._is_compilation_arch_compatible(var_values[0], target_arch)
            )
        
        # Unknown variable - assume compatible
        return True

    def _is_compilation_arch_compatible(self, compile_arch: str, target_arch: str) -> bool:
        """Check if compilation architecture is compatible with target architecture."""
        # Define compatibility rules for compilation
        compatibility_matrix = {
            "gfx908": ["gfx908"],  # MI100 - exact match only
            "gfx90a": ["gfx90a"],  # MI200 - exact match only
            "gfx940": ["gfx940"],  # MI300A - exact match only
            "gfx941": ["gfx941"],  # MI300X - exact match only
            "gfx942": ["gfx942"],  # MI300X - exact match only
        }
        
        compatible_archs = compatibility_matrix.get(compile_arch, [compile_arch])
        return target_arch in compatible_archs

    def _build_model_single_arch(
        self, 
        model_info: typing.Dict,
        credentials: typing.Dict,
        clean_cache: bool,
        registry: str,
        phase_suffix: str,
        batch_build_metadata: typing.Optional[dict]
    ) -> typing.List[typing.Dict]:
        """Build model using existing single architecture flow."""
        
        # Use existing build logic - MAD_SYSTEM_GPU_ARCHITECTURE comes from additional_context
        # or Dockerfile defaults
        dockerfiles = self._get_dockerfiles_for_model(model_info)
        
        results = []
        for dockerfile in dockerfiles:
            build_info = self.build_image(
                model_info, 
                dockerfile, 
                credentials,
                clean_cache, 
                phase_suffix
            )
            
            # Extract GPU architecture from build args or context for manifest
            gpu_arch = self._get_effective_gpu_architecture(model_info, dockerfile)
            if gpu_arch:
                build_info["gpu_architecture"] = gpu_arch
            
            # Handle registry push (existing logic)
            if registry:
                try:
                    registry_image = self._create_registry_image_name(
                        build_info["docker_image"], registry, batch_build_metadata, model_info
                    )
                    self.push_image(build_info["docker_image"], registry, credentials, registry_image)
                    build_info["registry_image"] = registry_image
                except Exception as e:
                    build_info["push_error"] = str(e)
            
            results.append(build_info)
        
        return results

    def _get_effective_gpu_architecture(self, model_info: typing.Dict, dockerfile_path: str) -> str:
        """Get effective GPU architecture for single arch builds."""
        # Check if MAD_SYSTEM_GPU_ARCHITECTURE is in build args from additional_context
        if ("docker_build_arg" in self.context.ctx and 
            "MAD_SYSTEM_GPU_ARCHITECTURE" in self.context.ctx["docker_build_arg"]):
            return self.context.ctx["docker_build_arg"]["MAD_SYSTEM_GPU_ARCHITECTURE"]
        
        # Try to extract from Dockerfile defaults
        try:
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            
            # Look for ARG or ENV declarations
            patterns = [
                r"ARG\s+MAD_SYSTEM_GPU_ARCHITECTURE=([^\s\n]+)",
                r"ENV\s+MAD_SYSTEM_GPU_ARCHITECTURE=([^\s\n]+)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    return match.group(1).strip('"\'')
        except Exception:
            pass
        
        return None

    def _infer_gpu_vendor_from_dockerfile(self, dockerfile: str) -> str:
        """Infer GPU vendor from dockerfile path.
        
        Args:
            dockerfile: Path to dockerfile (e.g., "docker/dummy.ubuntu.amd.Dockerfile")
            
        Returns:
            GPU vendor string: "AMD", "NVIDIA", or ""
        """
        dockerfile_lower = dockerfile.lower()
        
        # Check for explicit vendor indicators in filename
        if '.amd.' in dockerfile_lower or dockerfile_lower.endswith('.amd'):
            return "AMD"
        elif '.nvidia.' in dockerfile_lower or dockerfile_lower.endswith('.nvidia'):
            return "NVIDIA"
        
        # Try to infer from base image in Dockerfile
        try:
            with open(dockerfile, 'r') as f:
                content = f.read()
            
            # Look for base image indicators
            if 'FROM' in content:
                if 'rocm' in content.lower() or 'amd' in content.lower():
                    return "AMD"
                elif 'nvidia' in content.lower() or 'cuda' in content.lower():
                    return "NVIDIA"
        except Exception:
            pass
        
        # Default to empty (legacy - will be treated as compatible with all)
        return ""

    def _create_base_image_name(self, model_info: typing.Dict, dockerfile: str) -> str:
        """Create base image name from model info and dockerfile."""
        # Extract dockerfile context suffix (e.g., "ubuntu.amd" from "dummy.ubuntu.amd.Dockerfile")
        dockerfile_name = os.path.basename(dockerfile)
        if '.' in dockerfile_name:
            # Remove the .Dockerfile extension and get context
            context_parts = dockerfile_name.replace('.Dockerfile', '').split('.')[1:]  # Skip model name
            context_suffix = '.'.join(context_parts) if context_parts else 'default'
        else:
            context_suffix = 'default'
        
        # Create base image name: ci-{model}_{model}.{context}
        return f"ci-{model_info['name']}_{model_info['name']}.{context_suffix}"

    def _create_registry_image_name(
        self, 
        image_name: str, 
        registry: str, 
        batch_build_metadata: typing.Optional[dict], 
        model_info: typing.Dict
    ) -> str:
        """Create registry image name."""
        if batch_build_metadata and model_info["name"] in batch_build_metadata:
            meta = batch_build_metadata[model_info["name"]]
            if meta.get("registry_image"):
                return meta["registry_image"]
        
        # Default registry naming
        return self._determine_registry_image_name(image_name, registry)

    def _create_arch_registry_image_name(
        self, 
        image_name: str, 
        gpu_arch: str, 
        registry: str, 
        batch_build_metadata: typing.Optional[dict], 
        model_info: typing.Dict
    ) -> str:
        """Create architecture-specific registry image name."""
        # For multi-arch builds, add architecture to the tag
        if batch_build_metadata and model_info["name"] in batch_build_metadata:
            meta = batch_build_metadata[model_info["name"]]
            if meta.get("registry_image"):
                # Append architecture to existing registry image
                return f"{meta['registry_image']}_{gpu_arch}"
        
        # Default arch-specific registry naming
        base_registry_name = self._determine_registry_image_name(image_name, registry)
        return f"{base_registry_name}"  # Architecture already in image_name

    def _determine_registry_image_name(
        self, docker_image: str, registry: str, credentials: typing.Dict = None
    ) -> str:
        """Determine the registry image name that would be used for pushing.

        Args:
            docker_image: The local docker image name
            registry: Registry URL (e.g., "localhost:5000", "docker.io", or empty for DockerHub)
            credentials: Optional credentials dictionary for registry authentication

        Returns:
            str: The full registry image name that would be used
        """
        if not registry:
            return docker_image

        # Determine registry image name based on registry type
        if registry.lower() in ["docker.io", "dockerhub"]:
            # For DockerHub, always use format: repository:tag
            # Try to get repository from credentials, fallback to default if not available
            if (
                credentials
                and "dockerhub" in credentials
                and "repository" in credentials["dockerhub"]
            ):
                registry_image = (
                    f"{credentials['dockerhub']['repository']}:{docker_image}"
                )
            else:
                registry_image = docker_image
        else:
            # For other registries (local, AWS ECR, etc.), use format: registry/repository:tag
            registry_key = registry
            if (
                credentials
                and registry_key in credentials
                and "repository" in credentials[registry_key]
            ):
                registry_image = f"{registry}/{credentials[registry_key]['repository']}:{docker_image}"
            else:
                # Fallback to just registry/imagename if no repository specified
                registry_image = f"{registry}/{docker_image}"

        return registry_image

    def _is_compilation_arch_compatible(self, compile_arch: str, target_arch: str) -> bool:
        """Check if compilation architecture is compatible with target architecture."""
        # Define compatibility rules for compilation
        compatibility_matrix = {
            "gfx908": ["gfx908"],  # MI100 - exact match only
            "gfx90a": ["gfx90a"],  # MI200 - exact match only
            "gfx940": ["gfx940"],  # MI300A - exact match only
            "gfx941": ["gfx941"],  # MI300X - exact match only
            "gfx942": ["gfx942"],  # MI300X - exact match only
        }
        
        compatible_archs = compatibility_matrix.get(compile_arch, [compile_arch])
        return target_arch in compatible_archs

    def _build_model_for_arch(
        self, 
        model_info: typing.Dict,
        gpu_arch: str,
        credentials: typing.Dict,
        clean_cache: bool,
        registry: str,
        phase_suffix: str,
        batch_build_metadata: typing.Optional[dict]
    ) -> typing.List[typing.Dict]:
        """Build model for specific GPU architecture with smart image naming."""
        
        # Find dockerfiles
        dockerfiles = self._get_dockerfiles_for_model(model_info)
        
        arch_results = []
        for dockerfile in dockerfiles:
            # When using --target-archs, always add architecture suffix regardless of GPU variables
            # This ensures consistent naming for multi-architecture builds
            base_image_name = self._create_base_image_name(model_info, dockerfile)
            arch_image_name = f"{base_image_name}_{gpu_arch}"
            
            # Set MAD_SYSTEM_GPU_ARCHITECTURE for this build
            arch_build_args = {"MAD_SYSTEM_GPU_ARCHITECTURE": gpu_arch}
            
            # Build the image
            build_info = self.build_image(
                model_info, 
                dockerfile, 
                credentials,
                clean_cache, 
                phase_suffix,
                additional_build_args=arch_build_args,
                override_image_name=arch_image_name
            )
            
            # Add architecture metadata
            build_info["gpu_architecture"] = gpu_arch
            
            # Handle registry push with architecture-specific tagging
            if registry:
                registry_image = self._determine_registry_image_name(
                    arch_image_name, registry, credentials
                )
                try:
                    self.push_image(arch_image_name, registry, credentials, registry_image)
                    build_info["registry_image"] = registry_image
                except Exception as e:
                    build_info["push_error"] = str(e)
            
            arch_results.append(build_info)
        
        return arch_results

    def _build_model_single_arch(
        self, 
        model_info: typing.Dict,
        credentials: typing.Dict,
        clean_cache: bool,
        registry: str,
        phase_suffix: str,
        batch_build_metadata: typing.Optional[dict]
    ) -> typing.List[typing.Dict]:
        """Build model using existing single architecture flow."""
        
        # Find dockerfiles for this model
        dockerfiles = self._get_dockerfiles_for_model(model_info)

        results = []
        for dockerfile in dockerfiles:
            build_info = self.build_image(
                model_info, 
                dockerfile, 
                credentials,
                clean_cache, 
                phase_suffix
            )
            
            # Extract GPU architecture from build args or context for manifest
            gpu_arch = self._get_effective_gpu_architecture(model_info, dockerfile)
            if gpu_arch:
                build_info["gpu_architecture"] = gpu_arch
            
            # Handle registry push (existing logic)
            if registry:
                registry_image = self._determine_registry_image_name(
                    build_info["docker_image"], registry, credentials
                )
                try:
                    self.push_image(build_info["docker_image"], registry, credentials, registry_image)
                    build_info["registry_image"] = registry_image
                except Exception as e:
                    build_info["push_error"] = str(e)
            
            results.append(build_info)
        
        return results

    def _get_effective_gpu_architecture(self, model_info: typing.Dict, dockerfile_path: str) -> str:
        """Get effective GPU architecture for single arch builds."""
        # Check if MAD_SYSTEM_GPU_ARCHITECTURE is in build args from additional_context
        if ("docker_build_arg" in self.context.ctx and 
            "MAD_SYSTEM_GPU_ARCHITECTURE" in self.context.ctx["docker_build_arg"]):
            return self.context.ctx["docker_build_arg"]["MAD_SYSTEM_GPU_ARCHITECTURE"]
        
        # Try to extract from Dockerfile defaults
        try:
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            
            # Look for ARG or ENV declarations
            patterns = [
                r"ARG\s+MAD_SYSTEM_GPU_ARCHITECTURE=([^\s\n]+)",
                r"ENV\s+MAD_SYSTEM_GPU_ARCHITECTURE=([^\s\n]+)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    return match.group(1).strip('"\'')
        except Exception:
            pass
        
        return None
