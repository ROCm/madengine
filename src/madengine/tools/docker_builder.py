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
import typing
from contextlib import redirect_stdout, redirect_stderr
from madengine.core.console import Console
from madengine.core.context import Context
from madengine.utils.ops import PythonicTee


class DockerBuilder:
    """Class responsible for building Docker images for models."""
    
    def __init__(self, context: Context, console: Console = None, live_output: bool = False):
        """Initialize the Docker Builder.
        
        Args:
            context: The MADEngine context
            console: Optional console instance
            live_output: Whether to show live output
        """
        self.context = context
        self.console = console or Console(live_output=live_output)
        self.live_output = live_output
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
    
    def build_image(self, model_info: typing.Dict, dockerfile: str, 
                   credentials: typing.Dict = None, clean_cache: bool = False,
                   phase_suffix: str = "") -> typing.Dict:
        """Build a Docker image for the given model.
        
        Args:
            model_info: The model information dictionary
            dockerfile: Path to the Dockerfile
            credentials: Optional credentials dictionary
            clean_cache: Whether to use --no-cache
            phase_suffix: Suffix for log file name (e.g., ".build" or "")
            
        Returns:
            dict: Build information including image name, build duration, etc.
        """
        # Generate image name first
        image_docker_name = (
            model_info["name"].replace("/", "_").lower()
            + "_"
            + os.path.basename(dockerfile).replace(".Dockerfile", "")
        )
        
        docker_image = "ci-" + image_docker_name
        
        # Create log file for this build
        cur_docker_file_basename = os.path.basename(dockerfile).replace(".Dockerfile", "")
        log_file_path = (
            model_info["name"].replace("/", "_")
            + "_"
            + cur_docker_file_basename
            + phase_suffix
            + ".live.log"
        )
        # Replace / with _ in log file path (already done above, but keeping for safety)
        log_file_path = log_file_path.replace("/", "_")
        
        print(f"\n🔨 Starting Docker build for model: {model_info['name']}")
        print(f"📁 Dockerfile: {dockerfile}")
        print(f"🏷️  Target image: {docker_image}")
        print(f"📝 Build log: {log_file_path}")
        print(f"{'='*80}")
        
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
            with redirect_stdout(PythonicTee(outlog, self.live_output)), redirect_stderr(PythonicTee(outlog, self.live_output)):
                print(f"🔨 Executing build command...")
                self.console.sh(build_command, timeout=None)
                
                build_duration = time.time() - build_start_time
                
                print(f"⏱️  Build Duration: {build_duration:.2f} seconds")
                print(f"🏷️  MAD_CONTAINER_IMAGE is {docker_image}")
                print(f"✅ Docker build completed successfully")
                print(f"{'='*80}")
                
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
                        f"docker manifest inspect {base_docker} | grep digest | head -n 1 | cut -d \\\" -f 4"
                    )
                    print(f"BASE DOCKER SHA is {docker_sha}")
                except Exception as e:
                    print(f"Warning: Could not get docker SHA: {e}")
        
        build_info = {
            "docker_image": docker_image,
            "dockerfile": dockerfile,
            "base_docker": base_docker,
            "docker_sha": docker_sha,
            "build_duration": build_duration,
            "build_command": build_command,
            "log_file": log_file_path
        }
        
        # Store built image info
        self.built_images[docker_image] = build_info
        
        # Store model info linked to the built image
        self.built_models[docker_image] = model_info
        
        print(f"Successfully built image: {docker_image}")
        
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
                error_msg += f"\nPlease add {registry_key} credentials to credential.json:\n"
                error_msg += "{\n"
                error_msg += f'  "{registry_key}": {{\n'
                error_msg += f'    "repository": "your-repository",\n'
                error_msg += f'    "username": "your-{registry_key}-username",\n'
                error_msg += f'    "password": "your-{registry_key}-password"\n'
                error_msg += "  }\n"
                error_msg += "}"
            print(error_msg)
            raise RuntimeError(error_msg)
            
        creds = credentials[registry_key]
        
        if "username" not in creds or "password" not in creds:
            error_msg = f"Invalid credentials format for registry: {registry_key}"
            error_msg += f"\nCredentials must contain 'username' and 'password' fields"
            print(error_msg)
            raise RuntimeError(error_msg)
            
        # Ensure credential values are strings
        username = str(creds['username'])
        password = str(creds['password'])
            
        # Perform docker login
        login_command = f"echo '{password}' | docker login"
        
        if registry and registry.lower() not in ["docker.io", "dockerhub"]:
            login_command += f" {registry}"
            
        login_command += f" --username {username} --password-stdin"
        
        try:
            self.console.sh(login_command, secret=True)
            print(f"Successfully logged in to registry: {registry or 'DockerHub'}")
        except Exception as e:
            print(f"Failed to login to registry {registry}: {e}")
            raise

    def push_image(self, docker_image: str, registry: str = None, credentials: typing.Dict = None) -> str:
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
        registry_image = self._determine_registry_image_name(docker_image, registry, credentials)
        
        try:
            # Tag the image if different from local name
            if registry_image != docker_image:
                tag_command = f"docker tag {docker_image} {registry_image}"
                print(f"🏷️  Tagging image: {tag_command}")
                self.console.sh(tag_command)
            
            # Push the image
            push_command = f"docker push {registry_image}"
            print(f"\n🚀 Starting docker push to registry...")
            print(f"📤 Registry: {registry}")
            print(f"🏷️  Image: {registry_image}")
            self.console.sh(push_command)
            
            print(f"✅ Successfully pushed image to registry: {registry_image}")
            print(f"{'='*80}")
            return registry_image
            
        except Exception as e:
            print(f"Failed to push image {docker_image} to registry {registry}: {e}")
            raise
    
    def export_build_manifest(self, output_file: str = "build_manifest.json", registry: typing.Optional[str] = None) -> None:
        """Export enhanced build information to a manifest file.
        
        This creates a comprehensive build manifest that includes all necessary
        information for deployment, reducing the need for separate execution configs.
        
        Args:
            output_file: Path to output manifest file
            registry: Registry used for building (added to manifest metadata)
        """
        # Extract credentials from models
        credentials_required = list(set([
            model.get("cred", "") for model in self.built_models.values() 
            if model.get("cred", "") != ""
        ]))

        # Do not override per-image registry here; just export as is
        built_images_with_registry = self.built_images

        manifest = {
            "built_images": built_images_with_registry,
            "built_models": self.built_models,
            "context": {
                "docker_env_vars": self.context.ctx.get("docker_env_vars", {}),
                "docker_mounts": self.context.ctx.get("docker_mounts", {}),
                "docker_build_arg": self.context.ctx.get("docker_build_arg", {}),
                "gpu_vendor": self.context.ctx.get("gpu_vendor", ""),
                "docker_gpus": self.context.ctx.get("docker_gpus", "")
            },
            "credentials_required": credentials_required
        }

        # Add multi-node args to context if present
        if "build_multi_node_args" in self.context.ctx:
            manifest["context"]["multi_node_args"] = self.context.ctx["build_multi_node_args"]

        # Add push failure summary if any pushes failed
        push_failures = []
        for image_name, build_info in self.built_images.items():
            if "push_failed" in build_info and build_info["push_failed"]:
                push_failures.append({
                    "image": image_name,
                    "intended_registry_image": build_info.get("registry_image"),
                    "error": build_info.get("push_error")
                })

        if push_failures:
            manifest["push_failures"] = push_failures

        with open(output_file, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"Build manifest exported to: {output_file}")
        if push_failures:
            print(f"Warning: {len(push_failures)} image(s) failed to push to registry")
            for failure in push_failures:
                print(f"  - {failure['image']} -> {failure['intended_registry_image']}: {failure['error']}")
    
    def build_all_models(self, models: typing.List[typing.Dict], 
                        credentials: typing.Dict = None, 
                        clean_cache: bool = False,
                        registry: str = None,
                        phase_suffix: str = "",
                        batch_build_metadata: typing.Optional[dict] = None) -> typing.Dict:
        """Build images for all models.
        
        Args:
            models: List of model information dictionaries
            credentials: Optional credentials dictionary
            clean_cache: Whether to use --no-cache
            registry: Optional registry to push images to
            phase_suffix: Suffix for log file name (e.g., ".build" or "")
            
        Returns:
            dict: Summary of all built images
        """
        print(f"Building Docker images for {len(models)} models...")
        
        build_summary = {
            "successful_builds": [],
            "failed_builds": [],
            "total_build_time": 0
        }
        
        for model_info in models:
            try:
                # If batch_build_metadata is provided, override registry and registry_image for this model
                model_registry = registry
                model_registry_image = None
                if batch_build_metadata and model_info["name"] in batch_build_metadata:
                    meta = batch_build_metadata[model_info["name"]]
                    if meta.get("registry"):
                        model_registry = meta["registry"]
                    if meta.get("registry_image"):
                        model_registry_image = meta["registry_image"]

                # Find dockerfiles for this model
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

                if not dockerfiles:
                    print(f"No matching dockerfiles found for model {model_info['name']}")
                    continue
                
                # Build each dockerfile

                for dockerfile in dockerfiles.keys():
                    try:
                        build_info = self.build_image(
                            model_info, dockerfile, credentials, clean_cache, phase_suffix
                        )

                        # Determine registry image name and add to manifest before push operations
                        registry_image = None
                        if model_registry_image:
                            registry_image = model_registry_image
                        elif model_registry:
                            registry_image = self._determine_registry_image_name(
                                build_info["docker_image"], model_registry, credentials
                            )
                        if registry_image:
                            build_info["registry_image"] = registry_image
                            if build_info["docker_image"] in self.built_images:
                                self.built_images[build_info["docker_image"]]["registry_image"] = registry_image

                        # Now attempt to push to registry if registry is set
                        if model_registry and registry_image:
                            try:
                                actual_registry_image = self.push_image(
                                    build_info["docker_image"], model_registry, credentials
                                )
                                if actual_registry_image != registry_image:
                                    print(f"Warning: Pushed image name {actual_registry_image} differs from intended {registry_image}")
                            except Exception as push_error:
                                print(f"Failed to push {build_info['docker_image']} to registry: {push_error}")
                                build_info["push_failed"] = True
                                build_info["push_error"] = str(push_error)
                                if build_info["docker_image"] in self.built_images:
                                    self.built_images[build_info["docker_image"]]["push_failed"] = True
                                    self.built_images[build_info["docker_image"]]["push_error"] = str(push_error)

                        build_summary["successful_builds"].append({
                            "model": model_info["name"],
                            "dockerfile": dockerfile,
                            "build_info": build_info
                        })

                        build_summary["total_build_time"] += build_info["build_duration"]

                    except Exception as e:
                        print(f"Failed to build {dockerfile} for model {model_info['name']}: {e}")
                        build_summary["failed_builds"].append({
                            "model": model_info["name"],
                            "dockerfile": dockerfile,
                            "error": str(e)
                        })
                        
            except Exception as e:
                print(f"Error processing model {model_info['name']}: {e}")
                build_summary["failed_builds"].append({
                    "model": model_info["name"],
                    "error": str(e)
                })
        
        print(f"\nBuild Summary:")
        print(f"  Successful builds: {len(build_summary['successful_builds'])}")
        print(f"  Failed builds: {len(build_summary['failed_builds'])}")
        print(f"  Total build time: {build_summary['total_build_time']:.2f} seconds")
        
        return build_summary

    def _determine_registry_image_name(self, docker_image: str, registry: str, credentials: typing.Dict = None) -> str:
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
            if credentials and "dockerhub" in credentials and "repository" in credentials["dockerhub"]:
                registry_image = f"{credentials['dockerhub']['repository']}:{docker_image}"
            else:
                registry_image = docker_image
        else:
            # For other registries (local, AWS ECR, etc.), use format: registry/repository:tag
            registry_key = registry
            if credentials and registry_key in credentials and "repository" in credentials[registry_key]:
                registry_image = f"{registry}/{credentials[registry_key]['repository']}:{docker_image}"
            else:
                # Fallback to just registry/imagename if no repository specified
                registry_image = f"{registry}/{docker_image}"
        
        return registry_image
