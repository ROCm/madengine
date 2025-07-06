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
    
    def __init__(self, args):
        """Initialize the distributed orchestrator.
        
        Args:
            args: Command-line arguments
        """
        self.args = args
        self.console = Console(live_output=getattr(args, 'live_output', True))
        
        # Initialize context
        self.context = Context(
            additional_context=getattr(args, 'additional_context', None),
            additional_context_file=getattr(args, 'additional_context_file', None),
        )
        
        # Initialize data provider if data config exists
        data_json_file = getattr(args, 'data_config_file_name', 'data.json')
        if os.path.exists(data_json_file):
            self.data = Data(
                self.context,
                filename=data_json_file,
                force_mirrorlocal=getattr(args, 'force_mirror_local', False),
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
            print(f"Warning: Could not load credentials: {e}")
    
    def build_phase(self, registry: str = None, clean_cache: bool = False, 
                   manifest_output: str = "build_manifest.json") -> typing.Dict:
        """Execute the build phase - build all Docker images.
        
        Args:
            registry: Optional registry to push images to
            clean_cache: Whether to use --no-cache for builds
            manifest_output: Output file for build manifest
            
        Returns:
            dict: Build summary
        """
        print("=" * 60)
        print("STARTING BUILD PHASE")
        print("=" * 60)
        
        print(f"Building models with args {self.args}")
        
        # Discover models
        discover_models = DiscoverModels(args=self.args)
        models = discover_models.run()
        
        print(f"Discovered {len(models)} models to build")
        
        # Copy scripts for building
        self._copy_scripts()
        
        # Initialize builder
        builder = DockerBuilder(self.context, self.console, live_output=getattr(self.args, 'live_output', False))
        
        # Determine phase suffix for log files
        phase_suffix = ".build" if hasattr(self.args, '_separate_phases') and self.args._separate_phases else ""
        
        # Build all images
        build_summary = builder.build_all_models(
            models, self.credentials, clean_cache, registry, phase_suffix
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
            registry = manifest["registry"]
            print(f"Auto-detected registry from manifest: {registry}")
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
                        
                        # Pull image if from registry (either from CLI arg or manifest)
                        if "registry_image" in build_info:
                            # Use registry from CLI if provided, otherwise extract from registry_image
                            effective_registry = registry
                            if not effective_registry and build_info["registry_image"]:
                                # Extract registry from the registry_image format
                                registry_parts = build_info["registry_image"].split('/')
                                if len(registry_parts) > 1 and '.' in registry_parts[0]:
                                    effective_registry = registry_parts[0]
                                elif build_info["registry_image"].startswith('docker.io/') or '/' in build_info["registry_image"]:
                                    effective_registry = "docker.io"
                            
                            if effective_registry:
                                print(f"Pulling image from registry: {build_info['registry_image']}")
                                actual_image = runner.pull_image(
                                    build_info["registry_image"], image_name, effective_registry, self.credentials
                                )
                            else:
                                # Registry image exists but no valid registry found, use as-is
                                print(f"Using registry image as-is: {build_info['registry_image']}")
                                actual_image = build_info["registry_image"]
                        elif registry:
                            # Registry specified but no registry_image in manifest - attempt to construct registry image name
                            # This handles cases where manifest has registry info but images weren't actually pushed
                            if registry.lower() in ["docker.io", "dockerhub"]:
                                # For DockerHub, we need username from credentials
                                if self.credentials and "dockerhub" in self.credentials and "username" in self.credentials["dockerhub"]:
                                    registry_image_name = f"{self.credentials['dockerhub']['username']}/{image_name}"
                                else:
                                    registry_image_name = image_name
                            else:
                                registry_image_name = f"{registry}/{image_name}"
                            
                            print(f"Attempting to pull constructed registry image: {registry_image_name}")
                            try:
                                actual_image = runner.pull_image(
                                    registry_image_name, image_name, registry, self.credentials
                                )
                            except Exception as e:
                                print(f"Failed to pull from registry, falling back to local image: {e}")
                                actual_image = image_name
                        else:
                            actual_image = image_name
                        
                        # Run the container
                        run_results = runner.run_container(
                            model_info, actual_image, build_info, 
                            keep_alive=keep_alive, timeout=timeout, phase_suffix=phase_suffix
                        )
                        
                        execution_summary["successful_runs"].append(run_results)
                        execution_summary["total_execution_time"] += run_results.get("test_duration", 0)
                        
                        print(f"Successfully completed: {model_info['name']} -> {run_results['status']}")
                        
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
                        
                        # Pull image if from registry (either from CLI arg or manifest)
                        if "registry_image" in build_info:
                            # Use registry from CLI if provided, otherwise extract from registry_image
                            effective_registry = registry
                            if not effective_registry and build_info["registry_image"]:
                                # Extract registry from the registry_image format
                                registry_parts = build_info["registry_image"].split('/')
                                if len(registry_parts) > 1 and '.' in registry_parts[0]:
                                    effective_registry = registry_parts[0]
                                elif build_info["registry_image"].startswith('docker.io/') or '/' in build_info["registry_image"]:
                                    effective_registry = "docker.io"
                            
                            if effective_registry:
                                print(f"Pulling image from registry: {build_info['registry_image']}")
                                actual_image = runner.pull_image(
                                    build_info["registry_image"], image_name, effective_registry, self.credentials
                                )
                            else:
                                # Registry image exists but no valid registry found, use as-is
                                print(f"Using registry image as-is: {build_info['registry_image']}")
                                actual_image = build_info["registry_image"]
                        elif registry:
                            # Registry specified but no registry_image in manifest - attempt to construct registry image name
                            if registry.lower() in ["docker.io", "dockerhub"]:
                                # For DockerHub, we need username from credentials
                                if self.credentials and "dockerhub" in self.credentials and "username" in self.credentials["dockerhub"]:
                                    registry_image_name = f"{self.credentials['dockerhub']['username']}/{image_name}"
                                else:
                                    registry_image_name = image_name
                            else:
                                registry_image_name = f"{registry}/{image_name}"
                            
                            print(f"Attempting to pull constructed registry image: {registry_image_name}")
                            try:
                                actual_image = runner.pull_image(
                                    registry_image_name, image_name, registry, self.credentials
                                )
                            except Exception as e:
                                print(f"Failed to pull from registry, falling back to local image: {e}")
                                actual_image = image_name
                        else:
                            actual_image = image_name
                        
                        # Run the container
                        run_results = runner.run_container(
                            model_info, actual_image, build_info, 
                            keep_alive=keep_alive, timeout=timeout, phase_suffix=phase_suffix
                        )
                        
                        execution_summary["successful_runs"].append(run_results)
                        execution_summary["total_execution_time"] += run_results.get("test_duration", 0)
                        
                        print(f"Successfully completed: {model_name} -> {run_results['status']}")
                        
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
    
    def export_execution_config(self, models: typing.List[typing.Dict], 
                              output_file: str = "execution_config.json") -> None:
        """Export execution configuration for external orchestrators.
        
        Args:
            models: List of model configurations
            output_file: Output configuration file
        """
        config = {
            "models": models,
            "context": {
                "docker_env_vars": self.context.ctx.get("docker_env_vars", {}),
                "docker_mounts": self.context.ctx.get("docker_mounts", {}),
                "gpu_vendor": self.context.ctx.get("gpu_vendor", ""),
                "docker_gpus": self.context.ctx.get("docker_gpus", ""),
            },
            "credentials_required": [
                model.get("cred", "") for model in models 
                if model.get("cred", "") != ""
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Execution configuration exported to: {output_file}")
    
    def cleanup(self) -> None:
        """Cleanup the scripts/common directory."""
        # check the directory exists
        if os.path.exists("scripts/common"):
            # check tools.json exists in scripts/common directory
            if os.path.exists("scripts/common/tools.json"):
                # remove the scripts/common/tools.json file
                self.console.sh("rm -rf scripts/common/tools.json")
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


def create_ansible_playbook(manifest_file: str = "build_manifest.json",
                          execution_config: str = "execution_config.json",
                          playbook_file: str = "madengine_distributed.yml") -> None:
    """Create an Ansible playbook for distributed execution.
    
    Args:
        manifest_file: Build manifest file
        execution_config: Execution configuration file  
        playbook_file: Output Ansible playbook file
    """
    playbook_content = f"""---
# MADEngine Distributed Execution Playbook
# Generated automatically for distributed model execution

- name: MADEngine Distributed Model Execution
  hosts: gpu_nodes
  become: yes
  vars:
    manifest_file: "{manifest_file}"
    execution_config: "{execution_config}"
    madengine_workspace: "/tmp/madengine_distributed"
    
  tasks:
    - name: Create MADEngine workspace
      file:
        path: "{{{{ madengine_workspace }}}}"
        state: directory
        mode: '0755'
    
    - name: Copy build manifest to nodes
      copy:
        src: "{{{{ manifest_file }}}}"
        dest: "{{{{ madengine_workspace }}}}/{{{{ manifest_file }}}}"
    
    - name: Copy execution config to nodes  
      copy:
        src: "{{{{ execution_config }}}}"
        dest: "{{{{ madengine_workspace }}}}/{{{{ execution_config }}}}"
    
    - name: Pull Docker images from registry
      shell: |
        cd {{{{ madengine_workspace }}}}
        python3 -c "
        import json
        with open('{{{{ manifest_file }}}}', 'r') as f:
            manifest = json.load(f)
        for image_name, build_info in manifest['built_images'].items():
            if 'registry_image' in build_info:
                print(f'Pulling {{{{ build_info[\"registry_image\"] }}}}')
                import subprocess
                subprocess.run(['docker', 'pull', build_info['registry_image']], check=True)
                subprocess.run(['docker', 'tag', build_info['registry_image'], image_name], check=True)
        "
      when: inventory_hostname in groups['gpu_nodes']
    
    - name: Run MADEngine containers
      shell: |
        cd {{{{ madengine_workspace }}}}
        # This would call your ContainerRunner
        python3 -c "
        from madengine.tools.distributed_orchestrator import DistributedOrchestrator
        import argparse
        
        # Create minimal args for runner
        args = argparse.Namespace()
        args.live_output = True
        args.additional_context = None
        args.additional_context_file = None
        args.data_config_file_name = 'data.json'
        args.force_mirror_local = False
        
        orchestrator = DistributedOrchestrator(args)
        execution_summary = orchestrator.run_phase(
            manifest_file='{{{{ manifest_file }}}}',
            timeout=7200,
            keep_alive=False
        )
        print(f'Execution completed: {{{{ execution_summary }}}}')
        "
      when: inventory_hostname in groups['gpu_nodes']
      register: execution_results
    
    - name: Display execution results
      debug:
        var: execution_results.stdout_lines
      when: execution_results is defined
"""
    
    with open(playbook_file, 'w') as f:
        f.write(playbook_content)
    
    print(f"Ansible playbook created: {playbook_file}")


def create_kubernetes_manifests(manifest_file: str = "build_manifest.json",
                               execution_config: str = "execution_config.json",
                               namespace: str = "madengine") -> None:
    """Create Kubernetes manifests for distributed execution.
    
    Args:
        manifest_file: Build manifest file
        execution_config: Execution configuration file
        namespace: Kubernetes namespace
    """
    
    # ConfigMap for configuration files
    configmap_yaml = f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: madengine-config
  namespace: {namespace}
data:
  manifest.json: |
    # Content would be loaded from {manifest_file}
  execution-config.json: |
    # Content would be loaded from {execution_config}
---
apiVersion: v1
kind: Namespace
metadata:
  name: {namespace}
"""
    
    # Job template for model execution
    job_yaml = f"""apiVersion: batch/v1
kind: Job
metadata:
  name: madengine-model-execution
  namespace: {namespace}
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: madengine-runner
        image: madengine/distributed-runner:latest
        command: ["/bin/bash"]
        args: ["-c", "python3 -m madengine.tools.distributed_orchestrator run-phase --manifest-file=/config/manifest.json"]
        volumeMounts:
        - name: config-volume
          mountPath: /config
        - name: docker-socket
          mountPath: /var/run/docker.sock
        resources:
          limits:
            nvidia.com/gpu: 1  # Adjust based on model requirements
          requests:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: NVIDIA_VISIBLE_DEVICES
          value: "all"
        - name: NVIDIA_DRIVER_CAPABILITIES
          value: "compute,utility"
      volumes:
      - name: config-volume
        configMap:
          name: madengine-config
      - name: docker-socket
        hostPath:
          path: /var/run/docker.sock
          type: Socket
      nodeSelector:
        accelerator: nvidia-tesla-v100  # Adjust based on your GPU nodes
"""
    
    with open(f"k8s-madengine-configmap.yaml", 'w') as f:
        f.write(configmap_yaml)
    
    with open(f"k8s-madengine-job.yaml", 'w') as f:
        f.write(job_yaml)
    
    print(f"Kubernetes manifests created:")
    print(f"  - k8s-madengine-configmap.yaml")
    print(f"  - k8s-madengine-job.yaml")
