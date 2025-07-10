"""Orchestrator generation module for MADEngine distributed execution.

This module provides high-level interfaces for generating distributed
execution configurations using the template system.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

from .template_generator import TemplateGenerator


class OrchestatorGenerator:
    """High-level interface for generating distributed execution configurations."""
    
    def __init__(self, template_dir: Optional[str] = None, values_dir: Optional[str] = None):
        """Initialize the orchestrator generator.
        
        Args:
            template_dir: Custom template directory path
            values_dir: Custom values directory path
        """
        self.template_generator = TemplateGenerator(template_dir, values_dir)
    
    def generate_complete_ansible_setup(self, 
                                       manifest_file: str,
                                       environment: str = "default",
                                       output_dir: str = "ansible-setup") -> Dict[str, str]:
        """Generate complete Ansible setup including playbook, script, and inventory.
        
        Args:
            manifest_file: Path to build manifest JSON file
            environment: Environment name for values
            output_dir: Output directory for generated files
            
        Returns:
            dict: Dictionary mapping file types to generated file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        generated_files = {}
        
        # Generate playbook
        playbook_file = os.path.join(output_dir, "madengine_playbook.yml")
        self.template_generator.generate_ansible_playbook(
            manifest_file, environment, playbook_file
        )
        generated_files["playbook"] = playbook_file
        
        # Generate execution script
        script_file = os.path.join(output_dir, "execute_models.py")
        self.template_generator.generate_execution_script(
            manifest_file, environment, script_file
        )
        generated_files["script"] = script_file
        
        # Generate inventory file
        inventory_file = os.path.join(output_dir, "inventory.yml")
        self._generate_ansible_inventory(manifest_file, environment, inventory_file)
        generated_files["inventory"] = inventory_file
        
        # Generate ansible.cfg
        config_file = os.path.join(output_dir, "ansible.cfg")
        self._generate_ansible_config(environment, config_file)
        generated_files["config"] = config_file
        
        return generated_files
    
    def generate_complete_k8s_setup(self, 
                                   manifest_file: str,
                                   environment: str = "default",
                                   output_dir: str = "k8s-setup") -> Dict[str, List[str]]:
        """Generate complete Kubernetes setup including manifests and deployment scripts.
        
        Args:
            manifest_file: Path to build manifest JSON file
            environment: Environment name for values
            output_dir: Output directory for generated files
            
        Returns:
            dict: Dictionary mapping resource types to generated file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate manifests
        manifests_dir = os.path.join(output_dir, "manifests")
        manifest_files = self.template_generator.generate_kubernetes_manifests(
            manifest_file, environment, manifests_dir
        )
        
        # Generate deployment script
        deploy_script = os.path.join(output_dir, "deploy.sh")
        self._generate_k8s_deploy_script(environment, manifests_dir, deploy_script)
        
        # Generate cleanup script
        cleanup_script = os.path.join(output_dir, "cleanup.sh")
        self._generate_k8s_cleanup_script(environment, manifests_dir, cleanup_script)
        
        return {
            "manifests": manifest_files,
            "deploy_script": deploy_script,
            "cleanup_script": cleanup_script
        }
    
    def generate_execution_pipeline(self,
                                   manifest_file: str,
                                   environment: str = "default",
                                   output_dir: str = "pipeline") -> Dict[str, str]:
        """Generate a complete execution pipeline with monitoring.
        
        Args:
            manifest_file: Path to build manifest JSON file
            environment: Environment name for values
            output_dir: Output directory for generated files
            
        Returns:
            dict: Dictionary mapping component types to generated file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        generated_files = {}
        
        # Generate main execution script
        main_script = os.path.join(output_dir, "run_pipeline.py")
        self._generate_pipeline_script(manifest_file, environment, main_script)
        generated_files["main_script"] = main_script
        
        # Generate monitoring script
        monitor_script = os.path.join(output_dir, "monitor_execution.py")
        self._generate_monitoring_script(manifest_file, environment, monitor_script)
        generated_files["monitor_script"] = monitor_script
        
        # Generate configuration
        config_file = os.path.join(output_dir, "pipeline_config.json")
        self._generate_pipeline_config(manifest_file, environment, config_file)
        generated_files["config"] = config_file
        
        return generated_files
    
    def validate_manifest(self, manifest_file: str) -> Dict[str, Any]:
        """Validate build manifest for completeness.
        
        Args:
            manifest_file: Path to build manifest JSON file
            
        Returns:
            dict: Validation results
        """
        if not os.path.exists(manifest_file):
            return {"valid": False, "error": f"Manifest file not found: {manifest_file}"}
        
        try:
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            
            validation_results = {
                "valid": True,
                "warnings": [],
                "errors": []
            }
            
            # Check required fields
            required_fields = ["built_images", "context"]
            for field in required_fields:
                if field not in manifest:
                    validation_results["errors"].append(f"Missing required field: {field}")
                    validation_results["valid"] = False
            
            # Check for built images
            if "built_images" in manifest:
                if not manifest["built_images"]:
                    validation_results["warnings"].append("No built images found in manifest")
                else:
                    for image_name, image_info in manifest["built_images"].items():
                        if "docker_image" not in image_info:
                            validation_results["warnings"].append(f"Image {image_name} missing docker_image field")
            
            # Check context
            if "context" in manifest:
                context = manifest["context"]
                if "gpu_vendor" not in context:
                    validation_results["warnings"].append("GPU vendor not specified in context")
            
            return validation_results
            
        except json.JSONDecodeError as e:
            return {"valid": False, "error": f"Invalid JSON in manifest: {e}"}
        except Exception as e:
            return {"valid": False, "error": f"Error reading manifest: {e}"}
    
    def _generate_ansible_inventory(self, manifest_file: str, environment: str, output_file: str):
        """Generate Ansible inventory file."""
        # Load values to get host configuration
        values = self.template_generator.load_values(environment)
        
        # Load manifest for additional context
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        gpu_vendor = manifest.get("context", {}).get("gpu_vendor", "")
        
        inventory_content = f"""# MADEngine Ansible Inventory
# Generated for environment: {environment}
# GPU Vendor: {gpu_vendor}

[gpu_nodes]
# Add your GPU nodes here
# gpu-node-1 ansible_host=192.168.1.10 ansible_user=ubuntu
# gpu-node-2 ansible_host=192.168.1.11 ansible_user=ubuntu

[gpu_nodes:vars]
madengine_environment={environment}
gpu_vendor={gpu_vendor}
madengine_registry={manifest.get('registry', '')}

[all:vars]
ansible_python_interpreter=/usr/bin/python3
ansible_ssh_common_args='-o StrictHostKeyChecking=no'
"""
        
        with open(output_file, 'w') as f:
            f.write(inventory_content)
    
    def _generate_ansible_config(self, environment: str, output_file: str):
        """Generate Ansible configuration file."""
        config_content = f"""# MADEngine Ansible Configuration
# Generated for environment: {environment}

[defaults]
inventory = inventory.yml
host_key_checking = False
stdout_callback = yaml
stderr_callback = yaml
remote_user = ubuntu
private_key_file = ~/.ssh/id_rsa
timeout = 30
log_path = ./ansible.log

[ssh_connection]
ssh_args = -o ForwardAgent=yes -o ControlMaster=auto -o ControlPersist=60s
pipelining = True
"""
        
        with open(output_file, 'w') as f:
            f.write(config_content)
    
    def _generate_k8s_deploy_script(self, environment: str, manifests_dir: str, output_file: str):
        """Generate Kubernetes deployment script."""
        script_content = f"""#!/bin/bash
# MADEngine Kubernetes Deployment Script
# Generated for environment: {environment}

set -e

MANIFESTS_DIR="{manifests_dir}"
NAMESPACE="madengine-{environment}"

echo "Deploying MADEngine to Kubernetes..."
echo "Environment: {environment}"
echo "Namespace: $NAMESPACE"

# Apply manifests in order
if [ -f "$MANIFESTS_DIR/namespace.yaml" ]; then
    echo "Creating namespace..."
    kubectl apply -f "$MANIFESTS_DIR/namespace.yaml"
fi

if [ -f "$MANIFESTS_DIR/configmap.yaml" ]; then
    echo "Creating configmap..."
    kubectl apply -f "$MANIFESTS_DIR/configmap.yaml"
fi

if [ -f "$MANIFESTS_DIR/service.yaml" ]; then
    echo "Creating service..."
    kubectl apply -f "$MANIFESTS_DIR/service.yaml"
fi

if [ -f "$MANIFESTS_DIR/job.yaml" ]; then
    echo "Creating job..."
    kubectl apply -f "$MANIFESTS_DIR/job.yaml"
fi

echo "Deployment complete!"
echo "Monitor the job with: kubectl get jobs -n $NAMESPACE"
echo "View logs with: kubectl logs -n $NAMESPACE -l app.kubernetes.io/name=madengine"
"""
        
        with open(output_file, 'w') as f:
            f.write(script_content)
        
        os.chmod(output_file, 0o755)
    
    def _generate_k8s_cleanup_script(self, environment: str, manifests_dir: str, output_file: str):
        """Generate Kubernetes cleanup script."""
        script_content = f"""#!/bin/bash
# MADEngine Kubernetes Cleanup Script
# Generated for environment: {environment}

set -e

MANIFESTS_DIR="{manifests_dir}"
NAMESPACE="madengine-{environment}"

echo "Cleaning up MADEngine from Kubernetes..."
echo "Environment: {environment}"
echo "Namespace: $NAMESPACE"

# Delete resources
if [ -f "$MANIFESTS_DIR/job.yaml" ]; then
    echo "Deleting job..."
    kubectl delete -f "$MANIFESTS_DIR/job.yaml" --ignore-not-found=true
fi

if [ -f "$MANIFESTS_DIR/service.yaml" ]; then
    echo "Deleting service..."
    kubectl delete -f "$MANIFESTS_DIR/service.yaml" --ignore-not-found=true
fi

if [ -f "$MANIFESTS_DIR/configmap.yaml" ]; then
    echo "Deleting configmap..."
    kubectl delete -f "$MANIFESTS_DIR/configmap.yaml" --ignore-not-found=true
fi

if [ -f "$MANIFESTS_DIR/namespace.yaml" ]; then
    echo "Deleting namespace..."
    kubectl delete -f "$MANIFESTS_DIR/namespace.yaml" --ignore-not-found=true
fi

echo "Cleanup complete!"
"""
        
        with open(output_file, 'w') as f:
            f.write(script_content)
        
        os.chmod(output_file, 0o755)
    
    def _generate_pipeline_script(self, manifest_file: str, environment: str, output_file: str):
        """Generate pipeline execution script."""
        script_content = f"""#!/usr/bin/env python3
\"\"\"
MADEngine Execution Pipeline
Generated for environment: {environment}
\"\"\"

import os
import sys
import json
import time
import subprocess
from datetime import datetime

def main():
    \"\"\"Main pipeline execution function.\"\"\"
    print("=" * 80)
    print("MADEngine Execution Pipeline")
    print("=" * 80)
    print(f"Started: {{datetime.now().isoformat()}}")
    print(f"Environment: {environment}")
    
    # Load configuration
    with open('pipeline_config.json', 'r') as f:
        config = json.load(f)
    
    # Execute based on orchestrator type
    orchestrator_type = config.get('orchestrator_type', 'ansible')
    
    if orchestrator_type == 'ansible':
        return run_ansible_pipeline(config)
    elif orchestrator_type == 'k8s':
        return run_k8s_pipeline(config)
    else:
        print(f"Unknown orchestrator type: {{orchestrator_type}}")
        return 1

def run_ansible_pipeline(config):
    \"\"\"Run Ansible-based pipeline.\"\"\"
    print("Running Ansible pipeline...")
    
    # Run ansible playbook
    cmd = [
        'ansible-playbook',
        '-i', 'inventory.yml',
        'madengine_playbook.yml'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Ansible execution completed successfully")
        return 0
    else:
        print(f"Ansible execution failed: {{result.stderr}}")
        return 1

def run_k8s_pipeline(config):
    \"\"\"Run Kubernetes-based pipeline.\"\"\"
    print("Running Kubernetes pipeline...")
    
    # Deploy to Kubernetes
    result = subprocess.run(['./deploy.sh'], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Kubernetes deployment completed successfully")
        return 0
    else:
        print(f"Kubernetes deployment failed: {{result.stderr}}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
"""
        
        with open(output_file, 'w') as f:
            f.write(script_content)
        
        os.chmod(output_file, 0o755)
    
    def _generate_monitoring_script(self, manifest_file: str, environment: str, output_file: str):
        """Generate monitoring script."""
        script_content = f"""#!/usr/bin/env python3
\"\"\"
MADEngine Execution Monitoring
Generated for environment: {environment}
\"\"\"

import os
import sys
import json
import time
import subprocess
from datetime import datetime

def main():
    \"\"\"Main monitoring function.\"\"\"
    print("=" * 80)
    print("MADEngine Execution Monitor")
    print("=" * 80)
    print(f"Started: {{datetime.now().isoformat()}}")
    print(f"Environment: {environment}")
    
    # Load configuration
    with open('pipeline_config.json', 'r') as f:
        config = json.load(f)
    
    orchestrator_type = config.get('orchestrator_type', 'ansible')
    
    if orchestrator_type == 'k8s':
        return monitor_k8s_execution(config)
    else:
        print("Monitoring not implemented for this orchestrator type")
        return 0

def monitor_k8s_execution(config):
    \"\"\"Monitor Kubernetes execution.\"\"\"
    namespace = config.get('namespace', 'madengine-{environment}')
    
    print(f"Monitoring namespace: {{namespace}}")
    
    while True:
        try:
            # Check job status
            result = subprocess.run([
                'kubectl', 'get', 'jobs', '-n', namespace,
                '-o', 'json'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                jobs = json.loads(result.stdout)
                for job in jobs.get('items', []):
                    name = job['metadata']['name']
                    status = job.get('status', {{}})
                    
                    if status.get('succeeded', 0) > 0:
                        print(f"Job {{name}} completed successfully")
                        return 0
                    elif status.get('failed', 0) > 0:
                        print(f"Job {{name}} failed")
                        return 1
                    else:
                        print(f"Job {{name}} still running...")
            
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("Monitoring interrupted by user")
            return 0
        except Exception as e:
            print(f"Error monitoring: {{e}}")
            return 1

if __name__ == '__main__':
    sys.exit(main())
"""
        
        with open(output_file, 'w') as f:
            f.write(script_content)
        
        os.chmod(output_file, 0o755)
    
    def _generate_pipeline_config(self, manifest_file: str, environment: str, output_file: str):
        """Generate pipeline configuration."""
        # Load manifest for context
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        config = {
            "environment": environment,
            "orchestrator_type": "ansible",  # Default to ansible
            "namespace": f"madengine-{environment}",
            "manifest_file": manifest_file,
            "registry": manifest.get("registry", ""),
            "gpu_vendor": manifest.get("context", {}).get("gpu_vendor", ""),
            "monitoring": {
                "enabled": True,
                "interval": 30
            },
            "timeouts": {
                "execution": 7200,
                "monitoring": 14400
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)


# Convenience functions for backward compatibility
def generate_ansible_setup(manifest_file: str, environment: str = "default", 
                          output_dir: str = "ansible-setup") -> Dict[str, str]:
    """Generate complete Ansible setup."""
    generator = OrchestatorGenerator()
    return generator.generate_complete_ansible_setup(manifest_file, environment, output_dir)


def generate_k8s_setup(manifest_file: str, environment: str = "default", 
                       output_dir: str = "k8s-setup") -> Dict[str, List[str]]:
    """Generate complete Kubernetes setup."""
    generator = OrchestatorGenerator()
    return generator.generate_complete_k8s_setup(manifest_file, environment, output_dir)
