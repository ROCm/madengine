"""Kubernetes infrastructure for cloud deployments."""

import os
import json
import subprocess
import time
import yaml
from typing import Dict, Any
from pathlib import Path

from madengine.distribute.infrastructures.base import BaseInfrastructure, ExecutionResult


class K8sInfrastructure(BaseInfrastructure):
    """
    Kubernetes infrastructure for cloud-native distributed execution.
    
    Generates PyTorchJob manifests and deploys to Kubernetes cluster.
    """
    
    @property
    def infrastructure_name(self) -> str:
        return "k8s"
    
    def get_nodes(self):
        """
        K8s doesn't need explicit nodes - Kubernetes scheduler handles node selection.
        
        Returns:
            Empty list since K8s manages nodes automatically
        """
        return []
    
    def get_master_node(self):
        """
        K8s doesn't have a traditional master node concept.
        PyTorchJob manages master/worker coordination.
        
        Returns:
            None
        """
        return None
    
    def generate_orchestration(self, launcher) -> Dict[str, Any]:
        """
        Generate Kubernetes manifests (PyTorchJob).
        
        Args:
            launcher: Launcher instance
            
        Returns:
            Dict with manifest paths
        """
        self.log("Generating Kubernetes orchestration files...")
        
        # Get K8s config
        infra_config = self.get_infrastructure_config()
        namespace = infra_config.get('namespace', 'madengine')
        
        # Generate unique job name to allow concurrent jobs
        import time
        job_name = f"madengine-distributed-{int(time.time())}"
        
        # Generate PyTorchJob manifest with the job name
        pytorchjob_path = self._generate_pytorchjob(
            launcher, namespace, infra_config, job_name
        )
        
        # Generate ConfigMap for manifest
        configmap_path = self._generate_configmap(namespace)
        
        return {
            'type': 'k8s',
            'pytorchjob': str(pytorchjob_path),
            'configmap': str(configmap_path),
            'namespace': namespace,
            'job_name': job_name  # Store job name for monitoring
        }
    
    def _generate_pytorchjob(
        self,
        launcher,
        namespace: str,
        infra_config: Dict[str, Any],
        job_name: str
    ) -> Path:
        """Generate PyTorchJob manifest.
        
        Args:
            launcher: Launcher instance
            namespace: Kubernetes namespace
            infra_config: Infrastructure configuration
            job_name: Unique job name
            
        Returns:
            Path to generated PyTorchJob manifest
        """
        pytorchjob_path = self.output_dir / "pytorchjob.yaml"
        
        total_replicas = launcher.get_nodes_required()
        gpus_per_replica = launcher.get_processes_per_node()
        
        # Get resources
        resources = infra_config.get('resources', {})
        requests = resources.get('requests', {
            'cpu': '16',
            'memory': '128Gi',
            'amd.com/gpu': str(gpus_per_replica)
        })
        limits = resources.get('limits', requests.copy())
        
        # Get node selector
        node_selector = infra_config.get('node_selector', {})
        
        # Get image from manifest
        models = self.manifest.get('models', [])
        image = models[0]['registry_image'] if models else 'ubuntu:latest'
        
        # Get global config
        global_config = self.get_global_config()
        mad_workspace = global_config.get('mad_workspace', '/workspace/MAD')
        shared_fs = global_config.get('shared_filesystem', '/data')
        
        # Build PyTorchJob spec
        
        pytorchjob = {
            'apiVersion': 'kubeflow.org/v1',
            'kind': 'PyTorchJob',
            'metadata': {
                'name': job_name,
                'namespace': namespace
            },
            'spec': {
                'pytorchReplicaSpecs': {
                    'Master': {
                        'replicas': 1,
                        'restartPolicy': 'OnFailure',
                        'template': {
                            'spec': {
                                'containers': [{
                                    'name': 'pytorch',
                                    'image': image,
                                    'command': ['/bin/bash', '-c'],
                                    'args': [self._generate_container_script(mad_workspace, shared_fs)],
                                    'resources': {
                                        'requests': requests,
                                        'limits': limits
                                    },
                                    'volumeMounts': [
                                        {
                                            'name': 'manifest',
                                            'mountPath': '/config',
                                            'readOnly': True
                                        }
                                    ]
                                }],
                                'volumes': [
                                    {
                                        'name': 'manifest',
                                        'configMap': {
                                            'name': 'madengine-manifest'
                                        }
                                    }
                                ]
                            }
                        }
                    },
                    'Worker': {
                        'replicas': total_replicas - 1,
                        'restartPolicy': 'OnFailure',
                        'template': {
                            'spec': {
                                'containers': [{
                                    'name': 'pytorch',
                                    'image': image,
                                    'command': ['/bin/bash', '-c'],
                                    'args': [self._generate_container_script(mad_workspace, shared_fs)],
                                    'resources': {
                                        'requests': requests,
                                        'limits': limits
                                    },
                                    'volumeMounts': [
                                        {
                                            'name': 'manifest',
                                            'mountPath': '/config',
                                            'readOnly': True
                                        }
                                    ]
                                }],
                                'volumes': [
                                    {
                                        'name': 'manifest',
                                        'configMap': {
                                            'name': 'madengine-manifest'
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        }
        
        # Add node selector if specified
        if node_selector:
            pytorchjob['spec']['pytorchReplicaSpecs']['Master']['template']['spec']['nodeSelector'] = node_selector
            pytorchjob['spec']['pytorchReplicaSpecs']['Worker']['template']['spec']['nodeSelector'] = node_selector
        
        with open(pytorchjob_path, 'w') as f:
            yaml.dump(pytorchjob, f, default_flow_style=False)
        
        self.log(f"Generated PyTorchJob: {pytorchjob_path}")
        
        return pytorchjob_path
    
    def _generate_container_script(self, mad_workspace: str, shared_fs: str) -> str:
        """Generate container startup script."""
        
        # Get multi_node_args from manifest
        additional_context = self.manifest.get('additional_context', {}).copy()
        multi_node_args = additional_context.get('multi_node_args', {}).copy()
        
        script = f"""
set -e

echo "========================================="
echo "PyTorchJob Container Starting"
echo "RANK: $RANK"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "========================================="

# Setup MAD workspace
cd /workspace
if [ ! -d "MAD" ]; then
    git clone https://github.com/ROCm/MAD.git
    cd MAD
    python3 -m venv venv
    source venv/bin/activate
    pip install madengine
else
    cd MAD
    source venv/bin/activate
fi

# Copy manifest from ConfigMap
cp /config/build_manifest.json ./

# Build additional_context with PyTorchJob env vars
cat > /tmp/additional_context.json <<'EOF'
{json.dumps(additional_context, indent=2)}
EOF

# Update with PyTorchJob environment variables
python3 <<'PYTHON_SCRIPT'
import json
import os

with open('/tmp/additional_context.json', 'r') as f:
    ctx = json.load(f)

# Update multi_node_args with PyTorchJob values
if 'multi_node_args' in ctx:
    ctx['multi_node_args']['NODE_RANK'] = os.environ.get('RANK', '0')
    ctx['multi_node_args']['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    ctx['multi_node_args']['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')

with open('/tmp/additional_context.json', 'w') as f:
    json.dump(ctx, f)
PYTHON_SCRIPT

echo "Starting training..."

# Run madengine
madengine-cli run \\
  --manifest-file build_manifest.json \\
  --additional-context "$(cat /tmp/additional_context.json)" \\
  --verbose

exit_code=$?
echo "Container completed with exit code: $exit_code"
exit $exit_code
"""
        return script
    
    def _generate_configmap(self, namespace: str) -> Path:
        """Generate ConfigMap for manifest."""
        
        configmap_path = self.output_dir / "configmap.yaml"
        
        # Read manifest
        manifest_data = json.dumps(self.manifest, indent=2)
        
        configmap = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': 'madengine-manifest',
                'namespace': namespace
            },
            'data': {
                'build_manifest.json': manifest_data
            }
        }
        
        with open(configmap_path, 'w') as f:
            yaml.dump(configmap, f, default_flow_style=False)
        
        self.log(f"Generated ConfigMap: {configmap_path}")
        
        return configmap_path
    
    def execute(self, orchestration: Dict[str, Any]) -> ExecutionResult:
        """
        Execute distributed workload via Kubernetes.
        
        Applies manifests and waits for PyTorchJob to complete.
        """
        self.log("Executing distributed workload via Kubernetes...", force=True)
        
        result = ExecutionResult()
        
        pytorchjob = orchestration['pytorchjob']
        configmap = orchestration['configmap']
        namespace = orchestration['namespace']
        job_name = orchestration.get('job_name', 'madengine-distributed')
        
        start_time = time.time()
        
        try:
            # Apply ConfigMap
            self.log("Creating ConfigMap...")
            subprocess.run(
                ['kubectl', 'apply', '-f', configmap],
                check=True,
                capture_output=True
            )
            
            # Apply PyTorchJob
            self.log("Creating PyTorchJob...")
            subprocess.run(
                ['kubectl', 'apply', '-f', pytorchjob],
                check=True,
                capture_output=True
            )
            
            # Wait for PyTorchJob to complete
            self.log("Waiting for PyTorchJob to complete...", force=True)
            
            while True:
                time.sleep(30)
                
                # Check job status
                check_cmd = [
                    'kubectl', 'get', 'pytorchjob', job_name,
                    '-n', namespace, '-o', 'json'
                ]
                
                check_result = subprocess.run(
                    check_cmd,
                    capture_output=True,
                    text=True
                )
                
                if check_result.returncode != 0:
                    break
                
                job_status = json.loads(check_result.stdout)
                conditions = job_status.get('status', {}).get('conditions', [])
                
                for condition in conditions:
                    if condition['type'] == 'Succeeded' and condition['status'] == 'True':
                        result.success = True
                        break
                    elif condition['type'] == 'Failed' and condition['status'] == 'True':
                        result.success = False
                        result.errors.append("PyTorchJob failed")
                        break
                
                if result.success or result.errors:
                    break
                
                if self.verbose:
                    self.log("PyTorchJob still running...")
            
            result.execution_time = time.time() - start_time
            
            if result.success:
                result.successful_nodes = 1
                result.total_nodes = 1
                self.log("PyTorchJob completed successfully", force=True)
            else:
                result.failed_nodes = 1
                result.total_nodes = 1
                self.log("PyTorchJob failed", force=True)
            
            # Retrieve logs
            self.log("Retrieving pod logs...")
            
            logs_cmd = [
                'kubectl', 'logs', '-n', namespace,
                '-l', f'job-name={job_name}',
                '--tail=-1'
            ]
            
            logs_result = subprocess.run(
                logs_cmd,
                capture_output=True,
                text=True
            )
            
            # Save logs
            with open(self.output_dir / 'k8s_logs.txt', 'w') as f:
                f.write(logs_result.stdout)
            
        except FileNotFoundError:
            result.success = False
            result.errors.append("kubectl command not found. Is kubectl installed?")
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            result.execution_time = time.time() - start_time
        
        self.log(f"Execution completed in {result.execution_time:.2f}s", force=True)
        
        return result

