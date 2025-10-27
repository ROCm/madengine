"""Kubernetes launcher (PyTorchJob)."""

import json
from typing import Dict, Any
from madengine.distribute.launchers.base import BaseLauncher


class K8sLauncher(BaseLauncher):
    """
    Kubernetes launcher using PyTorchJob CRD.
    
    Uses Kubeflow PyTorchJob for coordinated distributed training.
    PyTorchJob automatically handles master/worker coordination.
    """
    
    def validate_config(self) -> None:
        """Validate k8s configuration."""
        # replicas = master + workers
        if 'replicas' not in self.config:
            self.config['replicas'] = self.config.get('nnodes', 1)
        
        # workers = replicas - 1 (master)
        if 'workers' not in self.config:
            self.config['workers'] = max(0, self.config['replicas'] - 1)
    
    def generate_launch_command(
        self,
        node_rank: int,
        master_addr: str,
        total_nodes: int,
        manifest_path: str
    ) -> str:
        """
        Generate command for PyTorchJob pod.
        
        PyTorchJob automatically sets:
        - MASTER_ADDR, MASTER_PORT
        - RANK, WORLD_SIZE
        - PET_NNODES, PET_NODE_RANK
        """
        nproc_per_node = self.config.get('gpus_per_replica', 8)
        
        # Build additional context using PyTorchJob env vars
        additional_context = self.manifest.get('additional_context', {}).copy()
        
        # PyTorchJob sets these automatically via env vars
        multi_node_args = {
            'RUNNER': 'torchrun',
            'MASTER_ADDR': '$MASTER_ADDR',
            'MASTER_PORT': '$MASTER_PORT',
            'NNODES': '$WORLD_SIZE',
            'NODE_RANK': '$RANK',
            'MAD_RUNTIME_NGPUS': str(nproc_per_node),
            'NCCL_SOCKET_IFNAME': self.config.get('network_interface', 'eth0'),
            'GLOO_SOCKET_IFNAME': self.config.get('network_interface', 'eth0'),
        }
        
        additional_context['multi_node_args'] = multi_node_args
        
        cmd = f"""#!/bin/bash
set -e

# PyTorchJob environment
echo "PyTorchJob Pod: RANK=$RANK, WORLD_SIZE=$WORLD_SIZE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Ensure MAD is set up
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

# Copy manifest
cp /config/build_manifest.json ./

# Run with PyTorchJob environment
madengine-cli run \\
  --manifest-file build_manifest.json \\
  --additional-context '{json.dumps(additional_context)}' \\
  --verbose
"""
        
        return cmd
    
    def get_required_env_vars(self, node_rank: int, master_addr: str) -> Dict[str, str]:
        """
        Get environment variables for Kubernetes.
        
        PyTorchJob sets most of these automatically.
        """
        network_interface = self.config.get('network_interface', 'eth0')
        
        env_vars = {
            'NCCL_SOCKET_IFNAME': network_interface,
            'GLOO_SOCKET_IFNAME': network_interface,
            'NCCL_DEBUG': 'INFO',
        }
        
        # PyTorchJob automatically sets:
        # - MASTER_ADDR, MASTER_PORT
        # - RANK, WORLD_SIZE
        # - PET_NNODES, PET_NODE_RANK
        
        return env_vars
    
    def get_nodes_required(self) -> int:
        """Get number of replicas (pods) required."""
        return self.config.get('replicas', 1)
    
    def get_processes_per_node(self) -> int:
        """Get number of GPUs per replica."""
        return self.config.get('gpus_per_replica', 8)
    
    def get_master_replicas(self) -> int:
        """Get number of master replicas (always 1 for PyTorchJob)."""
        return 1
    
    def get_worker_replicas(self) -> int:
        """Get number of worker replicas."""
        return self.config.get('workers', self.get_nodes_required() - 1)

