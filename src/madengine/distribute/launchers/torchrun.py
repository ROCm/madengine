"""Torchrun launcher for PyTorch DDP."""

import json
from typing import Dict, Any
from madengine.distribute.launchers.base import BaseLauncher


class TorchrunLauncher(BaseLauncher):
    """
    Torchrun launcher for PyTorch Distributed Data Parallel.
    
    Uses torchrun (elastic launch) for coordinated multi-node training.
    All nodes run torchrun with appropriate NODE_RANK.
    
    Uses existing multi_node_args pattern for backward compatibility.
    """
    
    def validate_config(self) -> None:
        """Validate torchrun configuration in multi_node_args."""
        required = ['NNODES', 'MAD_RUNTIME_NGPUS']
        for field in required:
            if field not in self.config:
                raise ValueError(f"torchrun launcher missing required field in multi_node_args: {field}")
    
    def generate_launch_command(
        self,
        node_rank: int,
        master_addr: str,
        total_nodes: int,
        manifest_path: str
    ) -> str:
        """
        Generate torchrun command for a specific node.
        
        Uses existing multi_node_args pattern from manifest and adds
        node-specific values (NODE_RANK, MASTER_ADDR).
        
        Example output:
            madengine-cli run \\
              --manifest-file build_manifest.json \\
              --additional-context '{"multi_node_args": {...}}' \\
              --verbose
        """
        # Start with base multi_node_args from manifest
        additional_context = self.manifest.get('additional_context', {}).copy()
        multi_node_args = additional_context.get('multi_node_args', {}).copy()
        
        # Add/override node-specific values
        multi_node_args['NODE_RANK'] = str(node_rank)
        multi_node_args['MASTER_ADDR'] = master_addr
        
        # Ensure RUNNER is set
        if 'RUNNER' not in multi_node_args:
            multi_node_args['RUNNER'] = 'torchrun'
        
        # Update additional_context with complete multi_node_args
        additional_context['multi_node_args'] = multi_node_args
        
        # Generate madengine-cli run command (same as legacy!)
        cmd = f"""madengine-cli run \\
  --manifest-file {manifest_path} \\
  --additional-context '{json.dumps(additional_context)}' \\
  --verbose"""
        
        return cmd
    
    def get_required_env_vars(self, node_rank: int, master_addr: str) -> Dict[str, str]:
        """Get environment variables for torchrun."""
        master_port = self.config.get('MASTER_PORT', '29500')
        network_interface = self.config.get('NCCL_SOCKET_IFNAME', 'eth0')
        
        env_vars = {
            'MASTER_ADDR': master_addr,
            'MASTER_PORT': master_port,
            'NODE_RANK': str(node_rank),
            'WORLD_SIZE': str(self.get_nodes_required()),
            'NCCL_SOCKET_IFNAME': network_interface,
            'GLOO_SOCKET_IFNAME': self.config.get('GLOO_SOCKET_IFNAME', network_interface),
            'NCCL_DEBUG': 'INFO',
        }
        
        return env_vars
    
    def get_nodes_required(self) -> int:
        """Get number of nodes required."""
        return int(self.config.get('NNODES', 1))
    
    def get_processes_per_node(self) -> int:
        """Get number of processes (GPUs) per node."""
        return int(self.config.get('MAD_RUNTIME_NGPUS', 8))

