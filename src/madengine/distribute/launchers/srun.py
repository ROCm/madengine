"""SLURM srun launcher."""

import json
from typing import Dict, Any
from madengine.distribute.launchers.base import BaseLauncher


class SrunLauncher(BaseLauncher):
    """
    SLURM srun launcher for HPC clusters.
    
    Uses srun to launch processes in SLURM-allocated nodes.
    Works within sbatch scripts.
    """
    
    def validate_config(self) -> None:
        """Validate srun configuration."""
        required = ['nodes']
        for field in required:
            if field not in self.config:
                raise ValueError(f"srun launcher missing required field: {field}")
    
    def generate_launch_command(
        self,
        node_rank: int,
        master_addr: str,
        total_nodes: int,
        manifest_path: str
    ) -> str:
        """
        Generate command to run within srun context.
        
        This is used within the sbatch script where srun handles distribution.
        """
        nproc_per_node = self.config.get('ntasks_per_node', 8)
        
        # Build additional context with SLURM-aware multi_node_args
        additional_context = self.manifest.get('additional_context', {}).copy()
        
        # In SLURM context, use SLURM environment variables
        multi_node_args = {
            'RUNNER': 'torchrun',
            'MASTER_ADDR': '$MASTER_ADDR',  # Set by sbatch script
            'MASTER_PORT': self.config.get('master_port', '29500'),
            'NNODES': '$SLURM_NNODES',
            'NODE_RANK': '$SLURM_PROCID',
            'MAD_RUNTIME_NGPUS': str(nproc_per_node),
            'NCCL_SOCKET_IFNAME': self.config.get('network_interface', 'ib0'),
            'GLOO_SOCKET_IFNAME': self.config.get('network_interface', 'ib0'),
        }
        
        additional_context['multi_node_args'] = multi_node_args
        
        # Generate madengine run command
        # Note: additional_context will be expanded by shell
        cmd = f"""madengine-cli run \\
  --manifest-file {manifest_path} \\
  --additional-context "$(cat <<EOF
{json.dumps(additional_context, indent=2)}
EOF
)" \\
  --verbose"""
        
        return cmd
    
    def get_required_env_vars(self, node_rank: int, master_addr: str) -> Dict[str, str]:
        """Get environment variables for srun."""
        network_interface = self.config.get('network_interface', 'ib0')
        
        env_vars = {
            'NCCL_SOCKET_IFNAME': network_interface,
            'GLOO_SOCKET_IFNAME': network_interface,
            'NCCL_DEBUG': 'INFO',
        }
        
        # SLURM sets these automatically
        # SLURM_PROCID, SLURM_NNODES, SLURM_NODELIST, etc.
        
        return env_vars
    
    def get_nodes_required(self) -> int:
        """Get number of nodes required."""
        return self.config.get('nodes', 1)
    
    def get_processes_per_node(self) -> int:
        """Get number of tasks per node."""
        return self.config.get('ntasks_per_node', 1)

