"""MPI launcher for distributed training."""

import json
from typing import Dict, Any
from madengine.distribute.launchers.base import BaseLauncher


class MPIRunLauncher(BaseLauncher):
    """
    MPI launcher for distributed training.
    
    Uses mpirun to launch processes across nodes.
    Only master node (rank 0) runs mpirun command.
    """
    
    def validate_config(self) -> None:
        """Validate mpirun configuration."""
        if 'np' not in self.config:
            raise ValueError("mpirun launcher missing required field: np (total processes)")
    
    def generate_launch_command(
        self,
        node_rank: int,
        master_addr: str,
        total_nodes: int,
        manifest_path: str
    ) -> str:
        """
        Generate mpirun command.
        
        Only rank 0 (master) runs mpirun. Workers wait for MPI to spawn them.
        """
        if node_rank == 0:
            # Master node runs mpirun
            np = self.config.get('np', total_nodes * self.config.get('ppn', 8))
            hostfile = self.config.get('hostfile', '/tmp/mpi_hostfile')
            mca_params = self.config.get('mca', 'btl_tcp_if_include=eth0')
            
            # Build additional context for madengine
            additional_context = self.manifest.get('additional_context', {}).copy()
            additional_context['launcher'] = {
                'type': 'mpirun',
                'node_rank': node_rank
            }
            
            cmd = f"""# Master node - run mpirun
mpirun -np {np} \\
  -hostfile {hostfile} \\
  --mca {mca_params} \\
  --bind-to none \\
  --map-by slot \\
  -x NCCL_SOCKET_IFNAME={self.config.get('network_interface', 'eth0')} \\
  -x NCCL_DEBUG=INFO \\
  python -m madengine.run_from_manifest {manifest_path}"""
        else:
            # Worker nodes - just wait (MPI will spawn processes)
            cmd = """# Worker node - waiting for MPI master
echo "Worker node {node_rank} - MPI processes will be spawned by master"
# Keep container alive
tail -f /dev/null"""
        
        return cmd
    
    def get_required_env_vars(self, node_rank: int, master_addr: str) -> Dict[str, str]:
        """Get environment variables for mpirun."""
        network_interface = self.config.get('network_interface', 'eth0')
        
        env_vars = {
            'MASTER_ADDR': master_addr,
            'NODE_RANK': str(node_rank),
            'NCCL_SOCKET_IFNAME': network_interface,
            'NCCL_DEBUG': 'INFO',
        }
        
        return env_vars
    
    def get_nodes_required(self) -> int:
        """Get number of nodes required."""
        np = self.config.get('np')
        ppn = self.config.get('ppn', 8)
        return (np + ppn - 1) // ppn  # Ceiling division
    
    def get_processes_per_node(self) -> int:
        """Get number of processes per node."""
        return self.config.get('ppn', 8)

