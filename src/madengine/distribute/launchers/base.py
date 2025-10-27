"""Base launcher interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseLauncher(ABC):
    """
    Base class for distributed launchers.
    
    Launchers define HOW processes communicate (torchrun, mpirun, etc.).
    """
    
    def __init__(self, launcher_config: Dict[str, Any], manifest: Dict[str, Any]):
        """
        Initialize launcher.
        
        Args:
            launcher_config: Launcher configuration from manifest
            manifest: Full build manifest
        """
        self.config = launcher_config
        self.manifest = manifest
        self.launcher_type = launcher_config.get('type')
    
    @abstractmethod
    def generate_launch_command(
        self,
        node_rank: int,
        master_addr: str,
        total_nodes: int,
        manifest_path: str
    ) -> str:
        """
        Generate the launch command for a specific node.
        
        Args:
            node_rank: Rank of this node (0 = master)
            master_addr: Master node address
            total_nodes: Total number of nodes
            manifest_path: Path to manifest file
            
        Returns:
            Launch command string
        """
        pass
    
    @abstractmethod
    def get_required_env_vars(self, node_rank: int, master_addr: str) -> Dict[str, str]:
        """
        Get required environment variables for this launcher.
        
        Args:
            node_rank: Rank of this node
            master_addr: Master node address
            
        Returns:
            Dict of environment variables
        """
        pass
    
    def validate_config(self) -> None:
        """
        Validate launcher configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    def get_nodes_required(self) -> int:
        """Get number of nodes required for this launcher."""
        return self.config.get('nnodes', self.config.get('nodes', 1))
    
    def get_processes_per_node(self) -> int:
        """Get number of processes per node."""
        return self.config.get('nproc_per_node', self.config.get('ppn', 1))

