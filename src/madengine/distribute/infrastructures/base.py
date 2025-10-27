"""Base infrastructure interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path


class ExecutionResult:
    """Result of distributed execution."""
    
    def __init__(self):
        self.success = False
        self.total_nodes = 0
        self.successful_nodes = 0
        self.failed_nodes = 0
        self.node_results = {}
        self.errors = []
        self.execution_time = 0.0
        
    def add_node_result(self, node_name: str, success: bool, output: str = "", error: str = ""):
        """Add result for a specific node."""
        self.node_results[node_name] = {
            'success': success,
            'output': output,
            'error': error
        }
        
        if success:
            self.successful_nodes += 1
        else:
            self.failed_nodes += 1
            if error:
                self.errors.append(f"{node_name}: {error}")
        
        self.total_nodes = len(self.node_results)
        self.success = (self.failed_nodes == 0)


class BaseInfrastructure(ABC):
    """
    Base class for infrastructure implementations.
    
    Infrastructures define HOW to reach nodes (SSH, Ansible, etc.).
    """
    
    def __init__(
        self,
        inventory_data: Dict[str, Any],
        manifest: Dict[str, Any],
        output_dir: str,
        verbose: bool = False
    ):
        """
        Initialize infrastructure.
        
        Args:
            inventory_data: Loaded inventory data
            manifest: Build manifest
            output_dir: Directory for generated files
            verbose: Enable verbose logging
        """
        self.inventory_data = inventory_data
        self.manifest = manifest
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def generate_orchestration(self, launcher) -> Dict[str, Any]:
        """
        Generate infrastructure-specific orchestration files.
        
        Args:
            launcher: Launcher instance (torchrun, mpirun, etc.)
            
        Returns:
            Dict with generated file paths and metadata
        """
        pass
    
    @abstractmethod
    def execute(self, orchestration: Dict[str, Any]) -> ExecutionResult:
        """
        Execute the distributed workload.
        
        Args:
            orchestration: Output from generate_orchestration()
            
        Returns:
            ExecutionResult with status and node results
        """
        pass
    
    def get_nodes(self) -> List[Dict[str, Any]]:
        """Get list of nodes from inventory."""
        return self.inventory_data.get('nodes', [])
    
    def get_master_node(self) -> Dict[str, Any]:
        """Get master node (first in list)."""
        nodes = self.get_nodes()
        if not nodes:
            raise ValueError("No nodes found in inventory")
        return nodes[0]
    
    def get_global_config(self) -> Dict[str, Any]:
        """Get global configuration from inventory."""
        return self.inventory_data.get('global', {})
    
    def get_infrastructure_config(self) -> Dict[str, Any]:
        """Get infrastructure-specific configuration."""
        infra_config = self.inventory_data.get('infrastructure', {})
        return infra_config.get(self.infrastructure_name, {})
    
    @property
    @abstractmethod
    def infrastructure_name(self) -> str:
        """Return infrastructure name (ssh, ansible, slurm, k8s)."""
        pass
    
    def log(self, message: str, force: bool = False):
        """Log message if verbose is enabled."""
        if self.verbose or force:
            print(f"[{self.infrastructure_name.upper()}] {message}")

