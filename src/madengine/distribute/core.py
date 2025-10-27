"""Core distribute orchestrator."""

import json
import os
import tempfile
from typing import Dict, Any, Optional
from pathlib import Path

from madengine.distribute.inventory import InventoryLoader
from madengine.distribute.launchers import (
    TorchrunLauncher,
    MPIRunLauncher,
    SrunLauncher,
    K8sLauncher,
)
from madengine.distribute.infrastructures import (
    SSHInfrastructure,
    AnsibleInfrastructure,
    SlurmInfrastructure,
    K8sInfrastructure,
)
from madengine.distribute.infrastructures.base import ExecutionResult


class DistributeOrchestrator:
    """
    Main orchestrator for distributed workload execution.
    
    Coordinates launcher and infrastructure to execute distributed training/inference.
    """
    
    # Launcher registry
    LAUNCHER_MAP = {
        'torchrun': TorchrunLauncher,
        'mpirun': MPIRunLauncher,
        'srun': SrunLauncher,
        'k8s': K8sLauncher,
    }
    
    # Infrastructure registry
    INFRASTRUCTURE_MAP = {
        'ssh': SSHInfrastructure,
        'ansible': AnsibleInfrastructure,
        'slurm': SlurmInfrastructure,
        'k8s': K8sInfrastructure,
    }
    
    def __init__(
        self,
        infrastructure: str,
        manifest_file: str,
        inventory_file: str,
        output_dir: Optional[str] = None,
        dry_run: bool = False,
        verbose: bool = False
    ):
        """
        Initialize orchestrator.
        
        Args:
            infrastructure: Infrastructure type (ssh, ansible, slurm, k8s)
            manifest_file: Path to build manifest
            inventory_file: Path to inventory file
            output_dir: Output directory for generated files
            dry_run: Generate files but don't execute
            verbose: Enable verbose logging
        """
        self.infrastructure_type = infrastructure.lower()
        self.manifest_file = manifest_file
        self.inventory_file = inventory_file
        self.dry_run = dry_run
        self.verbose = verbose
        
        # Setup output directory
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.cleanup_needed = False
        else:
            self.output_dir = Path(tempfile.mkdtemp(prefix=f"madengine_{infrastructure}_"))
            self.cleanup_needed = True
        
        # Load configuration
        self.manifest = self._load_manifest()
        self.inventory_data = InventoryLoader.load(inventory_file)
        
        # Validate
        self._validate()
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load build manifest file."""
        if not os.path.exists(self.manifest_file):
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_file}")
        
        with open(self.manifest_file, 'r') as f:
            manifest = json.load(f)
        
        # Store manifest path for reference
        manifest['_path'] = os.path.abspath(self.manifest_file)
        
        return manifest
    
    def _validate(self):
        """Validate configuration."""
        # Validate infrastructure type
        if self.infrastructure_type not in self.INFRASTRUCTURE_MAP:
            raise ValueError(
                f"Unsupported infrastructure: {self.infrastructure_type}. "
                f"Supported: {', '.join(self.INFRASTRUCTURE_MAP.keys())}"
            )
        
        # Validate multi_node_args exists in manifest
        additional_context = self.manifest.get('additional_context', {})
        if 'multi_node_args' not in additional_context:
            raise ValueError(
                "manifest missing 'multi_node_args' in additional_context. "
                "Please configure launcher settings during build phase."
            )
        
        multi_node_args = additional_context['multi_node_args']
        
        # Validate RUNNER is specified
        runner_type = multi_node_args.get('RUNNER', 'torchrun').lower()
        if runner_type not in self.LAUNCHER_MAP:
            raise ValueError(
                f"Unsupported RUNNER: {runner_type}. "
                f"Supported: {', '.join(self.LAUNCHER_MAP.keys())}"
            )
        
        # Validate inventory has nodes (except for K8s which manages its own)
        nodes = self.inventory_data.get('nodes', [])
        
        # K8s infrastructure doesn't require nodes in inventory
        if self.infrastructure_type != 'k8s':
            if not nodes:
                raise ValueError("Inventory must contain at least one node")
            
            # Validate NNODES doesn't exceed available nodes
            nnodes_required = int(multi_node_args.get('NNODES', 1))
            nodes_available = len(nodes)
            
            if nnodes_required > nodes_available:
                raise ValueError(
                    f"NNODES={nnodes_required} in multi_node_args exceeds "
                    f"available nodes in inventory ({nodes_available}). "
                    f"Please add more nodes to inventory or reduce NNODES."
                )
    
    def execute(self) -> ExecutionResult:
        """
        Execute distributed workload.
        
        Returns:
            ExecutionResult with status and details
        """
        try:
            # Phase 1: Create launcher
            if self.verbose:
                print(f"\n=== Phase 1: Initializing Launcher ===")
            
            launcher = self._create_launcher()
            
            if self.verbose:
                print(f"Launcher: {launcher.__class__.__name__}")
                print(f"Nodes required: {launcher.get_nodes_required()}")
                print(f"Processes per node: {launcher.get_processes_per_node()}")
            
            # Phase 2: Create infrastructure
            if self.verbose:
                print(f"\n=== Phase 2: Initializing Infrastructure ===")
            
            infrastructure = self._create_infrastructure()
            
            if self.verbose:
                print(f"Infrastructure: {infrastructure.__class__.__name__}")
                print(f"Output directory: {self.output_dir}")
            
            # Phase 3: Generate orchestration files
            if self.verbose:
                print(f"\n=== Phase 3: Generating Orchestration Files ===")
            
            orchestration = infrastructure.generate_orchestration(launcher)
            
            if self.verbose:
                print(f"Generated files in: {self.output_dir}")
            
            # Phase 4: Execute (unless dry run)
            if self.dry_run:
                if self.verbose:
                    print(f"\n=== Dry Run Mode ===")
                    print(f"Files generated in: {self.output_dir}")
                    print(f"Skipping execution")
                
                result = ExecutionResult()
                result.success = True
                result.total_nodes = 0
                return result
            
            if self.verbose:
                print(f"\n=== Phase 4: Executing Distributed Workload ===")
            
            result = infrastructure.execute(orchestration)
            
            if self.verbose:
                print(f"\n=== Execution Complete ===")
                print(f"Success: {result.success}")
                print(f"Total nodes: {result.total_nodes}")
                print(f"Successful: {result.successful_nodes}")
                print(f"Failed: {result.failed_nodes}")
                print(f"Execution time: {result.execution_time:.2f}s")
            
            return result
            
        finally:
            # Cleanup temporary directory if needed
            if self.cleanup_needed and not self.dry_run:
                import shutil
                if self.verbose:
                    print(f"\nCleaning up: {self.output_dir}")
                shutil.rmtree(self.output_dir, ignore_errors=True)
    
    def _create_launcher(self):
        """Create launcher instance."""
        additional_context = self.manifest.get('additional_context', {})
        multi_node_args = additional_context.get('multi_node_args', {})
        
        runner_type = multi_node_args.get('RUNNER', 'torchrun').lower()
        
        launcher_class = self.LAUNCHER_MAP[runner_type]
        launcher = launcher_class(
            launcher_config=multi_node_args,
            manifest=self.manifest
        )
        
        # Validate configuration
        launcher.validate_config()
        
        return launcher
    
    def _create_infrastructure(self):
        """Create infrastructure instance."""
        infrastructure_class = self.INFRASTRUCTURE_MAP[self.infrastructure_type]
        
        infrastructure = infrastructure_class(
            inventory_data=self.inventory_data,
            manifest=self.manifest,
            output_dir=str(self.output_dir),
            verbose=self.verbose
        )
        
        return infrastructure


def distribute_workload(
    infrastructure: str,
    manifest_file: str,
    inventory_file: str,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
    verbose: bool = False
) -> ExecutionResult:
    """
    Convenience function to distribute workload.
    
    Args:
        infrastructure: Infrastructure type (ssh, ansible, slurm, k8s)
        manifest_file: Path to build manifest
        inventory_file: Path to inventory file
        output_dir: Optional output directory
        dry_run: Generate files but don't execute
        verbose: Enable verbose logging
        
    Returns:
        ExecutionResult with status and details
    """
    orchestrator = DistributeOrchestrator(
        infrastructure=infrastructure,
        manifest_file=manifest_file,
        inventory_file=inventory_file,
        output_dir=output_dir,
        dry_run=dry_run,
        verbose=verbose
    )
    
    return orchestrator.execute()

