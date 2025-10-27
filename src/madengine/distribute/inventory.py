"""
Inventory file loader and parser.

Handles loading and validating infrastructure inventory files.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml


class InventoryLoader:
    """Loads and validates infrastructure inventory files."""
    
    @staticmethod
    def load(inventory_path: str) -> Dict[str, Any]:
        """
        Load inventory file (YAML or JSON).
        
        Args:
            inventory_path: Path to inventory file
            
        Returns:
            Dict containing inventory data
            
        Raises:
            FileNotFoundError: If inventory file doesn't exist
            ValueError: If inventory format is invalid
        """
        if not os.path.exists(inventory_path):
            raise FileNotFoundError(f"Inventory file not found: {inventory_path}")
        
        # Load based on extension
        inventory_path = Path(inventory_path)
        if inventory_path.suffix in ['.yml', '.yaml']:
            with open(inventory_path) as f:
                data = yaml.safe_load(f)
        elif inventory_path.suffix == '.json':
            with open(inventory_path) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported inventory format: {inventory_path.suffix}")
        
        # Validate structure
        InventoryLoader._validate(data)
        
        return data
    
    @staticmethod
    def _validate(data: Dict[str, Any]) -> None:
        """
        Validate inventory structure.
        
        Args:
            data: Inventory data
            
        Raises:
            ValueError: If structure is invalid
        """
        if 'nodes' not in data:
            raise ValueError("Inventory must contain 'nodes' section")
        
        nodes = data['nodes']
        if not isinstance(nodes, list) or len(nodes) == 0:
            raise ValueError("'nodes' must be a non-empty list")
        
        # Validate each node
        required_fields = ['hostname', 'address']
        for i, node in enumerate(nodes):
            for field in required_fields:
                if field not in node:
                    raise ValueError(f"Node {i} missing required field: {field}")
    
    @staticmethod
    def get_master_node(inventory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the master/rank-0 node.
        
        Args:
            inventory: Inventory data
            
        Returns:
            Master node dict
        """
        # First node is master by default
        return inventory['nodes'][0]
    
    @staticmethod
    def get_worker_nodes(inventory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get worker nodes (all nodes except first).
        
        Args:
            inventory: Inventory data
            
        Returns:
            List of worker node dicts
        """
        return inventory['nodes'][1:]
    
    @staticmethod
    def get_all_nodes(inventory: Dict[str, Any], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all nodes.
        
        Args:
            inventory: Inventory data
            limit: Optional limit on number of nodes
            
        Returns:
            List of node dicts
        """
        nodes = inventory['nodes']
        if limit:
            nodes = nodes[:limit]
        return nodes
    
    @staticmethod
    def get_node_count(inventory: Dict[str, Any]) -> int:
        """Get total number of nodes."""
        return len(inventory['nodes'])
    
    @staticmethod
    def get_global_config(inventory: Dict[str, Any]) -> Dict[str, Any]:
        """Get global configuration section."""
        return inventory.get('global', {})
    
    @staticmethod
    def get_infrastructure_config(inventory: Dict[str, Any], infrastructure: str) -> Dict[str, Any]:
        """Get infrastructure-specific configuration."""
        infra_config = inventory.get('infrastructure', {})
        return infra_config.get(infrastructure, {})

