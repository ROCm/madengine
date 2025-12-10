#!/usr/bin/env python3
"""
Configuration loader with multi-layer merging for deployments.

Layers (low to high priority):
1. System defaults (built-in presets)
2. User file (--additional-context-file)
3. User CLI (--additional-context)

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy


class ConfigLoader:
    """Smart configuration loader with preset support."""
    
    PRESET_DIR = Path(__file__).parent / "presets"
    
    @classmethod
    def load_preset(cls, preset_path: str) -> Dict[str, Any]:
        """
        Load a preset JSON file.
        
        Args:
            preset_path: Relative path to preset file from PRESET_DIR
            
        Returns:
            Dict containing preset configuration, or empty dict if not found
        """
        full_path = cls.PRESET_DIR / preset_path
        if not full_path.exists():
            return {}
        
        try:
            with open(full_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load preset {preset_path}: {e}")
            return {}
    
    @classmethod
    def deep_merge(cls, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries. Override wins conflicts.
        Nested dicts are merged, lists/primitives are replaced.
        Special handling: env_vars are merged (not replaced).
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = deepcopy(base)
        
        for key, value in override.items():
            # Skip documentation/comment fields from base if override has them
            if key.startswith('_'):
                result[key] = deepcopy(value)
                continue
                
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = cls.deep_merge(result[key], value)
            else:
                # Replace with override value
                result[key] = deepcopy(value)
        
        return result
    
    @classmethod
    def detect_profile_needs(cls, config: Dict) -> Dict[str, bool]:
        """
        Detect what profiles/optimizations are needed.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dict with flags: is_single_gpu, is_multi_gpu, is_multi_node, is_distributed
        """
        distributed = config.get("distributed", {})
        gpu_count = config.get("k8s", {}).get("gpu_count", 1)
        nnodes = distributed.get("nnodes", 1)
        
        is_distributed = distributed.get("enabled", False) or distributed.get("launcher")
        is_multi_gpu = gpu_count > 1 or is_distributed
        is_multi_node = nnodes > 1
        
        return {
            "is_single_gpu": gpu_count == 1 and not is_distributed,
            "is_multi_gpu": is_multi_gpu and not is_multi_node,
            "is_multi_node": is_multi_node,
            "is_distributed": is_distributed
        }
    
    @classmethod
    def select_profile(cls, config: Dict, needs: Dict[str, bool]) -> Optional[str]:
        """
        Auto-select k8s profile based on configuration needs.
        
        Args:
            config: Configuration dictionary
            needs: Profile needs from detect_profile_needs()
            
        Returns:
            Profile filename or None
        """
        if needs["is_multi_node"]:
            return "k8s/profiles/multi-node.json"
        elif needs["is_multi_gpu"]:
            return "k8s/profiles/multi-gpu.json"
        elif needs["is_single_gpu"]:
            return "k8s/profiles/single-gpu.json"
        
        return None
    
    @classmethod
    def load_k8s_config(cls, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load complete k8s configuration with multi-layer merging.
        
        Layers:
        1. Base k8s defaults
        2. GPU vendor base preset
        3. GPU vendor multi-GPU preset (if needed)
        4. Profile preset (single-gpu/multi-gpu/multi-node)
        5. User configuration (already merged from file + CLI)
        
        Args:
            user_config: User-provided configuration (merged from file + CLI)
            
        Returns:
            Complete configuration with all defaults applied
        """
        # Layer 1: Base defaults
        config = cls.load_preset("k8s/defaults.json")
        
        # Merge user config temporarily to detect requirements
        temp_config = cls.deep_merge(config, user_config)
        needs = cls.detect_profile_needs(temp_config)
        
        # Layer 2: GPU vendor base preset
        gpu_vendor = temp_config.get("gpu_vendor", "AMD").upper()
        vendor_file = f"k8s/gpu-vendors/{gpu_vendor.lower()}.json"
        vendor_preset = cls.load_preset(vendor_file)
        config = cls.deep_merge(config, vendor_preset)
        
        # Layer 3: GPU vendor multi-GPU optimizations (AMD only, when needed)
        if gpu_vendor == "AMD" and (needs["is_multi_gpu"] or needs["is_multi_node"]):
            amd_multi_preset = cls.load_preset("k8s/gpu-vendors/amd-multi-gpu.json")
            config = cls.deep_merge(config, amd_multi_preset)
        
        # Layer 4: Profile preset based on detected needs
        profile_file = cls.select_profile(temp_config, needs)
        if profile_file:
            profile_preset = cls.load_preset(profile_file)
            config = cls.deep_merge(config, profile_preset)
        
        # Layer 5: User configuration (highest priority)
        config = cls.deep_merge(config, user_config)
        
        return config
    
    @classmethod
    def load_slurm_config(cls, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load complete SLURM configuration with defaults.
        
        Args:
            user_config: User-provided configuration
            
        Returns:
            Complete configuration with defaults applied
        """
        config = cls.load_preset("slurm/defaults.json")
        return cls.deep_merge(config, user_config)
    
    @classmethod
    def infer_and_validate_deploy_type(cls, user_config: Dict[str, Any]) -> str:
        """
        Infer deployment type from config structure and validate for conflicts.
        
        Convention over Configuration: Presence of k8s/slurm field indicates deployment intent.
        
        Args:
            user_config: User configuration dictionary
            
        Returns:
            Deployment type: "k8s", "slurm", or "local"
            
        Raises:
            ValueError: If conflicting deployment configs present
        """
        has_k8s = "k8s" in user_config or "kubernetes" in user_config
        has_slurm = "slurm" in user_config
        explicit_deploy = user_config.get("deploy", "").lower()
        
        # Validation Rule 1: Can't have both k8s and slurm configs
        if has_k8s and has_slurm:
            raise ValueError(
                "Conflicting deployment configuration: Both 'k8s' and 'slurm' fields present. "
                "Please specify only one deployment target."
            )
        
        # Validation Rule 2: If explicit deploy set, it must match config presence
        if explicit_deploy:
            if explicit_deploy in ["k8s", "kubernetes"] and not has_k8s:
                raise ValueError(
                    f"Conflicting deployment: 'deploy' field is '{explicit_deploy}' but no 'k8s' config present. "
                    "Either add 'k8s' config or remove 'deploy' field."
                )
            if explicit_deploy == "slurm" and not has_slurm:
                raise ValueError(
                    f"Conflicting deployment: 'deploy' field is 'slurm' but no 'slurm' config present. "
                    "Either add 'slurm' config or remove 'deploy' field."
                )
            if explicit_deploy == "local" and (has_k8s or has_slurm):
                raise ValueError(
                    f"Conflicting deployment: 'deploy' field is 'local' but k8s/slurm config present. "
                    "Remove k8s/slurm config for local execution."
                )
        
        # Infer deployment type from config presence
        if has_k8s:
            return "k8s"
        elif has_slurm:
            return "slurm"
        else:
            return "local"
    
    @classmethod
    def load_config(cls, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load configuration with auto-inferred deploy type and validation.
        
        Infers deployment type from presence of k8s/slurm fields.
        Validates for conflicting configurations.
        Applies appropriate defaults based on deployment type.
        
        Args:
            user_config: User configuration (from file + CLI merge)
            
        Returns:
            Complete configuration with defaults and deploy field set
            
        Raises:
            ValueError: If conflicting deployment configs present
        """
        # Infer and validate deployment type
        deploy_type = cls.infer_and_validate_deploy_type(user_config)
        
        # Set deploy field (for internal use in manifest)
        user_config["deploy"] = deploy_type
        
        # Apply appropriate defaults based on deployment type
        if deploy_type == "k8s":
            return cls.load_k8s_config(user_config)
        elif deploy_type == "slurm":
            return cls.load_slurm_config(user_config)
        else:
            # Local - return as-is with deploy field added
            return user_config

