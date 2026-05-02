"""
GPU Configuration Resolution Utility

Provides hierarchical GPU count resolution with clear precedence rules
to handle inconsistencies between model definitions, deployment configs,
and runtime overrides.

Priority (highest to lowest):
1. Runtime config (--additional-context at run time)
2. Deployment config (k8s.gpu_count / slurm.gpus_per_node)
3. Model definition (n_gpus in models.json)
4. System default (1)

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import warnings
from typing import Dict, Any, Optional, Tuple


class GPUConfigResolver:
    """
    Resolves GPU count from multiple configuration sources with clear precedence.
    
    Handles various field names (n_gpus, gpu_count, gpus_per_node) and provides
    validation to catch configuration mismatches early.
    """
    
    # All recognized field names for GPU count
    GPU_FIELD_ALIASES = [
        "gpus_per_node",  # SLURM, preferred standard
        "gpu_count",      # Kubernetes
        "n_gpus",         # Legacy model.json
        "num_gpus",       # Alternative
        "ngpus",          # Alternative
    ]
    
    @classmethod
    def resolve_gpu_count(
        cls,
        model_info: Optional[Dict[str, Any]] = None,
        deployment_config: Optional[Dict[str, Any]] = None,
        runtime_override: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ) -> Tuple[int, str]:
        """
        Resolve GPU count from multiple sources with clear precedence.
        
        Args:
            model_info: Model configuration from models.json
            deployment_config: Deployment configuration (slurm/k8s section)
            runtime_override: Runtime override from --additional-context
            validate: Whether to validate and warn about mismatches
        
        Returns:
            Tuple of (gpu_count, source) where source indicates which config was used
            
        Examples:
            >>> # Priority 1: Runtime override
            >>> count, source = GPUConfigResolver.resolve_gpu_count(
            ...     model_info={"n_gpus": "1"},
            ...     deployment_config={"slurm": {"gpus_per_node": 8}},
            ...     runtime_override={"gpus_per_node": 4}
            ... )
            >>> count, source
            (4, 'runtime_override')
            
            >>> # Priority 2: Deployment config
            >>> count, source = GPUConfigResolver.resolve_gpu_count(
            ...     model_info={"n_gpus": "1"},
            ...     deployment_config={"slurm": {"gpus_per_node": 8}}
            ... )
            >>> count, source
            (8, 'deployment_config.slurm.gpus_per_node')
            
            >>> # Priority 3: Model definition
            >>> count, source = GPUConfigResolver.resolve_gpu_count(
            ...     model_info={"n_gpus": "2"}
            ... )
            >>> count, source
            (2, 'model_info.n_gpus')
        """
        sources = []  # Track all sources for validation
        
        # Priority 1: Runtime override
        if runtime_override:
            gpu_count = cls._extract_gpu_count(runtime_override, "runtime_override")
            if gpu_count is not None:
                sources.append(("runtime_override", gpu_count))
                if validate:
                    cls._validate_consistency(sources, model_info, deployment_config)
                return gpu_count, "runtime_override"
        
        # Priority 2: Deployment-specific config
        if deployment_config:
            # Check for SLURM config
            if "slurm" in deployment_config:
                gpu_count = cls._extract_gpu_count(
                    deployment_config["slurm"], 
                    "deployment_config.slurm"
                )
                if gpu_count is not None:
                    sources.append(("deployment_config.slurm.gpus_per_node", gpu_count))
                    if validate:
                        cls._validate_consistency(sources, model_info, deployment_config)
                    return gpu_count, "deployment_config.slurm.gpus_per_node"
            
            # Check for K8s config
            if "k8s" in deployment_config or "kubernetes" in deployment_config:
                k8s_config = deployment_config.get("k8s") or deployment_config.get("kubernetes")
                gpu_count = cls._extract_gpu_count(k8s_config, "deployment_config.k8s")
                if gpu_count is not None:
                    sources.append(("deployment_config.k8s.gpu_count", gpu_count))
                    if validate:
                        cls._validate_consistency(sources, model_info, deployment_config)
                    return gpu_count, "deployment_config.k8s.gpu_count"
        
        # Priority 3: Model definition
        if model_info:
            gpu_count = cls._extract_gpu_count(model_info, "model_info")
            if gpu_count is not None:
                sources.append(("model_info.n_gpus", gpu_count))
                if validate:
                    cls._validate_consistency(sources, model_info, deployment_config)
                return gpu_count, "model_info.n_gpus"
        
        # Priority 4: Default
        return 1, "default"
    
    @classmethod
    def _extract_gpu_count(
        cls, 
        config: Dict[str, Any], 
        context: str
    ) -> Optional[int]:
        """
        Extract GPU count from config dict, trying all known field names.
        
        Args:
            config: Configuration dictionary
            context: Context string for warning messages
            
        Returns:
            GPU count as integer, or None if not found
        """
        if not config:
            return None
        
        found_fields = []
        for field_name in cls.GPU_FIELD_ALIASES:
            if field_name in config:
                found_fields.append((field_name, config[field_name]))
        
        if not found_fields:
            return None
        
        # Warn if multiple GPU fields found
        if len(found_fields) > 1:
            field_list = ", ".join([f"{name}={val}" for name, val in found_fields])
            print(
                f"⚠️  Multiple GPU fields in {context}: {field_list}. "
                f"Using {found_fields[0][0]}={found_fields[0][1]}"
            )
        
        # Convert to int (handle string values like "8")
        try:
            return int(found_fields[0][1])
        except (ValueError, TypeError):
            print(
                f"⚠️  Invalid GPU count in {context}: {found_fields[0][1]}. Using default."
            )
            return None
    
    @classmethod
    def _validate_consistency(
        cls,
        sources: list,
        model_info: Optional[Dict[str, Any]],
        deployment_config: Optional[Dict[str, Any]],
    ) -> None:
        """
        Validate consistency between different GPU count sources.
        
        Warns if there are mismatches that might indicate configuration errors.
        
        Args:
            sources: List of (source_name, gpu_count) tuples found so far
            model_info: Model configuration for additional checks
            deployment_config: Deployment configuration for additional checks
        """
        if not sources:
            return
        
        # Collect all GPU counts from all sources
        all_counts = {}
        
        # Add already resolved source
        for source_name, count in sources:
            all_counts[source_name] = count
        
        # Check model_info
        if model_info:
            model_gpu = cls._extract_gpu_count(model_info, "model_info")
            if model_gpu is not None:
                all_counts["model_info.n_gpus"] = model_gpu
        
        # Check deployment config
        if deployment_config:
            if "slurm" in deployment_config:
                slurm_gpu = cls._extract_gpu_count(
                    deployment_config["slurm"], "slurm"
                )
                if slurm_gpu is not None:
                    all_counts["deployment_config.slurm.gpus_per_node"] = slurm_gpu
            
            if "k8s" in deployment_config or "kubernetes" in deployment_config:
                k8s_config = deployment_config.get("k8s") or deployment_config.get("kubernetes")
                k8s_gpu = cls._extract_gpu_count(k8s_config, "k8s")
                if k8s_gpu is not None:
                    all_counts["deployment_config.k8s.gpu_count"] = k8s_gpu
        
        # Check for mismatches
        unique_counts = set(all_counts.values())
        if len(unique_counts) > 1:
            mismatch_details = ", ".join([f"{k}={v}" for k, v in all_counts.items()])
            # Determine if this is likely intentional (deployment override) or an error
            is_deployment_override = (
                sources[0][0].startswith("runtime_override") or
                sources[0][0].startswith("deployment_config")
            )
            
            if is_deployment_override:
                # This is normal - deployment config overriding model default
                # Use print instead of warnings.warn for cleaner output
                print(
                    f"ℹ️  GPU configuration override: {sources[0][0]}={sources[0][1]} "
                    f"(overriding model default: {mismatch_details.split(',')[-1].strip()})"
                )
            else:
                # Potentially unexpected mismatch - use warning for actual errors
                warnings.warn(
                    f"\n⚠️  GPU count mismatch detected: {mismatch_details}\n"
                    f"   Using: {sources[0][0]}={sources[0][1]}\n"
                    f"   Precedence: runtime_override > deployment_config > model_info > default",
                    UserWarning,
                    stacklevel=4
                )


def resolve_runtime_gpus(
    model_info: Dict[str, Any],
    additional_context: Dict[str, Any],
) -> int:
    """
    Convenience function for resolving GPU count at runtime.
    
    This is the main entry point for runtime GPU resolution.
    
    Args:
        model_info: Model configuration from manifest
        additional_context: Additional context from CLI or config files
        
    Returns:
        Resolved GPU count as integer
        
    Example:
        >>> model_info = {"name": "my_model", "n_gpus": "1"}
        >>> additional_context = {"slurm": {"gpus_per_node": 8}}
        >>> gpu_count = resolve_runtime_gpus(model_info, additional_context)
        >>> gpu_count
        8
    """
    # Extract deployment config from additional_context
    deployment_config = additional_context.get("deployment_config", {})
    
    # Also check for direct slurm/k8s keys in additional_context
    if "slurm" in additional_context:
        if not deployment_config:
            deployment_config = {}
        deployment_config["slurm"] = additional_context["slurm"]
    
    if "k8s" in additional_context or "kubernetes" in additional_context:
        if not deployment_config:
            deployment_config = {}
        deployment_config["k8s"] = additional_context.get("k8s") or additional_context.get("kubernetes")
    
    # Check for direct runtime GPU override (in additional_context or deployment_config)
    runtime_override = None
    for field in GPUConfigResolver.GPU_FIELD_ALIASES:
        if field in additional_context:
            runtime_override = {field: additional_context[field]}
            break
        # Also check in deployment_config top-level (for SLURM local manifest)
        if deployment_config and field in deployment_config:
            runtime_override = {field: deployment_config[field]}
            break
    
    gpu_count, source = GPUConfigResolver.resolve_gpu_count(
        model_info=model_info,
        deployment_config=deployment_config,
        runtime_override=runtime_override,
        validate=True,
    )
    
    print(f"ℹ️  Resolved GPU count: {gpu_count} (from {source})")
    
    return gpu_count

