"""Config Parser Module for MAD Engine.

This module provides utilities to parse configuration files from model arguments
and load them in various formats (CSV, JSON, YAML).

Handles multiple repository patterns:
- Standalone repos (MAD, MAD-private): ./scripts/model/configs/
- Submodule in MAD-internal: ./scripts/{MAD|MAD-private}/model/configs/

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import re
import json
import logging
import typing
from pathlib import Path

import pandas as pd

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


class ConfigParser:
    """Parser for model configuration files.
    
    This class handles parsing configuration files in various formats
    (CSV, JSON, YAML) that are referenced in model arguments.
    
    Supports three usage patterns when run from MAD-internal CI:
    1. MAD-internal models: ./scripts/model/configs/
    2. MAD submodule: ./scripts/MAD/model/configs/
    3. MAD-private submodule: ./scripts/MAD-private/model/configs/
    
    Also works when run standalone in MAD or MAD-private repos.
    """
    
    # Known repository/submodule names to detect
    KNOWN_REPOS = ['MAD', 'MAD-private', 'MAD-internal']
    
    def __init__(self, scripts_base_dir: typing.Optional[str] = None):
        """Initialize ConfigParser.
        
        Args:
            scripts_base_dir: Base directory for scripts 
                             (e.g., "scripts/MAD-private/pyt_atom")
        """
        self.scripts_base_dir = scripts_base_dir
        self._path_cache = {}  # Cache resolved paths
    
    def _extract_repo_root(self, path: str) -> typing.Optional[str]:
        """Extract repository root from a scripts path.
        
        Examples:
            "scripts/MAD-private/pyt_atom" -> "scripts/MAD-private"
            "scripts/MAD/vllm" -> "scripts/MAD"
            "scripts/model" -> "scripts"
            
        Args:
            path: Full or partial path containing scripts directory
            
        Returns:
            Repository root path, or None if not identifiable
        """
        if not path:
            return None
        
        parts = Path(path).parts
        
        # Find 'scripts' in the path
        try:
            scripts_idx = parts.index('scripts')
        except ValueError:
            return None
        
        # Check if next part after 'scripts' is a known repo name
        if scripts_idx + 1 < len(parts):
            next_part = parts[scripts_idx + 1]
            if next_part in self.KNOWN_REPOS:
                # It's a submodule: scripts/MAD-private or scripts/MAD
                return os.path.join(*parts[:scripts_idx + 2])
            else:
                # It's MAD-internal's own models: scripts/model -> scripts
                return os.path.join(*parts[:scripts_idx + 1])
        
        # Just 'scripts' directory
        return os.path.join(*parts[:scripts_idx + 1])
    
    def _build_candidate_paths(
        self, 
        config_path: str, 
        model_scripts_path: str = None
    ) -> typing.List[str]:
        """Build list of candidate paths to try in priority order.
        
        Args:
            config_path: Relative config path (e.g., "configs/default.csv")
            model_scripts_path: Path to model script file
            
        Returns:
            List of full paths to try, in order of priority
        """
        candidates = []
        
        # Priority 1: Relative to model's immediate directory
        # scripts/MAD-private/pyt_atom + configs/default.csv
        if model_scripts_path:
            scripts_dir = os.path.dirname(model_scripts_path)
            if scripts_dir:
                candidates.append(os.path.join(scripts_dir, config_path))
        
        # Priority 2: Relative to scripts_base_dir
        # scripts/MAD-private/pyt_atom + configs/default.csv
        if self.scripts_base_dir:
            candidates.append(os.path.join(self.scripts_base_dir, config_path))
        
        # Priority 3: Relative to repository root (for shared configs)
        # This handles: scripts/MAD-private/pyt_atom -> scripts/MAD-private/configs/
        if self.scripts_base_dir:
            repo_root = self._extract_repo_root(self.scripts_base_dir)
            if repo_root:
                candidates.append(os.path.join(repo_root, config_path))
        
        if model_scripts_path:
            scripts_dir = os.path.dirname(model_scripts_path)
            if scripts_dir:
                repo_root = self._extract_repo_root(scripts_dir)
                if repo_root:
                    candidates.append(os.path.join(repo_root, config_path))
        
        # Priority 4: Walk up from model's directory
        # Try parent directories up to repo root
        if model_scripts_path:
            scripts_dir = os.path.dirname(model_scripts_path)
            repo_root = self._extract_repo_root(scripts_dir)
            if repo_root and scripts_dir:
                candidates.extend(
                    self._walk_up_between(config_path, scripts_dir, repo_root)
                )
        
        # Priority 5: Walk up from scripts_base_dir
        if self.scripts_base_dir:
            repo_root = self._extract_repo_root(self.scripts_base_dir)
            if repo_root:
                candidates.extend(
                    self._walk_up_between(config_path, self.scripts_base_dir, repo_root)
                )
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for path in candidates:
            normalized = os.path.normpath(path)
            if normalized not in seen:
                seen.add(normalized)
                unique_candidates.append(normalized)
        
        return unique_candidates
    
    def _walk_up_between(
        self, 
        config_path: str, 
        start_dir: str, 
        stop_dir: str
    ) -> typing.List[str]:
        """Generate candidate paths by walking up from start to stop directory.
        
        Args:
            config_path: Relative config path
            start_dir: Starting directory
            stop_dir: Stop at this directory (inclusive)
            
        Returns:
            List of candidate paths
        """
        candidates = []
        current = os.path.abspath(start_dir)
        stop = os.path.abspath(stop_dir)
        
        while current.startswith(stop):
            parent = os.path.dirname(current)
            if parent == current:  # Reached root
                break
            current = parent
            candidates.append(os.path.join(current, config_path))
            if current == stop:  # Reached stop directory
                break
        
        return candidates
    
    def parse_config_from_args(
        self, 
        args_string: str, 
        model_scripts_path: str = None
    ) -> typing.Optional[str]:
        """Extract and resolve config file path from model arguments.
        
        Resolution strategy:
        1. If absolute path -> verify it exists
        2. Try model's immediate directory
        3. Try scripts_base_dir
        4. Try repository root (scripts/MAD-private/, scripts/MAD/, scripts/)
        5. Walk up from model directory to repo root
        
        This handles all cases:
        - MAD-internal models: scripts/model/configs/default.csv
        - MAD submodule: scripts/MAD/model/configs/default.csv
        - MAD-private submodule: scripts/MAD-private/model/configs/default.csv
        - Shared configs at repo level: scripts/MAD-private/configs/default.csv
        
        Args:
            args_string: The args field from models.json
            model_scripts_path: Path to the model's script file (e.g., run.py)
            
        Returns:
            Full path to config file, or None if not found
        """
        if not args_string:
            return None
        
        # Look for --config argument
        config_match = re.search(r'--config\s+([^\s]+)', args_string)
        if not config_match:
            return None
        
        config_path = config_match.group(1)
        
        # Check cache first
        cache_key = f"{config_path}::{model_scripts_path}::{self.scripts_base_dir}"
        if cache_key in self._path_cache:
            cached_path = self._path_cache[cache_key]
            if os.path.exists(cached_path):
                return cached_path
            else:
                del self._path_cache[cache_key]
        
        # Handle absolute paths
        if os.path.isabs(config_path):
            if os.path.exists(config_path):
                self._path_cache[cache_key] = config_path
                return config_path
            else:
                LOGGER.warning(f"Absolute config path does not exist: {config_path}")
                return None
        
        # Build and try candidate paths
        candidates = self._build_candidate_paths(config_path, model_scripts_path)
        
        for candidate in candidates:
            LOGGER.debug(f"Trying config path: {candidate}")
            if os.path.exists(candidate):
                LOGGER.info(f"Found config file at: {candidate}")
                self._path_cache[cache_key] = candidate
                return candidate
        
        # Not found
        LOGGER.warning(
            f"Config file not found: {config_path}\n"
            f"  model_scripts_path: {model_scripts_path}\n"
            f"  scripts_base_dir: {self.scripts_base_dir}\n"
            f"  Tried {len(candidates)} locations:\n"
            + "\n".join(f"    - {c}" for c in candidates[:5])
            + (f"\n    ... and {len(candidates)-5} more" if len(candidates) > 5 else "")
        )
        return None
    
    def load_config_file(
        self, 
        config_path: str
    ) -> typing.Optional[typing.Union[typing.List[dict], dict]]:
        """Load and parse a configuration file.
        
        Args:
            config_path: Full path to the config file
            
        Returns:
            For CSV: List of dicts (one per row, excluding empty rows)
            For JSON/YAML: Dict or list as-is from file
            None if file cannot be loaded
        """
        if not config_path or not os.path.exists(config_path):
            return None
        
        file_ext = Path(config_path).suffix.lower()
        
        try:
            if file_ext == '.csv':
                return self._load_csv(config_path)
            elif file_ext == '.json':
                return self._load_json(config_path)
            elif file_ext in ['.yaml', '.yml']:
                return self._load_yaml(config_path)
            else:
                LOGGER.warning(f"Unsupported config file format: {file_ext}")
                return None
        except Exception as e:
            LOGGER.error(f"Error loading config file {config_path}: {e}")
            return None
    
    def _load_csv(self, config_path: str) -> typing.List[dict]:
        """Load CSV config file.
        
        Args:
            config_path: Path to CSV file
            
        Returns:
            List of dicts, one per row (excluding completely empty rows)
        """
        df = pd.read_csv(config_path)
        
        # Remove rows that are completely empty (all NaN)
        # This handles blank lines in CSV files
        df = df.dropna(how='all')
        
        # Convert NaN to None for JSON serialization
        df = df.where(pd.notnull(df), None)
        
        # Convert to list of dicts
        configs = df.to_dict(orient='records')
        
        LOGGER.info(f"Loaded {len(configs)} config entries from {config_path}")
        
        return configs
    
    def _load_json(self, config_path: str) -> typing.Union[dict, list]:
        """Load JSON config file.
        
        Args:
            config_path: Path to JSON file
            
        Returns:
            Dict or list from JSON file
        """
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _load_yaml(self, config_path: str) -> typing.Union[dict, list]:
        """Load YAML config file.
        
        Args:
            config_path: Path to YAML file
            
        Returns:
            Dict or list from YAML file
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is not installed. Cannot load YAML config files.")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def match_config_to_result(
        self, 
        configs_list: typing.List[dict], 
        result_data: dict, 
        model_name: str
    ) -> typing.Optional[dict]:
        """Match a specific result to its corresponding config.
        
        For CSV configs with multiple rows (like vllm), match based on
        model name and other identifiable fields.
        
        Args:
            configs_list: List of config dicts (from CSV rows)
            result_data: Single result row data
            model_name: The model name from result
            
        Returns:
            Matching config dict, or None if no match found
        """
        if not configs_list:
            return None
        
        # For single config, return it
        if len(configs_list) == 1:
            return configs_list[0]
        
        # For multiple configs, try to match based on common fields
        for config in configs_list:
            # Try to match on 'model' field if it exists in both
            if 'model' in config and 'model' in result_data:
                # Compare normalized versions
                config_model = str(config['model']).replace('/', '_').replace('-', '_').lower()
                result_model = str(result_data['model']).replace('/', '_').replace('-', '_').lower()
                if config_model in result_model or result_model in config_model:
                    # Additional checks for benchmark type if available
                    if 'benchmark' in config and 'benchmark' in result_data:
                        if config['benchmark'] == result_data['benchmark']:
                            return config
                    else:
                        return config
        
        # If no match found, return first config as fallback
        LOGGER.warning(f"Could not match config for result: {model_name}. Using first config.")
        return configs_list[0]
    
    def parse_and_load(
        self, 
        args_string: str, 
        model_scripts_path: str = None
    ) -> typing.Optional[typing.Union[typing.List[dict], dict]]:
        """Parse config path from args and load the config file.
        
        Convenience method that combines parse_config_from_args and load_config_file.
        
        Args:
            args_string: The args field from models.json
            model_scripts_path: Path to the model's script file
            
        Returns:
            Config data (list of dicts for CSV, dict for JSON/YAML), or None
        """
        config_path = self.parse_config_from_args(args_string, model_scripts_path)
        if not config_path:
            return None
        
        return self.load_config_file(config_path)


def get_config_parser(scripts_base_dir: typing.Optional[str] = None) -> ConfigParser:
    """Factory function to create a ConfigParser instance.
    
    Args:
        scripts_base_dir: Base directory for scripts
        
    Returns:
        ConfigParser instance
    """
    return ConfigParser(scripts_base_dir=scripts_base_dir)
