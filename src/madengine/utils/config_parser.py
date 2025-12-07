"""Config Parser Module for MAD Engine.

This module provides utilities to parse configuration files from model arguments
and load them in various formats (CSV, JSON, YAML).

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
    """
    
    def __init__(self, scripts_base_dir: typing.Optional[str] = None):
        """Initialize ConfigParser.
        
        Args:
            scripts_base_dir: Base directory for scripts (e.g., ~/amd/MAD-private/scripts)
        """
        self.scripts_base_dir = scripts_base_dir
    
    def parse_config_from_args(self, args_string: str, model_scripts_path: str = None) -> typing.Optional[str]:
        """Extract config file path from model arguments.
        
        Args:
            args_string: The args field from models.json
            model_scripts_path: Path to the model's script directory
            
        Returns:
            Full path to config file, or None if no config found
        """
        if not args_string:
            return None
        
        # Look for --config argument
        config_match = re.search(r'--config\s+([^\s]+)', args_string)
        if not config_match:
            return None
        
        config_path = config_match.group(1)
        
        # If it's already an absolute path, return it
        if os.path.isabs(config_path):
            return config_path if os.path.exists(config_path) else None
        
        # Try to resolve relative path
        # First, try relative to model scripts directory
        if model_scripts_path:
            scripts_dir = os.path.dirname(model_scripts_path)
            full_path = os.path.join(scripts_dir, config_path)
            if os.path.exists(full_path):
                return full_path
        
        # Try relative to scripts_base_dir
        if self.scripts_base_dir:
            full_path = os.path.join(self.scripts_base_dir, config_path)
            if os.path.exists(full_path):
                return full_path
        
        LOGGER.warning(f"Config file not found: {config_path}")
        return None
    
    def load_config_file(self, config_path: str) -> typing.Optional[typing.Union[typing.List[dict], dict]]:
        """Load and parse a configuration file.
        
        Args:
            config_path: Full path to the config file
            
        Returns:
            For CSV: List of dicts (one per row)
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
            List of dicts, one per row
        """
        df = pd.read_csv(config_path)
        # Convert NaN to None for JSON serialization
        df = df.where(pd.notnull(df), None)
        # Convert to list of dicts
        return df.to_dict(orient='records')
    
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
        # Extract model identifier from result model name
        # e.g., "pyt_vllm_llama-3.1-8b_perf_meta-llama_Llama-3.1-8B-Instruct" 
        # should match config with model="meta-llama/Llama-3.1-8B-Instruct"
        
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
            model_scripts_path: Path to the model's script directory
            
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

