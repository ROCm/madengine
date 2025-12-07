"""Module to update the perf_entry_super.json file with enhanced performance data.

This module is used to update the perf_entry_super.json file with performance data
that includes configuration information from config files.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in imports
import json
import os
import typing
# third-party imports
import pandas as pd
# MAD Engine imports
from madengine.utils.config_parser import ConfigParser


def read_json(js: str) -> dict:
    """Read a JSON file.
    
    Args:
        js: The path to the JSON file.
    
    Returns:
        The JSON dictionary.
    """
    with open(js, 'r') as f:
        return json.load(f)


def write_json(data: typing.Union[dict, list], output_path: str) -> None:
    """Write data to a JSON file.
    
    Args:
        data: The data to write (dict or list).
        output_path: The path to the output JSON file.
    """
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_perf_super_json(perf_super_json: str) -> list:
    """Load existing perf_entry_super.json file.
    
    Args:
        perf_super_json: Path to perf_entry_super.json file.
    
    Returns:
        List of performance records, or empty list if file doesn't exist.
    """
    if not os.path.exists(perf_super_json):
        return []
    
    try:
        data = read_json(perf_super_json)
        # Ensure it's a list
        if isinstance(data, list):
            return data
        else:
            return [data]
    except Exception as e:
        print(f"Warning: Could not load existing perf_entry_super.json: {e}")
        return []


def handle_multiple_results_super(
        perf_super_list: list,
        multiple_results: str,
        common_info: str,
        model_name: str,
        config_parser: ConfigParser
    ) -> list:
    """Handle multiple results with config matching.
    
    Args:
        perf_super_list: List of existing performance records.
        multiple_results: The path to the multiple results CSV file.
        common_info: The path to the common info JSON file.
        model_name: The model name.
        config_parser: ConfigParser instance for loading configs.
        
    Returns:
        Updated list of performance records with configs.
    """
    # Load multiple results CSV
    multiple_results_df = pd.read_csv(multiple_results)
    multiple_results_df.columns = multiple_results_df.columns.str.strip()
    
    # Check required columns
    required_cols = ['model', 'performance', 'metric']
    for col in required_cols:
        if col not in multiple_results_df.columns:
            raise RuntimeError(f"{multiple_results} file is missing the {col} column")
    
    # Load common info
    common_info_json = read_json(common_info)
    
    # Parse config file from args if present
    configs_data = None
    if 'args' in common_info_json and common_info_json['args']:
        # Try to extract config path from args
        scripts_path = common_info_json.get('pipeline', '')
        configs_data = config_parser.parse_and_load(
            common_info_json['args'],
            scripts_path
        )
    
    # Process each result row
    for result_row in multiple_results_df.to_dict(orient="records"):
        record = common_info_json.copy()
        
        # Update model name
        result_model = result_row.pop("model")
        record["model"] = f"{model_name}_{result_model}"
        
        # Update with result data
        record.update(result_row)
        
        # Set status based on performance
        if record.get("performance") is not None and pd.notna(record.get("performance")):
            record["status"] = "SUCCESS"
        else:
            record["status"] = "FAILURE"
        
        # Match config to this specific result
        if configs_data:
            if isinstance(configs_data, list):
                # For CSV configs with multiple rows, try to match
                matched_config = config_parser.match_config_to_result(
                    configs_data,
                    result_row,
                    result_model
                )
                record["configs"] = matched_config
            else:
                # For JSON/YAML configs, use as-is
                record["configs"] = configs_data
        else:
            record["configs"] = None
        
        perf_super_list.append(record)
    
    return perf_super_list


def handle_single_result_super(
        perf_super_list: list,
        single_result: str
    ) -> list:
    """Handle a single result.
    
    Args:
        perf_super_list: List of existing performance records.
        single_result: The path to the single result JSON file.
    
    Returns:
        Updated list of performance records.
    """
    single_result_json = read_json(single_result)
    
    # Ensure configs field exists (may be None)
    if "configs" not in single_result_json:
        single_result_json["configs"] = None
    
    perf_super_list.append(single_result_json)
    return perf_super_list


def handle_exception_result_super(
        perf_super_list: list,
        exception_result: str
    ) -> list:
    """Handle an exception result.
    
    Args:
        perf_super_list: List of existing performance records.
        exception_result: The path to the exception result JSON file.
    
    Returns:
        Updated list of performance records.
    """
    exception_result_json = read_json(exception_result)
    
    # Ensure configs field exists (may be None)
    if "configs" not in exception_result_json:
        exception_result_json["configs"] = None
    
    perf_super_list.append(exception_result_json)
    return perf_super_list


def update_perf_super_json(
        perf_super_json: str,
        multiple_results: typing.Optional[str] = None,
        single_result: typing.Optional[str] = None,
        exception_result: typing.Optional[str] = None,
        common_info: typing.Optional[str] = None,
        model_name: typing.Optional[str] = None,
        scripts_base_dir: typing.Optional[str] = None,
    ) -> None:
    """Update the perf_entry_super.json file with the latest performance data.
    
    Args:
        perf_super_json: Path to perf_entry_super.json file.
        multiple_results: Path to multiple results CSV file.
        single_result: Path to single result JSON file.
        exception_result: Path to exception result JSON file.
        common_info: Path to common info JSON file.
        model_name: The model name.
        scripts_base_dir: Base directory for scripts (for config file resolution).
    """
    print(f"Updating perf_entry_super.json with enhanced performance data")
    
    # Load existing perf_entry_super.json
    perf_super_list = load_perf_super_json(perf_super_json)
    
    # Create config parser
    config_parser = ConfigParser(scripts_base_dir=scripts_base_dir)
    
    # Handle different result types
    if multiple_results:
        perf_super_list = handle_multiple_results_super(
            perf_super_list,
            multiple_results,
            common_info,
            model_name,
            config_parser,
        )
    elif single_result:
        perf_super_list = handle_single_result_super(perf_super_list, single_result)
    elif exception_result:
        perf_super_list = handle_exception_result_super(
            perf_super_list, exception_result
        )
    else:
        print("No results to update in perf_entry_super.json")
        return
    
    # Write updated perf_entry_super.json
    write_json(perf_super_list, perf_super_json)
    print(f"Successfully updated {perf_super_json}")

