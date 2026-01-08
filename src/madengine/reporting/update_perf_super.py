"""Module to update the perf_super.json file with enhanced performance data.

This module is used to update the perf_super.json file (cumulative) with performance data
that includes configuration information from config files, and provides CSV/JSON export.
It also generates perf_entry_super.json (latest run only) for consistency with perf_entry.json.

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


def read_json(js: str) -> typing.Union[dict, list]:
    """Read a JSON file.
    
    Args:
        js: The path to the JSON file.
    
    Returns:
        The JSON dictionary or list.
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
    """Load existing perf_super.json file (cumulative).
    
    Args:
        perf_super_json: Path to perf_super.json file.
    
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
        print(f"Warning: Could not load existing {perf_super_json}: {e}")
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
        
        # Extract standard performance/metric columns
        record["performance"] = result_row.pop("performance")
        record["metric"] = result_row.pop("metric")
        
        # Put remaining metrics into multi_results
        # Exclude internal fields that shouldn't be in multi_results
        extra_metrics = {k: v for k, v in result_row.items() 
                         if k not in ["status"] and pd.notna(v)}
        if extra_metrics:
            record["multi_results"] = extra_metrics
        else:
            record["multi_results"] = None
        
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
    
    # Ensure multi_results field exists (may be None)
    if "multi_results" not in single_result_json:
        single_result_json["multi_results"] = None
    
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
    
    # Ensure multi_results field exists (may be None)
    if "multi_results" not in exception_result_json:
        exception_result_json["multi_results"] = None
    
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
    ) -> int:
    """Update the perf_super.json file (cumulative) with the latest performance data.
    
    Args:
        perf_super_json: Path to perf_super.json file (cumulative).
        multiple_results: Path to multiple results CSV file.
        single_result: Path to single result JSON file.
        exception_result: Path to exception result JSON file.
        common_info: Path to common info JSON file.
        model_name: The model name.
        scripts_base_dir: Base directory for scripts (for config file resolution).
        
    Returns:
        Number of entries added in this update.
    """
    print("\n" + "=" * 80)
    print("📊 UPDATING PERFORMANCE SUPERSET DATABASE")
    print("=" * 80)
    print(f"📂 Target file: {perf_super_json}")
    
    # Load existing perf_super.json
    perf_super_list = load_perf_super_json(perf_super_json)
    initial_count = len(perf_super_list)
    
    # Create config parser
    config_parser = ConfigParser(scripts_base_dir=scripts_base_dir)
    
    # Handle different result types
    if multiple_results:
        print("🔄 Processing multiple results with configs...")
        perf_super_list = handle_multiple_results_super(
            perf_super_list,
            multiple_results,
            common_info,
            model_name,
            config_parser,
        )
    elif single_result:
        print("🔄 Processing single result with configs...")
        perf_super_list = handle_single_result_super(perf_super_list, single_result)
    elif exception_result:
        print("⚠️  Processing exception result...")
        perf_super_list = handle_exception_result_super(
            perf_super_list, exception_result
        )
    else:
        print("ℹ️  No results to update in perf_super.json")
        return 0
    
    # Write updated perf_super.json
    write_json(perf_super_list, perf_super_json)
    entries_added = len(perf_super_list) - initial_count
    print(f"✅ Successfully updated: {perf_super_json} (added {entries_added} entries)")
    print("=" * 80 + "\n")
    
    return entries_added


def generate_perf_entry_super_json(
    perf_super_json: str = "perf_super.json",
    perf_entry_super_json: str = "perf_entry_super.json",
    num_entries: int = 1
) -> None:
    """Generate perf_entry_super.json (latest entries) from perf_super.json (cumulative).
    
    Args:
        perf_super_json: Path to cumulative JSON source
        perf_entry_super_json: Path to entry JSON output (latest entries only)
        num_entries: Number of latest entries to include
    """
    if not os.path.exists(perf_super_json):
        print(f"⚠️  {perf_super_json} not found, skipping entry JSON generation")
        return
    
    data = read_json(perf_super_json)
    if not isinstance(data, list):
        data = [data]
    
    if not data:
        print(f"⚠️  {perf_super_json} is empty, skipping entry JSON generation")
        return
    
    # Take the latest num_entries entries
    entry_data = data[-num_entries:] if num_entries > 0 else [data[-1]]
    
    # Write to perf_entry_super.json
    write_json(entry_data, perf_entry_super_json)
    print(f"✅ Generated entry JSON: {perf_entry_super_json} ({len(entry_data)} entries)")


def convert_super_json_to_csv(
    perf_super_json: str,
    output_csv: str,
    entry_only: bool = False,
    num_entries: int = 1
) -> None:
    """Convert JSON to CSV format.
    
    Args:
        perf_super_json: Path to JSON source
        output_csv: Output CSV path
        entry_only: If True, only convert latest entries; if False, convert all
        num_entries: Number of latest entries to include when entry_only=True
    """
    # Load JSON list
    if not os.path.exists(perf_super_json):
        print(f"⚠️  {perf_super_json} not found, skipping CSV generation")
        return
    
    data = read_json(perf_super_json)
    if not isinstance(data, list):
        data = [data]
    
    if not data:
        print(f"⚠️  {perf_super_json} is empty, skipping CSV generation")
        return
    
    if entry_only and data:
        # Take the latest num_entries entries
        data = data[-num_entries:] if num_entries > 0 else [data[-1]]
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Serialize complex fields to JSON strings
    if 'configs' in df.columns:
        df['configs'] = df['configs'].apply(
            lambda x: json.dumps(x) if x is not None else None
        )
    
    if 'multi_results' in df.columns:
        df['multi_results'] = df['multi_results'].apply(
            lambda x: json.dumps(x) if x is not None else None
        )
    
    # Write to CSV
    df.to_csv(output_csv, index=False)
    print(f"✅ Generated CSV: {output_csv} ({len(df)} entries)")


def update_perf_super_csv(
    perf_super_json: str = "perf_super.json",
    perf_super_csv: str = "perf_super.csv",
    num_entries: int = 1
) -> None:
    """Generate perf_entry_super.json, perf_entry_super.csv and perf_super.csv from perf_super.json.
    
    Args:
        perf_super_json: Path to cumulative JSON source (perf_super.json)
        perf_super_csv: Path to cumulative CSV (perf_super.csv)
        num_entries: Number of latest entries to include in perf_entry_super.*
    """
    print("\n" + "=" * 80)
    print("📄 GENERATING FILES FROM PERFORMANCE SUPERSET")
    print("=" * 80)
    
    # Generate perf_entry_super.json (latest entries from current run)
    generate_perf_entry_super_json(
        perf_super_json=perf_super_json,
        perf_entry_super_json="perf_entry_super.json",
        num_entries=num_entries
    )
    
    # Generate perf_entry_super.csv (latest entries from current run)
    convert_super_json_to_csv(
        "perf_entry_super.json",  # Use the entry JSON as source
        "perf_entry_super.csv",
        entry_only=False  # Read all from entry JSON (already filtered)
    )
    
    # Generate perf_super.csv (all entries)
    convert_super_json_to_csv(
        perf_super_json,
        perf_super_csv,
        entry_only=False
    )
    
    print("=" * 80 + "\n")

