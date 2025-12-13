#!/usr/bin/env python3
"""
Utility functions for madengine CLI

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from madengine.core.errors import ErrorHandler, set_error_handler
from .constants import ExitCode


# Initialize Rich console
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Setup Rich logging configuration and unified error handler."""
    log_level = logging.DEBUG if verbose else logging.INFO

    # Setup rich logging handler
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=verbose,
        markup=True,
        rich_tracebacks=True,
    )

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich_handler],
    )

    # Setup unified error handler
    error_handler = ErrorHandler(console=console, verbose=verbose)
    set_error_handler(error_handler)


def split_comma_separated_tags(tags: List[str]) -> List[str]:
    """Split comma-separated tags into individual tags.
    
    Handles both formats:
    - Multiple flags: --tags dummy --tags multi → ['dummy', 'multi']
    - Comma-separated: --tags dummy,multi → ['dummy', 'multi']
    
    Args:
        tags: List of tag strings (may contain comma-separated values)
        
    Returns:
        List of individual tag strings
    """
    if not tags:
        return []
    
    processed_tags = []
    for tag in tags:
        # Split by comma and strip whitespace
        split_tags = [t.strip() for t in tag.split(',') if t.strip()]
        processed_tags.extend(split_tags)
    
    return processed_tags


def create_args_namespace(**kwargs) -> object:
    """Create an argparse.Namespace-like object from keyword arguments."""

    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return Args(**kwargs)


def save_summary_with_feedback(
    summary: Dict, output_path: Optional[str], summary_type: str
) -> None:
    """Save summary to file with user feedback."""
    if output_path:
        try:
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2)
            console.print(
                f"💾 {summary_type} summary saved to: [cyan]{output_path}[/cyan]"
            )
        except IOError as e:
            console.print(f"❌ Failed to save {summary_type} summary: [red]{e}[/red]")
            raise typer.Exit(ExitCode.FAILURE)


def display_results_table(summary: Dict, title: str, show_gpu_arch: bool = False) -> None:
    """Display results in a formatted table with each model as a separate row."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Index", justify="right", style="dim")
    table.add_column("Status", style="bold")
    table.add_column("Model", style="cyan")
    
    # Add GPU Architecture column if multi-arch build was used
    if show_gpu_arch:
        table.add_column("GPU Architecture", style="yellow")

    successful = summary.get("successful_builds", summary.get("successful_runs", []))
    failed = summary.get("failed_builds", summary.get("failed_runs", []))

    # Helper function to extract model name from build result
    def extract_model_name(item):
        if isinstance(item, dict):
            # Prioritize direct model name field if available
            if "model" in item:
                return item["model"]
            elif "name" in item:
                return item["name"]
            # Fallback to extracting from docker_image for backward compatibility
            elif "docker_image" in item:
                # Extract model name from docker image name
                # e.g., "ci-dummy_dummy.ubuntu.amd" -> "dummy"
                # e.g., "ci-dummy_dummy.ubuntu.amd_gfx908" -> "dummy"
                docker_image = item["docker_image"]
                if docker_image.startswith("ci-"):
                    # Remove ci- prefix and extract model name
                    parts = docker_image[3:].split("_")
                    if len(parts) >= 2:
                        model_name = parts[0]  # First part is the model name
                    else:
                        model_name = parts[0] if parts else docker_image
                else:
                    model_name = docker_image
                return model_name
        return str(item)[:20]  # Fallback

    # Helper function to extract GPU architecture
    def extract_gpu_arch(item):
        if isinstance(item, dict) and "gpu_architecture" in item:
            return item["gpu_architecture"]
        return "N/A"

    # Add successful builds/runs
    row_index = 1
    for item in successful:
        model_name = extract_model_name(item)
        if show_gpu_arch:
            gpu_arch = extract_gpu_arch(item)
            table.add_row(str(row_index), "✅ Success", model_name, gpu_arch)
        else:
            table.add_row(str(row_index), "✅ Success", model_name)
        row_index += 1

    # Add failed builds/runs
    for item in failed:
        if isinstance(item, dict):
            model_name = item.get("model", "Unknown")
            if show_gpu_arch:
                gpu_arch = item.get("architecture", "N/A")
                table.add_row(str(row_index), "❌ Failed", model_name, gpu_arch)
            else:
                table.add_row(str(row_index), "❌ Failed", model_name)
        else:
            if show_gpu_arch:
                table.add_row(str(row_index), "❌ Failed", str(item), "N/A")
            else:
                table.add_row(str(row_index), "❌ Failed", str(item))
        row_index += 1

    # Show empty state if no results
    if not successful and not failed:
        if show_gpu_arch:
            table.add_row("1", "ℹ️ No items", "", "")
        else:
            table.add_row("1", "ℹ️ No items", "")

    console.print(table)


def display_performance_table(perf_csv_path: str = "perf.csv", session_start_row: int = None) -> None:
    """Display performance metrics from perf.csv file.
    
    Shows all historical runs with visual markers for current session runs.
    
    Args:
        perf_csv_path: Path to the performance CSV file
        session_start_row: Optional row number to filter from (for current session only)
    """
    if not os.path.exists(perf_csv_path):
        console.print(f"[yellow]⚠️  Performance CSV not found: {perf_csv_path}[/yellow]")
        return
    
    try:
        import pandas as pd
        from madengine.utils.session_tracker import SessionTracker
        
        # Read CSV file
        df = pd.read_csv(perf_csv_path)
        
        if df.empty:
            console.print("[yellow]⚠️  Performance CSV is empty[/yellow]")
            return
        
        # Get session_start_row to mark current runs (don't filter, just mark)
        total_rows = len(df)
        
        # Try parameter first, then fall back to marker file
        if session_start_row is None:
            session_start_row = SessionTracker.load_session_marker_for_csv(perf_csv_path)
        
        # Count current session runs for title
        if session_start_row is not None and session_start_row < total_rows:
            current_run_count = total_rows - session_start_row
            title = f"📊 Performance Results (all {total_rows} runs, {current_run_count} from current session)"
        else:
            title = f"📊 Performance Results (all {total_rows} runs)"
        
        # Create performance table
        perf_table = Table(
            title=title,
            show_header=True,
            header_style="bold magenta"
        )
        
        # Add columns (with "Run" marker column as first column)
        perf_table.add_column("Run", justify="center", width=4)  # Marker column for current session
        perf_table.add_column("Index", justify="right", style="dim")
        perf_table.add_column("Model", style="cyan")
        perf_table.add_column("Topology", justify="center", style="blue")
        perf_table.add_column("Deployment", justify="center", style="cyan")
        perf_table.add_column("GPU Arch", style="yellow")
        perf_table.add_column("Performance", justify="right", style="green")
        perf_table.add_column("Metric", style="green")
        perf_table.add_column("Status", style="bold")
        perf_table.add_column("Duration", justify="right", style="blue")
        perf_table.add_column("Data Name", style="magenta")
        perf_table.add_column("Data Provider", style="magenta")        
        
        # Helper function to format duration
        def format_duration(duration):
            if pd.isna(duration) or duration == "":
                return "N/A"
            try:
                dur = float(duration)
                if dur < 1:
                    return f"{dur*1000:.0f}ms"
                elif dur < 60:
                    return f"{dur:.2f}s"
                else:
                    return f"{dur/60:.1f}m"
            except (ValueError, TypeError):
                return "N/A"
        
        # Helper function to format performance
        def format_performance(perf):
            if pd.isna(perf) or perf == "":
                return "N/A"
            try:
                val = float(perf)
                if val >= 1000:
                    return f"{val:,.0f}"
                elif val >= 10:
                    return f"{val:.1f}"
                else:
                    return f"{val:.2f}"
            except (ValueError, TypeError):
                return str(perf)
        
        # Add rows from dataframe
        for idx, row in df.iterrows():
            # Determine if this is a current session run
            is_current_run = (session_start_row is not None and idx >= session_start_row)
            run_marker = "[bold green]➤[/]" if is_current_run else ""  # Arrow marker for current runs
            
            model = str(row.get("model", "Unknown"))
            dataname = str(row.get("dataname", "")) if not pd.isna(row.get("dataname")) and row.get("dataname") != "" else "N/A"
            data_provider_type = str(row.get("data_provider_type", "")) if not pd.isna(row.get("data_provider_type")) and row.get("data_provider_type") != "" else "N/A"
            
            # Format topology: Always show "NxG" format for consistency
            # Examples: "1N×1G" (single node, single GPU), "1N×4G" (single node, 4 GPUs), "2N×2G" (2 nodes, 2 GPUs each)
            n_gpus = row.get("n_gpus", 1)
            nnodes = row.get("nnodes", 1)
            gpus_per_node = row.get("gpus_per_node", n_gpus)
            
            # Determine topology display format
            try:
                nnodes_int = int(nnodes) if not pd.isna(nnodes) and str(nnodes) != "" else 1
                gpus_per_node_int = int(gpus_per_node) if not pd.isna(gpus_per_node) and str(gpus_per_node) != "" else int(n_gpus) if not pd.isna(n_gpus) else 1
                
                # Always show NxG format for consistency
                topology = f"{nnodes_int}N×{gpus_per_node_int}G"
            except (ValueError, TypeError):
                # Fallback if parsing fails
                topology = "N/A"
            
            deployment_type = str(row.get("deployment_type", "local")) if not pd.isna(row.get("deployment_type")) and row.get("deployment_type") != "" else "local"
            gpu_arch = str(row.get("gpu_architecture", "N/A"))
            performance = format_performance(row.get("performance", ""))
            metric = str(row.get("metric", "")) if not pd.isna(row.get("metric")) else ""
            
            status = str(row.get("status", "UNKNOWN"))
            duration = format_duration(row.get("test_duration", ""))
            
            # Color-code status
            if status == "SUCCESS":
                status_display = "✅ Success"
            elif status == "FAILURE":
                status_display = "❌ Failed"
            else:
                status_display = f"⚠️  {status}"
            
            perf_table.add_row(
                run_marker,         # Marker column showing ➤ for current runs
                str(idx),
                model,
                topology,
                deployment_type,
                gpu_arch,
                performance,
                metric,
                status_display,
                duration,
                dataname,
                data_provider_type
            )
        
        console.print()  # Add blank line
        console.print(perf_table)
        
        # Print summary statistics
        total_runs = len(df)
        successful_runs = len(df[df["status"] == "SUCCESS"])
        failed_runs = len(df[df["status"] == "FAILURE"])
        
        console.print()
        console.print(f"[bold]Summary:[/bold] {total_runs} total runs, "
                     f"[green]{successful_runs} successful[/green], "
                     f"[red]{failed_runs} failed[/red]")
        
    except ImportError:
        console.print("[yellow]⚠️  pandas not installed. Install with: pip install pandas[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Error reading performance CSV: {e}[/red]")

