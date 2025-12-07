#!/usr/bin/env python3
"""
Modern CLI for madengine Distributed Orchestrator

Production-ready command-line interface built with Typer and Rich
for building and running models in distributed scenarios.
"""

import ast
import json
import logging
import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    from typing import Annotated  # Python 3.9+
except ImportError:
    from typing_extensions import Annotated  # Python 3.8

import typer
from rich import print as rprint
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.traceback import install

# Install rich traceback handler for better error displays
install(show_locals=True)

# Initialize Rich console
console = Console()

# Import madengine components
from madengine.orchestration.build_orchestrator import BuildOrchestrator
from madengine.orchestration.run_orchestrator import RunOrchestrator
from madengine.utils.discover_models import DiscoverModels
# Legacy runner imports removed (Phase 5 cleanup) - replaced by deployment/ architecture
from madengine.core.errors import (
    ErrorHandler,
    set_error_handler,
    BuildError,
    ConfigurationError,
    DiscoveryError,
    RuntimeError as MADRuntimeError,
)

# Initialize the main Typer app
app = typer.Typer(
    name="madengine-cli",
    help="🚀 madengine Distributed Orchestrator - Build and run AI models in distributed scenarios",
    rich_markup_mode="rich",
    add_completion=False,
    no_args_is_help=True,
)

# Legacy sub-applications removed (Phase 5 cleanup)
# - generate_app: Replaced by new deployment/ architecture
# - runner_app: Replaced by new deployment/ architecture
# Use: madengine-cli run --additional-context '{"deploy": "slurm"}' instead

# Constants
DEFAULT_MANIFEST_FILE = "build_manifest.json"
DEFAULT_PERF_OUTPUT = "perf.csv"
DEFAULT_DATA_CONFIG = "data.json"
DEFAULT_TOOLS_CONFIG = "./scripts/common/tools.json"
DEFAULT_ANSIBLE_OUTPUT = "madengine_distributed.yml"
DEFAULT_TIMEOUT = -1
DEFAULT_INVENTORY_FILE = "inventory.yml"
DEFAULT_RUNNER_REPORT = "runner_report.json"


# Exit codes
class ExitCode:
    SUCCESS = 0
    FAILURE = 1
    BUILD_FAILURE = 2
    RUN_FAILURE = 3
    INVALID_ARGS = 4


# Valid values for validation
VALID_GPU_VENDORS = ["AMD", "NVIDIA", "INTEL"]
VALID_GUEST_OS = ["UBUNTU", "CENTOS", "ROCKY"]


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


def process_batch_manifest(batch_manifest_file: str) -> Dict[str, List[str]]:
    """Process batch manifest file and extract model tags based on build_new flag.

    Args:
        batch_manifest_file: Path to the input batch.json file

    Returns:
        Dict containing 'build_tags' and 'all_tags' lists

    Raises:
        FileNotFoundError: If the manifest file doesn't exist
        ValueError: If the manifest format is invalid
    """
    if not os.path.exists(batch_manifest_file):
        raise FileNotFoundError(f"Batch manifest file not found: {batch_manifest_file}")

    try:
        with open(batch_manifest_file, "r") as f:
            manifest_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in batch manifest file: {e}")

    if not isinstance(manifest_data, list):
        raise ValueError("Batch manifest must be a list of model objects")

    build_tags = []  # Models that need to be built (build_new=true)
    all_tags = []  # All models in the manifest

    for i, model in enumerate(manifest_data):
        if not isinstance(model, dict):
            raise ValueError(f"Model entry {i} must be a dictionary")

        if "model_name" not in model:
            raise ValueError(f"Model entry {i} missing required 'model_name' field")

        model_name = model["model_name"]
        build_new = model.get("build_new", False)

        all_tags.append(model_name)
        if build_new:
            build_tags.append(model_name)

    return {
        "build_tags": build_tags,
        "all_tags": all_tags,
        "manifest_data": manifest_data,
    }


def validate_additional_context(
    additional_context: str,
    additional_context_file: Optional[str] = None,
) -> Dict[str, str]:
    """
    Validate and parse additional context.

    Args:
        additional_context: JSON string containing additional context
        additional_context_file: Optional file containing additional context

    Returns:
        Dict containing parsed additional context

    Raises:
        typer.Exit: If validation fails
    """
    context = {}

    # Load from file first
    if additional_context_file:
        try:
            with open(additional_context_file, "r") as f:
                context = json.load(f)
            console.print(
                f"✅ Loaded additional context from file: [cyan]{additional_context_file}[/cyan]"
            )
        except (FileNotFoundError, json.JSONDecodeError) as e:
            console.print(f"❌ Failed to load additional context file: [red]{e}[/red]")
            raise typer.Exit(ExitCode.INVALID_ARGS)

    # Parse string context (overrides file)
    if additional_context and additional_context != "{}":
        try:
            string_context = json.loads(additional_context)
            context.update(string_context)
            console.print("✅ Loaded additional context from command line")
        except json.JSONDecodeError as e:
            console.print(f"❌ Invalid JSON in additional context: [red]{e}[/red]")
            console.print("💡 Please provide valid JSON format")
            raise typer.Exit(ExitCode.INVALID_ARGS)

    if not context:
        console.print("❌ [red]No additional context provided[/red]")
        console.print(
            "💡 For build operations, you must provide additional context with gpu_vendor and guest_os"
        )

        # Show example usage
        example_panel = Panel(
            """[bold cyan]Example usage:[/bold cyan]
madengine-cli build --tags dummy --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'

[bold cyan]Or using a file:[/bold cyan]
madengine-cli build --tags dummy --additional-context-file context.json

[bold cyan]Required fields:[/bold cyan]
• gpu_vendor: [green]AMD[/green], [green]NVIDIA[/green], [green]INTEL[/green]
• guest_os: [green]UBUNTU[/green], [green]CENTOS[/green], [green]ROCKY[/green]""",
            title="Additional Context Help",
            border_style="blue",
        )
        console.print(example_panel)
        raise typer.Exit(ExitCode.INVALID_ARGS)

    # Validate required fields
    required_fields = ["gpu_vendor", "guest_os"]
    missing_fields = [field for field in required_fields if field not in context]

    if missing_fields:
        console.print(
            f"❌ Missing required fields: [red]{', '.join(missing_fields)}[/red]"
        )
        console.print(
            "💡 Both gpu_vendor and guest_os are required for build operations"
        )
        raise typer.Exit(ExitCode.INVALID_ARGS)

    # Validate gpu_vendor
    gpu_vendor = context["gpu_vendor"].upper()
    if gpu_vendor not in VALID_GPU_VENDORS:
        console.print(f"❌ Invalid gpu_vendor: [red]{context['gpu_vendor']}[/red]")
        console.print(
            f"💡 Supported values: [green]{', '.join(VALID_GPU_VENDORS)}[/green]"
        )
        raise typer.Exit(ExitCode.INVALID_ARGS)

    # Validate guest_os
    guest_os = context["guest_os"].upper()
    if guest_os not in VALID_GUEST_OS:
        console.print(f"❌ Invalid guest_os: [red]{context['guest_os']}[/red]")
        console.print(
            f"💡 Supported values: [green]{', '.join(VALID_GUEST_OS)}[/green]"
        )
        raise typer.Exit(ExitCode.INVALID_ARGS)

    console.print(
        f"✅ Context validated: [green]{gpu_vendor}[/green] + [green]{guest_os}[/green]"
    )
    return context


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


def _process_batch_manifest_entries(
    batch_data: Dict,
    manifest_output: str,
    registry: Optional[str],
    guest_os: Optional[str],
    gpu_vendor: Optional[str],
) -> None:
    """Process batch manifest and add entries for all models to build_manifest.json.

    Args:
        batch_data: Processed batch manifest data
        manifest_output: Path to the build manifest file
        registry: Registry used for the build
        guest_os: Guest OS for the build
        gpu_vendor: GPU vendor for the build
    """

    # Load the existing build manifest
    if os.path.exists(manifest_output):
        with open(manifest_output, "r") as f:
            build_manifest = json.load(f)
        # Remove top-level registry if present
        build_manifest.pop("registry", None)
    else:
        # Create a minimal manifest structure
        build_manifest = {
            "built_images": {},
            "built_models": {},
            "context": {},
            "credentials_required": [],
        }

    # Process each model in the batch manifest
    for model_entry in batch_data["manifest_data"]:
        model_name = model_entry["model_name"]
        build_new = model_entry.get("build_new", False)
        model_registry_image = model_entry.get("registry_image", "")
        model_registry = model_entry.get("registry", "")

        # If the model was not built (build_new=false), create an entry for it
        if not build_new:
            # Find the model configuration by discovering models with this tag
            try:
                # Create a temporary args object to discover the model
                temp_args = create_args_namespace(
                    tags=[model_name],
                    registry=registry,
                    additional_context="{}",
                    additional_context_file=None,
                    clean_docker_cache=False,
                    manifest_output=manifest_output,
                    live_output=False,
                    verbose=False,
                    _separate_phases=True,
                )

                discover_models = DiscoverModels(args=temp_args)
                models = discover_models.run()

                for model_info in models:
                    if model_info["name"] == model_name:
                        # Get dockerfile
                        dockerfile = model_info.get("dockerfile")
                        dockerfile_specified = (
                            f"{dockerfile}.{guest_os.lower()}.{gpu_vendor.lower()}"
                        )
                        dockerfile_matched_list = glob.glob(f"{dockerfile_specified}.*")

                        # Check the matched list
                        if not dockerfile_matched_list:
                            console.print(
                                f"Warning: No Dockerfile found for {dockerfile_specified}"
                            )
                            raise FileNotFoundError(
                                f"No Dockerfile found for {dockerfile_specified}"
                            )
                        else:
                            dockerfile_matched = dockerfile_matched_list[0].split("/")[-1].replace(".Dockerfile", "")

                        # Create a synthetic image name for this model
                        synthetic_image_name = f"ci-{model_name}_{dockerfile_matched}"

                        # Add to built_images (even though it wasn't actually built)
                        build_manifest["built_images"][synthetic_image_name] = {
                            "docker_image": synthetic_image_name,
                            "dockerfile": model_info.get("dockerfile"),
                            "base_docker": "",  # No base since not built
                            "docker_sha": "",  # No SHA since not built
                            "build_duration": 0,
                            "build_command": f"# Skipped build for {model_name} (build_new=false)",
                            "log_file": f"{model_name}_{dockerfile_matched}.build.skipped.log",
                            "registry_image": (
                                model_registry_image
                                or f"{model_registry or registry or 'dockerhub'}/{synthetic_image_name}"
                                if model_registry_image or model_registry or registry
                                else ""
                            ),
                            "registry": model_registry or registry or "dockerhub",
                        }

                        # Add to built_models - include all discovered model fields
                        model_entry = model_info.copy()  # Start with all fields from discovered model

                        # Ensure minimum required fields have fallback values
                        model_entry.setdefault("name", model_name)
                        model_entry.setdefault("dockerfile", f"docker/{model_name}")
                        model_entry.setdefault("scripts", f"scripts/{model_name}/run.sh")
                        model_entry.setdefault("n_gpus", "1")
                        model_entry.setdefault("owner", "")
                        model_entry.setdefault("training_precision", "")
                        model_entry.setdefault("tags", [])
                        model_entry.setdefault("args", "")
                        model_entry.setdefault("cred", "")

                        build_manifest["built_models"][synthetic_image_name] = model_entry
                        break

            except Exception as e:
                console.print(f"Warning: Could not process model {model_name}: {e}")
                # Create a minimal entry anyway
                synthetic_image_name = f"ci-{model_name}_{dockerfile_matched}"
                build_manifest["built_images"][synthetic_image_name] = {
                    "docker_image": synthetic_image_name,
                    "dockerfile": f"docker/{model_name}",
                    "base_docker": "",
                    "docker_sha": "",
                    "build_duration": 0,
                    "build_command": f"# Skipped build for {model_name} (build_new=false)",
                    "log_file": f"{model_name}_{dockerfile_matched}.build.skipped.log",
                    "registry_image": model_registry_image or "",
                    "registry": model_registry or registry or "dockerhub",
                }
                build_manifest["built_models"][synthetic_image_name] = {
                    "name": model_name,
                    "dockerfile": f"docker/{model_name}",
                    "scripts": f"scripts/{model_name}/run.sh",
                    "n_gpus": "1",
                    "owner": "",
                    "training_precision": "",
                    "tags": [],
                    "args": "",
                }

    # Save the updated manifest
    with open(manifest_output, "w") as f:
        json.dump(build_manifest, f, indent=2)

    console.print(
        f"✅ Added entries for all models from batch manifest to {manifest_output}"
    )


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


def display_performance_table(perf_csv_path: str = "perf.csv") -> None:
    """Display performance metrics from perf.csv file.
    
    Args:
        perf_csv_path: Path to the performance CSV file
    """
    if not os.path.exists(perf_csv_path):
        console.print(f"[yellow]⚠️  Performance CSV not found: {perf_csv_path}[/yellow]")
        return
    
    try:
        import pandas as pd
        
        # Read CSV file
        df = pd.read_csv(perf_csv_path)
        
        if df.empty:
            console.print("[yellow]⚠️  Performance CSV is empty[/yellow]")
            return
        
        # Create performance table
        perf_table = Table(
            title="📊 Performance Results",
            show_header=True,
            header_style="bold magenta"
        )
        
        # Add columns
        perf_table.add_column("Index", justify="right", style="dim")
        perf_table.add_column("Model", style="cyan")
        perf_table.add_column("Topology", justify="center", style="blue")  # Changed from "GPUs"
        perf_table.add_column("Deployment", justify="center", style="cyan")
        perf_table.add_column("GPU Arch", style="yellow")
        perf_table.add_column("Performance", justify="right", style="green")
        perf_table.add_column("Metric", style="green")
        perf_table.add_column("Efficiency", justify="right", style="yellow")  # NEW
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
            
            # Format scaling efficiency
            scaling_efficiency = row.get("scaling_efficiency", "")
            if not pd.isna(scaling_efficiency) and scaling_efficiency != "":
                try:
                    efficiency_val = float(scaling_efficiency)
                    efficiency_display = f"{efficiency_val:.1f}%"
                except (ValueError, TypeError):
                    efficiency_display = "N/A"
            else:
                efficiency_display = "N/A"
            
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
                str(idx),
                model,
                topology,           # Changed from n_gpus
                deployment_type,
                gpu_arch,
                performance,
                metric,
                efficiency_display, # NEW
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


@app.command()
def build(
    tags: Annotated[
        List[str],
        typer.Option("--tags", "-t", help="Model tags to build (can specify multiple)"),
    ] = [],
    target_archs: Annotated[
        List[str],
        typer.Option(
            "--target-archs", 
            "-a", 
            help="Target GPU architectures to build for (e.g., gfx908,gfx90a,gfx942). If not specified, builds single image with MAD_SYSTEM_GPU_ARCHITECTURE from additional_context or detected GPU architecture."
        ),
    ] = [],
    registry: Annotated[
        Optional[str],
        typer.Option("--registry", "-r", help="Docker registry to push images to"),
    ] = None,
    batch_manifest: Annotated[
        Optional[str],
        typer.Option(
            "--batch-manifest", help="Input batch.json file for batch build mode"
        ),
    ] = None,
    additional_context: Annotated[
        str,
        typer.Option(
            "--additional-context", "-c", help="Additional context as JSON string"
        ),
    ] = "{}",
    additional_context_file: Annotated[
        Optional[str],
        typer.Option(
            "--additional-context-file",
            "-f",
            help="File containing additional context JSON",
        ),
    ] = None,
    clean_docker_cache: Annotated[
        bool,
        typer.Option("--clean-docker-cache", help="Rebuild images without using cache"),
    ] = False,
    manifest_output: Annotated[
        str,
        typer.Option("--manifest-output", "-m", help="Output file for build manifest"),
    ] = DEFAULT_MANIFEST_FILE,
    summary_output: Annotated[
        Optional[str],
        typer.Option(
            "--summary-output", "-s", help="Output file for build summary JSON"
        ),
    ] = None,
    live_output: Annotated[
        bool, typer.Option("--live-output", "-l", help="Print output in real-time")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """
    🔨 Build Docker images for models in distributed scenarios.

    This command builds Docker images for the specified model tags and optionally
    pushes them to a registry. Additional context with gpu_vendor and guest_os
    is required for build-only operations.
    """
    setup_logging(verbose)

    # Process tags to handle comma-separated values
    # Supports both: --tags dummy --tags multi AND --tags dummy,multi
    processed_tags = split_comma_separated_tags(tags)
    
    # Validate mutually exclusive options
    if batch_manifest and processed_tags:
        console.print(
            "❌ [bold red]Error: Cannot specify both --batch-manifest and --tags options[/bold red]"
        )
        raise typer.Exit(ExitCode.INVALID_ARGS)

    # Process batch manifest if provided
    batch_data = None
    effective_tags = processed_tags
    batch_build_metadata = None

    # There are 2 scenarios for batch builds and single builds
    # - Batch builds: Use the batch manifest to determine which models to build
    # - Single builds: Use the tags directly
    if batch_manifest:
        # Process the batch manifest
        if verbose:
            console.print(f"[DEBUG] Processing batch manifest: {batch_manifest}")
        try:
            batch_data = process_batch_manifest(batch_manifest)
            if verbose:
                console.print(f"[DEBUG] batch_data: {batch_data}")

            effective_tags = batch_data["build_tags"]
            # Build a mapping of model_name -> registry_image/registry for build_new models
            batch_build_metadata = {}
            for model in batch_data["manifest_data"]:
                if model.get("build_new", False):
                    batch_build_metadata[model["model_name"]] = {
                        "registry_image": model.get("registry_image"),
                        "registry": model.get("registry"),
                    }
            if verbose:
                console.print(f"[DEBUG] batch_build_metadata: {batch_build_metadata}")

            console.print(
                Panel(
                    f"� [bold cyan]Batch Build Mode[/bold cyan]\n"
                    f"Input manifest: [yellow]{batch_manifest}[/yellow]\n"
                    f"Total models: [yellow]{len(batch_data['all_tags'])}[/yellow]\n"
                    f"Models to build: [yellow]{len(batch_data['build_tags'])}[/yellow] ({', '.join(batch_data['build_tags']) if batch_data['build_tags'] else 'none'})\n"
                    f"Registry: [yellow]{registry or 'Local only'}[/yellow]",
                    title="Batch Build Configuration",
                    border_style="blue",
                )
            )
        except (FileNotFoundError, ValueError) as e:
            console.print(
                f"❌ [bold red]Error processing batch manifest: {e}[/bold red]"
            )
            raise typer.Exit(ExitCode.INVALID_ARGS)
    else:
        console.print(
            Panel(
                f"�🔨 [bold cyan]Building Models[/bold cyan]\n"
                f"Tags: [yellow]{', '.join(processed_tags) if processed_tags else 'All models'}[/yellow]\n"
                f"Registry: [yellow]{registry or 'Local only'}[/yellow]",
                title="Build Configuration",
                border_style="blue",
            )
        )

    try:
        # Validate additional context
        validate_additional_context(additional_context, additional_context_file)

        # Create arguments object
        args = create_args_namespace(
            tags=effective_tags,
            target_archs=target_archs,
            registry=registry,
            additional_context=additional_context,
            additional_context_file=additional_context_file,
            clean_docker_cache=clean_docker_cache,
            manifest_output=manifest_output,
            live_output=live_output,
            verbose=verbose,
            _separate_phases=True,
            batch_build_metadata=batch_build_metadata if batch_build_metadata else None,
        )

        # Initialize orchestrator in build-only mode
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing build orchestrator...", total=None)
            
            # Use new BuildOrchestrator
            orchestrator = BuildOrchestrator(args)
            progress.update(task, description="Building models...")

            # Execute build workflow
            manifest_file = orchestrator.execute(
                registry=registry,
                clean_cache=clean_docker_cache,
                manifest_output=manifest_output,
                batch_build_metadata=batch_build_metadata,
            )
            
            # Load build summary for display
            with open(manifest_output, 'r') as f:
                manifest = json.load(f)
                build_summary = manifest.get("summary", {})
            
            progress.update(task, description="Build completed!")

        # Handle batch manifest post-processing
        if batch_data:
            with console.status("Processing batch manifest..."):
                additional_context = getattr(args, "additional_context", None)
                if isinstance(additional_context, str):
                    additional_context = json.loads(additional_context)
                guest_os = (
                    additional_context.get("guest_os") if additional_context else None
                )
                gpu_vendor = (
                    additional_context.get("gpu_vendor") if additional_context else None
                )
                _process_batch_manifest_entries(
                    batch_data, manifest_output, registry, guest_os, gpu_vendor
                )

        # Display results
        # Check if target_archs was used to show GPU architecture column
        show_gpu_arch = bool(target_archs)
        display_results_table(build_summary, "Build Results", show_gpu_arch)

        # Save summary
        save_summary_with_feedback(build_summary, summary_output, "Build")

        # Check results and exit with appropriate code
        failed_builds = len(build_summary.get("failed_builds", []))
        successful_builds = len(build_summary.get("successful_builds", []))
        
        if failed_builds == 0:
            console.print(
                "🎉 [bold green]All builds completed successfully![/bold green]"
            )
            raise typer.Exit(ExitCode.SUCCESS)
        elif successful_builds > 0:
            # Partial success
            console.print(
                f"⚠️  [bold yellow]Partial success: "
                f"{successful_builds} built, {failed_builds} failed[/bold yellow]"
            )
            console.print(
                "💡 [dim]Successful builds are available in build_manifest.json[/dim]"
            )
            raise typer.Exit(ExitCode.BUILD_FAILURE)  # Non-zero exit for CI/CD
        else:
            # All failed
            console.print(
                f"💥 [bold red]All builds failed[/bold red]"
            )
            raise typer.Exit(ExitCode.BUILD_FAILURE)

    except typer.Exit:
        raise
    except BuildError as e:
        # Specific build error handling
        console.print(f"💥 [bold red]Build error: {e}[/bold red]")
        if hasattr(e, 'suggestions') and e.suggestions:
            console.print("\n💡 [cyan]Suggestions:[/cyan]")
            for suggestion in e.suggestions:
                console.print(f"  • {suggestion}")
        raise typer.Exit(ExitCode.BUILD_FAILURE)
        
    except ConfigurationError as e:
        # Configuration errors
        console.print(f"⚙️  [bold red]Configuration error: {e}[/bold red]")
        if hasattr(e, 'suggestions') and e.suggestions:
            console.print("\n💡 [cyan]Suggestions:[/cyan]")
            for suggestion in e.suggestions:
                console.print(f"  • {suggestion}")
        raise typer.Exit(ExitCode.INVALID_ARGS)
        
    except DiscoveryError as e:
        # Model discovery errors
        console.print(f"🔍 [bold red]Discovery error: {e}[/bold red]")
        console.print("💡 Check MODEL_DIR or models.json configuration")
        raise typer.Exit(ExitCode.FAILURE)
        
    except KeyboardInterrupt:
        console.print("\n🛑 [yellow]Build cancelled by user[/yellow]")
        raise typer.Exit(ExitCode.FAILURE)
        
    except PermissionError as e:
        console.print(f"🔒 [bold red]Permission denied: {e}[/bold red]")
        console.print("💡 Check file/directory permissions or run with appropriate privileges")
        raise typer.Exit(ExitCode.FAILURE)
        
    except FileNotFoundError as e:
        console.print(f"📁 [bold red]File not found: {e}[/bold red]")
        console.print("💡 Check that all required files exist")
        raise typer.Exit(ExitCode.FAILURE)
        
    except Exception as e:
        console.print(f"💥 [bold red]Unexpected error: {e}[/bold red]")
        if verbose:
            console.print_exception()
        
        from madengine.core.errors import handle_error, create_error_context
        context = create_error_context(
            operation="build",
            phase="build",
            component="build_command"
        )
        handle_error(e, context=context)
        raise typer.Exit(ExitCode.FAILURE)


@app.command()
def run(
    tags: Annotated[
        List[str],
        typer.Option("--tags", "-t", help="Model tags to run (can specify multiple)"),
    ] = [],
    manifest_file: Annotated[
        str, typer.Option("--manifest-file", "-m", help="Build manifest file path")
    ] = "",
    registry: Annotated[
        Optional[str], typer.Option("--registry", "-r", help="Docker registry URL")
    ] = None,
    timeout: Annotated[
        int,
        typer.Option(
            "--timeout",
            help="Timeout for model run in seconds (-1 for default, 0 for no timeout)",
        ),
    ] = DEFAULT_TIMEOUT,
    additional_context: Annotated[
        str,
        typer.Option(
            "--additional-context", "-c", help="Additional context as JSON string"
        ),
    ] = "{}",
    additional_context_file: Annotated[
        Optional[str],
        typer.Option(
            "--additional-context-file",
            "-f",
            help="File containing additional context JSON",
        ),
    ] = None,
    keep_alive: Annotated[
        bool,
        typer.Option("--keep-alive", help="Keep Docker containers alive after run"),
    ] = False,
    keep_model_dir: Annotated[
        bool, typer.Option("--keep-model-dir", help="Keep model directory after run")
    ] = False,
    skip_model_run: Annotated[
        bool, typer.Option("--skip-model-run", help="Skip running the model")
    ] = False,
    clean_docker_cache: Annotated[
        bool,
        typer.Option(
            "--clean-docker-cache",
            help="Rebuild images without using cache (for full workflow)",
        ),
    ] = False,
    manifest_output: Annotated[
        str,
        typer.Option(
            "--manifest-output", help="Output file for build manifest (full workflow)"
        ),
    ] = DEFAULT_MANIFEST_FILE,
    summary_output: Annotated[
        Optional[str],
        typer.Option("--summary-output", "-s", help="Output file for summary JSON"),
    ] = None,
    live_output: Annotated[
        bool, typer.Option("--live-output", "-l", help="Print output in real-time")
    ] = False,
    output: Annotated[
        str, typer.Option("--output", "-o", help="Performance output file")
    ] = DEFAULT_PERF_OUTPUT,
    ignore_deprecated_flag: Annotated[
        bool, typer.Option("--ignore-deprecated", help="Force run deprecated models")
    ] = False,
    data_config_file_name: Annotated[
        str, typer.Option("--data-config", help="Custom data configuration file")
    ] = DEFAULT_DATA_CONFIG,
    tools_json_file_name: Annotated[
        str, typer.Option("--tools-config", help="Custom tools JSON configuration")
    ] = DEFAULT_TOOLS_CONFIG,
    generate_sys_env_details: Annotated[
        bool,
        typer.Option("--sys-env-details", help="Generate system config env details"),
    ] = True,
    force_mirror_local: Annotated[
        Optional[str],
        typer.Option("--force-mirror-local", help="Path to force local data mirroring"),
    ] = None,
    disable_skip_gpu_arch: Annotated[
        bool,
        typer.Option(
            "--disable-skip-gpu-arch",
            help="Disable skipping models based on GPU architecture",
        ),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """
    🚀 Run model containers in distributed scenarios.

    If manifest-file is provided and exists, runs execution phase only.
    Otherwise runs the complete workflow (build + run).
    """
    setup_logging(verbose)

    # Process tags to handle comma-separated values
    processed_tags = split_comma_separated_tags(tags)

    # Input validation
    if timeout < -1:
        console.print(
            "❌ [red]Timeout must be -1 (default) or a positive integer[/red]"
        )
        raise typer.Exit(ExitCode.INVALID_ARGS)

    try:
        # Check if we're doing execution-only or full workflow
        manifest_exists = manifest_file and os.path.exists(manifest_file)

        if manifest_exists:
            console.print(
                Panel(
                    f"🚀 [bold cyan]Running Models (Execution Only)[/bold cyan]\n"
                    f"Manifest: [yellow]{manifest_file}[/yellow]\n"
                    f"Registry: [yellow]{registry or 'Auto-detected'}[/yellow]\n"
                    f"Timeout: [yellow]{timeout if timeout != -1 else 'Default'}[/yellow]s",
                    title="Execution Configuration",
                    border_style="green",
                )
            )

            # Create arguments object for execution only
            args = create_args_namespace(
                tags=processed_tags,
                manifest_file=manifest_file,
                registry=registry,
                timeout=timeout,
                additional_context=additional_context,
                additional_context_file=additional_context_file,
                keep_alive=keep_alive,
                keep_model_dir=keep_model_dir,
                skip_model_run=skip_model_run,
                live_output=live_output,
                output=output,
                ignore_deprecated_flag=ignore_deprecated_flag,
                data_config_file_name=data_config_file_name,
                tools_json_file_name=tools_json_file_name,
                generate_sys_env_details=generate_sys_env_details,
                force_mirror_local=force_mirror_local,
                disable_skip_gpu_arch=disable_skip_gpu_arch,
                verbose=verbose,
                _separate_phases=True,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Initializing execution orchestrator...", total=None
                )
                
                # Use new RunOrchestrator
                orchestrator = RunOrchestrator(args)
                progress.update(task, description="Running models...")

                execution_summary = orchestrator.execute(
                    manifest_file=manifest_file,
                    tags=None,  # manifest-only mode
                    registry=registry,
                    timeout=timeout,
                )
                progress.update(task, description="Execution completed!")

            # Display results summary
            display_results_table(execution_summary, "Execution Results")
            
            # Display detailed performance metrics from CSV
            display_performance_table(getattr(args, "output", DEFAULT_PERF_OUTPUT))
            
            save_summary_with_feedback(execution_summary, summary_output, "Execution")

            failed_runs = len(execution_summary.get("failed_runs", []))
            if failed_runs == 0:
                console.print(
                    "🎉 [bold green]All model executions completed successfully![/bold green]"
                )
                raise typer.Exit(ExitCode.SUCCESS)
            else:
                console.print(
                    f"💥 [bold red]Execution failed for {failed_runs} models[/bold red]"
                )
                raise typer.Exit(ExitCode.RUN_FAILURE)

        else:
            # Check if MAD_CONTAINER_IMAGE is provided - this enables local image mode
            additional_context_dict = {}
            try:
                if additional_context and additional_context != "{}":
                    additional_context_dict = json.loads(additional_context)
            except json.JSONDecodeError:
                try:
                    # Try parsing as Python dict literal
                    additional_context_dict = ast.literal_eval(additional_context)
                except (ValueError, SyntaxError):
                    console.print(
                        f"❌ [red]Invalid additional_context format: {additional_context}[/red]"
                    )
                    raise typer.Exit(ExitCode.INVALID_ARGS)
            
            # Load additional context from file if provided
            if additional_context_file and os.path.exists(additional_context_file):
                try:
                    with open(additional_context_file, 'r') as f:
                        file_context = json.load(f)
                        additional_context_dict.update(file_context)
                except json.JSONDecodeError:
                    console.print(
                        f"❌ [red]Invalid JSON format in {additional_context_file}[/red]"
                    )
                    raise typer.Exit(ExitCode.INVALID_ARGS)

            # MAD_CONTAINER_IMAGE handling is now done in RunOrchestrator
            # Full workflow (may include MAD_CONTAINER_IMAGE mode)
            if manifest_file:
                console.print(
                    f"⚠️  Manifest file [yellow]{manifest_file}[/yellow] not found, running complete workflow"
                )

            console.print(
                Panel(
                    f"🔨🚀 [bold cyan]Complete Workflow (Build + Run)[/bold cyan]\n"
                    f"Tags: [yellow]{', '.join(processed_tags) if processed_tags else 'All models'}[/yellow]\n"
                    f"Registry: [yellow]{registry or 'Local only'}[/yellow]\n"
                    f"Timeout: [yellow]{timeout if timeout != -1 else 'Default'}[/yellow]s",
                    title="Workflow Configuration",
                    border_style="magenta",
                )
            )

            # Create arguments object for full workflow
            args = create_args_namespace(
                tags=processed_tags,
                registry=registry,
                timeout=timeout,
                additional_context=additional_context,
                additional_context_file=additional_context_file,
                keep_alive=keep_alive,
                keep_model_dir=keep_model_dir,
                skip_model_run=skip_model_run,
                clean_docker_cache=clean_docker_cache,
                manifest_output=manifest_output,
                live_output=live_output,
                output=output,
                ignore_deprecated_flag=ignore_deprecated_flag,
                data_config_file_name=data_config_file_name,
                tools_json_file_name=tools_json_file_name,
                generate_sys_env_details=generate_sys_env_details,
                force_mirror_local=force_mirror_local,
                disable_skip_gpu_arch=disable_skip_gpu_arch,
                verbose=verbose,
                _separate_phases=True,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Initializing workflow orchestrator...", total=None
                )
                
                # Use new RunOrchestrator (handles build+run automatically when tags provided)
                orchestrator = RunOrchestrator(args)
                
                progress.update(task, description="Building and running models...")
                execution_summary = orchestrator.execute(
                    manifest_file=None,  # Triggers build phase
                    tags=processed_tags,
                    registry=registry,
                    timeout=timeout,
                )
                progress.update(task, description="Workflow completed!")

            # Load build summary from generated manifest
            with open(manifest_output, 'r') as f:
                manifest = json.load(f)
                build_summary = manifest.get("summary", {})

            # Combine summaries
            workflow_summary = {
                "build_phase": build_summary,
                "run_phase": execution_summary,
                "overall_success": (
                    len(build_summary.get("failed_builds", [])) == 0
                    and len(execution_summary.get("failed_runs", [])) == 0
                ),
            }

            # Display results
            display_results_table(build_summary, "Build Results")
            display_results_table(execution_summary, "Execution Results")
            
            # Display detailed performance metrics from CSV
            display_performance_table(getattr(args, "output", DEFAULT_PERF_OUTPUT))
            
            save_summary_with_feedback(workflow_summary, summary_output, "Workflow")

            if workflow_summary["overall_success"]:
                console.print(
                    "🎉 [bold green]Complete workflow finished successfully![/bold green]"
                )
                raise typer.Exit(ExitCode.SUCCESS)
            else:
                failed_runs = len(execution_summary.get("failed_runs", []))
                if failed_runs > 0:
                    console.print(
                        f"💥 [bold red]Workflow completed but {failed_runs} model executions failed[/bold red]"
                    )
                    raise typer.Exit(ExitCode.RUN_FAILURE)
                else:
                    console.print(
                        "💥 [bold red]Workflow failed for unknown reasons[/bold red]"
                    )
                    raise typer.Exit(ExitCode.FAILURE)

    except typer.Exit:
        raise
    except MADRuntimeError as e:
        # Runtime execution errors
        console.print(f"💥 [bold red]Runtime error: {e}[/bold red]")
        if hasattr(e, 'suggestions') and e.suggestions:
            console.print("\n💡 [cyan]Suggestions:[/cyan]")
            for suggestion in e.suggestions:
                console.print(f"  • {suggestion}")
        raise typer.Exit(ExitCode.RUN_FAILURE)
        
    except ConfigurationError as e:
        # Configuration errors
        console.print(f"⚙️  [bold red]Configuration error: {e}[/bold red]")
        if hasattr(e, 'suggestions') and e.suggestions:
            console.print("\n💡 [cyan]Suggestions:[/cyan]")
            for suggestion in e.suggestions:
                console.print(f"  • {suggestion}")
        raise typer.Exit(ExitCode.INVALID_ARGS)
        
    except KeyboardInterrupt:
        console.print("\n🛑 [yellow]Run cancelled by user[/yellow]")
        raise typer.Exit(ExitCode.FAILURE)
        
    except FileNotFoundError as e:
        console.print(f"📁 [bold red]File not found: {e}[/bold red]")
        console.print("💡 Check manifest file path and required files")
        raise typer.Exit(ExitCode.FAILURE)
        
    except Exception as e:
        console.print(f"💥 [bold red]Run process failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        
        from madengine.core.errors import handle_error, create_error_context
        context = create_error_context(
            operation="run",
            phase="run",
            component="run_command"
        )
        handle_error(e, context=context)
        raise typer.Exit(ExitCode.FAILURE)


@app.command()
def discover(
    tags: Annotated[
        List[str],
        typer.Option("--tags", "-t", help="Model tags to discover (can specify multiple)"),
    ] = [],
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """
    🔍 Discover all models in the project.

    This command discovers all available models in the project based on the
    specified tags. If no tags are provided, all models will be discovered.
    """
    setup_logging(verbose)

    # Process tags to handle comma-separated values
    processed_tags = split_comma_separated_tags(tags)

    console.print(
        Panel(
            f"🔍 [bold cyan]Discovering Models[/bold cyan]\n"
            f"Tags: [yellow]{processed_tags if processed_tags else 'All models'}[/yellow]",
            title="Model Discovery",
            border_style="blue",
        )
    )

    try:
        # Create args namespace similar to mad.py
        args = create_args_namespace(tags=processed_tags)
        
        # Use DiscoverModels class
        # Note: DiscoverModels prints output directly and returns None
        discover_models_instance = DiscoverModels(args=args)
        result = discover_models_instance.run()
        
        console.print("✅ [bold green]Model discovery completed successfully[/bold green]")

    except Exception as e:
        console.print(f"💥 [bold red]Model discovery failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(ExitCode.FAILURE)




@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool, typer.Option("--version", help="Show version and exit")
    ] = False,
) -> None:
    """
    🚀 madengine Distributed Orchestrator

    Modern CLI for building and running AI models in distributed scenarios.
    Built with Typer and Rich for a beautiful, production-ready experience.
    """
    if version:
        # You might want to get the actual version from your package
        console.print(
            "🚀 [bold cyan]madengine-cli[/bold cyan] version [green]1.0.0[/green]"
        )
        raise typer.Exit()

    # If no command is provided, show help
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        ctx.exit()


def cli_main() -> None:
    """Entry point for the CLI application."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n🛑 [yellow]Operation cancelled by user[/yellow]")
        sys.exit(ExitCode.FAILURE)
    except Exception as e:
        console.print(f"💥 [bold red]Unexpected error: {e}[/bold red]")
        console.print_exception()
        sys.exit(ExitCode.FAILURE)


if __name__ == "__main__":
    cli_main()


# ============================================================================
