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
from madengine.tools.distributed_orchestrator import DistributedOrchestrator
from madengine.tools.discover_models import DiscoverModels
from madengine.runners.orchestrator_generation import (
    generate_ansible_setup,
    generate_k8s_setup,
    generate_slurm_setup,
)
from madengine.runners.factory import RunnerFactory
from madengine.core.errors import ErrorHandler, set_error_handler

# Initialize the main Typer app
app = typer.Typer(
    name="madengine-cli",
    help="🚀 madengine Distributed Orchestrator - Build and run AI models in distributed scenarios",
    rich_markup_mode="rich",
    add_completion=False,
    no_args_is_help=True,
)

# Sub-applications for organized commands
generate_app = typer.Typer(
    name="generate",
    help="📋 Generate orchestration files (Slurm, Ansible, Kubernetes)",
    rich_markup_mode="rich",
)
app.add_typer(generate_app, name="generate")

# Runner application for distributed execution
runner_app = typer.Typer(
    name="runner",
    help="🚀 Distributed runner for orchestrated execution across multiple nodes (SSH, Slurm, Ansible, Kubernetes)",
    rich_markup_mode="rich",
)
app.add_typer(runner_app, name="runner")

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
                    output="perf.csv",
                    ignore_deprecated_flag=False,
                    data_config_file_name="data.json",
                    tools_json_file_name="scripts/common/tools.json",
                    generate_sys_env_details=True,
                    force_mirror_local=None,
                    disable_skip_gpu_arch=False,
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
    🔨 Build Docker images for models in distributed scenarios.

    This command builds Docker images for the specified model tags and optionally
    pushes them to a registry. Additional context with gpu_vendor and guest_os
    is required for build-only operations.
    """
    setup_logging(verbose)

    # Validate mutually exclusive options
    if batch_manifest and tags:
        console.print(
            "❌ [bold red]Error: Cannot specify both --batch-manifest and --tags options[/bold red]"
        )
        raise typer.Exit(ExitCode.INVALID_ARGS)

    # Process batch manifest if provided
    batch_data = None
    effective_tags = tags
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
                f"Tags: [yellow]{', '.join(tags) if tags else 'All models'}[/yellow]\n"
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
            output=output,
            ignore_deprecated_flag=ignore_deprecated_flag,
            data_config_file_name=data_config_file_name,
            tools_json_file_name=tools_json_file_name,
            generate_sys_env_details=generate_sys_env_details,
            force_mirror_local=force_mirror_local,
            disable_skip_gpu_arch=disable_skip_gpu_arch,
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
            orchestrator = DistributedOrchestrator(args, build_only_mode=True)
            progress.update(task, description="Building models...")

            # Prepare build phase arguments
            build_phase_kwargs = dict(
                registry=registry,
                clean_cache=clean_docker_cache,
                manifest_output=manifest_output,
            )
            # Pass batch_build_metadata to build_phase if present
            if batch_build_metadata:
                build_phase_kwargs["batch_build_metadata"] = batch_build_metadata

            build_summary = orchestrator.build_phase(**build_phase_kwargs)
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

        # Check results and exit
        failed_builds = len(build_summary.get("failed_builds", []))
        if failed_builds == 0:
            console.print(
                "🎉 [bold green]All builds completed successfully![/bold green]"
            )
            raise typer.Exit(ExitCode.SUCCESS)
        else:
            console.print(
                f"💥 [bold red]Build failed for {failed_builds} models[/bold red]"
            )
            raise typer.Exit(ExitCode.BUILD_FAILURE)

    except typer.Exit:
        raise
    except Exception as e:
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
                tags=tags,
                manifest_file=manifest_file,
                registry=registry,
                timeout=timeout,
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
                orchestrator = DistributedOrchestrator(args)
                progress.update(task, description="Running models...")

                execution_summary = orchestrator.run_phase(
                    manifest_file=manifest_file,
                    registry=registry,
                    timeout=timeout,
                    keep_alive=keep_alive,
                )
                progress.update(task, description="Execution completed!")

            # Display results
            display_results_table(execution_summary, "Execution Results")
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

            # Check for MAD_CONTAINER_IMAGE in additional context
            mad_container_image = additional_context_dict.get("MAD_CONTAINER_IMAGE")
            
            if mad_container_image:
                # Local image mode - skip build phase and generate manifest
                console.print(
                    Panel(
                        f"🏠📦 [bold cyan]Local Image Mode (Skip Build + Run)[/bold cyan]\n"
                        f"Container Image: [yellow]{mad_container_image}[/yellow]\n"
                        f"Tags: [yellow]{', '.join(tags) if tags else 'All models'}[/yellow]\n"
                        f"Timeout: [yellow]{timeout if timeout != -1 else 'Default'}[/yellow]s\n"
                        f"[dim]Note: Build phase will be skipped, using local image[/dim]",
                        title="Local Image Configuration",
                        border_style="blue",
                    )
                )

                # Create arguments object for local image mode
                args = create_args_namespace(
                    tags=tags,
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
                        "Initializing local image orchestrator...", total=None
                    )
                    orchestrator = DistributedOrchestrator(args)

                    # Generate manifest for local image (skip build phase)
                    progress.update(task, description="Generating manifest for local image...")
                    build_summary = orchestrator.generate_local_image_manifest(
                        container_image=mad_container_image,
                        manifest_output=manifest_output,
                    )

                    # Run phase with local image
                    progress.update(task, description="Running models with local image...")
                    execution_summary = orchestrator.run_phase(
                        manifest_file=manifest_output,
                        registry=registry,
                        timeout=timeout,
                        keep_alive=keep_alive,
                    )
                    progress.update(task, description="Local image workflow completed!")

                # Combine summaries for local image mode
                workflow_summary = {
                    "build_phase": build_summary,
                    "run_phase": execution_summary,
                    "local_image_mode": True,
                    "container_image": mad_container_image,
                    "overall_success": len(execution_summary.get("failed_runs", [])) == 0,
                }

                # Display results
                display_results_table(execution_summary, "Local Image Execution Results")
                save_summary_with_feedback(workflow_summary, summary_output, "Local Image Workflow")

                if workflow_summary["overall_success"]:
                    console.print(
                        "🎉 [bold green]Local image workflow finished successfully![/bold green]"
                    )
                    raise typer.Exit(ExitCode.SUCCESS)
                else:
                    failed_runs = len(execution_summary.get("failed_runs", []))
                    console.print(
                        f"💥 [bold red]Local image workflow completed but {failed_runs} model executions failed[/bold red]"
                    )
                    raise typer.Exit(ExitCode.RUN_FAILURE)

            else:
                # Full workflow
                if manifest_file:
                    console.print(
                        f"⚠️  Manifest file [yellow]{manifest_file}[/yellow] not found, running complete workflow"
                    )

                console.print(
                    Panel(
                        f"🔨🚀 [bold cyan]Complete Workflow (Build + Run)[/bold cyan]\n"
                        f"Tags: [yellow]{', '.join(tags) if tags else 'All models'}[/yellow]\n"
                        f"Registry: [yellow]{registry or 'Local only'}[/yellow]\n"
                        f"Timeout: [yellow]{timeout if timeout != -1 else 'Default'}[/yellow]s",
                        title="Workflow Configuration",
                        border_style="magenta",
                    )
                )

                # Create arguments object for full workflow
                args = create_args_namespace(
                tags=tags,
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
                # Build phase
                task = progress.add_task(
                    "Initializing workflow orchestrator...", total=None
                )
                orchestrator = DistributedOrchestrator(args)

                progress.update(task, description="Building models...")
                build_summary = orchestrator.build_phase(
                    registry=registry,
                    clean_cache=clean_docker_cache,
                    manifest_output=manifest_output,
                )

                failed_builds = len(build_summary.get("failed_builds", []))
                if failed_builds > 0:
                    progress.update(task, description="Build failed!")
                    console.print(
                        f"💥 [bold red]Build failed for {failed_builds} models, aborting workflow[/bold red]"
                    )
                    display_results_table(build_summary, "Build Results")
                    raise typer.Exit(ExitCode.BUILD_FAILURE)

                # Run phase
                progress.update(task, description="Running models...")
                execution_summary = orchestrator.run_phase(
                    manifest_file=manifest_output,
                    registry=registry,
                    timeout=timeout,
                    keep_alive=keep_alive,
                )
                progress.update(task, description="Workflow completed!")

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
    except Exception as e:
        console.print(f"💥 [bold red]Run process failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
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

    console.print(
        Panel(
            f"🔍 [bold cyan]Discovering Models[/bold cyan]\n"
            f"Tags: [yellow]{tags if tags else 'All models'}[/yellow]",
            title="Model Discovery",
            border_style="blue",
        )
    )

    try:
        # Create args namespace similar to mad.py
        args = create_args_namespace(tags=tags)
        
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


@generate_app.command("ansible")
def generate_ansible(
    manifest_file: Annotated[
        str, typer.Option("--manifest-file", "-m", help="Build manifest file")
    ] = DEFAULT_MANIFEST_FILE,
    environment: Annotated[
        str, typer.Option("--environment", "-e", help="Environment configuration")
    ] = "default",
    output: Annotated[
        str, typer.Option("--output", "-o", help="Output Ansible playbook file")
    ] = DEFAULT_ANSIBLE_OUTPUT,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """
    📋 Generate Ansible playbook for distributed execution.

    Uses the enhanced build manifest as the primary configuration source
    with environment-specific values for customization.
    """
    setup_logging(verbose)

    console.print(
        Panel(
            f"📋 [bold cyan]Generating Ansible Playbook[/bold cyan]\n"
            f"Manifest: [yellow]{manifest_file}[/yellow]\n"
            f"Environment: [yellow]{environment}[/yellow]\n"
            f"Output: [yellow]{output}[/yellow]",
            title="Ansible Generation",
            border_style="blue",
        )
    )

    try:
        # Validate input files
        if not os.path.exists(manifest_file):
            console.print(
                f"❌ [bold red]Manifest file not found: {manifest_file}[/bold red]"
            )
            raise typer.Exit(ExitCode.FAILURE)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating Ansible playbook...", total=None)

            # Use the new template system
            result = generate_ansible_setup(
                manifest_file=manifest_file,
                environment=environment,
                output_dir=str(Path(output).parent),
            )

            progress.update(task, description="Ansible playbook generated!")

        console.print(
            f"✅ [bold green]Ansible setup generated successfully:[/bold green]"
        )
        for file_type, file_path in result.items():
            console.print(f"  📄 {file_type}: [cyan]{file_path}[/cyan]")

    except Exception as e:
        console.print(
            f"💥 [bold red]Failed to generate Ansible playbook: {e}[/bold red]"
        )
        if verbose:
            console.print_exception()
        raise typer.Exit(ExitCode.FAILURE)


@generate_app.command("k8s")
def generate_k8s(
    manifest_file: Annotated[
        str, typer.Option("--manifest-file", "-m", help="Build manifest file")
    ] = DEFAULT_MANIFEST_FILE,
    environment: Annotated[
        str, typer.Option("--environment", "-e", help="Environment configuration")
    ] = "default",
    output_dir: Annotated[
        str, typer.Option("--output-dir", "-o", help="Output directory for manifests")
    ] = "k8s-setup",
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """
    ☸️  Generate Kubernetes manifests for distributed execution.

    Uses the enhanced build manifest as the primary configuration source
    with environment-specific values for customization.
    """
    setup_logging(verbose)

    console.print(
        Panel(
            f"☸️  [bold cyan]Generating Kubernetes Manifests[/bold cyan]\n"
            f"Manifest: [yellow]{manifest_file}[/yellow]\n"
            f"Environment: [yellow]{environment}[/yellow]\n"
            f"Output Directory: [yellow]{output_dir}[/yellow]",
            title="Kubernetes Generation",
            border_style="blue",
        )
    )

    try:
        # Validate input files
        if not os.path.exists(manifest_file):
            console.print(
                f"❌ [bold red]Manifest file not found: {manifest_file}[/bold red]"
            )
            raise typer.Exit(ExitCode.FAILURE)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating Kubernetes manifests...", total=None)

            # Use the new template system
            result = generate_k8s_setup(
                manifest_file=manifest_file,
                environment=environment,
                output_dir=output_dir,
            )

            progress.update(task, description="Kubernetes manifests generated!")

        console.print(
            f"✅ [bold green]Kubernetes setup generated successfully:[/bold green]"
        )
        for file_type, file_paths in result.items():
            console.print(f"  📄 {file_type}:")
            if isinstance(file_paths, list):
                for file_path in file_paths:
                    console.print(f"    - [cyan]{file_path}[/cyan]")
            else:
                console.print(f"    - [cyan]{file_paths}[/cyan]")

    except Exception as e:
        console.print(
            f"💥 [bold red]Failed to generate Kubernetes manifests: {e}[/bold red]"
        )
        if verbose:
            console.print_exception()
        raise typer.Exit(ExitCode.FAILURE)


@generate_app.command("slurm")
def generate_slurm(
    manifest_file: Annotated[
        str,
        typer.Option(
            "--manifest-file",
            "-m",
            help="📄 Path to build manifest JSON file",
        ),
    ] = "build_manifest.json",
    environment: Annotated[
        str,
        typer.Option(
            "--environment",
            "-e",
            help="🌍 Environment configuration (default, dev, prod, test)",
        ),
    ] = "default",
    output_dir: Annotated[
        str,
        typer.Option(
            "--output-dir",
            "-o",
            help="📂 Output directory for generated SLURM files",
        ),
    ] = "slurm-setup",
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """
    🖥️ Generate SLURM job scripts and configuration for distributed execution.

    Creates job array scripts, individual job scripts, inventory configuration,
    and submission helper scripts for SLURM cluster execution.

    Example:
        madengine-cli generate slurm --manifest-file build_manifest.json --environment prod --output-dir slurm-setup
    """
    setup_logging(verbose)

    console.print(
        Panel(
            f"🖥️ [bold cyan]Generating SLURM Setup[/bold cyan]\n"
            f"📄 Manifest: {manifest_file}\n"
            f"🌍 Environment: {environment}\n"
            f"📂 Output: {output_dir}",
            title="SLURM Generation",
            border_style="blue",
        )
    )

    # Validate manifest file exists
    if not os.path.exists(manifest_file):
        console.print(f"❌ [bold red]Manifest file not found: {manifest_file}[/bold red]")
        raise typer.Exit(ExitCode.FAILURE)

    try:
        with console.status("[bold green]Generating SLURM configuration..."):
            # Generate complete SLURM setup
            result = generate_slurm_setup(
                manifest_file=manifest_file,
                environment=environment,
                output_dir=output_dir,
            )

        # Display success message with generated files
        console.print(f"✅ [bold green]SLURM setup generated successfully![/bold green]")
        console.print(f"📁 [cyan]Setup directory:[/cyan] {output_dir}")
        
        console.print("\n📋 [cyan]Generated files:[/cyan]")
        for file_type, file_path in result.items():
            if file_type == "individual_jobs":
                console.print(f"  • [yellow]{file_type}:[/yellow] {len(file_path)} job scripts")
                for job_script in file_path[:3]:  # Show first 3
                    console.print(f"    - {os.path.basename(job_script)}")
                if len(file_path) > 3:
                    console.print(f"    - ... and {len(file_path) - 3} more")
            else:
                console.print(f"  • [yellow]{file_type}:[/yellow] {file_path}")

        console.print(
            f"\n💡 [dim]Next step:[/dim] [cyan]madengine-cli runner slurm --inventory {os.path.join(output_dir, 'inventory.yml')} --job-scripts-dir {output_dir}[/cyan]"
        )

    except FileNotFoundError as e:
        console.print(
            f"💥 [bold red]File not found: {e}[/bold red]"
        )
        raise typer.Exit(ExitCode.FAILURE)
    except Exception as e:
        console.print(
            f"💥 [bold red]Failed to generate SLURM setup: {e}[/bold red]"
        )
        if verbose:
            console.print_exception()
        raise typer.Exit(ExitCode.FAILURE)


@generate_app.command("list")
def list_templates(
    template_dir: Annotated[
        Optional[str], typer.Option("--template-dir", help="Custom template directory")
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """
    📋 List available templates.

    Shows all available Jinja2 templates organized by type (ansible, k8s, etc.).
    """
    setup_logging(verbose)

    console.print(
        Panel(
            f"📋 [bold cyan]Available Templates[/bold cyan]",
            title="Template Listing",
            border_style="blue",
        )
    )

    try:
        # Create template generator
        from madengine.runners.template_generator import TemplateGenerator

        generator = TemplateGenerator(template_dir)

        templates = generator.list_templates()

        if not templates:
            console.print("❌ [yellow]No templates found[/yellow]")
            raise typer.Exit(ExitCode.SUCCESS)

        # Display templates in a formatted table
        table = Table(
            title="Available Templates", show_header=True, header_style="bold magenta"
        )
        table.add_column("Type", style="cyan")
        table.add_column("Templates", style="yellow")

        for template_type, template_files in templates.items():
            files_str = "\n".join(template_files) if template_files else "No templates"
            table.add_row(template_type.upper(), files_str)

        console.print(table)

    except Exception as e:
        console.print(f"💥 [bold red]Failed to list templates: {e}[/bold red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(ExitCode.FAILURE)


@generate_app.command("validate")
def validate_template(
    template_path: Annotated[
        str, typer.Argument(help="Path to template file to validate")
    ],
    template_dir: Annotated[
        Optional[str], typer.Option("--template-dir", help="Custom template directory")
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """
    ✅ Validate template syntax.

    Validates Jinja2 template syntax and checks for common issues.
    """
    setup_logging(verbose)

    console.print(
        Panel(
            f"✅ [bold cyan]Validating Template[/bold cyan]\n"
            f"Template: [yellow]{template_path}[/yellow]",
            title="Template Validation",
            border_style="green",
        )
    )

    try:
        # Create template generator
        from madengine.runners.template_generator import TemplateGenerator

        generator = TemplateGenerator(template_dir)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Validating template...", total=None)

            is_valid = generator.validate_template(template_path)

            progress.update(task, description="Validation completed!")

        if is_valid:
            console.print(
                f"✅ [bold green]Template validation successful:[/bold green]"
            )
            console.print(f"  📄 Template: [cyan]{template_path}[/cyan]")
            console.print(f"  🎯 Syntax: [green]Valid[/green]")
        else:
            console.print(f"❌ [bold red]Template validation failed:[/bold red]")
            console.print(f"  📄 Template: [cyan]{template_path}[/cyan]")
            console.print(f"  🎯 Syntax: [red]Invalid[/red]")
            raise typer.Exit(ExitCode.FAILURE)

    except Exception as e:
        console.print(f"💥 [bold red]Failed to validate template: {e}[/bold red]")
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
# RUNNER COMMANDS
# ============================================================================


@runner_app.command("ssh")
def runner_ssh(
    inventory_file: Annotated[
        str,
        typer.Option(
            "--inventory",
            "-i",
            help="🗂️ Path to inventory file (YAML or JSON format)",
        ),
    ] = DEFAULT_INVENTORY_FILE,
    manifest_file: Annotated[
        str,
        typer.Option(
            "--manifest-file",
            "-m",
            help="📋 Build manifest file (generated by 'madengine-cli build')",
        ),
    ] = DEFAULT_MANIFEST_FILE,
    report_output: Annotated[
        str,
        typer.Option(
            "--report-output",
            help="📊 Output file for execution report",
        ),
    ] = DEFAULT_RUNNER_REPORT,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="🔍 Enable verbose logging",
        ),
    ] = False,
):
    """
    🔐 Execute models across multiple nodes using SSH.

    Distributes pre-built build manifest (created by 'madengine-cli build')
    to remote nodes based on inventory configuration and executes
    'madengine-cli run' remotely through SSH client.

    The build manifest contains all configuration (tags, timeout, registry, etc.)
    so only inventory and manifest file paths are needed.

    Example:
        madengine-cli runner ssh --inventory nodes.yml --manifest-file build_manifest.json
    """
    setup_logging(verbose)

    try:
        # Validate input files
        if not os.path.exists(inventory_file):
            console.print(
                f"❌ [bold red]Inventory file not found: {inventory_file}[/bold red]"
            )
            raise typer.Exit(ExitCode.FAILURE)

        if not os.path.exists(manifest_file):
            console.print(
                f"❌ [bold red]Build manifest file not found: {manifest_file}[/bold red]"
            )
            console.print(
                "💡 Generate it first using: [cyan]madengine-cli build[/cyan]"
            )
            raise typer.Exit(ExitCode.FAILURE)

        # Create SSH runner
        console.print("🚀 [bold blue]Starting SSH distributed execution[/bold blue]")

        with console.status("Initializing SSH runner..."):
            runner = RunnerFactory.create_runner(
                "ssh", inventory_path=inventory_file, console=console, verbose=verbose
            )

        # Execute workload (minimal spec - most info is in the manifest)
        console.print(f"� Distributing manifest: [cyan]{manifest_file}[/cyan]")
        console.print(f"📋 Using inventory: [cyan]{inventory_file}[/cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Executing SSH distributed workload...", total=None
            )

            # Create minimal workload spec (most info is in the manifest)
            from madengine.runners.base import WorkloadSpec

            workload = WorkloadSpec(
                model_tags=[],  # Not needed - in manifest
                manifest_file=manifest_file,  # This is the key input
                timeout=3600,  # Default timeout, actual timeout from manifest
                registry=None,  # Auto-detected from manifest
                additional_context={},
                node_selector={},
                parallelism=1,
            )

            result = runner.run(workload)

        # Display results
        _display_runner_results(result, "SSH")

        # Generate report
        report_path = runner.generate_report(report_output)
        console.print(
            f"📊 Execution report saved to: [bold green]{report_path}[/bold green]"
        )

        # Exit with appropriate code
        if result.failed_executions == 0:
            console.print(
                "✅ [bold green]All executions completed successfully[/bold green]"
            )
            raise typer.Exit(code=ExitCode.SUCCESS)
        else:
            console.print(
                f"❌ [bold red]{result.failed_executions} execution(s) failed[/bold red]"
            )
            raise typer.Exit(code=ExitCode.RUN_FAILURE)

    except ImportError as e:
        console.print(f"💥 [bold red]SSH runner not available: {e}[/bold red]")
        console.print(
            "Install SSH dependencies: [bold cyan]pip install paramiko scp[/bold cyan]"
        )
        raise typer.Exit(code=ExitCode.FAILURE)
    except Exception as e:
        console.print(f"💥 [bold red]SSH execution failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=ExitCode.RUN_FAILURE)


@runner_app.command("ansible")
def runner_ansible(
    inventory_file: Annotated[
        str,
        typer.Option(
            "--inventory",
            "-i",
            help="🗂️ Path to inventory file (YAML or JSON format)",
        ),
    ] = DEFAULT_INVENTORY_FILE,
    playbook_file: Annotated[
        str,
        typer.Option(
            "--playbook",
            help="📋 Path to Ansible playbook file (generated by 'madengine-cli generate ansible')",
        ),
    ] = DEFAULT_ANSIBLE_OUTPUT,
    report_output: Annotated[
        str,
        typer.Option(
            "--report-output",
            help="📊 Output file for execution report",
        ),
    ] = DEFAULT_RUNNER_REPORT,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="🔍 Enable verbose logging",
        ),
    ] = False,
):
    """
    ⚡ Execute models across cluster using Ansible.

    Runs pre-generated Ansible playbook (created by 'madengine-cli generate ansible')
    with inventory file leveraging ansible-runner to distribute
    workload for parallel execution of models on cluster.

    The playbook contains all configuration (tags, timeout, registry, etc.)
    so only inventory and playbook paths are needed.

    Example:
        madengine-cli runner ansible --inventory cluster.yml --playbook madengine_distributed.yml
    """
    setup_logging(verbose)

    try:
        # Validate input files
        if not os.path.exists(inventory_file):
            console.print(
                f"❌ [bold red]Inventory file not found: {inventory_file}[/bold red]"
            )
            raise typer.Exit(ExitCode.FAILURE)

        if not os.path.exists(playbook_file):
            console.print(
                f"❌ [bold red]Playbook file not found: {playbook_file}[/bold red]"
            )
            console.print(
                "💡 Generate it first using: [cyan]madengine-cli generate ansible[/cyan]"
            )
            raise typer.Exit(ExitCode.FAILURE)

        # Create Ansible runner
        console.print(
            "🚀 [bold blue]Starting Ansible distributed execution[/bold blue]"
        )

        with console.status("Initializing Ansible runner..."):
            runner = RunnerFactory.create_runner(
                "ansible",
                inventory_path=inventory_file,
                playbook_path=playbook_file,
                console=console,
                verbose=verbose,
            )

        # Execute workload (no workload spec needed - everything is in the playbook)
        console.print(f"� Executing playbook: [cyan]{playbook_file}[/cyan]")
        console.print(f"📋 Using inventory: [cyan]{inventory_file}[/cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Executing Ansible playbook...", total=None)

            # Create minimal workload spec (most info is in the playbook)
            from madengine.runners.base import WorkloadSpec

            workload = WorkloadSpec(
                model_tags=[],  # Not needed - in playbook
                manifest_file="",  # Not needed - in playbook
            )

            result = runner.run(workload)

        # Display results
        _display_runner_results(result, "Ansible")

        # Generate report
        report_path = runner.generate_report(report_output)
        console.print(
            f"📊 Execution report saved to: [bold green]{report_path}[/bold green]"
        )

        # Exit with appropriate code
        if result.failed_executions == 0:
            console.print(
                "✅ [bold green]All executions completed successfully[/bold green]"
            )
            raise typer.Exit(code=ExitCode.SUCCESS)
        else:
            console.print(
                f"❌ [bold red]{result.failed_executions} execution(s) failed[/bold red]"
            )
            raise typer.Exit(code=ExitCode.RUN_FAILURE)

    except ImportError as e:
        console.print(f"💥 [bold red]Ansible runner not available: {e}[/bold red]")
        console.print(
            "Install Ansible dependencies: [bold cyan]pip install ansible-runner[/bold cyan]"
        )
        raise typer.Exit(code=ExitCode.FAILURE)
    except Exception as e:
        console.print(f"💥 [bold red]Ansible execution failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=ExitCode.RUN_FAILURE)


@runner_app.command("k8s")
def runner_k8s(
    inventory_file: Annotated[
        str,
        typer.Option(
            "--inventory",
            "-i",
            help="🗂️ Path to inventory file (YAML or JSON format)",
        ),
    ] = DEFAULT_INVENTORY_FILE,
    manifests_dir: Annotated[
        str,
        typer.Option(
            "--manifests-dir",
            "-d",
            help="📁 Directory containing Kubernetes manifests (generated by 'madengine-cli generate k8s')",
        ),
    ] = "k8s-setup",
    kubeconfig: Annotated[
        Optional[str],
        typer.Option(
            "--kubeconfig",
            help="⚙️ Path to kubeconfig file",
        ),
    ] = None,
    report_output: Annotated[
        str,
        typer.Option(
            "--report-output",
            help="📊 Output file for execution report",
        ),
    ] = DEFAULT_RUNNER_REPORT,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="🔍 Enable verbose logging",
        ),
    ] = False,
):
    """
    ☸️ Execute models across Kubernetes cluster.

    Runs pre-generated Kubernetes manifests (created by 'madengine-cli generate k8s')
    with inventory file leveraging kubernetes python client to distribute
    workload for parallel execution of models on cluster.

    The manifests contain all configuration (tags, timeout, registry, etc.)
    so only inventory and manifests directory paths are needed.

    Example:
        madengine-cli runner k8s --inventory cluster.yml --manifests-dir k8s-setup
    """
    setup_logging(verbose)

    try:
        # Validate input files/directories
        if not os.path.exists(inventory_file):
            console.print(
                f"❌ [bold red]Inventory file not found: {inventory_file}[/bold red]"
            )
            raise typer.Exit(ExitCode.FAILURE)

        if not os.path.exists(manifests_dir):
            console.print(
                f"❌ [bold red]Manifests directory not found: {manifests_dir}[/bold red]"
            )
            console.print(
                "💡 Generate it first using: [cyan]madengine-cli generate k8s[/cyan]"
            )
            raise typer.Exit(ExitCode.FAILURE)

        # Create Kubernetes runner
        console.print(
            "🚀 [bold blue]Starting Kubernetes distributed execution[/bold blue]"
        )

        with console.status("Initializing Kubernetes runner..."):
            runner = RunnerFactory.create_runner(
                "k8s",
                inventory_path=inventory_file,
                manifests_dir=manifests_dir,
                kubeconfig_path=kubeconfig,
                console=console,
                verbose=verbose,
            )

        # Execute workload (no workload spec needed - everything is in the manifests)
        console.print(f"☸️  Applying manifests from: [cyan]{manifests_dir}[/cyan]")
        console.print(f"📋 Using inventory: [cyan]{inventory_file}[/cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Executing Kubernetes manifests...", total=None)

            # Create minimal workload spec (most info is in the manifests)
            from madengine.runners.base import WorkloadSpec

            workload = WorkloadSpec(
                model_tags=[],  # Not needed - in manifests
                manifest_file="",  # Not needed - in manifests
            )

            result = runner.run(workload)

        # Display results
        _display_runner_results(result, "Kubernetes")

        # Generate report
        report_path = runner.generate_report(report_output)
        console.print(
            f"📊 Execution report saved to: [bold green]{report_path}[/bold green]"
        )

        # Exit with appropriate code
        if result.failed_executions == 0:
            console.print(
                "✅ [bold green]All executions completed successfully[/bold green]"
            )
            raise typer.Exit(code=ExitCode.SUCCESS)
        else:
            console.print(
                f"❌ [bold red]{result.failed_executions} execution(s) failed[/bold red]"
            )
            raise typer.Exit(code=ExitCode.RUN_FAILURE)

    except ImportError as e:
        console.print(f"💥 [bold red]Kubernetes runner not available: {e}[/bold red]")
        console.print(
            "Install Kubernetes dependencies: [bold cyan]pip install kubernetes[/bold cyan]"
        )
        raise typer.Exit(code=ExitCode.FAILURE)
    except Exception as e:
        console.print(f"💥 [bold red]Kubernetes execution failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=ExitCode.RUN_FAILURE)


@runner_app.command("slurm")
def runner_slurm(
    inventory: Annotated[
        str,
        typer.Option(
            "--inventory",
            "-i",
            help="📋 Path to SLURM inventory file (generated by 'madengine-cli generate slurm')",
        ),
    ],
    job_scripts_dir: Annotated[
        str,
        typer.Option(
            "--job-scripts-dir",
            "-j",
            help="📂 Directory containing generated SLURM job scripts",
        ),
    ],
    timeout: Annotated[
        int,
        typer.Option(
            "--timeout",
            "-t",
            help="⏰ Execution timeout in seconds",
        ),
    ] = 3600,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """
    🖥️ Run distributed workload using pre-generated SLURM job scripts.

    Runs pre-generated SLURM job scripts (created by 'madengine-cli generate slurm')
    for distributed model execution across SLURM cluster nodes.

    Example:
        madengine-cli runner slurm --inventory cluster.yml --job-scripts-dir slurm-setup
    """
    setup_logging(verbose)

    console.print(
        Panel(
            f"🖥️ [bold cyan]SLURM Distributed Execution[/bold cyan]\n"
            f"📋 Inventory: {inventory}\n"
            f"📂 Job Scripts: {job_scripts_dir}\n"
            f"⏰ Timeout: {timeout}s",
            title="SLURM Runner",
            border_style="blue",
        )
    )

    try:
        # Validate input files/directories
        if not os.path.exists(inventory):
            console.print(
                f"❌ [bold red]Inventory file not found: {inventory}[/bold red]"
            )
            raise typer.Exit(ExitCode.FAILURE)

        if not os.path.exists(job_scripts_dir):
            console.print(
                f"❌ [bold red]Job scripts directory not found: {job_scripts_dir}[/bold red]"
            )
            console.print(
                "💡 Generate it first using: [cyan]madengine-cli generate slurm[/cyan]"
            )
            raise typer.Exit(ExitCode.FAILURE)

        # Create SLURM runner
        console.print(
            "🚀 [bold blue]Starting SLURM distributed execution[/bold blue]"
        )

        with console.status("Initializing SLURM runner..."):
            runner = RunnerFactory.create_runner(
                "slurm",
                inventory_path=inventory,
                job_scripts_dir=job_scripts_dir,
                console=console,
                verbose=verbose,
            )

        # Create minimal workload spec for SLURM runner
        from madengine.runners.base import WorkloadSpec
        workload = WorkloadSpec(
            model_tags=["slurm-execution"],  # Will be determined from job scripts
            manifest_file="",  # Not needed for pre-generated scripts
            timeout=timeout,
        )

        # Execute the workload
        with console.status("🔄 Executing SLURM workload..."):
            result = runner.run(workload)

        # Display results
        _display_runner_results(result, "SLURM")

        # Display success/failure message
        if result.successful_executions > 0:
            console.print(
                f"✅ [bold green]SLURM execution completed with {result.successful_executions} successful tasks[/bold green]"
            )
        
        if result.failed_executions > 0:
            console.print(
                f"⚠️  [bold yellow]{result.failed_executions} tasks failed[/bold yellow]"
            )
            
        # Exit with appropriate code
        if result.successful_executions == 0:
            raise typer.Exit(code=ExitCode.RUN_FAILURE)

    except KeyboardInterrupt:
        console.print("\n⚠️  [bold yellow]SLURM execution interrupted by user[/bold yellow]")
        raise typer.Exit(code=ExitCode.FAILURE)
    except Exception as e:
        console.print(f"💥 [bold red]SLURM execution failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=ExitCode.RUN_FAILURE)


def _display_runner_results(result, runner_type: str):
    """Display runner execution results in a formatted table.

    Args:
        result: DistributedResult object
        runner_type: Type of runner (SSH, Ansible, Kubernetes)
    """
    console.print(f"\n📊 [bold blue]{runner_type} Execution Results[/bold blue]")

    # Summary table
    summary_table = Table(title="Execution Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="magenta")

    summary_table.add_row("Total Nodes", str(result.total_nodes))
    summary_table.add_row("Successful Executions", str(result.successful_executions))
    summary_table.add_row("Failed Executions", str(result.failed_executions))
    summary_table.add_row("Total Duration", f"{result.total_duration:.2f}s")

    console.print(summary_table)

    # Detailed results table
    if result.node_results:
        results_table = Table(title="Detailed Results")
        results_table.add_column("Node", style="cyan")
        results_table.add_column("Model", style="yellow")
        results_table.add_column("Status", style="green")
        results_table.add_column("Duration", style="magenta")
        results_table.add_column("Error", style="red")

        for exec_result in result.node_results:
            status_color = "green" if exec_result.status == "SUCCESS" else "red"
            status_text = f"[{status_color}]{exec_result.status}[/{status_color}]"

            results_table.add_row(
                exec_result.node_id,
                exec_result.model_tag,
                status_text,
                f"{exec_result.duration:.2f}s",
                exec_result.error_message or "",
            )

        console.print(results_table)
