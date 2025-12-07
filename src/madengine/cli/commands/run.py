#!/usr/bin/env python3
"""
Run command for madengine CLI

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import ast
import json
import os
from typing import List, Optional

import typer
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

try:
    from typing import Annotated  # Python 3.9+
except ImportError:
    from typing_extensions import Annotated  # Python 3.8

from madengine.orchestration.run_orchestrator import RunOrchestrator
from madengine.core.errors import (
    ConfigurationError,
    RuntimeError as MADRuntimeError,
)

from ..constants import (
    ExitCode,
    DEFAULT_MANIFEST_FILE,
    DEFAULT_PERF_OUTPUT,
    DEFAULT_DATA_CONFIG,
    DEFAULT_TOOLS_CONFIG,
    DEFAULT_TIMEOUT,
)
from ..utils import (
    console,
    setup_logging,
    split_comma_separated_tags,
    create_args_namespace,
    save_summary_with_feedback,
    display_results_table,
    display_performance_table,
)


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

