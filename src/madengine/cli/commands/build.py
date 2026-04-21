#!/usr/bin/env python3
"""
Build command for madengine CLI

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
from typing import List, Optional

import typer
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

try:
    from typing import Annotated  # Python 3.9+
except ImportError:
    from typing_extensions import Annotated  # Python 3.8

from madengine.orchestration.build_orchestrator import BuildOrchestrator
from madengine.core.errors import BuildError, ConfigurationError, DiscoveryError

from ..constants import ExitCode, DEFAULT_MANIFEST_FILE
from ..utils import (
    console,
    setup_logging,
    split_comma_separated_tags,
    create_args_namespace,
    save_summary_with_feedback,
    display_results_table,
)
from ..validators import validate_additional_context, process_batch_manifest, process_batch_manifest_entries


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
    use_image: Annotated[
        Optional[str],
        typer.Option(
            "--use-image",
            is_flag=True,
            flag_value="auto",
            help="Skip Docker build and use pre-built image. Optionally specify image name, or omit to auto-detect from model card's DOCKER_IMAGE_NAME"
        ),
    ] = None,
    build_on_compute: Annotated[
        bool,
        typer.Option(
            "--build-on-compute",
            help="Build Docker images on SLURM compute node instead of login node"
        ),
    ] = False,
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
    
    if additional_context_file and additional_context!="{}":
        console.print(
            "❌ [bold red]Error: Cannot specify both --additional-context-file and --additional-context options[/bold red]"
        )
        raise typer.Exit(ExitCode.INVALID_ARGS)

    if use_image and registry:
        console.print(
            "❌ [bold red]Error: Cannot specify both --use-image and --registry options[/bold red]\n"
            "[yellow]Use --use-image for pre-built external images.[/yellow]\n"
            "[yellow]Use --registry to push locally built images.[/yellow]"
        )
        raise typer.Exit(ExitCode.INVALID_ARGS)

    if use_image and build_on_compute:
        console.print(
            "❌ [bold red]Error: Cannot specify both --use-image and --build-on-compute options[/bold red]\n"
            "[yellow]--use-image skips Docker build entirely.[/yellow]\n"
            "[yellow]--build-on-compute builds on SLURM compute nodes.[/yellow]"
        )
        raise typer.Exit(ExitCode.INVALID_ARGS)

    if build_on_compute and not registry:
        console.print(
            "❌ [bold red]Error: --build-on-compute requires --registry option[/bold red]\n"
            "[yellow]Build on compute node pushes image to registry.[/yellow]\n"
            "[yellow]Run phase will pull image in parallel on all nodes.[/yellow]\n"
            "[dim]Example: --build-on-compute --registry docker.io/myorg[/dim]"
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
                    f"📦 [bold cyan]Batch Build Mode[/bold cyan]\n"
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
                f"🔨 [bold cyan]Building Models[/bold cyan]\n"
                f"Tags: [yellow]{', '.join(processed_tags) if processed_tags else 'All models'}[/yellow]\n"
                f"Registry: [yellow]{registry or 'Local only'}[/yellow]",
                title="Build Configuration",
                border_style="blue",
            )
        )

    try:
        validated_context = validate_additional_context(
            additional_context, additional_context_file
        )

        # Create arguments object
        args = create_args_namespace(
            tags=effective_tags,
            target_archs=target_archs,
            registry=registry,
            additional_context=repr(validated_context),
            additional_context_file=None,
            clean_docker_cache=clean_docker_cache,
            manifest_output=manifest_output,
            live_output=live_output,
            verbose=verbose,
            _separate_phases=True,
            batch_build_metadata=batch_build_metadata if batch_build_metadata else None,
            use_image=use_image,
            build_on_compute=build_on_compute,
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
                use_image=use_image,
                build_on_compute=build_on_compute,
            )
            
            # Load build summary for display
            with open(manifest_output, 'r') as f:
                manifest = json.load(f)
                build_summary = manifest.get("summary", {})
            
            progress.update(task, description="Build completed!")

        # Handle batch manifest post-processing
        if batch_data:
            with console.status("Processing batch manifest..."):
                guest_os = validated_context.get("guest_os")
                gpu_vendor = validated_context.get("gpu_vendor")
                process_batch_manifest_entries(
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

