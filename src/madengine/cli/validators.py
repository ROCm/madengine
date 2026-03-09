#!/usr/bin/env python3
"""
Validation functions for madengine CLI

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import glob
import json
import os
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel

from madengine.utils.discover_models import DiscoverModels
from .constants import ExitCode, VALID_GPU_VENDORS, VALID_GUEST_OS
from .utils import create_args_namespace


# Initialize Rich console
console = Console()


def validate_additional_context(
    additional_context: str,
    additional_context_file: Optional[str] = None,
    use_image: Optional[str] = None,
) -> Dict[str, str]:
    """
    Validate and parse additional context.

    Args:
        additional_context: JSON string containing additional context
        additional_context_file: Optional file containing additional context
        use_image: Optional pre-built image to use (skips required field validation)

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

    # When using pre-built image, gpu_vendor and guest_os are optional
    # They will be extracted from model card env_vars if needed
    using_prebuilt_image = use_image and use_image.lower() not in ["none", ""]
    
    if using_prebuilt_image:
        # For pre-built images, context is optional - will use model card env_vars
        if not context:
            console.print("ℹ️  No additional context provided (using pre-built image)")
            console.print("💡 gpu_vendor and guest_os will be read from model card env_vars if needed")
            return {}
        
        # If context provided, validate any gpu_vendor/guest_os values present
        if "gpu_vendor" in context:
            gpu_vendor = context["gpu_vendor"].upper()
            if gpu_vendor not in VALID_GPU_VENDORS:
                console.print(f"❌ Invalid gpu_vendor: [red]{context['gpu_vendor']}[/red]")
                console.print(f"💡 Supported values: [green]{', '.join(VALID_GPU_VENDORS)}[/green]")
                raise typer.Exit(ExitCode.INVALID_ARGS)
        
        if "guest_os" in context:
            guest_os = context["guest_os"].upper()
            if guest_os not in VALID_GUEST_OS:
                console.print(f"❌ Invalid guest_os: [red]{context['guest_os']}[/red]")
                console.print(f"💡 Supported values: [green]{', '.join(VALID_GUEST_OS)}[/green]")
                raise typer.Exit(ExitCode.INVALID_ARGS)
        
        console.print("✅ Context validated (pre-built image mode)")
        return context

    # For Dockerfile builds, require gpu_vendor and guest_os
    if not context:
        console.print("❌ [red]No additional context provided[/red]")
        console.print(
            "💡 For build operations, you must provide additional context with gpu_vendor and guest_os"
        )

        # Show example usage
        example_panel = Panel(
            """[bold cyan]Example usage:[/bold cyan]
madengine build --tags dummy --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'

[bold cyan]Or using a file:[/bold cyan]
madengine build --tags dummy --additional-context-file context.json

[bold cyan]Required fields:[/bold cyan]
• gpu_vendor: [green]AMD[/green], [green]NVIDIA[/green]
• guest_os: [green]UBUNTU[/green], [green]CENTOS[/green]

[bold cyan]Or use a pre-built image:[/bold cyan]
madengine build --tags dummy --use-image auto""",
            title="Additional Context Help",
            border_style="blue",
        )
        console.print(example_panel)
        raise typer.Exit(ExitCode.INVALID_ARGS)

    # Validate required fields for Dockerfile builds
    required_fields = ["gpu_vendor", "guest_os"]
    missing_fields = [field for field in required_fields if field not in context]

    if missing_fields:
        console.print(
            f"❌ Missing required fields: [red]{', '.join(missing_fields)}[/red]"
        )
        console.print(
            "💡 Both gpu_vendor and guest_os are required for Dockerfile builds"
        )
        console.print(
            "💡 Or use --use-image to skip Dockerfile build"
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


def process_batch_manifest_entries(
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

