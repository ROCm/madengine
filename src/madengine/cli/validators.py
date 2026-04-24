#!/usr/bin/env python3
"""
Validation functions for madengine CLI

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import ast
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich.console import Console

from madengine.utils.discover_models import DiscoverModels
from madengine.core.additional_context_defaults import (
    DEFAULT_GPU_VENDOR,
    DEFAULT_GUEST_OS,
    apply_build_context_defaults,
)
from .constants import ExitCode, VALID_GPU_VENDORS, VALID_GUEST_OS
from .utils import create_args_namespace


# Initialize Rich console
console = Console()

_EXAMPLE_ADDITIONAL_CONTEXT = (
    '--additional-context \'{"docker_build_arg": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx942"}}\''
)


def parse_additional_context_cli_string(additional_context: str) -> Dict[str, Any]:
    """
    Parse --additional-context string: JSON first, then Python literal (single-quoted dicts).
    """
    if not additional_context or additional_context.strip() == "{}":
        return {}
    try:
        parsed = json.loads(additional_context)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(additional_context)
        except (ValueError, SyntaxError) as e:
            console.print(
                f"❌ Invalid additional_context format: [red]{e}[/red]"
            )
            console.print(
                "💡 Use JSON or a Python dict literal, e.g. "
                + _EXAMPLE_ADDITIONAL_CONTEXT
            )
            raise typer.Exit(ExitCode.INVALID_ARGS)
    if not isinstance(parsed, dict):
        console.print(
            "❌ additional_context must be a JSON object at the top level, not a list or scalar."
        )
        raise typer.Exit(ExitCode.INVALID_ARGS)
    return parsed


def merge_additional_context_from_sources(
    additional_context: str,
    additional_context_file: Optional[str],
) -> Tuple[Dict[str, Any], bool]:
    """
    Load file first, then overlay CLI string (CLI wins).

    Returns:
        (merged dict, loaded_from_cli_non_empty) for messaging.
    """
    context: Dict[str, Any] = {}
    if additional_context_file:
        try:
            with open(additional_context_file, "r") as f:
                context = json.load(f)
            if not isinstance(context, dict):
                console.print(
                    "❌ additional-context file must contain a JSON object at the top level."
                )
                raise typer.Exit(ExitCode.INVALID_ARGS)
            console.print(
                f"✅ Loaded additional context from file: [cyan]{additional_context_file}[/cyan]"
            )
        except (FileNotFoundError, json.JSONDecodeError) as e:
            console.print(f"❌ Failed to load additional context file: [red]{e}[/red]")
            raise typer.Exit(ExitCode.INVALID_ARGS)

    cli_non_empty = bool(additional_context and additional_context.strip() != "{}")
    if cli_non_empty:
        string_context = parse_additional_context_cli_string(additional_context)
        context.update(string_context)
        console.print("✅ Loaded additional context from command line")

    return context, cli_non_empty


def _fail_structure(key: str, expected: str) -> None:
    console.print(
        f"❌ Invalid additional context: key [red]{key}[/red] must be {expected}."
    )
    console.print(f"💡 Example: {_EXAMPLE_ADDITIONAL_CONTEXT}")
    raise typer.Exit(ExitCode.INVALID_ARGS)


def validate_additional_context_structure(context: Dict[str, Any]) -> None:
    """Validate types of known keys after defaults are applied."""
    if "docker_build_arg" in context:
        v = context["docker_build_arg"]
        if not isinstance(v, dict):
            _fail_structure("docker_build_arg", "an object (string values)")
        for bk, bv in v.items():
            if not isinstance(bk, str):
                _fail_structure("docker_build_arg", "an object with string keys")
            if not isinstance(bv, (str, int, float, bool)):
                _fail_structure(
                    f"docker_build_arg['{bk}']",
                    "a string, number, or boolean (Docker build-arg values)",
                )

    if "docker_env_vars" in context and not isinstance(
        context["docker_env_vars"], dict
    ):
        _fail_structure("docker_env_vars", "an object")

    if "docker_mounts" in context and not isinstance(context["docker_mounts"], dict):
        _fail_structure("docker_mounts", "an object")

    if "env_vars" in context and not isinstance(context["env_vars"], dict):
        _fail_structure("env_vars", "an object")

    if "tools" in context and not isinstance(context["tools"], list):
        _fail_structure("tools", "an array")

    if "pre_scripts" in context and not isinstance(context["pre_scripts"], list):
        _fail_structure("pre_scripts", "an array")

    for nest in (
        "k8s",
        "slurm",
        "kubernetes",
        "distributed",
        "vllm",
        "deployment_config",
        "deploy",
    ):
        if nest in context and not isinstance(context[nest], dict):
            _fail_structure(nest, "an object")

    if "docker_gpus" in context and not isinstance(context["docker_gpus"], str):
        _fail_structure("docker_gpus", "a string")

    if "MAD_CONTAINER_IMAGE" in context and not isinstance(
        context["MAD_CONTAINER_IMAGE"], str
    ):
        _fail_structure("MAD_CONTAINER_IMAGE", "a string")

    if "timeout" in context and not isinstance(
        context["timeout"], (int, float, type(None))
    ):
        _fail_structure("timeout", "a number")

    if "debug" in context and not isinstance(context["debug"], (bool, type(None))):
        _fail_structure("debug", "a boolean")

    if "gpu_vendor" in context and not isinstance(context["gpu_vendor"], str):
        _fail_structure("gpu_vendor", "a string")

    if "guest_os" in context and not isinstance(context["guest_os"], str):
        _fail_structure("guest_os", "a string")

    if "log_error_pattern_scan" in context and not isinstance(
        context["log_error_pattern_scan"], (bool, str, int, float, type(None))
    ):
        _fail_structure(
            "log_error_pattern_scan",
            "a boolean, string, number, or null",
        )

    if "log_error_benign_patterns" in context:
        lebp = context["log_error_benign_patterns"]
        if not isinstance(lebp, list) or not all(
            isinstance(x, str) for x in lebp
        ):
            _fail_structure(
                "log_error_benign_patterns",
                "an array of strings",
            )

    if "log_error_patterns" in context:
        lep = context["log_error_patterns"]
        if not isinstance(lep, list) or not lep or not all(
            isinstance(x, str) for x in lep
        ):
            _fail_structure(
                "log_error_patterns",
                "a non-empty array of strings",
            )


def _normalize_docker_build_arg_values(context: Dict[str, Any]) -> None:
    dba = context.get("docker_build_arg")
    if not isinstance(dba, dict):
        return
    for k in list(dba.keys()):
        v = dba[k]
        if not isinstance(v, str):
            dba[k] = str(v)


def _validate_gpu_vendor_guest_after_defaults(context: Dict[str, Any]) -> None:
    """Validate gpu_vendor / guest_os enums (expects keys present after defaults)."""
    gpu_vendor = context["gpu_vendor"].upper()
    if gpu_vendor not in VALID_GPU_VENDORS:
        console.print(f"❌ Invalid gpu_vendor: [red]{context['gpu_vendor']}[/red]")
        console.print(
            f"💡 Supported values: [green]{', '.join(VALID_GPU_VENDORS)}[/green]"
        )
        raise typer.Exit(ExitCode.INVALID_ARGS)

    guest_os = context["guest_os"].upper()
    if guest_os not in VALID_GUEST_OS:
        console.print(f"❌ Invalid guest_os: [red]{context['guest_os']}[/red]")
        console.print(
            f"💡 Supported values: [green]{', '.join(VALID_GUEST_OS)}[/green]"
        )
        raise typer.Exit(ExitCode.INVALID_ARGS)

    # Canonical form for Context.filter() and downstream (exact string match)
    context["gpu_vendor"] = gpu_vendor
    context["guest_os"] = guest_os

    console.print(
        f"✅ Context validated: [green]{gpu_vendor}[/green] + [green]{guest_os}[/green]"
    )


def finalize_additional_context_dict(
    context: Dict[str, Any],
    *,
    print_defaults_banner: bool = True,
) -> Dict[str, Any]:
    """
    Apply gpu_vendor/guest_os defaults, validate structure and enums.
    Mutates context in place.
    """
    missing_gpu = "gpu_vendor" not in context
    missing_guest = "guest_os" not in context
    apply_build_context_defaults(context)
    defaults_applied = []
    if missing_gpu:
        defaults_applied.append(("gpu_vendor", DEFAULT_GPU_VENDOR))
    if missing_guest:
        defaults_applied.append(("guest_os", DEFAULT_GUEST_OS))

    if print_defaults_banner and defaults_applied:
        console.print("\nℹ️  [cyan]Using default values for build configuration:[/cyan]")
        for field, value in defaults_applied:
            console.print(f"   • {field}: [green]{value}[/green] (default)")
        console.print(
            "\n💡 [dim]To customize, use --additional-context "
            '\'{"gpu_vendor": "NVIDIA", "guest_os": "CENTOS"}\'[/dim]\n'
        )

    validate_additional_context_structure(context)
    _normalize_docker_build_arg_values(context)
    _validate_gpu_vendor_guest_after_defaults(context)
    return context


def additional_context_needs_cli_validation(
    additional_context: str,
    additional_context_file: Optional[str],
) -> bool:
    """True when the user supplied a non-empty context (file and/or CLI string)."""
    if additional_context_file:
        return True
    if additional_context and additional_context.strip() != "{}":
        return True
    return False


def validate_additional_context(
    additional_context: str,
    additional_context_file: Optional[str] = None,
) -> Dict[str, Any]:
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
    context, _ = merge_additional_context_from_sources(
        additional_context, additional_context_file
    )
    finalize_additional_context_dict(context)
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
            # Initialize with a safe fallback so the except block can always reference it
            dockerfile_matched = "unknown"
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
                            dockerfile_matched = (
                                dockerfile_matched_list[0]
                                .split("/")[-1]
                                .replace(".Dockerfile", "")
                            )

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
                        model_entry = (
                            model_info.copy()
                        )  # Start with all fields from discovered model

                        # Ensure minimum required fields have fallback values
                        model_entry.setdefault("name", model_name)
                        model_entry.setdefault("dockerfile", f"docker/{model_name}")
                        model_entry.setdefault(
                            "scripts", f"scripts/{model_name}/run.sh"
                        )
                        model_entry.setdefault("n_gpus", "1")
                        model_entry.setdefault("owner", "")
                        model_entry.setdefault("training_precision", "")
                        model_entry.setdefault("tags", [])
                        model_entry.setdefault("args", "")
                        model_entry.setdefault("cred", "")

                        build_manifest["built_models"][
                            synthetic_image_name
                        ] = model_entry
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
