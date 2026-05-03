#!/usr/bin/env python3
"""
Pure image filtering logic for run orchestrator.

Returns filtered image dicts and lists of skipped items so the orchestrator
can handle logging and CSV writes. No I/O or console here.
"""

from typing import Any, Dict, List, Tuple


def filter_images_by_gpu_compatibility(
    built_images: Dict[str, Any],
    runtime_gpu_vendor: str,
    runtime_gpu_arch: str,
) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
    """
    Filter images compatible with runtime GPU vendor and architecture.

    Args:
        built_images: Dictionary of built images from manifest.
        runtime_gpu_vendor: Runtime GPU vendor (AMD, NVIDIA, NONE).
        runtime_gpu_arch: Runtime GPU architecture (gfx90a, sm_90, etc.).

    Returns:
        (compatible_images, skipped_list) where skipped_list is
        [(model_name, reason), ...] for orchestrator to log.
    """
    compatible: Dict[str, Any] = {}
    skipped: List[Tuple[str, str]] = []

    for model_name, image_info in built_images.items():
        image_gpu_vendor = image_info.get("gpu_vendor", "")
        image_arch = image_info.get("gpu_architecture", "")

        if not image_gpu_vendor:
            compatible[model_name] = image_info
            continue

        if runtime_gpu_vendor == "NONE" or image_gpu_vendor == runtime_gpu_vendor:
            if image_arch:
                if image_arch == runtime_gpu_arch:
                    compatible[model_name] = image_info
                else:
                    skipped.append(
                        (
                            model_name,
                            f"architecture mismatch ({image_arch} != {runtime_gpu_arch})",
                        )
                    )
            else:
                compatible[model_name] = image_info
        else:
            skipped.append(
                (
                    model_name,
                    f"GPU vendor mismatch ({image_gpu_vendor} != {runtime_gpu_vendor})",
                )
            )

    return compatible, skipped


def filter_images_by_skip_gpu_arch(
    built_images: Dict[str, Any],
    built_models: Dict[str, Any],
    runtime_gpu_arch: str,
    disable_skip: bool = False,
) -> Tuple[Dict[str, Any], List[Tuple[str, Dict, str]]]:
    """
    Filter out models that should skip the current GPU architecture.

    Implements skip_gpu_arch from model definitions.

    Args:
        built_images: Dictionary of built images from manifest.
        built_models: Dictionary of model metadata from manifest.
        runtime_gpu_arch: Runtime GPU architecture (gfx90a, A100, etc.).
        disable_skip: If True, return all images and empty skipped list.

    Returns:
        (compatible_images, skipped_list) where skipped_list is
        [(model_name, image_info, gpu_arch), ...] for orchestrator to call
        _write_skipped_status(model_name, image_info, gpu_arch).
    """
    if disable_skip:
        return built_images, []

    compatible: Dict[str, Any] = {}
    skipped: List[Tuple[str, Dict, str]] = []

    for model_name, image_info in built_images.items():
        model_info = built_models.get(model_name, {})
        skip_gpu_arch_str = model_info.get("skip_gpu_arch", "")

        if not skip_gpu_arch_str:
            compatible[model_name] = image_info
            continue

        skip_list = [arch.strip() for arch in skip_gpu_arch_str.split(",")]
        sys_gpu_arch = runtime_gpu_arch
        if sys_gpu_arch and "NVIDIA" in sys_gpu_arch:
            # Normalize "NVIDIA A100-SXM4-40GB" -> "A100", "NVIDIA H100 PCIe" -> "H100"
            # (mirrors get_gpu_arch() in tests/fixtures/utils.py)
            after_prefix = sys_gpu_arch[sys_gpu_arch.index("NVIDIA") + len("NVIDIA ") :]
            sys_gpu_arch = after_prefix.split("-")[0].split()[0]

        if sys_gpu_arch in skip_list:
            skipped.append((model_name, image_info, runtime_gpu_arch))
        else:
            compatible[model_name] = image_info

    return compatible, skipped
