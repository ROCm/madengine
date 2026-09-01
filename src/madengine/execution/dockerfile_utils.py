#!/usr/bin/env python3
"""
Pure helpers for parsing Dockerfile GPU variables and validating target architecture.

Used by DockerBuilder for build-phase validation. No I/O or context dependency.
"""

import re
import typing

# GPU architecture variables used in MAD/DLM Dockerfiles
GPU_ARCH_VARIABLES = [
    "MAD_SYSTEM_GPU_ARCHITECTURE",
    "PYTORCH_ROCM_ARCH",
    "GPU_TARGETS",
    "GFX_COMPILATION_ARCH",
    "GPU_ARCHS",
]


def parse_dockerfile_gpu_variables(
    dockerfile_content: str,
) -> typing.Dict[str, typing.List[str]]:
    """Parse GPU architecture variables from Dockerfile content."""
    gpu_variables: typing.Dict[str, typing.List[str]] = {}

    for var_name in GPU_ARCH_VARIABLES:
        arg_pattern = rf"ARG\s+{var_name}=([^\s\n]+)"
        arg_matches = re.findall(arg_pattern, dockerfile_content, re.IGNORECASE)
        env_pattern = rf"ENV\s+{var_name}[=\s]+([^\s\n]+)"
        env_matches = re.findall(env_pattern, dockerfile_content, re.IGNORECASE)

        all_matches = arg_matches + env_matches
        if all_matches:
            raw_value = all_matches[-1].strip("\"'")
            parsed_values = parse_gpu_variable_value(var_name, raw_value)
            if parsed_values:
                gpu_variables[var_name] = parsed_values

    return gpu_variables


def parse_gpu_variable_value(var_name: str, raw_value: str) -> typing.List[str]:
    """Parse GPU variable value based on variable type and format."""
    architectures: typing.List[str] = []

    if var_name in ["GPU_TARGETS", "GPU_ARCHS", "PYTORCH_ROCM_ARCH"]:
        if ";" in raw_value:
            architectures = [a.strip() for a in raw_value.split(";") if a.strip()]
        elif "," in raw_value:
            architectures = [a.strip() for a in raw_value.split(",") if a.strip()]
        else:
            architectures = [raw_value.strip()]
    else:
        architectures = [raw_value.strip()]

    normalized_archs = []
    for arch in architectures:
        normalized = normalize_architecture_name(arch)
        if normalized:
            normalized_archs.append(normalized)
    return normalized_archs


def normalize_architecture_name(arch: str) -> typing.Optional[str]:
    """Normalize architecture name to standard format."""
    arch = arch.lower().strip()
    if arch.startswith("gfx"):
        return arch
    if arch in ["mi100", "mi-100"]:
        return "gfx908"
    if arch in ["mi200", "mi-200", "mi210", "mi250"]:
        return "gfx90a"
    if arch in ["mi300", "mi-300", "mi300a"]:
        return "gfx940"
    if arch in ["mi300x", "mi-300x"]:
        return "gfx942"
    if arch.startswith("mi"):
        return arch
    return arch if arch else None


def is_target_arch_compatible_with_variable(
    var_name: str,
    var_values: typing.List[str],
    target_arch: str,
) -> bool:
    """
    Check if target architecture is compatible with a GPU variable's values.
    """
    if var_name == "MAD_SYSTEM_GPU_ARCHITECTURE":
        return True
    if var_name in ["PYTORCH_ROCM_ARCH", "GPU_TARGETS", "GPU_ARCHS"]:
        return target_arch in var_values
    if var_name == "GFX_COMPILATION_ARCH":
        return len(var_values) == 1 and (
            var_values[0] == target_arch
            or is_compilation_arch_compatible(var_values[0], target_arch)
        )
    return True


def dockerfile_requires_explicit_mad_arch_build_arg(dockerfile_content: str) -> bool:
    """
    True when the Dockerfile declares ARG MAD_SYSTEM_GPU_ARCHITECTURE without a
    non-empty default (user must pass --build-arg or rely on a non-empty ARG/ENV default).

    If the variable is not declared as ARG, returns False. If any ARG line gives a
    non-empty default, returns False (Dockerfile supplies a default).
    """
    found_arg = False
    has_nonempty_default = False
    has_bare_or_empty_default = False
    for raw_line in dockerfile_content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        if not upper.startswith("ARG") or "MAD_SYSTEM_GPU_ARCHITECTURE" not in upper:
            continue
        found_arg = True
        m = re.match(
            r"ARG\s+MAD_SYSTEM_GPU_ARCHITECTURE\s*=\s*(\S+)",
            line,
            re.IGNORECASE,
        )
        if m and m.group(1).strip("\"'"):
            has_nonempty_default = True
            continue
        m_bare = re.match(
            r"ARG\s+MAD_SYSTEM_GPU_ARCHITECTURE\s*$",
            line,
            re.IGNORECASE,
        )
        if m_bare:
            has_bare_or_empty_default = True
            continue
        m_empty = re.match(
            r"ARG\s+MAD_SYSTEM_GPU_ARCHITECTURE\s*=\s*$",
            line,
            re.IGNORECASE,
        )
        if m_empty:
            has_bare_or_empty_default = True
    if not found_arg:
        return False
    if has_nonempty_default:
        return False
    return has_bare_or_empty_default


def is_compilation_arch_compatible(compile_arch: str, target_arch: str) -> bool:
    """Check if compilation architecture is compatible with target architecture."""
    compatibility_matrix = {
        "gfx908": ["gfx908"],
        "gfx90a": ["gfx90a"],
        "gfx940": ["gfx940"],
        "gfx941": ["gfx941"],
        "gfx942": ["gfx942"],
    }
    compatible_archs = compatibility_matrix.get(compile_arch, [compile_arch])
    return target_arch in compatible_archs
