#!/usr/bin/env python3
"""
Per-cluster fact profiles for SLURM deployments.

The existing SLURM presets describe the *shape* of a job — how many nodes, how long, which
launcher. What they cannot describe is the cluster the job lands on, so cluster facts ended
up baked into the shape: `slurm/profiles/multi-node.json` sets `NCCL_IB_DISABLE=1` and
`NCCL_SOCKET_IFNAME=eth0`, which quietly puts every multi-node run on TCP over an interface
that may not exist, on a fabric that may well be RoCE.

A cluster profile carries the facts instead: whether the scheduler advertises GPU GRES,
which NICs carry the collectives, the transport variables that make RDMA work, and the node
facts (GPU vendor, GPUs per node, architecture) that a submit node without GPUs cannot
discover for itself. Profiles are selected by `slurm.cluster_profile`, merge after the shape
profile and before the user's own configuration, and a site can keep its own profile outside
the repository — a bundled profile names a hardware archetype, never someone's cluster.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from madengine.core.errors import ErrorContext, ValidationError

CLUSTER_DIR = Path(__file__).parent / "slurm" / "clusters"

#: Colon-separated directories holding site profiles, searched before the bundled ones.
PROFILE_PATH_ENV = "MADENGINE_CLUSTER_PROFILES"

#: Set when no profile is named in the configuration.
PROFILE_NAME_ENV = "MADENGINE_CLUSTER_PROFILE"


def available_profiles() -> List[str]:
    """
    Names of the profiles that can be selected, from the search path and the bundle.

    Returns:
        List[str]: profile names, sorted, without the .json suffix
    """
    names = set()
    for directory in _search_dirs():
        if directory.is_dir():
            names.update(path.stem for path in directory.glob("*.json"))
    return sorted(names)


def _search_dirs() -> List[Path]:
    """Directories to look in, site-first so a site can shadow a bundled archetype."""
    dirs = [Path(p) for p in os.environ.get(PROFILE_PATH_ENV, "").split(os.pathsep) if p]
    dirs.append(CLUSTER_DIR)
    return dirs


def resolve_profile_path(name: str) -> Path:
    """
    Find the file backing a profile reference.

    A reference that looks like a path (contains a separator or ends in `.json`) is taken as
    one, so a site can point at a profile it keeps next to its manifests. Anything else is a
    name looked up in the search path and then in the bundled archetypes.

    Args:
        name: profile name or path

    Returns:
        Path: the profile file

    Raises:
        ValidationError: no such profile
    """
    context = ErrorContext(
        operation="cluster profile lookup", component="deployment.presets"
    )

    if os.sep in name or name.endswith(".json"):
        path = Path(name).expanduser()
        if not path.is_file():
            raise ValidationError(
                f"Cluster profile not found: {path}",
                context=context,
                suggestions=[f"Bundled profiles: {', '.join(available_profiles())}"],
            )
        return path

    for directory in _search_dirs():
        candidate = directory / f"{name}.json"
        if candidate.is_file():
            return candidate

    raise ValidationError(
        f"Unknown cluster profile: {name}",
        context=context,
        suggestions=[
            f"Available profiles: {', '.join(available_profiles())}",
            f"Point {PROFILE_PATH_ENV} at a directory of site profiles, or give a path",
        ],
    )


def load_profile(name: str) -> Dict[str, Any]:
    """
    Load and validate one cluster profile.

    Args:
        name: profile name or path

    Returns:
        Dict[str, Any]: the profile

    Raises:
        ValidationError: the profile is missing, unparseable, or does not match the schema
    """
    import jsonschema

    from madengine.schemas import load_schema

    path = resolve_profile_path(name)
    context = ErrorContext(
        operation="cluster profile loading",
        component="deployment.presets",
        file_path=str(path),
    )

    try:
        with open(path) as f:
            profile = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"Cluster profile {path} is not valid JSON: {exc}", context=context, cause=exc
        ) from exc

    validator = jsonschema.Draft202012Validator(load_schema("cluster_profile.schema.json"))
    first = next(iter(sorted(validator.iter_errors(profile), key=lambda e: list(e.path))), None)
    if first is not None:
        pointer = "/" + "/".join(str(part) for part in first.absolute_path)
        raise ValidationError(
            f"Invalid cluster profile {path}: {pointer}: {first.message}", context=context
        )

    return profile


def _merge(base: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge one profile over a configuration.

    Nested dicts merge; a null value removes the key, which is how a profile says a variable
    inherited from a shape preset does not apply here — an interface name that does not exist
    on this cluster is worse than no interface name at all.

    Args:
        base: configuration so far
        profile: profile to apply

    Returns:
        Dict[str, Any]: the merged configuration
    """
    result = dict(base)
    for key, value in profile.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def selected_profiles(config: Dict[str, Any]) -> List[str]:
    """
    Profile references named by a configuration, in merge order.

    Args:
        config: SLURM configuration, before user overrides are applied

    Returns:
        List[str]: profile names or paths; empty when none is selected
    """
    selection: Union[str, Sequence[str], None] = (config.get("slurm") or {}).get(
        "cluster_profile"
    )
    if not selection:
        selection = os.environ.get(PROFILE_NAME_ENV) or None
    if not selection:
        return []
    if isinstance(selection, str):
        return [selection]
    return list(selection)


def apply_cluster_profiles(
    config: Dict[str, Any], selection: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """
    Merge the selected cluster profiles into a configuration.

    Several profiles may be named, and they merge left to right: cluster facts are
    orthogonal, so "this fabric" and "this scheduler advertises no GPU GRES" are separate
    statements rather than a combinatorial set of files.

    Args:
        config: configuration to merge into
        selection: profile references; taken from the configuration when omitted

    Returns:
        Dict[str, Any]: the configuration with profiles applied

    Raises:
        ValidationError: a named profile is missing or invalid
    """
    references = list(selection) if selection is not None else selected_profiles(config)
    for reference in references:
        profile = load_profile(reference)
        # Documentation keys describe the file, not the cluster.
        facts = {k: v for k, v in profile.items() if not k.startswith("_")}
        config = _merge(config, facts)
    return config
