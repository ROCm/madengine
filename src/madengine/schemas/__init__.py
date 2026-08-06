#!/usr/bin/env python3
"""
Declared shapes for the files madengine reads, and the validator that enforces them.

The build manifest used to be checked for three top-level keys and nothing else, so a
typo in a transport variable or a node count that disagreed with itself surfaced as a
failed multi-node job minutes later instead of as a startup error. `build_manifest.schema.json`
declares the shape; `validate_build_manifest` reports the first violation with its JSON
pointer, plus the cross-field checks a schema cannot express.

The schema is also where the deployment target is defined to live: under
`deployment_config`. A manifest that carries a top-level `slurm`/`k8s`/`distributed` block
is migrated into `deployment_config` with a warning, so the two used to disagree silently
and now cannot.

The result side is declared the same way. `perf_csv.schema.json` gives the columns of
`perf.csv` in order, replacing the header string that each of the three writers spelled
out for itself under a comment asking the reader to keep them in sync. Columns that
restate a manifest field point at it, so the row and the manifest are two views of one
shape rather than two lists that happen to agree today.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import functools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from madengine.core.errors import ErrorContext, ValidationError

SCHEMA_DIR = Path(__file__).parent

#: Blocks that describe *where* a run is deployed. They belong under `deployment_config`.
DEPLOYMENT_BLOCKS = ("slurm", "k8s", "kubernetes", "distributed")


def load_schema(name: str = "build_manifest.schema.json") -> Dict[str, Any]:
    """Load a bundled JSON Schema by file name."""
    with open(SCHEMA_DIR / name) as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def perf_csv_columns() -> Tuple[str, ...]:
    """
    Column names of `perf.csv`, in order.

    Order is the declaration order in `perf_csv.schema.json`: a row is written positionally
    into an existing file, so the schema owns the order as well as the names.

    Returns:
        Tuple[str, ...]: the column names
    """
    return tuple(load_schema("perf_csv.schema.json")["properties"])


@functools.lru_cache(maxsize=1)
def perf_csv_header() -> str:
    """
    The `perf.csv` header line, without a trailing newline.

    Returns:
        str: comma-separated column names
    """
    return ",".join(perf_csv_columns())


def unknown_perf_columns(row: Dict[str, Any]) -> List[str]:
    """
    Report keys of a result row that no column accepts.

    A row is written with `extrasaction="ignore"`, so a key the schema does not declare is
    dropped on the floor rather than reported. Callers that care can ask.

    Args:
        row: a result row keyed by column name

    Returns:
        List[str]: keys that are not declared columns, sorted
    """
    return sorted(set(row) - set(perf_csv_columns()))


def _pointer(path) -> str:
    """Render a jsonschema error path as a JSON pointer (RFC 6901)."""
    return "/" + "/".join(str(part) for part in path) if path else "/"


def migrate_top_level_deployment_blocks(manifest: Dict[str, Any]) -> List[str]:
    """
    Move top-level deployment blocks under `deployment_config`, in place.

    Historically a manifest could carry a top-level `slurm` block that selected the SLURM
    target while the values that took effect were the ones under `deployment_config`. Fold
    the former into the latter so there is one place to read.

    Args:
        manifest: parsed manifest, modified in place

    Returns:
        List[str]: human-readable warnings, one per migrated or ignored block
    """
    warnings: List[str] = []
    for block in DEPLOYMENT_BLOCKS:
        if block not in manifest:
            continue
        value = manifest.pop(block)
        deployment_config = manifest.setdefault("deployment_config", {})
        if block in deployment_config:
            warnings.append(
                f"manifest carries both '{block}' at the top level and under "
                f"'deployment_config'; the top-level block was ignored"
            )
        else:
            deployment_config[block] = value
            warnings.append(
                f"manifest carries '{block}' at the top level; it belongs under "
                f"'deployment_config' and was moved there"
            )
    return warnings


def _semantic_warnings(manifest: Dict[str, Any]) -> List[str]:
    """Cross-field checks that are advisory rather than fatal."""
    warnings: List[str] = []

    images = manifest.get("built_images") or {}
    models = manifest.get("built_models") or {}
    unused = sorted(set(images) - set(models))
    if unused:
        warnings.append(
            f"built_images entries with no matching built_models entry: {', '.join(unused)}"
        )

    deployment_config = manifest.get("deployment_config") or {}
    target = deployment_config.get("target")
    if target == "slurm" and "slurm" not in deployment_config:
        warnings.append(
            "deployment_config.target is 'slurm' but there is no deployment_config.slurm "
            "block; cluster defaults will be used"
        )
    return warnings


def _semantic_errors(manifest: Dict[str, Any]) -> List[str]:
    """Cross-field checks a JSON Schema cannot express, and that break a run."""
    errors: List[str] = []

    images = manifest.get("built_images") or {}
    models = manifest.get("built_models") or {}
    orphans = sorted(set(models) - set(images))
    if orphans:
        errors.append(
            f"built_models entries with no matching built_images entry: "
            f"{', '.join(orphans)}. The two dicts are joined by key."
        )

    deployment_config = manifest.get("deployment_config") or {}
    nodes = (deployment_config.get("slurm") or {}).get("nodes")
    nnodes = (deployment_config.get("distributed") or {}).get("nnodes")
    if nodes is not None and nnodes is not None and nodes != nnodes:
        errors.append(
            f"deployment_config.slurm.nodes ({nodes}) != "
            f"deployment_config.distributed.nnodes ({nnodes}); sbatch and the launcher "
            f"would disagree on the world size"
        )
    return errors


def validate_build_manifest(
    manifest: Dict[str, Any],
    source: Optional[str] = None,
    migrate: bool = True,
) -> List[str]:
    """
    Validate a build manifest against the bundled schema, fail-fast on the first error.

    Args:
        manifest: parsed manifest; modified in place when `migrate` is set
        source: manifest path, for error messages
        migrate: fold top-level deployment blocks into `deployment_config`

    Returns:
        List[str]: non-fatal warnings

    Raises:
        ValidationError: on the first schema violation or failed cross-field check
    """
    import jsonschema

    warnings = migrate_top_level_deployment_blocks(manifest) if migrate else []

    where = f" in {source}" if source else ""
    validator = jsonschema.Draft202012Validator(load_schema())
    first = next(iter(sorted(validator.iter_errors(manifest), key=lambda e: list(e.path))), None)
    if first is not None:
        raise ValidationError(
            f"Invalid manifest{where}: {_pointer(first.absolute_path)}: {first.message}",
            context=ErrorContext(
                operation="manifest validation",
                component="schemas",
                file_path=source,
            ),
            suggestions=[
                "Check the field against src/madengine/schemas/build_manifest.schema.json",
            ],
        )

    errors = _semantic_errors(manifest)
    if errors:
        raise ValidationError(
            f"Invalid manifest{where}: {errors[0]}",
            context=ErrorContext(
                operation="manifest validation",
                component="schemas",
                file_path=source,
            ),
        )

    return warnings + _semantic_warnings(manifest)
