#!/usr/bin/env python3
"""
Load a shell env file (the `mad.env` convention) into the run's environment.

Multi-node runs depend on a set of variables that describe *where things live* on the
cluster: `MODEL_DIR`, the cache roots, and `MAD_DOCKER_BUILDS`. Until now the only way to
supply them was to `source mad.env` in the same shell before every `madengine run`, and
forgetting produced failures far from the cause — an empty `MODEL_DIR` makes the run
script path resolve to nothing, and a `MAD_DOCKER_BUILDS` that is not on shared storage
makes every worker rebuild the image or fail to find it.

A manifest can now name the file (`deployment_config.env_file`) and madengine loads it
itself. The file is executed by `bash`, exactly as sourcing it would, so the usual shell
constructs (`${VAR:-default}`, `$(cat ~/.token)`, conditionals) behave the same — which
also means an env file is trusted input, on par with the manifest that names it.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Optional

from madengine.core.errors import ErrorContext, ValidationError

#: A shell that hangs (waiting on a prompt, say) must not hang the run.
SOURCE_TIMEOUT_SECONDS = 120

#: Separates the before/after environment dumps in the helper shell's output. Both dumps
#: come from the same shell, so anything bash sets on its own (COLUMNS, SHLVL, ...)
#: appears in both and is not mistaken for something the env file did.
_BOUNDARY = "__madengine_env_file_boundary__"

#: Bash's own bookkeeping, which differs between the two dumps for reasons unrelated to
#: the file's contents.
_SHELL_BOOKKEEPING = frozenset({"_", "SHLVL", "PWD", "OLDPWD"})

#: What each file already applied in this process, keyed by resolved path. A run reaches
#: :func:`apply_env_file` twice for the same manifest -- once in the orchestrator, once
#: when the deployment layer loads the manifest -- and sourcing a file again is not a
#: no-op: `PATH="$PATH:/opt/x"` appends a second time and `$(...)` runs a second time.
_APPLIED: Dict[str, Dict[str, str]] = {}


def resolve_env_file(env_file: str, base_dir: Optional[str] = None) -> Path:
    """The path an *env_file* names, relative paths resolved against *base_dir*."""
    path = Path(env_file)
    if not path.is_absolute() and base_dir:
        path = Path(base_dir) / path
    return path


def load_env_file(env_file: str, base_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Source an env file with bash and return the variables it sets or changes.

    Args:
        env_file: path to the file; relative paths resolve against `base_dir`
        base_dir: directory to resolve a relative `env_file` against (typically the
            manifest's directory), defaults to the current working directory

    Returns:
        Dict[str, str]: variables the file added or changed, relative to the environment
            madengine is running with

    Raises:
        ValidationError: the file is missing, or bash failed to source it
    """
    path = resolve_env_file(env_file, base_dir)

    context = ErrorContext(
        operation="env_file loading", component="core.env_file", file_path=str(path)
    )
    if not path.is_file():
        raise ValidationError(
            f"env_file not found: {path}",
            context=context,
            suggestions=[
                "deployment_config.env_file is resolved relative to the manifest",
            ],
        )

    # `set -a` is what makes plain `KEY=value` lines exported, matching what an operator
    # gets from `source mad.env` in a shell configured the usual way. Sourcing is checked
    # explicitly: a syntax error makes `.` fail but would otherwise be masked by the
    # `env` that follows it.
    script = (
        f"env -0; printf '%s\\0' {shlex.quote(_BOUNDARY)}; "
        f"set -a; . {shlex.quote(str(path))} || exit 42; env -0"
    )
    try:
        result = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            timeout=SOURCE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValidationError(
            f"Timed out after {SOURCE_TIMEOUT_SECONDS}s sourcing env_file: {path}",
            context=context,
            cause=exc,
        ) from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or f"bash exited with {result.returncode}"
        raise ValidationError(
            f"Failed to source env_file {path}: {detail}",
            context=context,
        )

    before_dump, _, after_dump = result.stdout.partition(f"{_BOUNDARY}\0")
    before = _parse_env_dump(before_dump)

    loaded: Dict[str, str] = {}
    for key, value in _parse_env_dump(after_dump).items():
        # A variable that already held this value is not something the file changed.
        if key in _SHELL_BOOKKEEPING or before.get(key) == value:
            continue
        loaded[key] = value
    return loaded


def _parse_env_dump(dump: str) -> Dict[str, str]:
    """
    Parse NUL-delimited `env -0` output into a mapping.

    Args:
        dump: raw `env -0` output

    Returns:
        Dict[str, str]: the variables in the dump
    """
    parsed: Dict[str, str] = {}
    for entry in dump.split("\0"):
        if not entry or "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        parsed[key] = value
    return parsed


def apply_env_file(env_file: str, base_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Load an env file and apply it to `os.environ`, as sourcing it would, once per process.

    The file wins over the inherited environment, so behaviour matches what the operator
    gets by sourcing it before the run. A second call for the same file returns what the
    first one applied without running it again: the submit side of a cluster run reaches
    this twice for one manifest, and a file is shell, not a list of assignments.

    Args:
        env_file: path to the file; relative paths resolve against `base_dir`
        base_dir: directory to resolve a relative `env_file` against

    Returns:
        Dict[str, str]: the variables that were applied
    """
    key = str(resolve_env_file(env_file, base_dir).resolve())
    if key in _APPLIED:
        return dict(_APPLIED[key])

    loaded = load_env_file(env_file, base_dir)
    os.environ.update(loaded)
    _APPLIED[key] = dict(loaded)
    return loaded
