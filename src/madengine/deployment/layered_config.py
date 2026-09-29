"""Layered configuration for distributed inference workloads.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

Distributed inference workloads are configured three different ways today, by three
different teams, and each format is owned by the team that authors it:

  1. scripts/<dir>/models.yaml          serve recipe per model x role x mode
  2. scripts/<dir>/configs/*.yaml       benchmark selection and its knobs
  3. cluster.sh                         site facts, as ${VAR:-default}

Those are not three ways of doing one thing. They are three layers, separated by
WHERE IN THE PIPELINE the value is created -- which is also who reviews a change
to it. They overlap only where one team had to reach into another's territory.

This module implements a fourth way: a single file carrying the layers that
travel WITH A MODEL -- how it is served, and what is measured. It exists so a
team that wants those in one reviewable place can have it, without any of the
other three changing.

It deliberately stops at the model boundary. Site facts and run shape are
configured per RUN, not per model, and madengine composes those from Hydra config
groups (scheduler, launcher, +profile, +env). A model card is the wrong place to
say what a cluster is.

It resolves to a plain environment dict -- exactly what ways 1-3 already consume --
so adopting it requires no change to any workload script. It is a front end, not a
migration.

madengine does NOT parse ways 1-3. Those files are read by the scripts that own
them, inside the container. See docs/distributed-config.md.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from madengine.deployment.config_loader import ConfigLoader

# Name looked for next to a card's models.json. Overridable per card with
# env_vars.MAD_CONFIG, which mirrors how the accuracy work locates a sibling
# acc.yaml next to --config.
DEFAULT_FILENAME = "mad-config.yaml"

SUPPORTED_VERSIONS = (1,)


class LayeredConfigError(ValueError):
    """A way-4 file exists but cannot be used.

    Raised rather than warned: a file that is present and malformed is a mistake
    someone wants to hear about, and silently ignoring it would resolve to a
    different configuration than the author intended.
    """


def _as_arg_map(value: Any) -> Dict[str, Any]:
    """Normalise serve flags to a map so a later layer can override ONE flag.

    Way 1 stores flags as an opaque string ("--disable-radix-cache --cuda-graph-bs
    8 16"), which cannot be merged -- overriding one flag means string surgery, and
    scripts/vllm_dissag/models.yaml says so itself: the moriio path "strips any
    duplicate from the yaml tp: string". A map makes layering work.

    A bare flag (no value) maps to True, matching how configs/*.yaml already writes
    "--trust-remote-code: true".
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        raise LayeredConfigError(
            f"serve flags must be a map or a string, got {type(value).__name__}"
        )

    out: Dict[str, Any] = {}
    tokens = shlex.split(value)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("-"):
            raise LayeredConfigError(
                f"expected a flag starting with '-', got {token!r} in {value!r}"
            )
        # Collect every following non-flag token: --cuda-graph-bs takes a list.
        values: List[str] = []
        i += 1
        while i < len(tokens) and not tokens[i].startswith("-"):
            values.append(tokens[i])
            i += 1
        if not values:
            out[token] = True
        elif len(values) == 1:
            out[token] = values[0]
        else:
            out[token] = " ".join(values)
    return out


def _env_of(section: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The env block of a section, or empty."""
    if not section:
        return {}
    env = section.get("env") or {}
    if not isinstance(env, dict):
        raise LayeredConfigError("'env' must be a map of NAME: value")
    return env


def find_config(
    model_info: Dict[str, Any], scripts_dir: Optional[Path]
) -> Optional[Path]:
    """Locate a way-4 file for this model, or None.

    env_vars.MAD_CONFIG wins; otherwise a sibling DEFAULT_FILENAME next to the
    card's scripts directory.

    Deliberately NOT read from the card's `args`: that field already means two
    different things to the two consumers (madengine passes it to the script,
    a plain sbatch submission passes it to sbatch), so putting a third
    meaning in it would make the ambiguity worse.
    """
    explicit = (model_info.get("env_vars") or {}).get("MAD_CONFIG")
    if explicit:
        path = Path(explicit)
        if not path.is_absolute() and scripts_dir is not None:
            path = scripts_dir / path
        return path if path.exists() else None

    if scripts_dir is None:
        return None
    candidate = scripts_dir / DEFAULT_FILENAME
    return candidate if candidate.exists() else None


def load_config(path: Path) -> Dict[str, Any]:
    """Parse and version-check a way-4 file."""
    try:
        with open(path, "r") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise LayeredConfigError(f"{path}: not valid YAML: {exc}") from exc
    except OSError as exc:
        raise LayeredConfigError(f"{path}: cannot be read: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise LayeredConfigError(
            f"{path}: top level must be a map, got {type(data).__name__}"
        )

    version = data.get("version")
    if version not in SUPPORTED_VERSIONS:
        raise LayeredConfigError(
            f"{path}: version {version!r} is not supported "
            f"(this madengine understands {list(SUPPORTED_VERSIONS)})"
        )
    return data


def resolve_serve_args(
    config: Dict[str, Any], mode: Optional[str] = None, role: Optional[str] = None
) -> Dict[str, Any]:
    """Resolve the model's serve flags for one role and mode.

    base -> modes[mode] -> roles[role] -> roles[role][mode], each overriding the
    last. This is the axis way 1 owns and way 2 has no concept of, so it has to
    survive into the merged format or disaggregation cannot be expressed.
    """
    model = config.get("model") or {}
    serve = model.get("serve") or {}
    args = _as_arg_map(serve.get("base"))

    if mode:
        modes = serve.get("modes") or {}
        args = ConfigLoader.deep_merge(args, _as_arg_map(modes.get(mode)))

    if role:
        roles = serve.get("roles") or {}
        role_block = roles.get(role) or {}
        if isinstance(role_block, (str, list)):
            args = ConfigLoader.deep_merge(args, _as_arg_map(role_block))
        else:
            # A role may carry flags directly and/or per-mode flags under it.
            direct = {k: v for k, v in role_block.items() if k.startswith("-")}
            args = ConfigLoader.deep_merge(args, _as_arg_map(direct))
            if mode and mode in role_block:
                args = ConfigLoader.deep_merge(args, _as_arg_map(role_block[mode]))
    return args


def _benchmark_entries(
    config: Dict[str, Any], kind: Optional[str]
) -> List[Dict[str, Any]]:
    """Benchmark entries that apply, task-level defaults first.

    An entry with no 'kind' is a default shared by every benchmark, matching the
    task-level/model-level inheritance the accuracy schema uses.
    """
    entries = config.get("benchmark") or []
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        raise LayeredConfigError("'benchmark' must be a list of entries")

    defaults = [e for e in entries if isinstance(e, dict) and not e.get("kind")]
    if kind is None:
        return defaults
    specific = [e for e in entries if isinstance(e, dict) and e.get("kind") == kind]
    return defaults + specific


def resolve_env(
    config: Dict[str, Any],
    model_env: Optional[Dict[str, Any]] = None,
    runtime_env: Optional[Dict[str, Any]] = None,
    benchmark: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve every layer to the environment the workload actually sees.

    Lowest precedence first:

        model         whoever tuned this model on this hardware
        benchmark     whoever defines the measurement
        model_env     the card's own env_vars
        runtime_env   additional_context env_vars / -e at submit time  (highest)

    The last two are placed above the file so an operator pinning something at
    submit time still wins, which is the property every one of the three existing
    formats already relies on.

    runtime_env is also where Hydra lands: --config translates to
    additional_context, so `--config +env=nccl_debug` outranks anything a card
    declares. That is the right way round -- a run-level override should beat a
    model default -- and it composes without either side knowing about the other.

    There is no 'site' layer. Site facts are a property of the run, and belong to
    a Hydra group or to cluster.sh; see docs/distributed-config.md.
    """
    merged: Dict[str, Any] = {}
    if config.get("site"):
        # Not silently ignored: a dropped setting that looks applied is the exact
        # failure this module exists to prevent.
        raise LayeredConfigError(
            "'site:' is no longer read here. Site facts belong to the run, not to a "
            "model card, and madengine now composes them from Hydra config groups "
            "(--config +profile=..., --config +env=...) or from cluster.sh.\n"
            "  Move each key to whichever of those owns it, and keep 'model:' and "
            "'benchmark:' here -- those are per-model and per-measurement, which no "
            "Hydra group can express."
        )
    merged = ConfigLoader.deep_merge(merged, _env_of(config.get("model")))
    for entry in _benchmark_entries(config, benchmark):
        merged = ConfigLoader.deep_merge(merged, _env_of(entry))
    merged = ConfigLoader.deep_merge(merged, dict(model_env or {}))
    merged = ConfigLoader.deep_merge(merged, dict(runtime_env or {}))

    # SLURM and docker -e both want strings; a YAML 8192 would otherwise arrive
    # as an int and fail on the way into the environment.
    return {str(k): str(v) for k, v in merged.items()}


def validate_card(model_info: Dict[str, Any]) -> List[str]:
    """Warnings about a model card's configuration, as plain strings.

    Only checks things that are genuinely ambiguous rather than merely unusual --
    a warning nobody can act on is noise.
    """
    warnings: List[str] = []
    args = (model_info.get("args") or "").strip()
    name = model_info.get("name", "<unnamed>")

    # `args` means different things to the two consumers: madengine hands it to
    # the script (bash <script> <args>), while submitting the card
    # script with plain sbatch hands it to sbatch. A distributed card carrying sbatch flags therefore
    # behaves differently depending on who runs it.
    if "distributed" in model_info and args:
        sbatch_like = {"-N", "-n", "--nodes", "--ntasks", "--ntasks-per-node", "--gres"}
        tokens = set(shlex.split(args))
        hit = sorted(tokens & sbatch_like)
        if hit:
            warnings.append(
                f"{name}: args={args!r} contains sbatch flag(s) {hit}. madengine passes "
                f"args to the model script, not to sbatch, so these reach the script as "
                f"positional arguments. Node count for a distributed card comes from "
                f"distributed.nnodes / slurm.nodes."
            )
    return warnings


def resolve_for_model(
    model_info: Dict[str, Any],
    scripts_dir: Optional[Path],
    runtime_env: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """Way-4 env for a model card, plus any warnings. Empty env when no file.

    Returns ({}, warnings) rather than raising when there is no way-4 file: ways
    1-3 are the common case and must stay untouched.
    """
    warnings = validate_card(model_info)
    path = find_config(model_info, scripts_dir)
    if path is None:
        return {}, warnings

    config = load_config(path)
    env = resolve_env(
        config,
        model_env=model_info.get("env_vars") or {},
        runtime_env=runtime_env or {},
        benchmark=(model_info.get("env_vars") or {}).get("BENCHMARK_SCRIPT"),
    )
    return env, warnings
