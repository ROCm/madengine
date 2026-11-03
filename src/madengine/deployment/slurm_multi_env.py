"""The submit-time environment of a slurm_multi card, with where each value came from.

A slurm_multi card runs its own launcher script (MAD's run_*.slurm), which owns its
environment: it fills site defaults, applies the model's recipe and records every
value with the layer that set it (MAD scripts/common/config/layers.sh). madengine's
part is the top layer, "submit time", which it builds from, weakest first:

    mad-config    the card's layered config (mad-config.yaml model: / benchmark:)
    docker-env-vars   the manifest's / additional context's docker_env_vars
    card          the card's env_vars
    additional-context   the run's env_vars, without keys only a preset set
    distributed   xP / yD from the card's sglang_disagg block, when not set above

and then the values madengine itself decides (the image the nodes pull, MAD_TOOLS,
the Kubernetes LOG_PATH), labelled "madengine".

Every path that runs a slurm_multi card -- SLURM, Spur (a SlurmDeployment), and
Kubernetes per pod -- builds this one environment here, so a card gets the same
submit layer wherever it runs. The source of each name travels with it as
MAD_ENV_SOURCES ("NAME=label,..."), which the launcher folds into its record.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config_loader import PRESET_ENV_KEYS
from .layered_config import resolve_for_model

SOURCES_VAR = "MAD_ENV_SOURCES"


class SubmitEnv:
    """Ordered env plus the source of each name; later set() wins, like dict.update."""

    def __init__(self):
        self.env: Dict[str, str] = {}
        self.sources: Dict[str, str] = {}

    def update(self, values: Optional[Dict[str, Any]], source: str) -> None:
        for k, v in (values or {}).items():
            self.set(k, v, source)

    def set(self, key: str, value: Any, source: str) -> None:
        self.env[key] = str(value)
        self.sources[key] = source

    def setdefault(self, key: str, value: Any, source: str) -> str:
        if key not in self.env:
            self.set(key, value, source)
        return self.env[key]

    def encoded_sources(self) -> str:
        return ",".join(f"{k}={v}" for k, v in sorted(self.sources.items()) if k != SOURCES_VAR)

    def final(self) -> Dict[str, str]:
        """The env to export, with MAD_ENV_SOURCES."""
        out = dict(self.env)
        out[SOURCES_VAR] = self.encoded_sources()
        return out


def user_env_vars(additional_context: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """additional_context.env_vars without keys only a preset set, and the keys dropped.

    Presets carry settings for madengine's own launchers; a slurm_multi script owns its
    environment and the same card submitted without madengine never receives them.
    """
    env = dict(additional_context.get("env_vars") or {})
    preset_only = additional_context.get(PRESET_ENV_KEYS) or []
    dropped = sorted(k for k in preset_only if k in env)
    for key in dropped:
        del env[key]
    return env, dropped


def submit_env(
    model_info: Dict[str, Any],
    additional_context: Dict[str, Any],
    scripts_dir: Optional[Path],
    distributed_config: Optional[Dict[str, Any]] = None,
) -> Tuple[SubmitEnv, List[str], List[str]]:
    """The card's submit layer: (env with sources, layered-config warnings, preset keys dropped).

    Raises LayeredConfigError when the card's layered config is refused.
    """
    user_env, dropped = user_env_vars(additional_context)
    out = SubmitEnv()
    layered_env, warnings = resolve_for_model(model_info, scripts_dir, runtime_env=user_env)
    out.update(layered_env, "mad-config")
    out.update(additional_context.get("docker_env_vars"), "docker-env-vars")
    out.update(model_info.get("env_vars"), "card")
    out.update(user_env, "additional-context")
    sglang = ((model_info.get("distributed") or {}).get("sglang_disagg")
              or (distributed_config or {}).get("sglang_disagg") or {})
    if sglang:
        out.setdefault("xP", sglang.get("prefill_nodes", 1), "distributed")
        out.setdefault("yD", sglang.get("decode_nodes", 1), "distributed")
    return out, warnings, dropped
