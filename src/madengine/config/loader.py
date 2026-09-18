"""Hydra-based config loader using the Compose API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from madengine.core.errors import ConfigurationError


class HydraConfigLoader:
    """Loads madengine config using Hydra's Compose API."""

    @staticmethod
    def load(config_args: List[str]) -> DictConfig:
        """Load and compose config from Hydra overrides and/or user YAML.

        Merge order (later wins):
            1. Package defaults (``configs/config.yaml`` + default groups)
            2. User YAML file, if provided
            3. Hydra group and ``key=value`` overrides from remaining ``--config`` args

        Args:
            config_args: Mix of Hydra overrides and optional user YAML path.

        Returns:
            Composed DictConfig with all merges applied.
        """
        user_file, overrides = HydraConfigLoader._parse_args(config_args)

        # Hydra needs a real filesystem directory, and configs/ always ships
        # inside the package, so resolve it relative to this module.
        config_dir = str(Path(__file__).resolve().parent.parent / "configs")

        GlobalHydra.instance().clear()

        CONFIG_GROUPS = {"platform", "scheduler", "hardware", "launcher"}

        group_overrides: List[str] = []
        value_overrides: List[str] = []
        for override in overrides:
            if override.startswith(("+", "~")):
                group_overrides.append(override)
                continue
            if "=" not in override:
                group_overrides.append(override)
                continue
            key = override.split("=", 1)[0]
            if "." not in key and key in CONFIG_GROUPS:
                group_overrides.append(override)
            else:
                value_overrides.append(override)

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            defaults_cfg = compose(config_name="config")
            OmegaConf.set_struct(defaults_cfg, False)
            cfg: DictConfig = defaults_cfg
            if user_file:
                user_cfg = OmegaConf.load(user_file)
                cfg = cast(DictConfig, OmegaConf.merge(cfg, user_cfg))
            if group_overrides:
                overlay_cfg = compose(config_name="config", overrides=group_overrides)
                HydraConfigLoader._overlay_changed(cfg, defaults_cfg, overlay_cfg)
            if value_overrides:
                OmegaConf.set_struct(cfg, False)
                cfg = cast(
                    DictConfig,
                    OmegaConf.merge(cfg, OmegaConf.from_dotlist(value_overrides)),
                )

        return cfg

    @staticmethod
    def _overlay_changed(
        target: DictConfig, baseline: DictConfig, updated: DictConfig
    ) -> None:
        """Copy values from ``updated`` that differ from ``baseline`` onto ``target``.

        This lets Hydra CLI/group overrides win over a user YAML file without
        resetting unrelated YAML keys back to package defaults.
        """
        OmegaConf.set_struct(target, False)
        HydraConfigLoader._merge_diff(
            target,
            OmegaConf.to_container(baseline, resolve=False) or {},
            OmegaConf.to_container(updated, resolve=False) or {},
            prefix=None,
        )

    @staticmethod
    def _merge_diff(
        target: DictConfig, old: Any, new: Any, prefix: Optional[str]
    ) -> None:
        if not isinstance(new, dict) or not isinstance(old, dict):
            if new != old:
                if prefix is None:
                    raise ConfigurationError("Cannot overlay a non-dict root config")
                OmegaConf.update(target, prefix, new, merge=False)
            return
        for key, val in new.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in old:
                OmegaConf.update(target, path, val, merge=True)
            elif val != old[key]:
                HydraConfigLoader._merge_diff(target, old[key], val, path)

    @staticmethod
    def _parse_args(config_args: List[str]) -> Tuple[Optional[str], List[str]]:
        """Separate user YAML file path from Hydra overrides."""
        user_file = None
        overrides: List[str] = []
        for arg in config_args:
            if (
                arg.endswith((".yaml", ".yml"))
                and "=" not in arg
                and not arg.startswith("+")
            ):
                if user_file:
                    raise ConfigurationError("Only one YAML config file allowed")
                user_file = arg
            else:
                overrides.append(arg)
        return user_file, overrides
