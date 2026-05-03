"""Hydra-based config loader using the Compose API."""

import importlib.resources
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from madengine.core.errors import ConfigurationError


class HydraConfigLoader:
    """Loads madengine config using Hydra's Compose API."""

    @staticmethod
    def load(config_args: list) -> DictConfig:
        """Load and compose config from Hydra overrides and/or user YAML.

        Args:
            config_args: Mix of Hydra overrides and optional user YAML path.

        Returns:
            Composed DictConfig with all merges applied.
        """
        user_file, overrides = HydraConfigLoader._parse_args(config_args)

        config_dir = str(
            Path(importlib.resources.files("madengine")) / "configs"
        )

        if not os.path.isdir(config_dir):
            config_dir = str(
                Path(__file__).parent.parent / "configs"
            )

        GlobalHydra.instance().clear()

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config", overrides=overrides)

        if user_file:
            user_cfg = OmegaConf.load(user_file)
            OmegaConf.set_struct(cfg, False)
            cfg = OmegaConf.merge(cfg, user_cfg)

        return cfg

    @staticmethod
    def _parse_args(config_args: list) -> tuple:
        """Separate user YAML file path from Hydra overrides."""
        user_file = None
        overrides = []
        for arg in config_args:
            if (
                arg.endswith((".yaml", ".yml"))
                and "=" not in arg
                and not arg.startswith("+")
            ):
                if user_file:
                    raise ConfigurationError(
                        "Only one YAML config file allowed"
                    )
                user_file = arg
            else:
                overrides.append(arg)
        return user_file, overrides
