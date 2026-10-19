"""Config-driven YAML configuration system for madengine."""

from madengine.config.loader import HydraConfigLoader
from madengine.config.schema import ConfigValidator
from madengine.config.translator import ConfigTranslator


def load_config(config_args: list) -> tuple:
    """Load config from Hydra overrides and/or user YAML file.

    Args:
        config_args: List of Hydra overrides and/or a YAML file path.

    Returns:
        Tuple of (additional_context dict, metadata dict).
    """
    cfg = HydraConfigLoader.load(config_args)
    errors = ConfigValidator.validate(cfg)
    if errors:
        from madengine.core.errors import ConfigurationError

        raise ConfigurationError(
            "Config validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    return ConfigTranslator.to_additional_context(cfg)
