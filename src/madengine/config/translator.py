"""Translates clean YAML config to internal additional_context format."""

from omegaconf import DictConfig, OmegaConf


class ConfigTranslator:
    """Maps YAML config keys to internal additional_context dict format."""

    KEY_MAP = {
        "docker.build_args": "docker_build_arg",
        "docker.env_vars": "docker_env_vars",
        "docker.mounts": "docker_mounts",
        "docker.gpus": "docker_gpus",
        "docker.cpus": "docker_cpus",
        "docker.additional_run_options": "additional_docker_run_options",
        "log_error.pattern_scan": "log_error_pattern_scan",
        "log_error.benign_patterns": "log_error_benign_patterns",
        "log_error.patterns": "log_error_patterns",
    }

    EXTRACTED_KEYS = {
        "model", "build", "platform", "output",
        "summary_output", "data_config", "live_output",
    }

    @classmethod
    def to_additional_context(cls, cfg: DictConfig) -> tuple:
        """Placeholder — implemented in Task 5."""
        return {}, {}
