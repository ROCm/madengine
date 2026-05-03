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
        "model",
        "build",
        "platform",
        "output",
        "summary_output",
        "data_config",
        "live_output",
    }

    @classmethod
    def to_additional_context(cls, cfg: DictConfig) -> tuple:
        """Convert DictConfig to (additional_context, metadata) tuple.

        Returns:
            additional_context: dict in the format expected by existing pipeline.
            metadata: dict with model.tags, build.registry, etc. for the CLI layer.
        """
        raw = OmegaConf.to_container(cfg, resolve=True)

        context = {}
        metadata = {}

        for key, value in raw.items():
            if key in cls.EXTRACTED_KEYS:
                metadata[key] = value
            elif key == "docker":
                for subkey, subval in value.items():
                    internal_key = cls.KEY_MAP.get(
                        f"docker.{subkey}", f"docker_{subkey}"
                    )
                    if subval is not None:
                        context[internal_key] = subval
            elif key == "log_error":
                for subkey, subval in value.items():
                    internal_key = cls.KEY_MAP.get(
                        f"log_error.{subkey}", f"log_error_{subkey}"
                    )
                    context[internal_key] = subval
            elif key == "runtime":
                metadata["runtime"] = value
            else:
                if value is not None:
                    context[key] = value

        model = metadata.get("model", {})
        if model and model.get("container_image"):
            context["MAD_CONTAINER_IMAGE"] = model["container_image"]

        return context, metadata
