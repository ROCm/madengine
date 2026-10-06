"""Translates clean YAML config to internal additional_context format."""

from typing import Any, Dict, Tuple, cast

from omegaconf import DictConfig, OmegaConf

from madengine.deployment.common import canonicalize_distributed_launcher


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

    OMIT_FROM_CONTEXT = {
        "defaults",
        "scheduler",
        "hardware",
        "launcher",
    }

    @classmethod
    def to_additional_context(
        cls, cfg: DictConfig
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Convert DictConfig to (additional_context, metadata) tuple.

        Returns:
            additional_context: dict in the format expected by existing pipeline.
            metadata: dict with model.tags, build.registry, etc. for the CLI layer.
        """
        raw = cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True))

        context: Dict[str, Any] = {}
        metadata: Dict[str, Any] = {}

        for key, value in raw.items():
            if key in cls.EXTRACTED_KEYS:
                metadata[key] = value
            elif key == "docker":
                if not isinstance(value, dict):
                    continue
                for subkey, subval in value.items():
                    internal_key = cls.KEY_MAP.get(
                        f"docker.{subkey}", f"docker_{subkey}"
                    )
                    if subval is None:
                        continue
                    if isinstance(subval, dict) and not subval:
                        continue
                    # False flags are CLI defaults; omit so JSON-equivalent
                    # context is not cluttered with docker_keep_alive: false.
                    if subkey in ("keep_alive", "clean_cache") and subval is False:
                        continue
                    context[internal_key] = subval
            elif key == "log_error":
                if not isinstance(value, dict):
                    continue
                for subkey, subval in value.items():
                    internal_key = cls.KEY_MAP.get(
                        f"log_error.{subkey}", f"log_error_{subkey}"
                    )
                    if isinstance(subval, list) and not subval:
                        continue
                    context[internal_key] = subval
            elif key == "runtime":
                # Device lists duplicate what Docker already applies from
                # gpu_vendor; keep metadata only so YAML does not invent a
                # runtime key that --additional-context never used.
                metadata["runtime"] = value
            elif key in cls.OMIT_FROM_CONTEXT:
                continue
            else:
                if value is None:
                    continue
                if isinstance(value, (list, dict)) and not value:
                    continue
                context[key] = value

        if metadata.get("live_output"):
            context["live_output"] = True

        model = metadata.get("model", {})
        if model and model.get("container_image"):
            context["MAD_CONTAINER_IMAGE"] = model["container_image"]

        dist = context.get("distributed")
        if isinstance(dist, dict) and dist.get("launcher"):
            dist["launcher"] = (
                canonicalize_distributed_launcher(dist["launcher"]) or dist["launcher"]
            )

        return context, metadata
