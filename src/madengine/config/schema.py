"""Config validation for composed Hydra configs."""

from omegaconf import DictConfig, OmegaConf


KNOWN_TOP_LEVEL_KEYS = {
    "defaults", "platform", "scheduler", "hardware", "launcher",
    "model", "docker", "build", "env_vars", "debug", "live_output",
    "log_error", "tools", "pre_scripts", "post_scripts",
    "encapsulate_script", "data_config", "output", "summary_output",
    "gpu_vendor", "guest_os", "runtime", "slurm", "k8s",
    "kubernetes", "distributed", "vllm", "sglang_disagg",
    "shared_data", "timeout", "gpu_type", "gpu_memory_gb",
    "gpus_per_node", "data",
}

SUPPORTED_PLATFORMS = {"docker"}


class ConfigValidator:
    """Validates composed config for consistency."""

    @staticmethod
    def validate(cfg: DictConfig) -> list:
        """Return list of validation errors (empty = valid)."""
        errors = []

        raw = OmegaConf.to_container(cfg, resolve=False) if isinstance(cfg, DictConfig) else {}

        if raw.get("slurm") and raw.get("k8s"):
            errors.append(
                "Cannot specify both 'slurm' and 'k8s' sections"
            )

        dist = raw.get("distributed")
        if isinstance(dist, dict):
            if dist.get("enabled") and not dist.get("launcher"):
                errors.append(
                    "distributed.enabled=true requires distributed.launcher"
                )
            nnodes = dist.get("nnodes")
            if nnodes is not None:
                if not isinstance(nnodes, int) or nnodes < 1:
                    errors.append(
                        "distributed.nnodes must be a positive integer"
                    )

        platform = raw.get("platform")
        if isinstance(platform, dict):
            ptype = platform.get("type")
            if ptype and ptype not in SUPPORTED_PLATFORMS:
                errors.append(
                    f"Platform '{ptype}' is not yet supported. "
                    f"Supported: {', '.join(sorted(SUPPORTED_PLATFORMS))}"
                )

        for key in raw:
            if key not in KNOWN_TOP_LEVEL_KEYS:
                errors.append(f"Unknown config key: '{key}'")

        return errors
