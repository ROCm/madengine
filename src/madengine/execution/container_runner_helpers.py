#!/usr/bin/env python3
"""
Pure helpers for container run flow (log paths, timeout resolution).

Extracted so run_container logic is easier to test and maintain.
"""

import typing


def resolve_run_timeout(
    model_info: typing.Dict,
    cli_timeout: int,
    default_cli_timeout: int = 7200,
) -> int:
    """
    Resolve effective run timeout from model config and CLI.

    - If model has a timeout and CLI is using default (7200), use model's timeout.
    - If CLI timeout is explicitly set (not default), it overrides model timeout.

    Args:
        model_info: Model info dict; may have "timeout" key.
        cli_timeout: Timeout from CLI.
        default_cli_timeout: Value considered "default" for CLI (typically 7200).

    Returns:
        Effective timeout in seconds.
    """
    if (
        "timeout" in model_info
        and model_info["timeout"] is not None
        and model_info["timeout"] > 0
        and cli_timeout == default_cli_timeout
    ):
        return model_info["timeout"]
    return cli_timeout


def make_run_log_file_path(
    model_info: typing.Dict,
    docker_image: str,
    phase_suffix: str = "",
) -> str:
    """
    Build the log file path for a container run.

    Derives dockerfile part from image name (strip ci- and model prefix),
    then: {model_safe}_{dockerfile_part}{phase_suffix}.live.log

    Args:
        model_info: Must have "name" key.
        docker_image: Full docker image name (e.g. ci-model_ubuntu.22.04).
        phase_suffix: Optional suffix (e.g. ".run").

    Returns:
        Log file path string with "/" replaced by "_".
    """
    image_name_without_ci = docker_image.replace("ci-", "")
    model_name_clean = model_info["name"].replace("/", "_").lower()

    if image_name_without_ci.startswith(model_name_clean + "_"):
        dockerfile_part = image_name_without_ci[len(model_name_clean + "_") :]
    else:
        dockerfile_part = image_name_without_ci

    log_file_path = (
        model_info["name"].replace("/", "_")
        + "_"
        + dockerfile_part
        + phase_suffix
        + ".live.log"
    )
    return log_file_path.replace("/", "_")
