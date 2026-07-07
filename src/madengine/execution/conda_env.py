#!/usr/bin/env python3
"""Conda environment lifecycle management for bare-metal execution.

This module manages conda/mamba environments used by the bare-metal execution
backend, which runs models directly on the host (no Docker). The conda env plays
the role the Docker image plays in the container path: it isolates model
dependencies from the host and from other models.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import shlex
import shutil
import typing

from madengine.core.console import Console


def resolve_conda_env_name(
    model_info: typing.Dict, bm_config: typing.Optional[typing.Dict] = None
) -> str:
    """Resolve the conda env name for a model.

    Priority: bare_metal config ``conda_env`` > model card ``conda_env`` >
    derived from the model name (``/`` replaced with ``_``).

    Args:
        model_info: Model definition dict.
        bm_config: The ``bare_metal`` block from additional_context.

    Returns:
        The conda environment name.
    """
    bm_config = bm_config or {}
    name = bm_config.get("conda_env") or model_info.get("conda_env")
    if name:
        return str(name)
    return "mad_" + str(model_info.get("name", "model")).replace("/", "_")


class CondaEnvManager:
    """Manage conda/mamba environments for bare-metal execution."""

    def __init__(
        self,
        console: typing.Optional[Console] = None,
        bm_config: typing.Optional[typing.Dict] = None,
    ) -> None:
        """Initialize the conda env manager.

        Args:
            console: Console for shell execution (created if not provided).
            bm_config: The ``bare_metal`` block from additional_context.
        """
        self.console = console or Console()
        self.bm_config = bm_config or {}
        self._conda_bin: typing.Optional[str] = None

    def detect_conda_bin(self) -> str:
        """Locate the conda (or mamba) executable.

        Respects ``bare_metal.conda_bin`` when set, else searches PATH for
        ``mamba`` then ``conda``. Caches the result.

        Returns:
            Path to the conda/mamba executable.

        Raises:
            RuntimeError: If no conda/mamba executable can be found.
        """
        if self._conda_bin:
            return self._conda_bin

        configured = self.bm_config.get("conda_bin")
        if configured:
            if os.path.isfile(configured) or shutil.which(configured):
                self._conda_bin = configured
                return self._conda_bin
            raise RuntimeError(
                f"Configured conda_bin '{configured}' not found or not executable."
            )

        for candidate in ("mamba", "conda"):
            found = shutil.which(candidate)
            if found:
                self._conda_bin = found
                return self._conda_bin

        raise RuntimeError(
            "No conda/mamba executable found on PATH. Install conda/mamba or set "
            "bare_metal.conda_bin in --additional-context."
        )

    def env_exists(self, env_name: str) -> bool:
        """Return True if a conda env with *env_name* already exists.

        Matches by exact env name in ``conda env list`` output (the last path
        component of each env directory, and any explicitly named env).

        Args:
            env_name: The conda environment name.

        Returns:
            True if the environment exists.
        """
        conda_bin = self.detect_conda_bin()
        try:
            output = self.console.sh(f"{shlex.quote(conda_bin)} env list", canFail=True)
        except Exception:
            return False

        for line in output.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Format: "<name>   [*]   <path>"  OR just a path.
            parts = line.split()
            if not parts:
                continue
            first = parts[0]
            if first == env_name:
                return True
            # Match on the basename of the env path (last column).
            last = parts[-1]
            if os.path.basename(last.rstrip("/")) == env_name:
                return True
        return False

    def conda_run_prefix(self, env_name: str) -> str:
        """Return the ``conda run`` command prefix for executing inside *env_name*.

        Uses ``--no-capture-output`` so the child process streams stdout/stderr
        directly (matching live-output behavior of the Docker path).

        Args:
            env_name: The conda environment name.

        Returns:
            Command prefix string, e.g. ``/path/conda run -n env --no-capture-output``.
        """
        conda_bin = self.detect_conda_bin()
        return (
            f"{shlex.quote(conda_bin)} run -n {shlex.quote(env_name)} "
            f"--no-capture-output"
        )

    def create_or_update(
        self, model_info: typing.Dict, timeout: typing.Optional[int] = 3600
    ) -> str:
        """Create or update the conda env for a model and install dependencies.

        Resolution:
          1. If ``environment_file`` is given -> ``conda env create/update -f``.
          2. Else -> ``conda create -n <env> python=<python_version or default>``.
          3. If ``setup_script`` is given -> run it inside the env.

        Idempotent: when the env already exists and ``reuse_env`` (default True)
        is set, creation is skipped (setup_script still runs so deps stay fresh).

        Args:
            model_info: Model definition dict (may carry conda_env,
                environment_file, python_version, setup_script).
            timeout: Per-command timeout in seconds (None disables it).

        Returns:
            The resolved conda environment name.

        Raises:
            RuntimeError: If env creation or dependency setup fails.
        """
        env_name = resolve_conda_env_name(model_info, self.bm_config)
        conda_bin = self.detect_conda_bin()
        conda_q = shlex.quote(conda_bin)

        environment_file = (
            self.bm_config.get("environment_file")
            or model_info.get("environment_file")
            or ""
        )
        python_version = (
            self.bm_config.get("python_version")
            or model_info.get("python_version")
            or ""
        )
        setup_script = (
            self.bm_config.get("setup_script") or model_info.get("setup_script") or ""
        )
        reuse_env = self.bm_config.get("reuse_env", True)

        already = self.env_exists(env_name)

        if already and reuse_env:
            print(f"ℹ️  Conda env '{env_name}' already exists; reusing (reuse_env).")
        else:
            if environment_file:
                if not os.path.isfile(environment_file):
                    raise RuntimeError(
                        f"environment_file not found: {environment_file}"
                    )
                # `env update` is idempotent and also creates when missing.
                verb = "update" if already else "create"
                self.console.sh(
                    f"{conda_q} env {verb} -n {shlex.quote(env_name)} "
                    f"-f {shlex.quote(environment_file)}",
                    timeout=timeout,
                )
            else:
                py = f"python={python_version}" if python_version else "python"
                self.console.sh(
                    f"{conda_q} create -y -n {shlex.quote(env_name)} {shlex.quote(py)}",
                    timeout=timeout,
                )
            print(f"✓ Conda env ready: {env_name}")

        if setup_script:
            if not os.path.isfile(setup_script):
                raise RuntimeError(f"setup_script not found: {setup_script}")
            prefix = self.conda_run_prefix(env_name)
            self.console.sh(
                f"{prefix} bash {shlex.quote(setup_script)}",
                timeout=timeout,
            )
            print(f"✓ Ran setup_script in env '{env_name}': {setup_script}")

        return env_name

    def remove(self, env_name: str, timeout: typing.Optional[int] = 600) -> None:
        """Remove a conda environment (best-effort).

        Args:
            env_name: The conda environment name.
            timeout: Command timeout in seconds.
        """
        conda_bin = self.detect_conda_bin()
        self.console.sh(
            f"{shlex.quote(conda_bin)} env remove -y -n {shlex.quote(env_name)}",
            canFail=True,
            timeout=timeout,
        )
