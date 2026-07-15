#!/usr/bin/env python3
"""Conda environment lifecycle management for bare-metal execution.

This module manages conda/mamba environments used by the bare-metal execution
backend, which runs models directly on the host (no Docker). The conda env plays
the role the Docker image plays in the container path: it isolates model
dependencies from the host and from other models.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import hashlib
import os
import platform
import shlex
import shutil
import stat
import typing
import urllib.request

from madengine.core.console import Console

# Base URL for downloading pinned micromamba static binaries when no
# conda/mamba is present on the node. micromamba is a drop-in for the
# conda/mamba CLI surface CondaEnvManager relies on.
_MICROMAMBA_BASE_URL = "https://micro.mamba.pm/api/micromamba"

# Base URL for TheRock's per-architecture ROCm/torch pip wheel indexes. The
# resolved index for a gfx arch is "<base>/<gfx_arch>/" and serves both the
# rocm[...] userspace packages and matching torch/torchvision wheels. Kept as a
# single constant because TheRock's preview URLs may shift.
_ROCM_NIGHTLIES_BASE_URL = "https://rocm.nightlies.amd.com/v2"


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


def _vendor_suffix(vendor: typing.Optional[str]) -> str:
    """Map a GPU vendor string to the file suffix used for vendor variants.

    Mirrors the Dockerfile convention (``*.amd.Dockerfile`` /
    ``*.nvidia.Dockerfile``): AMD -> ``amd``, NVIDIA -> ``nvidia``.

    Args:
        vendor: GPU vendor string (case-insensitive), e.g. ``"AMD"``.

    Returns:
        The lowercase suffix token, or ``""`` if the vendor is unknown.
    """
    v = str(vendor or "").strip().upper()
    if v == "AMD":
        return "amd"
    if v == "NVIDIA":
        return "nvidia"
    return ""


def resolve_rocm_index_url(index_url: str, gfx_arch: str) -> str:
    """Resolve the ROCm pip wheel index URL for a GPU architecture.

    An explicit *index_url* (anything other than ``"auto"``) is returned
    unchanged. ``"auto"`` (or empty) resolves to TheRock's per-arch index
    ``<base>/<gfx_arch>/`` using the detected gfx architecture.

    Args:
        index_url: Configured index URL, or ``"auto"`` to derive from the arch.
        gfx_arch: Detected GPU architecture (e.g. ``"gfx942"``).

    Returns:
        The resolved pip index URL.

    Raises:
        RuntimeError: If ``"auto"`` is requested but no gfx arch is available.
    """
    if index_url and index_url != "auto":
        return index_url
    if not gfx_arch:
        raise RuntimeError(
            "rocm.index_url=auto requires a detected GPU architecture "
            "(MAD_SYSTEM_GPU_ARCHITECTURE); set rocm.index_url explicitly instead."
        )
    return f"{_ROCM_NIGHTLIES_BASE_URL}/{gfx_arch}/"


def resolve_environment_file(base_path: str, vendor: typing.Optional[str]) -> str:
    """Resolve a vendor-specific variant of a file, else the base file.

    Given ``scripts/dummy/environment.yml`` and ``vendor="AMD"``, prefer
    ``scripts/dummy/environment.amd.yml`` when it exists, otherwise return the
    base path unchanged. Mirrors the Dockerfile suffix convention used by
    ``DockerBuilder._get_dockerfiles_for_model`` (suffix match only, no
    ``# CONTEXT`` build-arg filtering — there is no build-arg matrix here).

    Args:
        base_path: The base file path (e.g. an ``environment.yml``).
        vendor: GPU vendor string used to pick the variant.

    Returns:
        The vendor-specific path if it exists, else *base_path*.
    """
    if not base_path:
        return base_path
    suffix = _vendor_suffix(vendor)
    if not suffix:
        return base_path
    root, ext = os.path.splitext(base_path)
    candidate = f"{root}.{suffix}{ext}"
    if os.path.isfile(candidate):
        return candidate
    return base_path


def bootstrap_micromamba() -> str:
    """Download a pinned micromamba static binary when no conda/mamba exists.

    Fetches the micromamba binary for the host architecture into
    ``~/.cache/madengine/micromamba/micromamba`` and returns that path.
    Idempotent: if the cached binary already exists, the download is skipped.

    micromamba is a drop-in for the ``conda``/``mamba`` CLI surface used by
    :class:`CondaEnvManager` (``env list``, ``create``, ``env update``,
    ``run -n <env> --no-capture-output``, ``env remove``), so no other code
    needs to branch on which binary is in use.

    Returns:
        Path to the executable micromamba binary.

    Raises:
        RuntimeError: If the host architecture is unsupported or the download
            fails.
    """
    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "madengine", "micromamba"
    )
    target = os.path.join(cache_dir, "micromamba")
    if os.path.isfile(target) and os.access(target, os.X_OK):
        return target

    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        platform_tag = "linux-64"
    elif machine in ("aarch64", "arm64"):
        platform_tag = "linux-aarch64"
    else:
        raise RuntimeError(
            f"Unsupported architecture for micromamba bootstrap: {machine}"
        )

    url = f"{_MICROMAMBA_BASE_URL}/{platform_tag}/latest"
    os.makedirs(cache_dir, exist_ok=True)
    print(f"⤓ Bootstrapping micromamba for {platform_tag} from {url}")
    try:
        # The endpoint serves a tar.bz2 archive with the binary at
        # bin/micromamba; stream it and extract just that member.
        import tarfile

        tmp_archive = target + ".tar.bz2"
        urllib.request.urlretrieve(url, tmp_archive)  # noqa: S310 (trusted URL)
        with tarfile.open(tmp_archive, "r:bz2") as tar:
            member = tar.getmember("bin/micromamba")
            member.name = "micromamba"
            tar.extract(member, path=cache_dir)
        os.remove(tmp_archive)
    except Exception as exc:
        raise RuntimeError(f"Failed to bootstrap micromamba from {url}: {exc}") from exc

    mode = os.stat(target).st_mode
    os.chmod(target, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"✓ micromamba ready: {target}")
    return target


class CondaEnvManager:
    """Manage conda/mamba environments for bare-metal execution."""

    def __init__(
        self,
        console: typing.Optional[Console] = None,
        bm_config: typing.Optional[typing.Dict] = None,
        gpu_vendor: typing.Optional[str] = None,
        gpu_arch: typing.Optional[str] = None,
    ) -> None:
        """Initialize the conda env manager.

        Args:
            console: Console for shell execution (created if not provided).
            bm_config: The ``bare_metal`` block from additional_context.
            gpu_vendor: Detected GPU vendor (e.g. ``"AMD"``/``"NVIDIA"``) used to
                pick vendor-specific dependency files. Falls back to
                ``bare_metal.gpu_vendor`` when not given.
            gpu_arch: Detected GPU architecture (e.g. ``"gfx942"``) used to
                resolve ``rocm.index_url=auto``. Falls back to
                ``bare_metal.gpu_arch`` when not given.
        """
        self.console = console or Console()
        self.bm_config = bm_config or {}
        self.gpu_vendor = gpu_vendor or self.bm_config.get("gpu_vendor")
        self.gpu_arch = gpu_arch or self.bm_config.get("gpu_arch")
        self._conda_bin: typing.Optional[str] = None

    def detect_conda_bin(self) -> str:
        """Locate the conda (or mamba/micromamba) executable.

        Respects ``bare_metal.conda_bin`` when set, else searches PATH for
        ``micromamba``, ``mamba``, then ``conda``. When none is found, bootstraps
        a pinned micromamba binary (see :func:`bootstrap_micromamba`). Caches the
        result.

        Returns:
            Path to the conda/mamba/micromamba executable.

        Raises:
            RuntimeError: If a configured ``conda_bin`` is invalid, or micromamba
                bootstrap fails.
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

        for candidate in ("micromamba", "mamba", "conda"):
            found = shutil.which(candidate)
            if found:
                self._conda_bin = found
                return self._conda_bin

        # Nothing on PATH and no conda_bin configured: bootstrap micromamba so
        # bare-metal runs work with zero manual conda install.
        self._conda_bin = bootstrap_micromamba()
        return self._conda_bin

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
        # `--no-capture-output` is a conda/mamba flag that makes `run` stream the
        # child's stdout/stderr instead of buffering. micromamba does not accept
        # it (it streams by default) and forwards the unrecognized token into its
        # internal `exec`, producing "exec: --: invalid option". So only add the
        # flag for conda/mamba.
        is_micromamba = "micromamba" in os.path.basename(conda_bin).lower()
        capture_flag = "" if is_micromamba else " --no-capture-output"
        return f"{shlex.quote(conda_bin)} run -n {shlex.quote(env_name)}{capture_flag}"

    def _dep_hash_path(self, env_name: str) -> str:
        """Return the path of the dependency-hash stamp file for *env_name*."""
        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "madengine", "envhash"
        )
        return os.path.join(cache_dir, env_name)

    def _compute_dep_hash(self, files: typing.List[str]) -> str:
        """Hash the content of file-driven dependency inputs.

        Only existing files contribute. Returns ``""`` when no such files are
        given, which callers treat as "no file-driven deps" (pure reuse).

        Args:
            files: Candidate dependency file paths (environment_file,
                requirements_file); missing/empty entries are skipped.

        Returns:
            A hex digest, or ``""`` if no existing files were provided.
        """
        digest = hashlib.sha256()
        found_any = False
        for path in files:
            if not path or not os.path.isfile(path):
                continue
            found_any = True
            digest.update(path.encode("utf-8"))
            with open(path, "rb") as f:
                digest.update(f.read())
        return digest.hexdigest() if found_any else ""

    def _write_dep_hash(self, env_name: str, dep_hash: str) -> None:
        """Persist the dependency hash stamp for *env_name* (best-effort)."""
        if not dep_hash:
            return
        path = self._dep_hash_path(env_name)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(dep_hash)
        except OSError as exc:
            print(f"⚠️  Could not write dep-hash stamp for '{env_name}': {exc}")

    def _read_dep_hash(self, env_name: str) -> str:
        """Read the stored dependency hash stamp for *env_name* (``""`` if none)."""
        path = self._dep_hash_path(env_name)
        try:
            with open(path, "r") as f:
                return f.read().strip()
        except OSError:
            return ""

    def _install_rocm_wheels(
        self,
        env_name: str,
        rocm_config: typing.Dict,
        timeout: typing.Optional[int],
    ) -> None:
        """Install ROCm userspace (and optionally torch) pip wheels into the env.

        Resolves the per-arch index URL (``index_url=auto`` -> gfx-arch index)
        and pip-installs the configured packages, then torch/torchvision from the
        same index when ``torch`` is set.

        Args:
            env_name: The conda environment name.
            rocm_config: The resolved ``rocm`` block.
            timeout: Per-command timeout in seconds.

        Raises:
            RuntimeError: If the index URL cannot be resolved.
        """
        index_url = resolve_rocm_index_url(
            rocm_config.get("index_url", "auto"), self.gpu_arch
        )
        packages = rocm_config.get("packages") or ["rocm[libraries,devel]"]
        prefix = self.conda_run_prefix(env_name)
        url_q = shlex.quote(index_url)
        pkgs_q = " ".join(shlex.quote(p) for p in packages)
        self.console.sh(
            f"{prefix} pip install --index-url {url_q} {pkgs_q}",
            timeout=timeout,
        )
        print(f"✓ Installed ROCm wheels in '{env_name}' from {index_url}")
        if rocm_config.get("torch", False):
            self.console.sh(
                f"{prefix} pip install --index-url {url_q} torch torchvision",
                timeout=timeout,
            )
            print(f"✓ Installed torch/torchvision in '{env_name}' from {index_url}")

    def create_or_update(
        self, model_info: typing.Dict, timeout: typing.Optional[int] = 3600
    ) -> str:
        """Create or update the conda env for a model and install dependencies.

        Pipeline (each step optional and additive):
          1. Create/reuse the env: ``conda env create/update -f`` when
             ``environment_file`` is given, else
             ``conda create -n <env> python=<python_version or default>``.
          2. If ``rocm.enabled`` -> pip-install ROCm userspace (+ torch) from the
             gfx-arch index.
          3. If ``requirements_file`` -> ``pip install -r`` inside the env.
          4. If ``setup_script`` -> run it inside the env.

        Idempotency:
          - When the env exists and ``reuse_env`` (default True) is set, steps 1
            and 3 are skipped *unless* the content of ``environment_file`` /
            ``requirements_file`` changed since last run (tracked via a content-
            hash stamp), which forces an env update + requirements reinstall.
          - The ROCm-wheel install (step 2) is skip-on-reuse regardless of file
            hashes (it is not file-driven and reinstalling is slow); force a
            refresh with ``reuse_env: False``.
          - ``setup_script`` (step 4) always runs.

        Args:
            model_info: Model definition dict (may carry conda_env,
                environment_file, requirements_file, python_version, rocm,
                setup_script).
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
        if environment_file:
            environment_file = resolve_environment_file(
                environment_file, self.gpu_vendor
            )
        requirements_file = (
            self.bm_config.get("requirements_file")
            or model_info.get("requirements_file")
            or ""
        )
        if requirements_file:
            requirements_file = resolve_environment_file(
                requirements_file, self.gpu_vendor
            )
        python_version = (
            self.bm_config.get("python_version")
            or model_info.get("python_version")
            or ""
        )
        setup_script = (
            self.bm_config.get("setup_script") or model_info.get("setup_script") or ""
        )
        if setup_script:
            setup_script = resolve_environment_file(setup_script, self.gpu_vendor)
        rocm_config = self.bm_config.get("rocm") or model_info.get("rocm") or {}
        reuse_env = self.bm_config.get("reuse_env", True)

        already = self.env_exists(env_name)

        # File-driven deps (environment_file/requirements_file) invalidate reuse
        # when their content changes; ROCm wheels do not (see docstring).
        dep_hash = self._compute_dep_hash([environment_file, requirements_file])
        deps_changed = bool(dep_hash) and dep_hash != self._read_dep_hash(env_name)
        file_step_needed = (not already) or (not reuse_env) or deps_changed

        if already and reuse_env and not deps_changed:
            print(f"ℹ️  Conda env '{env_name}' already exists; reusing (reuse_env).")
        else:
            if already and reuse_env and deps_changed:
                print(
                    f"ℹ️  Dependency files changed for '{env_name}'; "
                    f"updating env despite reuse_env."
                )
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

        # ROCm wheels: skip-on-reuse (install only on fresh create or reuse_env=False).
        if rocm_config.get("enabled") and (not already or not reuse_env):
            self._install_rocm_wheels(env_name, rocm_config, timeout)

        if requirements_file and file_step_needed:
            if not os.path.isfile(requirements_file):
                raise RuntimeError(f"requirements_file not found: {requirements_file}")
            prefix = self.conda_run_prefix(env_name)
            self.console.sh(
                f"{prefix} pip install -r {shlex.quote(requirements_file)}",
                timeout=timeout,
            )
            print(f"✓ Installed requirements in '{env_name}': {requirements_file}")

        if file_step_needed:
            self._write_dep_hash(env_name, dep_hash)

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
