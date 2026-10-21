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

# Base URL for an opt-in micromamba download. The path below appends
# ``/<platform>/latest``; this is not a pinned build. Bare-metal detection
# does not download it. micromamba is a drop-in for the conda/mamba CLI
# surface CondaEnvManager relies on.
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

    Priority: conda config ``conda_env`` > model card ``conda_env`` >
    derived from the model name (``/`` replaced with ``_``).

    Args:
        model_info: Model definition dict.
        bm_config: The ``conda`` block from additional_context.

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
    """Download a micromamba static binary.

    Not used by :meth:`CondaEnvManager.detect_conda_bin`. Bare-metal runs require
    a conda, mamba, or micromamba already on ``PATH`` (or ``conda.conda_bin``).
    This helper remains for callers that explicitly opt in.

    Fetches the current micromamba binary for the host architecture from the
    upstream ``latest`` channel into ``~/.cache/madengine/micromamba/micromamba``
    and returns that path. This is not a pinned version. Idempotent: if the
    cached binary already exists, the download is skipped.

    micromamba is a drop-in for the ``conda``/``mamba`` CLI surface used by
    :class:`CondaEnvManager` (``env list``, ``create``, ``env update``,
    ``run -p <prefix> --no-capture-output``, ``env remove``), so no other code
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
    os.chmod(target, mode | stat.S_IXUSR)
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
            bm_config: The ``conda`` block from additional_context.
            gpu_vendor: Detected GPU vendor (e.g. ``"AMD"``/``"NVIDIA"``) used to
                pick vendor-specific dependency files. Falls back to
                ``conda.gpu_vendor`` when not given.
            gpu_arch: Detected GPU architecture (e.g. ``"gfx942"``) used to
                resolve ``rocm.index_url=auto``. Falls back to
                ``conda.gpu_arch`` when not given.
        """
        self.console = console or Console()
        self.bm_config = bm_config or {}
        self.gpu_vendor = gpu_vendor or self.bm_config.get("gpu_vendor")
        self.gpu_arch = gpu_arch or self.bm_config.get("gpu_arch")
        self._conda_bin: typing.Optional[str] = None

    def detect_conda_bin(self) -> str:
        """Locate the conda (or mamba/micromamba) executable.

        Respects ``conda.conda_bin`` when set, else searches PATH for
        ``micromamba``, ``mamba``, then ``conda``. Does not download a conda
        implementation. Caches the result.

        Returns:
            Path to the conda/mamba/micromamba executable.

        Raises:
            RuntimeError: If a configured ``conda_bin`` is invalid, or no conda
                implementation is on ``PATH``.
        """
        if self._conda_bin:
            return self._conda_bin

        configured = self.bm_config.get("conda_bin")
        if configured:
            if os.path.isfile(configured):
                if not os.access(configured, os.X_OK):
                    raise RuntimeError(
                        f"Configured conda_bin '{configured}' is not executable."
                    )
                self._conda_bin = os.path.abspath(configured)
                return self._conda_bin
            resolved = shutil.which(configured)
            if resolved:
                self._conda_bin = resolved
                return self._conda_bin
            raise RuntimeError(
                f"Configured conda_bin '{configured}' not found or not executable."
            )

        for candidate in ("micromamba", "mamba", "conda"):
            found = shutil.which(candidate)
            if found:
                self._conda_bin = found
                return self._conda_bin

        raise RuntimeError(
            "No conda, mamba, or micromamba executable found on PATH. "
            "Install one, or set conda.conda_bin to its path. "
            "madengine does not download a conda implementation."
        )

    def env_root(self) -> str:
        """Return the directory that holds bare-metal env prefixes.

        ``conda.env_root`` overrides the default
        ``~/.cache/madengine/envs``. Envs are prefixes under this directory, not
        named envs in the user's conda root, so jobs on a shared node do not
        collide with each other or with the user's interactive environments.
        """
        configured = self.bm_config.get("env_root")
        if configured:
            return os.path.abspath(str(configured))
        return os.path.join(os.path.expanduser("~"), ".cache", "madengine", "envs")

    def _safe_env_name(self, env_name: str) -> str:
        """Return a single path segment for *env_name*.

        Slashes and ``..`` are flattened so a user-supplied ``conda_env`` cannot
        escape :meth:`env_root` or the dependency-hash cache directory.
        """
        raw = str(env_name).strip().replace("\\", "/")
        safe = raw.replace("/", "_").replace("..", "_")
        if not safe or safe in (".", "_"):
            return "mad_env"
        return safe

    def env_prefix(self, env_name: str) -> str:
        """Return the on-disk prefix for *env_name* under :meth:`env_root`."""
        return os.path.join(self.env_root(), self._safe_env_name(env_name))

    def env_exists(self, env_name: str) -> bool:
        """Return True if the prefix for *env_name* is an existing conda env.

        A prefix counts only when it contains ``conda-meta``, which ``conda
        create -p`` writes. This does not consult ``conda env list``, so a
        same-named env in the user's conda root is not reused.

        Args:
            env_name: The conda environment name.

        Returns:
            True if the prefix exists.
        """
        return os.path.isdir(os.path.join(self.env_prefix(env_name), "conda-meta"))

    def conda_run_prefix(self, env_name: str) -> str:
        """Return the ``conda run`` command prefix for executing inside *env_name*.

        Uses ``-p <prefix>`` so the command targets the madengine-owned prefix.
        ``--no-capture-output`` makes conda/mamba stream the child's
        stdout/stderr. micromamba does not accept that flag (it streams by
        default) and would forward it into ``exec``.

        Args:
            env_name: The conda environment name.

        Returns:
            Command prefix string, e.g. ``/path/conda run -p /prefix --no-capture-output``.
        """
        conda_bin = self.detect_conda_bin()
        prefix = shlex.quote(self.env_prefix(env_name))
        is_micromamba = "micromamba" in os.path.basename(conda_bin).lower()
        capture_flag = "" if is_micromamba else " --no-capture-output"
        return f"{shlex.quote(conda_bin)} run -p {prefix}{capture_flag}"

    def _dep_hash_path(self, env_name: str) -> str:
        """Return the path of the dependency-hash stamp file for *env_name*."""
        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "madengine", "envhash"
        )
        return os.path.join(cache_dir, self._safe_env_name(env_name))

    def _compute_dep_hash(self, files: typing.List[str], extra: str = "") -> str:
        """Hash file-driven dependency inputs plus an optional extra token.

        Only existing files contribute. *extra* covers inputs that are not
        files (the resolved ROCm index, package list, and torch flag). Returns
        ``""`` when nothing contributed, which callers treat as "no tracked
        deps" (pure reuse).

        Args:
            files: Candidate dependency file paths (environment_file,
                requirements_file, setup_script); missing entries are skipped.
            extra: Additional stable text folded into the digest.

        Returns:
            A hex digest, or ``""`` if nothing contributed.
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
        if extra:
            found_any = True
            digest.update(extra.encode("utf-8"))
        return digest.hexdigest() if found_any else ""

    def _rocm_fingerprint(self, rocm_config: typing.Dict) -> str:
        """Stable token for the ROCm wheel install inputs.

        Included in the dependency hash so a changed index URL, package list,
        torch flag, or GPU architecture invalidates reuse. Returns ``""`` when
        the ROCm install is disabled.
        """
        if not rocm_config or not rocm_config.get("enabled"):
            return ""
        try:
            index_url = resolve_rocm_index_url(
                rocm_config.get("index_url", "auto"), self.gpu_arch
            )
        except RuntimeError:
            index_url = str(rocm_config.get("index_url", "auto"))
        packages = rocm_config.get("packages") or ["rocm[libraries,devel]"]
        torch_flag = "1" if rocm_config.get("torch") else "0"
        return "|".join(
            [
                index_url,
                ",".join(str(p) for p in packages),
                torch_flag,
                str(self.gpu_arch or ""),
            ]
        )

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

    def _delete_dep_hash(self, env_name: str) -> None:
        """Remove the dependency-hash stamp for *env_name* (best-effort)."""
        path = self._dep_hash_path(env_name)
        try:
            os.remove(path)
        except OSError:
            pass

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
          - Envs are created at ``<env_root>/<name>`` via ``conda create -p`` /
            ``conda env create -p``, not as named envs in the user conda root.
          - When the prefix exists and ``reuse_env`` (default True) is set,
            steps 1-4 are skipped unless ``environment_file``,
            ``requirements_file``, ``setup_script``, or the ROCm wheel inputs
            (index, packages, torch, GPU arch) changed since last run.
          - A python-only env that must be recreated is removed first.
            ``conda create -p`` on an existing prefix fails.
          - ``setup_script`` runs only when the env is created or its tracked
            inputs changed. Put a step that must run every invocation in
            ``pre_scripts``.

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
        prefix_path = self.env_prefix(env_name)
        prefix_q = shlex.quote(prefix_path)

        already = self.env_exists(env_name)

        # File contents and ROCm wheel inputs share one stamp. A mismatch
        # forces the create/update, pip, ROCm, and setup steps together.
        dep_hash = self._compute_dep_hash(
            [environment_file, requirements_file, setup_script],
            extra=self._rocm_fingerprint(rocm_config),
        )
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
                    f"{conda_q} env {verb} -p {prefix_q} "
                    f"-f {shlex.quote(environment_file)}",
                    timeout=timeout,
                )
            else:
                # `conda create -p` fails when the prefix already exists.
                if already:
                    self.remove(env_name, timeout=timeout)
                py = f"python={python_version}" if python_version else "python"
                # conda-forge is explicit so micromamba (no user .condarc) and
                # conda agree on where the interpreter comes from.
                self.console.sh(
                    f"{conda_q} create -y -c conda-forge -p {prefix_q} "
                    f"{shlex.quote(py)}",
                    timeout=timeout,
                )
            print(f"✓ Conda env ready: {env_name} ({prefix_path})")

        if rocm_config.get("enabled") and file_step_needed:
            self._install_rocm_wheels(env_name, rocm_config, timeout)

        if requirements_file and file_step_needed:
            if not os.path.isfile(requirements_file):
                raise RuntimeError(f"requirements_file not found: {requirements_file}")
            run_prefix = self.conda_run_prefix(env_name)
            self.console.sh(
                f"{run_prefix} pip install -r {shlex.quote(requirements_file)}",
                timeout=timeout,
            )
            print(f"✓ Installed requirements in '{env_name}': {requirements_file}")

        if setup_script and file_step_needed:
            if not os.path.isfile(setup_script):
                raise RuntimeError(f"setup_script not found: {setup_script}")
            run_prefix = self.conda_run_prefix(env_name)
            self.console.sh(
                f"{run_prefix} bash {shlex.quote(setup_script)}",
                timeout=timeout,
            )
            print(f"✓ Ran setup_script in env '{env_name}': {setup_script}")

        if file_step_needed:
            self._write_dep_hash(env_name, dep_hash)

        return env_name

    def remove(self, env_name: str, timeout: typing.Optional[int] = 600) -> None:
        """Remove a conda environment prefix and its dependency-hash stamp.

        A failed ``conda env remove`` raises. The hash stamp is deleted only
        after the prefix is gone, so a failed removal is not treated as a
        clean env on the next run.

        Args:
            env_name: The conda environment name.
            timeout: Command timeout in seconds.

        Raises:
            RuntimeError: If the prefix exists and ``conda env remove`` fails.
        """
        if self.env_exists(env_name):
            conda_bin = self.detect_conda_bin()
            self.console.sh(
                f"{shlex.quote(conda_bin)} env remove -y -p "
                f"{shlex.quote(self.env_prefix(env_name))}",
                timeout=timeout,
            )
        self._delete_dep_hash(env_name)
