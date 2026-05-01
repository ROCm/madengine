#!/usr/bin/env python3
"""
Pure helpers for container run flow (log paths, timeout resolution).

Extracted so run_container logic is easier to test and maintain.
"""

import re
import typing

# Default substrings matched in container run logs post-hoc (see ContainerRunner).
DEFAULT_LOG_ERROR_PATTERNS: typing.Tuple[str, ...] = (
    "OutOfMemoryError",
    "HIP out of memory",
    "CUDA out of memory",
    "RuntimeError:",
    "AssertionError:",
    "ValueError:",
    "SystemExit",
    "failed (exitcode:",
    "Traceback (most recent call last)",
    "FAILED",
    "Exception:",
    "ImportError:",
    "ModuleNotFoundError:",
)


def _coerce_bool(value: typing.Any, *, default: bool) -> bool:
    """Interpret JSON/CLI scalars as bool; fall back to *default* if None."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("0", "false", "no", "off", ""):
            return False
        if s in ("1", "true", "yes", "on"):
            return True
    return default


def _pick_context_over_model(
    model_info: typing.Dict,
    additional_context: typing.Dict,
    key: str,
    default: typing.Any = None,
) -> typing.Any:
    """Resolve key from model_info, overridden by additional_context when present."""
    ctx = additional_context or {}
    mi = model_info or {}
    if key in ctx:
        return ctx[key]
    if key in mi:
        return mi[key]
    return default


def resolve_log_error_scan_config(
    model_info: typing.Dict,
    additional_context: typing.Optional[typing.Dict] = None,
) -> typing.Tuple[bool, typing.List[str], typing.List[str]]:
    """
    Resolve whether to scan run logs for error substrings and which patterns to use.

    Keys (in ``additional_context`` and/or ``model_info``; context wins):

    - ``log_error_pattern_scan`` (default True): set False to skip post-run log error detection.
    - ``log_error_benign_patterns``: list of extra **literal** substrings; a log line containing
      any of them is excluded from error matching (not interpreted as regex).
    - ``log_error_patterns``: non-empty list of strings replaces the default error pattern list.

    Returns:
        (scan_enabled, error_patterns, extra_benign_patterns)
    """
    ctx = additional_context if additional_context is not None else {}
    mi = model_info if model_info is not None else {}

    scan_enabled = _coerce_bool(
        _pick_context_over_model(mi, ctx, "log_error_pattern_scan", True),
        default=True,
    )

    raw_benign_mi = mi.get("log_error_benign_patterns")
    raw_benign_ctx = ctx.get("log_error_benign_patterns")
    extra_benign: typing.List[str] = []
    for part in (raw_benign_mi, raw_benign_ctx):
        if isinstance(part, list):
            extra_benign.extend(str(x) for x in part if x is not None)

    custom_patterns = _pick_context_over_model(mi, ctx, "log_error_patterns", None)
    if (
        isinstance(custom_patterns, list)
        and len(custom_patterns) > 0
        and all(isinstance(x, str) for x in custom_patterns)
    ):
        error_patterns = list(custom_patterns)
    else:
        error_patterns = list(DEFAULT_LOG_ERROR_PATTERNS)

    return scan_enabled, error_patterns, extra_benign


def log_text_has_error_pattern(
    log_text: str,
    pattern: str,
    benign_substrings: typing.Sequence[str],
    benign_regexes: typing.Sequence[str] = (),
) -> bool:
    """
    Whether *log_text* contains a literal *pattern* on some line that is not excluded.

    Exclusions (same intent as the old ``grep -v -E | grep -F`` pipeline):

    - Meta lines mentioning our own ``grep`` / "Found error pattern" machinery.
    - *benign_substrings*: line skipped if any string appears as a **literal** substring.
    - *benign_regexes*: line skipped if any compiled regex matches (for built-in ROCProf rules).

    User-supplied benign entries should use *benign_substrings* only so regex metacharacters
    in config are not interpreted unless explicitly added to *benign_regexes*.
    """
    pattern_escaped = re.escape(pattern)
    try:
        meta_excl = re.compile(
            f"(grep -q.*{pattern_escaped}|Found error pattern.*{pattern_escaped})"
        )
    except re.error:
        return False

    compiled_benign: typing.List[re.Pattern[str]] = []
    for rx in benign_regexes:
        try:
            compiled_benign.append(re.compile(rx))
        except re.error:
            continue

    for line in log_text.splitlines():
        if meta_excl.search(line):
            continue
        if any(s in line for s in benign_substrings):
            continue
        if any(br.search(line) for br in compiled_benign):
            continue
        if pattern in line:
            return True
    return False


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


def _docker_image_ref_for_log_naming(docker_image: str) -> str:
    """
    Reduce a Docker image reference to a stable filename-safe log naming component.

    Build logs use ``{model}_{dockerfile}.build.live.log`` where the image tag is
    ``ci-{model_lower}_{dockerfile}``. For those CI-style refs, returning the
    tag preserves pairing with ``*.build.live.log``. For normal tagged refs such
    as ``ubuntu:22.04`` or ``myimage:latest``, collapsing to only the tag would
    cause collisions, so keep the digest-free reference and sanitize it for filenames.

    Rules:

    - Strip digest (``@sha256:...``) for naming purposes.
    - If the last path segment is ``name:tag`` and ``tag`` starts with ``ci-``,
      return only ``tag``.
    - Otherwise return the digest-free reference with ``:``, ``/``, and ``@``
      replaced by ``_``.

    Short names without ``/`` or ``:``, e.g. ``ci-model_ubuntu``, pass through
    unchanged (aside from digest stripping).
    """
    if not docker_image:
        return docker_image
    s = docker_image.strip()
    ref_without_digest = s.split("@", 1)[0]
    last_slash = ref_without_digest.rfind("/")
    tail = (
        ref_without_digest[last_slash + 1 :] if last_slash >= 0 else ref_without_digest
    )
    if ":" in tail:
        _, tag = tail.split(":", 1)
        if tag.startswith("ci-"):
            return tag
    return ref_without_digest.replace("/", "_").replace(":", "_").replace("@", "_")


def make_run_log_file_path(
    model_info: typing.Dict,
    docker_image: str,
    phase_suffix: str = "",
) -> str:
    """
    Build the log file path for a container run.

    Derives dockerfile part from image name (strip ci- and model prefix),
    then: {model_safe}_{dockerfile_part}{phase_suffix}.live.log

    ``docker_image`` may be a short tag (``ci-model_ubuntu``) or a full
    reference (``registry/repo/name:ci-model_ubuntu``); the latter is
    normalized so run logs align with ``DockerBuilder`` / ``*.build.live.log``.

    Args:
        model_info: Must have "name" key.
        docker_image: Docker image reference (short tag or registry/name:tag).
        phase_suffix: Optional suffix (e.g. ".run").

    Returns:
        Log file path string with "/" replaced by "_".
    """
    docker_image = _docker_image_ref_for_log_naming(docker_image)
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
