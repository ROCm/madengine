#!/usr/bin/env python3
"""
Registry image digest helpers for madengine.

Build pushes record the digest of the image they push; runs can optionally be
pinned to that digest so a registry tag that moved between build and run fails
loudly instead of silently resolving to a different image.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import re
import typing

from madengine.core.errors import ConfigurationError, create_error_context

# `docker push` prints e.g. "mytag: digest: sha256:<64 hex> size: 4738".
_PUSH_DIGEST_RE = re.compile(r"digest:\s*(sha256:[0-9a-f]{64})")

# `docker image inspect --format '{{index .RepoDigests 0}}'` prints "repo@sha256:<64 hex>".
_REPO_DIGEST_RE = re.compile(r"@(sha256:[0-9a-f]{64})")


def parse_push_digest(push_output: typing.Optional[str]) -> typing.Optional[str]:
    """Extract the pushed image digest from `docker push` output.

    Args:
        push_output: Combined stdout/stderr of the push command.

    Returns:
        The digest (``sha256:...``), or None if no digest line was present.
        When several digest lines appear the last one is returned, which is the
        digest of the reference the push command was invoked with.
    """
    if not push_output:
        return None
    matches = _PUSH_DIGEST_RE.findall(push_output)
    return matches[-1] if matches else None


def parse_repo_digest(inspect_output: typing.Optional[str]) -> typing.Optional[str]:
    """Extract the digest from a ``repo@sha256:...`` reference.

    Args:
        inspect_output: Output of
            ``docker image inspect --format '{{index .RepoDigests 0}}' <image>``.

    Returns:
        The digest (``sha256:...``), or None if the output held no digest
        (e.g. ``<no value>`` when the image has no RepoDigests entry).
    """
    if not inspect_output:
        return None
    match = _REPO_DIGEST_RE.search(inspect_output)
    return match.group(1) if match else None


def build_pinned_reference(registry_image: str, digest: str) -> str:
    """Build a digest-pinned image reference.

    Any existing tag or digest on ``registry_image`` is dropped. A port in the
    registry host (``localhost:5000/org/img``) is not mistaken for a tag because
    only the final path segment is inspected for ``:``.

    Args:
        registry_image: Image reference, with or without tag/digest.
        digest: Digest to pin to (``sha256:...``).

    Returns:
        A reference of the form ``repo@sha256:...``.
    """
    repo = registry_image.split("@", 1)[0]
    last_slash = repo.rfind("/")
    tail = repo[last_slash + 1 :]
    if ":" in tail:
        repo = repo[: last_slash + 1] + tail.split(":", 1)[0]
    return f"{repo}@{digest}"


def resolve_pinned_image(
    registry_image: str,
    image_digest: typing.Optional[str],
    require_pinned: bool,
    model_name: str = "",
) -> str:
    """Resolve the image reference to use for a run.

    Args:
        registry_image: Tagged registry reference from the build manifest.
        image_digest: Digest recorded at build time, if any.
        require_pinned: True when --require-pinned-image / require_pinned_image is set.
        model_name: Model name, used only to make the error message actionable.

    Returns:
        ``registry_image`` unchanged when enforcement is off, otherwise a
        digest-pinned reference.

    Raises:
        ConfigurationError: When enforcement is on but the manifest recorded no
            digest. Falling back to the tag is deliberately not done: silent
            degradation would defeat the guarantee the caller asked for.
    """
    if not require_pinned:
        return registry_image

    if not image_digest:
        raise ConfigurationError(
            f"--require-pinned-image is set but the build manifest records no "
            f"image digest for model '{model_name}' (image: {registry_image}). "
            f"Refusing to pull by tag.",
            context=create_error_context(
                operation="resolve_pinned_image",
                component="image_digest",
                additional_info={"model": model_name, "image": registry_image},
            ),
            suggestions=[
                "Rebuild with this version of madengine so the push digest is recorded",
                "Drop --require-pinned-image / require_pinned_image to pull by tag",
            ],
        )

    return build_pinned_reference(registry_image, image_digest)
