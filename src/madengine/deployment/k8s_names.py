#!/usr/bin/env python3
"""
Kubernetes-safe names for metadata.name, label values, and container names.

Model names from data.json may contain ``/``, spaces, or uppercase letters that
are invalid for ``metadata.name`` (RFC 1123 subdomain) or for label values.
Container names must be a single DNS label (no dots), stricter than Job names.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

from __future__ import annotations

import hashlib
import re
from typing import Final

# Kubernetes DNS subdomain total length (metadata.name)
_MAX_OBJECT_NAME_LEN: Final[int] = 253
# Label value max length
_MAX_LABEL_VALUE_LEN: Final[int] = 63
# Container / initContainer names: DNS label only (no dots); see Pod validation.
_MAX_DNS_LABEL_LEN: Final[int] = 63


def _trim_edges_alnum(s: str) -> str:
    """Ensure string starts and ends with [a-z0-9] (required for RFC1123 names)."""
    s = s.strip("-.")
    if not s:
        return "x"
    # Strip leading non-alphanumeric
    while s and not s[0].isalnum():
        s = s[1:]
    while s and not s[-1].isalnum():
        s = s[:-1]
    return s or "x"


def sanitize_k8s_object_name(prefix: str, raw_model_name: str, max_total_len: int = _MAX_OBJECT_NAME_LEN) -> str:
    """
    Build a valid ``metadata.name`` substring from a model name.

    Args:
        prefix: Leading segment (e.g. ``madengine``). May contain only chars valid in the final name.
        raw_model_name: Original model name (may include ``/``, ``_``, spaces).
        max_total_len: Maximum total length (default 253).

    Returns:
        A lowercase name safe for Kubernetes ``metadata.name`` (Job, PVC, Service, etc.).
    """
    raw = (raw_model_name or "").strip()
    pfx = (prefix or "").strip().lower()
    pfx = re.sub(r"[^a-z0-9.-]+", "-", pfx)
    pfx = re.sub(r"-+", "-", pfx).strip("-")
    if not pfx:
        pfx = "madengine"

    body = raw.lower()
    body = re.sub(r"[^a-z0-9.-]+", "-", body)
    body = re.sub(r"-+", "-", body).strip("-")
    if not body:
        body = "model"

    combined = f"{pfx}-{body}"
    combined = _trim_edges_alnum(combined)
    # Dots are allowed in RFC1123 but avoid double semantics; keep as-is if present
    if len(combined) <= max_total_len:
        return combined

    # Too long: stable short hash + truncated body
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    # room: prefix + "-" + digest + "-" + rest
    anchor = f"{pfx}-{digest}"
    room = max_total_len - len(anchor) - 1
    if room < 8:
        # Extreme: prefix alone too long — fall back to hash-only tail
        return _trim_edges_alnum(f"{digest}-{hashlib.sha256(raw.encode()).hexdigest()[:20]}")[:max_total_len]

    tail = body[:room] if room > 0 else ""
    tail = _trim_edges_alnum(tail) if tail else "m"
    out = f"{anchor}-{tail}"
    if len(out) > max_total_len:
        out = out[:max_total_len]
    return _trim_edges_alnum(out)


def sanitize_k8s_container_name(name_hint: str, max_len: int = _MAX_DNS_LABEL_LEN) -> str:
    """
    Sanitize for ``spec.containers[].name`` / initContainer names.

    Kubernetes rejects dots and other subdomain punctuation here: names must be a
    single DNS **label** (``[a-z0-9]([-a-z0-9]*[a-z0-9])?``), max 63 characters.
    Job/PVC ``metadata.name`` may still contain dots; do not reuse that string
    verbatim as a container name.
    """
    s = (name_hint or "").strip().lower()
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s:
        s = "madengine-main"
    s = _trim_edges_alnum(s)
    if len(s) > max_len:
        digest = hashlib.sha256((name_hint or "").encode("utf-8")).hexdigest()[:8]
        room = max_len - len(digest) - 1
        if room < 4:
            return digest[:max_len]
        head = s[:room]
        head = _trim_edges_alnum(head)
        out = f"{digest}-{head}"
        if len(out) > max_len:
            out = out[:max_len]
        return _trim_edges_alnum(out) or "m"
    return s


def sanitize_k8s_label_value(raw: str, max_len: int = _MAX_LABEL_VALUE_LEN) -> str:
    """
    Sanitize a string for use as a Kubernetes **label value** (max 63 chars).

    Label values must be empty or begin/end with alphanumeric, with ``-``, ``_``, ``.`` inside.
    """
    s = (raw or "").strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-_.")
    if not s:
        return "model"
    s = _trim_edges_alnum(s)
    if len(s) <= max_len:
        return s
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    # digest + '-' + remainder
    remainder = max_len - len(digest) - 1
    if remainder < 4:
        return digest[:max_len]
    tail = s[:remainder]
    tail = _trim_edges_alnum(tail)
    out = f"{digest}-{tail}"
    if len(out) > max_len:
        out = out[:max_len]
    return _trim_edges_alnum(out)
