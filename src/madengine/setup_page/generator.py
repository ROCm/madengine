#!/usr/bin/env python3
"""
Setup page generator.

Collects the models/tags from a repository's ``models.json`` (via
:class:`~madengine.utils.discover_models.DiscoverModels`) and renders the
:data:`~madengine.setup_page.schema.CONTEXT_SCHEMA` into a single, fully
self-contained HTML setup page.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import os
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from madengine.setup_page.schema import (
    CONTEXT_SCHEMA,
    SECTIONS,
    WORKLOADS,
    validate_schema,
    workload_encapsulate_script,
)

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
_TEMPLATE_NAME = "setup_page.html.j2"

DEFAULT_INSTALL_CMD = "pip install git+https://github.com/ROCm/madengine.git"


def _normalize_tags(tags: Any) -> List[str]:
    """Normalize a model's ``tags`` (list or comma-separated string) to a list."""
    if isinstance(tags, list):
        return [str(t).strip() for t in tags if str(t).strip()]
    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]
    return []


def _clean_repo_url(repo_url: str) -> str:
    """Strip trailing slashes and a ``.git`` suffix from a repo URL."""
    if not repo_url:
        return ""
    cleaned = repo_url.rstrip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[: -len(".git")]
    return cleaned


def collect_models() -> List[Dict[str, Any]]:
    """Discover models in the current working directory's ``models.json``.

    Returns a list of ``{"name": str, "tags": List[str]}`` entries, including
    models from ``scripts/*/models.json`` and ``scripts/*/get_models_json.py``.
    Falls back to reading the root ``models.json`` directly if full discovery
    fails (e.g. no ``scripts/`` directory).
    """
    from madengine.utils.discover_models import DiscoverModels

    class _Args:
        tags: List[str] = []

    models: List[Dict[str, Any]] = []
    try:
        discoverer = DiscoverModels(args=_Args())
        discoverer.discover_models()
        for entry in discoverer.models:
            models.append(
                {"name": entry.get("name", ""), "tags": _normalize_tags(entry.get("tags"))}
            )
        for custom in getattr(discoverer, "custom_models", []):
            models.append(
                {"name": custom.name, "tags": _normalize_tags(getattr(custom, "tags", []))}
            )
    except FileNotFoundError:
        raise
    except Exception:
        # Degrade gracefully to the root models.json (e.g. no scripts/ dir).
        root = os.path.join(os.getcwd(), "models.json")
        if os.path.exists(root):
            with open(root) as f:
                for entry in json.load(f):
                    models.append(
                        {
                            "name": entry.get("name", ""),
                            "tags": _normalize_tags(entry.get("tags")),
                        }
                    )
        else:
            raise

    # De-duplicate by name while preserving order.
    seen = set()
    unique: List[Dict[str, Any]] = []
    for model in models:
        if model["name"] and model["name"] not in seen:
            seen.add(model["name"])
            unique.append(model)
    return unique


def render_setup_page(
    models: List[Dict[str, Any]],
    title: str = "madengine Setup Picker",
    repo_url: str = "",
    install_cmd: str = DEFAULT_INSTALL_CMD,
    schema: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Render the setup page to a self-contained HTML string.

    Args:
        models: List of ``{"name", "tags"}`` model entries.
        title: Page title / heading.
        repo_url: Model repo URL (used to generate clone instructions). Any
            trailing ``.git`` is handled client-side.
        install_cmd: The madengine install command shown in step 1.
        schema: Context schema (defaults to :data:`CONTEXT_SCHEMA`).

    Returns:
        The rendered HTML document as a string.
    """
    schema = schema if schema is not None else CONTEXT_SCHEMA
    validate_schema(schema)

    workloads = [
        dict(workload, encapsulate=workload_encapsulate_script(workload["id"]))
        for workload in WORKLOADS
    ]

    data = {
        "models": models,
        "schema": schema,
        "sections": SECTIONS,
        "workloads": workloads,
        "repoUrl": _clean_repo_url(repo_url),
        "installCmd": install_cmd,
        "title": title,
    }
    # Escape "<" so the JSON payload is safe inside a <script> block.
    data_json = json.dumps(data).replace("<", "\\u003c")

    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template(_TEMPLATE_NAME)
    return template.render(title=title, data_json=data_json)


def generate_setup_page(
    output: str = "index.html",
    title: str = "madengine Setup Picker",
    repo_url: str = "",
    install_cmd: str = DEFAULT_INSTALL_CMD,
) -> str:
    """Collect models and write the rendered setup page to ``output``.

    Returns the output path written.
    """
    models = collect_models()
    html = render_setup_page(
        models=models, title=title, repo_url=repo_url, install_cmd=install_cmd
    )
    out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)
    return output
