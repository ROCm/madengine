"""
Kubernetes script and tool loading mixin.

Handles loading madengine common scripts, tool wrapper scripts, and Primus
experiment files for embedding into Kubernetes ConfigMaps. Since madengine
is not installed inside model Docker images, these scripts must be bundled
into the ConfigMap so the init container can recreate the expected layout.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import re
from pathlib import Path
from typing import Dict, List

from madengine.utils.path_utils import get_madengine_root

from .primus_backend import (
    infer_primus_examples_overlay_subdirs,
    merged_primus_config,
)


class KubernetesScriptsMixin:
    """Script and tool loading for Kubernetes ConfigMap embedding."""

    def gather_system_env_details(
        self, pre_scripts: List[Dict], model_name: str
    ) -> None:
        """
        Gather system environment details by adding rocEnvTool to pre-scripts.

        This ensures K8s deployment collects the same system info as local execution.

        Args:
            pre_scripts: List of pre-script configurations
            model_name: The model name (used for output file naming)
        """
        pre_env_details = {
            "path": "scripts/common/pre_scripts/run_rocenv_tool.sh",
            "args": model_name.replace("/", "_") + "_env"
        }
        pre_scripts.append(pre_env_details)
        self.console.print(f"[dim]Added rocEnvTool to pre-scripts with args: {pre_env_details['args']}[/dim]")

    def _add_tool_scripts(self, pre_scripts: List[Dict], post_scripts: List[Dict]) -> None:
        """
        Add tool pre/post scripts to execution lists (similar to local execution).

        Extracts pre_scripts and post_scripts from tools.json definitions and adds them
        to the pre_scripts and post_scripts lists for execution in K8s pods.

        Args:
            pre_scripts: List to append tool pre-scripts to
            post_scripts: List to append tool post-scripts to
        """
        tools_config = self._get_tools_config()
        if not tools_config:
            return

        tools_json_path = get_madengine_root() / "scripts" / "common" / "tools.json"
        if not tools_json_path.exists():
            return

        with open(tools_json_path, "r") as f:
            tools_definitions = json.load(f)

        for tool in tools_config:
            tool_name = tool.get("name")
            if not tool_name or tool_name not in tools_definitions.get("tools", {}):
                continue

            tool_def = tools_definitions["tools"][tool_name]

            if "pre_scripts" in tool_def:
                pre_scripts[:0] = tool_def["pre_scripts"]

            if "post_scripts" in tool_def:
                post_scripts.extend(tool_def["post_scripts"])

    def _load_common_scripts(self, script_list: List[Dict]) -> Dict[str, str]:
        """
        Load common script contents from madengine package for embedding in ConfigMap.

        Since madengine is not installed in model Docker images, we need to embed
        the common scripts (pre_scripts, post_scripts, and tool wrapper scripts) in the ConfigMap.

        Args:
            script_list: List of script configurations with 'path' field

        Returns:
            Dict mapping relative script paths to their contents
        """
        script_contents = {}
        madengine_root = get_madengine_root()

        for script_config in script_list:
            script_path = script_config.get("path", "")
            if not script_path:
                continue

            abs_script_path = madengine_root / script_path

            if abs_script_path.exists() and abs_script_path.is_file():
                with open(abs_script_path, "r") as f:
                    script_contents[script_path] = f.read()
                self.console.print(f"[dim]Loaded common script: {script_path}[/dim]")

                if "run_rocenv_tool.sh" in script_path:
                    rocenv_dir = abs_script_path.parent / "rocEnvTool"
                    if rocenv_dir.exists() and rocenv_dir.is_dir():
                        for py_file in rocenv_dir.glob("*.py"):
                            rel_path = f"scripts/common/pre_scripts/rocEnvTool/{py_file.name}"
                            with open(py_file, "r") as f:
                                script_contents[rel_path] = f.read()
                            self.console.print(f"[dim]Loaded rocEnvTool file: {rel_path}[/dim]")

                        for json_file in rocenv_dir.glob("*.json"):
                            rel_path = f"scripts/common/pre_scripts/rocEnvTool/{json_file.name}"
                            with open(json_file, "r") as f:
                                script_contents[rel_path] = f.read()
                            self.console.print(f"[dim]Loaded rocEnvTool file: {rel_path}[/dim]")
            else:
                self.console.print(f"[yellow]Warning: Script not found: {script_path} (at {abs_script_path})[/yellow]")

        tools_config = self._get_tools_config()
        if tools_config:
            self._load_tool_wrapper_scripts(script_contents, tools_config, madengine_root)

        return script_contents

    def _load_tool_wrapper_scripts(self, script_contents: Dict[str, str],
                                   tools_config: List[Dict], madengine_root: Path) -> None:
        """
        Load tool wrapper scripts and tools.json for K8s ConfigMap.

        This enables profiling tools like rocprof to work in K8s deployments.

        Args:
            script_contents: Dict to populate with script contents
            tools_config: List of tool configurations from manifest
            madengine_root: Path to madengine package root
        """
        tools_json_path = madengine_root / "scripts" / "common" / "tools.json"
        if tools_json_path.exists():
            with open(tools_json_path, "r") as f:
                tools_definitions = json.load(f)
                script_contents["scripts/common/tools.json"] = json.dumps(tools_definitions, indent=2)
            self.console.print(f"[dim]Loaded tools.json[/dim]")
        else:
            self.console.print(f"[yellow]Warning: tools.json not found at {tools_json_path}[/yellow]")
            return

        for tool in tools_config:
            tool_name = tool.get("name")
            if not tool_name:
                continue

            if tool_name not in tools_definitions.get("tools", {}):
                self.console.print(f"[yellow]Warning: Tool '{tool_name}' not found in tools.json[/yellow]")
                continue

            tool_def = tools_definitions["tools"][tool_name]

            cmd = tool.get("cmd", tool_def.get("cmd", ""))

            if "scripts/common/tools/" in cmd:
                parts = cmd.split()
                for part in parts:
                    if "scripts/common/tools/" in part:
                        script_rel_path = part.replace("../", "")
                        abs_script_path = madengine_root / script_rel_path

                        if abs_script_path.exists() and abs_script_path.is_file():
                            with open(abs_script_path, "r") as f:
                                script_contents[script_rel_path] = f.read()
                            self.console.print(f"[dim]Loaded tool script: {script_rel_path}[/dim]")

                            if script_rel_path.endswith('.py'):
                                tools_dir = abs_script_path.parent
                                utility_modules = ['amd_smi_utils.py', 'rocm_smi_utils.py', 'pynvml_utils.py']
                                for util_file in utility_modules:
                                    util_path = tools_dir / util_file
                                    if util_path.exists():
                                        util_rel_path = f"scripts/common/tools/{util_file}"
                                        if util_rel_path not in script_contents:
                                            with open(util_path, "r") as f:
                                                script_contents[util_rel_path] = f.read()
                                            self.console.print(f"[dim]Loaded tool utility module: {util_rel_path}[/dim]")
                        else:
                            self.console.print(f"[yellow]Warning: Tool script not found: {script_rel_path} (at {abs_script_path})[/yellow]")
                        break

            for script_config in tool_def.get("pre_scripts", []):
                script_path = script_config.get("path", "")
                if script_path and script_path not in script_contents:
                    abs_script_path = madengine_root / script_path
                    if abs_script_path.exists():
                        with open(abs_script_path, "r") as f:
                            script_contents[script_path] = f.read()
                        self.console.print(f"[dim]Loaded tool pre-script: {script_path}[/dim]")

            for script_config in tool_def.get("post_scripts", []):
                script_path = script_config.get("path", "")
                if script_path and script_path not in script_contents:
                    abs_script_path = madengine_root / script_path
                    if abs_script_path.exists():
                        with open(abs_script_path, "r") as f:
                            script_contents[script_path] = f.read()
                        self.console.print(f"[dim]Loaded tool post-script: {script_path}[/dim]")

            for script_config in tool_def.get("pre_scripts", []):
                script_path = script_config.get("path", "")
                if script_path:
                    abs_script_path = madengine_root / script_path
                    if abs_script_path.exists():
                        with open(abs_script_path, "r") as f:
                            script_content = f.read()
                            tool_refs = re.findall(r'(?:\.\./)?scripts/common/tools/[\w_]+\.py', script_content)
                            for tool_ref in tool_refs:
                                tool_script_path = tool_ref.strip('"\'').replace("../", "")
                                abs_tool_path = madengine_root / tool_script_path

                                if abs_tool_path.exists() and tool_script_path not in script_contents:
                                    with open(abs_tool_path, "r") as tf:
                                        script_contents[tool_script_path] = tf.read()
                                    self.console.print(f"[dim]Loaded tool dependency: {tool_script_path}[/dim]")

                                    if tool_script_path.endswith('.py'):
                                        tools_dir = abs_tool_path.parent
                                        utility_modules = ['amd_smi_utils.py', 'rocm_smi_utils.py', 'pynvml_utils.py']
                                        for util_file in utility_modules:
                                            util_path = tools_dir / util_file
                                            if util_path.exists():
                                                util_rel_path = f"scripts/common/tools/{util_file}"
                                                if util_rel_path not in script_contents:
                                                    with open(util_path, "r") as uf:
                                                        script_contents[util_rel_path] = uf.read()
                                                    self.console.print(f"[dim]Loaded utility module (from dependency): {util_rel_path}[/dim]")

    def _bundle_primus_k8s_examples_overlay(
        self, model_scripts_contents: Dict[str, str], model_name: str = ""
    ) -> None:
        """
        Add Primus experiment files from ``scripts/Primus`` into ``model_scripts_contents``
        using ConfigMap keys under ``Primus/...`` (not ``scripts/Primus/...``).

        The init container writes paths like ``/workspace/Primus/examples/...``, matching
        ``PRIMUS_ROOT=/workspace/Primus`` in the Primus Dockerfile. The Job volume hides
        image layers under ``/workspace``, so this bundle is what makes K8s runs work.

        Always includes when present:

        - ``requirements.txt`` (repo root; ``pip install -r`` from ``run_pretrain.sh``)
        - ``examples/scripts/`` (``prepare_experiment.py``, NCCL helper shells, etc.)
        - ``examples/run_pretrain.sh``
        - The backend subtree from ``distributed.primus.config_path`` (torchtitan,
          megatron, MaxText, ...).
        """
        manifest = getattr(self, "manifest", None)
        primus_cfg = merged_primus_config(
            manifest if isinstance(manifest, dict) else None,
            self.config.additional_context,
        )
        config_path = primus_cfg.get("config_path") or ""
        backend_hint = (primus_cfg.get("backend") or "").strip()
        subdirs = infer_primus_examples_overlay_subdirs(
            config_path,
            backend_hint=backend_hint,
            model_name=model_name or "",
        )
        cwd = Path.cwd()
        primus_repo = cwd / "scripts" / "Primus"
        if not primus_repo.is_dir():
            self.console.print(
                f"[yellow]Primus K8s: {primus_repo} not found — skipping Primus ConfigMap bundle.[/yellow]"
            )
            return

        def _add_primus_file(host_file: Path) -> bool:
            try:
                content = host_file.read_text(encoding="utf-8", errors="strict")
            except (UnicodeDecodeError, OSError):
                self.console.print(
                    f"[dim]Skipping non-text Primus file for K8s bundle: {host_file}[/dim]"
                )
                return False
            rel_under_repo = host_file.relative_to(primus_repo)
            key = str(Path("Primus") / rel_under_repo)
            model_scripts_contents[key] = content
            return True

        req = primus_repo / "requirements.txt"
        if req.is_file():
            if _add_primus_file(req):
                self.console.print("[dim]Primus K8s: bundled Primus/requirements.txt[/dim]")

        ex_scripts = primus_repo / "examples" / "scripts"
        if ex_scripts.is_dir():
            n_scripts = 0
            for f in ex_scripts.rglob("*"):
                if not f.is_file():
                    continue
                if _add_primus_file(f):
                    n_scripts += 1
            self.console.print(
                f"[dim]Primus K8s: bundled Primus/examples/scripts for ConfigMap ({n_scripts} files)[/dim]"
            )

        run_pre = primus_repo / "examples" / "run_pretrain.sh"
        if run_pre.is_file():
            if _add_primus_file(run_pre):
                self.console.print("[dim]Primus K8s: bundled Primus/examples/run_pretrain.sh[/dim]")

        for sub in subdirs:
            base = primus_repo / "examples" / sub
            if not base.is_dir():
                self.console.print(
                    f"[yellow]Primus K8s: scripts/Primus/examples/{sub} not found under {cwd} — "
                    "skipping that subtree.[/yellow]"
                )
                continue
            n = 0
            for f in base.rglob("*"):
                if not f.is_file():
                    continue
                if _add_primus_file(f):
                    n += 1
            self.console.print(
                f"[dim]Primus K8s: bundled Primus/examples/{sub} for ConfigMap ({n} files)[/dim]"
            )

    def _load_k8s_tools(self) -> Dict:
        """
        Load K8s-specific tools configuration.

        Returns:
            Dict with K8s tools configuration
        """
        k8s_tools_file = Path(__file__).parent.parent / "scripts" / "k8s" / "tools.json"

        if k8s_tools_file.exists():
            try:
                with open(k8s_tools_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.console.print(f"[yellow]Warning: Failed to load K8s tools config: {e}[/yellow]")
                return {}
        else:
            self.console.print(f"[yellow]Warning: K8s tools.json not found at {k8s_tools_file}[/yellow]")
            return {}
