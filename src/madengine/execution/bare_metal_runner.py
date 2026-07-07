#!/usr/bin/env python3
"""Bare-metal (conda) execution backend for madengine.

Runs a model's scripts directly on the host inside a conda environment, without
Docker. The conda env (created in the build phase by :class:`CondaEnvManager`)
provides dependency isolation; this runner wraps each script invocation in
``conda run -n <env>`` and reuses madengine's pre/post-script, performance
extraction, and ``perf.csv`` reporting.

This is the non-Docker sibling of ``ContainerRunner``. It is single-node/local
only. It parallels ``ContainerRunner._run_self_managed`` (which already runs
scripts on the host for self-managed launchers) but adds conda wrapping, GPU
environment setup, and full reporting.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import shlex
import time
import typing
from contextlib import redirect_stderr, redirect_stdout

from rich.console import Console as RichConsole

from madengine.core.console import Console
from madengine.core.context import Context
from madengine.core.dataprovider import Data
from madengine.execution.conda_env import CondaEnvManager, resolve_conda_env_name
from madengine.execution.container_runner_helpers import (
    make_run_log_file_path,
    resolve_run_timeout,
)
from madengine.execution.run_reporting import (
    determine_status,
    extract_performance_from_log,
    write_perf_records,
)
from madengine.reporting.update_perf_csv import PERF_CSV_HEADER, flatten_tags
from madengine.utils.config_parser import ConfigParser
from madengine.utils.gpu_config import resolve_runtime_gpus
from madengine.utils.ops import PythonicTee, file_print
from madengine.utils.path_utils import scripts_base_dir_from
from madengine.utils.run_details import get_build_number, get_pipeline


class BareMetalRunner:
    """Run models on bare metal inside a conda environment (no Docker)."""

    def __init__(
        self,
        context: Context = None,
        data: Data = None,
        console: Console = None,
        live_output: bool = False,
        additional_context: typing.Dict = None,
    ):
        """Initialize the bare-metal runner.

        Args:
            context: The madengine context (with runtime GPU detection).
            data: The data provider instance.
            console: Optional console instance.
            live_output: Whether to stream output live.
            additional_context: Additional configuration context.
        """
        self.context = context
        self.data = data
        self.console = console or Console(live_output=live_output)
        self.live_output = live_output
        self.rich_console = RichConsole()
        self.credentials = None
        self.perf_csv_path = "perf.csv"
        self.additional_context = additional_context or {}
        self.bm_config = self.additional_context.get("bare_metal", {}) or {}
        self.conda = CondaEnvManager(console=self.console, bm_config=self.bm_config)

    def set_perf_csv_path(self, path: str) -> None:
        """Set the perf.csv output path."""
        self.perf_csv_path = path

    def set_credentials(self, credentials: typing.Dict) -> None:
        """Set credentials for model execution."""
        self.credentials = credentials

    def ensure_perf_csv_exists(self) -> None:
        """Ensure perf.csv exists with the standard header."""
        if not os.path.exists(self.perf_csv_path):
            file_print(PERF_CSV_HEADER, filename=self.perf_csv_path, mode="w")
            print(f"Created performance CSV file: {self.perf_csv_path}")

    def _gpu_env(self, resolved_gpu_count: int) -> typing.Dict[str, str]:
        """Build GPU-visibility env vars for the resolved GPU count.

        On bare metal there is no Docker ``--device`` flag; instead we expose a
        GPU subset via ``HIP_VISIBLE_DEVICES`` (AMD) / ``CUDA_VISIBLE_DEVICES``
        (NVIDIA). A count of -1 (all) leaves visibility unrestricted.

        Args:
            resolved_gpu_count: Number of GPUs requested.

        Returns:
            Dict of GPU env vars (possibly empty).
        """
        env: typing.Dict[str, str] = {}
        if resolved_gpu_count is None or resolved_gpu_count < 0:
            return env
        vendor = ""
        if self.context:
            vendor = str(self.context.ctx.get("gpu_vendor", "")).upper()
        device_list = ",".join(str(i) for i in range(resolved_gpu_count))
        if "AMD" in vendor:
            env["HIP_VISIBLE_DEVICES"] = device_list
            env["ROCR_VISIBLE_DEVICES"] = device_list
        elif "NVIDIA" in vendor:
            env["CUDA_VISIBLE_DEVICES"] = device_list
        return env

    def _build_run_env(
        self, model_info: typing.Dict, resolved_gpu_count: int
    ) -> typing.Dict[str, str]:
        """Assemble the environment for the model script.

        Layers (later overrides earlier): host env, context docker_env_vars,
        GPU-visibility vars, model card env_vars, additional_context env_vars,
        MAD_MODEL_NAME / build number.

        Args:
            model_info: Model definition dict.
            resolved_gpu_count: Number of GPUs requested.

        Returns:
            Environment dict for subprocess execution.
        """
        env = os.environ.copy()

        if self.context and "docker_env_vars" in self.context.ctx:
            for key, value in self.context.ctx["docker_env_vars"].items():
                env[key] = str(value)

        env.update(self._gpu_env(resolved_gpu_count))
        env["MAD_RUNTIME_NGPUS"] = str(
            resolved_gpu_count if resolved_gpu_count is not None else ""
        )

        if model_info.get("env_vars"):
            for key, value in model_info["env_vars"].items():
                env[key] = str(value)
                print(f"  ENV: {key}=<set>")

        if self.additional_context.get("env_vars"):
            for key, value in self.additional_context["env_vars"].items():
                env[key] = str(value)
                print(f"  ENV: {key}=<set>")

        env["MAD_MODEL_NAME"] = model_info["name"]
        env["JENKINS_BUILD_NUMBER"] = get_build_number()
        multiple_results = model_info.get("multiple_results")
        if multiple_results:
            env["MAD_OUTPUT_CSV"] = multiple_results

        return env

    def _resolve_script(self, model_info: typing.Dict) -> typing.Tuple[str, str, str]:
        """Resolve script path, working directory, and interpreter.

        Mirrors ContainerRunner._run_self_managed: a ``.sh``/``.py`` path is used
        directly; a directory falls back to ``run.sh``.

        Args:
            model_info: Model definition dict.

        Returns:
            Tuple ``(script_path, working_dir, interpreter)`` where interpreter is
            "python3" or "bash".

        Raises:
            FileNotFoundError: If the resolved script does not exist.
        """
        scripts_arg = model_info["scripts"]
        if scripts_arg.endswith((".sh", ".slurm", ".py")):
            script_path = scripts_arg
        else:
            script_path = os.path.join(scripts_arg, "run.sh")

        if not os.path.isabs(script_path):
            script_path = os.path.join(os.getcwd(), script_path)

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")

        working_dir = os.path.dirname(script_path) or os.getcwd()
        interpreter = "python3" if script_path.endswith(".py") else "bash"
        return script_path, working_dir, interpreter

    def _sh_in_dir(
        self,
        cmd: str,
        env: typing.Dict[str, str],
        working_dir: str,
        timeout: typing.Optional[int],
        can_fail: bool,
    ) -> str:
        """Run *cmd* in *working_dir* via Console.sh, capturing output to the log.

        Console.sh captures the child's stdout/stderr and (in live mode) streams
        it; the returned text is re-printed by the caller so it is teed into the
        run-log file (subprocess fds are not affected by redirect_stdout, so this
        capture-then-print path is what actually lands output in the log).

        Args:
            cmd: Command to run.
            env: Environment dict for the child process.
            working_dir: Directory to cd into before running.
            timeout: Timeout in seconds (None disables).
            can_fail: If False, a nonzero exit raises RuntimeError.

        Returns:
            The captured command output.
        """
        full_cmd = f"cd {shlex.quote(working_dir)} && {cmd}"
        return self.console.sh(
            full_cmd,
            canFail=can_fail,
            timeout=timeout if timeout and timeout > 0 else None,
            env=env,
        )

    def _run_scripts(
        self,
        scripts: typing.List[typing.Dict],
        env_name: str,
        env: typing.Dict[str, str],
        working_dir: str,
        timeout: typing.Optional[int],
    ) -> None:
        """Run pre/post scripts inside the conda env.

        Each entry is ``{"path": ..., "args": ...}``. Scripts run from
        *working_dir* so relative ``scripts/common`` references resolve. A
        nonzero exit raises (pre/post scripts are setup steps that must succeed).

        Args:
            scripts: List of script descriptors.
            env_name: Conda env name.
            env: Environment dict.
            working_dir: Working directory for execution.
            timeout: Per-script timeout in seconds.
        """
        prefix = self.conda.conda_run_prefix(env_name)
        for script in scripts:
            script_path = script["path"].strip()
            script_args = script.get("args", "").strip() if "args" in script else ""
            args_q = (
                " ".join(shlex.quote(a) for a in shlex.split(script_args))
                if script_args
                else ""
            )
            cmd = f"{prefix} bash {shlex.quote(script_path)} {args_q}".rstrip()
            print(f"🔧 Pre/Post script: {cmd}")
            output = self._sh_in_dir(cmd, env, working_dir, timeout, can_fail=False)
            if not self.live_output:
                print(output)

    def _create_run_details(
        self,
        model_info: typing.Dict,
        build_info: typing.Dict,
        run_results: typing.Dict,
        resolved_gpu_count: int,
    ) -> typing.Dict:
        """Build a perf.csv run-details dict for a bare-metal run.

        Args:
            model_info: Model definition dict.
            build_info: Build info from the manifest entry.
            run_results: Accumulated run results (status, performance, etc.).
            resolved_gpu_count: Number of GPUs used.

        Returns:
            Run details dict compatible with update_perf_csv.
        """
        gpu_arch = ""
        if self.context:
            gpu_arch = (self.context.ctx.get("docker_env_vars") or {}).get(
                "MAD_SYSTEM_GPU_ARCHITECTURE", ""
            )
        n_gpus = str(resolved_gpu_count if resolved_gpu_count is not None else "")

        run_details = {
            "model": model_info["name"],
            "n_gpus": n_gpus,
            "nnodes": "1",
            "gpus_per_node": n_gpus,
            "training_precision": model_info.get("training_precision", ""),
            "pipeline": get_pipeline(),
            "args": model_info.get("args", ""),
            "tags": model_info.get("tags", ""),
            "docker_file": "",
            "base_docker": "",
            "docker_sha": "",
            "docker_image": "",
            "git_commit": run_results.get("git_commit", ""),
            "machine_name": run_results.get("machine_name", ""),
            "deployment_type": "bare_metal",
            "launcher": "conda",
            "gpu_architecture": gpu_arch,
            "performance": run_results.get("performance", ""),
            "metric": run_results.get("metric", ""),
            "relative_change": "",
            "status": run_results.get("status", "FAILURE"),
            "build_duration": build_info.get("build_duration", ""),
            "test_duration": run_results.get("test_duration", ""),
            "dataname": run_results.get("dataname", ""),
            "data_provider_type": run_results.get("data_provider_type", ""),
            "data_size": run_results.get("data_size", ""),
            "data_download_duration": run_results.get("data_download_duration", ""),
            "build_number": get_build_number(),
            "additional_docker_run_options": "",
        }
        flatten_tags(run_details)

        try:
            scripts_base_dir = scripts_base_dir_from(model_info.get("scripts", ""))
            config_parser = ConfigParser(scripts_base_dir=scripts_base_dir)
            run_details["configs"] = config_parser.parse_and_load(
                model_info.get("args", ""), model_info.get("scripts", "")
            )
        except Exception as e:
            print(f"⚠️  Warning: Could not parse config file: {e}")
            run_details["configs"] = None

        return run_details

    def run_model(
        self,
        model_info: typing.Dict,
        build_info: typing.Dict = None,
        timeout: int = 7200,
        skip_model_run: bool = False,
        phase_suffix: str = "",
    ) -> typing.Dict:
        """Run a single model on bare metal inside its conda env.

        Args:
            model_info: Model definition dict.
            build_info: Build info from the manifest entry.
            timeout: Execution timeout in seconds.
            skip_model_run: If True, run pre-scripts but skip the model script.
            phase_suffix: Suffix for the run-log file (e.g. ".run").

        Returns:
            Run results dict (status, performance, metric, test_duration, ...).
        """
        build_info = build_info or {}
        env_name = resolve_conda_env_name(model_info, self.bm_config)
        timeout = resolve_run_timeout(model_info, timeout)
        # Bare-metal logs are named like the Docker path but with an explicit
        # bare-metal marker instead of an image reference.
        log_file_path = make_run_log_file_path(
            model_info, f"bare_metal_{env_name}", phase_suffix
        )
        print(f"Run log will be written to: {log_file_path}")

        machine_name = self.console.sh("hostname")

        run_results = {
            "model": model_info["name"],
            "docker_image": "",
            "status": "FAILURE",
            "performance": "",
            "metric": "",
            "test_duration": 0,
            "machine_name": machine_name,
            "log_file": log_file_path,
        }

        self.rich_console.print(
            f"[bold green]🏃 Running model:[/bold green] "
            f"[bold cyan]{model_info['name']}[/bold cyan] "
            f"[dim]on bare metal in conda env[/dim] [yellow]{env_name}[/yellow]"
        )

        resolved_gpu_count = resolve_runtime_gpus(model_info, self.additional_context)
        env = self._build_run_env(model_info, resolved_gpu_count)

        # Collect pre/post scripts from context (populated from scripts/common).
        pre_scripts = (
            list(self.context.ctx.get("pre_scripts", [])) if self.context else []
        )
        post_scripts = (
            list(self.context.ctx.get("post_scripts", [])) if self.context else []
        )
        encapsulate = (
            self.context.ctx.get("encapsulate_script", "") if self.context else ""
        )

        try:
            script_path, working_dir, interpreter = self._resolve_script(model_info)
        except FileNotFoundError as e:
            run_results["status"] = "FAILURE"
            run_results["status_detail"] = str(e)
            self.rich_console.print(f"[red]✗ {e}[/red]")
            self._record(model_info, build_info, run_results, resolved_gpu_count)
            return run_results

        model_args = (
            self.context.ctx.get("model_args", model_info.get("args", ""))
            if self.context
            else model_info.get("args", "")
        )
        args_q = (
            " ".join(shlex.quote(a) for a in shlex.split(model_args))
            if model_args
            else ""
        )
        prefix = self.conda.conda_run_prefix(env_name)
        encap = f"{encapsulate} " if encapsulate else ""
        model_cmd = f"{prefix} {encap}{interpreter} {shlex.quote(script_path)} {args_q}".rstrip()

        test_start_time = time.time()
        try:
            with open(log_file_path, mode="w", buffering=1) as outlog:
                with redirect_stdout(
                    PythonicTee(outlog, self.live_output)
                ), redirect_stderr(PythonicTee(outlog, self.live_output)):
                    print(f"⏰ Setting timeout to {timeout} seconds.")
                    print(f"📂 Working directory: {working_dir}")
                    print(f"🐍 Conda env: {env_name}")

                    if pre_scripts:
                        self._run_scripts(
                            pre_scripts, env_name, env, working_dir, timeout
                        )

                    if skip_model_run:
                        run_results["status"] = "SKIPPED"
                        print("Skipping model run (--skip-model-run).")
                    else:
                        print(f"🚀 Executing: {model_cmd}")
                        print("=" * 80)
                        # canFail=True: a nonzero model exit is not fatal here;
                        # status is decided from perf metrics + log error scan
                        # (matches the Docker path's status semantics).
                        model_output = self._sh_in_dir(
                            model_cmd, env, working_dir, timeout, can_fail=True
                        )
                        if not self.live_output:
                            print(model_output)
                        print("=" * 80)

                    if post_scripts:
                        self._run_scripts(
                            post_scripts, env_name, env, working_dir, timeout
                        )

            run_results["test_duration"] = time.time() - test_start_time
            print(f"test_duration: {run_results['test_duration']:.2f}s")

            if not skip_model_run:
                performance, metric = extract_performance_from_log(log_file_path)
                run_results["performance"] = performance or ""
                run_results["metric"] = metric or ""
                run_results["status"] = determine_status(
                    log_file_path, performance, model_info, self.additional_context
                )
                if run_results["status"] == "SUCCESS":
                    self.rich_console.print("[green]Status: SUCCESS[/green]")
                else:
                    self.rich_console.print("[red]Status: FAILURE[/red]")

        except Exception as e:
            run_results["status"] = "FAILURE"
            run_results["status_detail"] = str(e)
            run_results["test_duration"] = time.time() - test_start_time
            self.rich_console.print(f"[red]✗ Bare-metal run failed: {e}[/red]")

        self._record(model_info, build_info, run_results, resolved_gpu_count)
        return run_results

    def _record(
        self,
        model_info: typing.Dict,
        build_info: typing.Dict,
        run_results: typing.Dict,
        resolved_gpu_count: int,
    ) -> None:
        """Write perf records for a run (skipped for SKIPPED / deferred perf)."""
        if run_results.get("status") == "SKIPPED":
            return
        if self.additional_context.get("skip_perf_collection", False):
            return
        self.ensure_perf_csv_exists()
        try:
            run_details = self._create_run_details(
                model_info, build_info, run_results, resolved_gpu_count
            )
            write_perf_records(
                run_details,
                model_info,
                self.perf_csv_path,
                run_results.get("status", "FAILURE"),
            )
            print(f"Updated perf.csv with result for {model_info['name']}")
        except Exception as e:
            self.rich_console.print(
                f"[yellow]Warning: Could not update perf.csv: {e}[/yellow]"
            )

    def run_models_from_manifest(
        self,
        manifest_file: str,
        registry: str = None,
        timeout: int = 7200,
        keep_alive: bool = False,
        keep_model_dir: bool = False,
        skip_model_run: bool = False,
        phase_suffix: str = "",
    ) -> typing.Dict:
        """Run all models from a build manifest on bare metal.

        Signature mirrors ContainerRunner.run_models_from_manifest so the
        orchestrator can call either interchangeably. ``registry``,
        ``keep_alive`` and ``keep_model_dir`` are Docker-only and ignored here.

        Args:
            manifest_file: Path to build_manifest.json.
            registry: Ignored (Docker-only).
            timeout: Execution timeout per model in seconds.
            keep_alive: Ignored (Docker-only).
            keep_model_dir: Ignored (Docker-only).
            skip_model_run: Whether to skip the model script invocation.
            phase_suffix: Suffix for log files (e.g. ".run").

        Returns:
            Execution summary: successful_runs, failed_runs, total_runs.
        """
        import json

        self.rich_console.print(
            f"[bold blue]📦 Loading manifest:[/bold blue] {manifest_file}"
        )
        with open(manifest_file, "r") as f:
            manifest = json.load(f)

        built_images = manifest.get("built_images", {})
        built_models = manifest.get("built_models", {})

        if "context" in manifest and isinstance(manifest["context"], dict):
            self.additional_context = {
                **(self.additional_context or {}),
                **manifest["context"],
            }
            self.bm_config = self.additional_context.get("bare_metal", {}) or {}
            self.conda.bm_config = self.bm_config

        if not built_models:
            self.rich_console.print("[yellow]⚠️  No models found in manifest[/yellow]")
            return {"successful_runs": [], "failed_runs": [], "total_runs": 0}

        keys = built_images.keys() if built_images else built_models.keys()

        successful_runs: typing.List[typing.Dict] = []
        failed_runs: typing.List[typing.Dict] = []

        for key in keys:
            model_info = built_models.get(key, {})
            if not model_info:
                self.rich_console.print(
                    f"[yellow]⚠️  No model info for {key}, skipping[/yellow]"
                )
                continue
            build_info = built_images.get(key, {}) if built_images else {}
            try:
                run_results = self.run_model(
                    model_info=model_info,
                    build_info=build_info,
                    timeout=timeout,
                    skip_model_run=skip_model_run,
                    phase_suffix=phase_suffix,
                )
                status = run_results.get("status", "FAILURE")
                if status in ("SUCCESS", "SKIPPED"):
                    successful_runs.append(
                        {
                            "model": model_info["name"],
                            "image": "bare_metal",
                            "status": status,
                            "performance": run_results.get("performance"),
                            "duration": run_results.get("test_duration"),
                        }
                    )
                else:
                    failed_runs.append(
                        {
                            "model": model_info["name"],
                            "image": "bare_metal",
                            "status": status,
                            "error": "Bare-metal execution failed - check logs",
                        }
                    )
                    self.rich_console.print(
                        f"[red]❌ Run failed for {model_info['name']}: {status}[/red]"
                    )
            except Exception as e:
                self.rich_console.print(
                    f"[red]❌ Failed to run {model_info.get('name', key)}: {e}[/red]"
                )
                failed_runs.append(
                    {
                        "model": model_info.get("name", key),
                        "image": "bare_metal",
                        "error": str(e),
                    }
                )

        self.rich_console.print("\n[bold]📊 Execution Summary:[/bold]")
        self.rich_console.print(
            f"  [green]✓ Successful:[/green] {len(successful_runs)}"
        )
        self.rich_console.print(f"  [red]✗ Failed:[/red] {len(failed_runs)}")

        return {
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "total_runs": len(successful_runs) + len(failed_runs),
        }
