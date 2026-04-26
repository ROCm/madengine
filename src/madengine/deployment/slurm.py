#!/usr/bin/env python3
"""
SLURM Deployment - HPC cluster deployment using CLI commands.

Uses subprocess to call SLURM CLI commands (sbatch, squeue, scancel).
No Python SLURM library required (zero dependencies).

**Assumption**: User has already SSH'd to SLURM login node manually.
madengine is executed ON the login node, not remotely.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseDeployment, DeploymentConfig, DeploymentResult, DeploymentStatus, create_jinja_env
from .primus_backend import infer_primus_backend_from_model_name, merged_primus_config
from .common import configure_multi_node_profiling, normalize_launcher
from .config_loader import ConfigLoader, apply_deployment_config
from .slurm_node_selector import SlurmNodeSelector
from madengine.utils.gpu_config import resolve_runtime_gpus
from madengine.utils.run_details import get_build_number, get_pipeline
from madengine.utils.path_utils import scripts_base_dir_from
import json


class SlurmDeployment(BaseDeployment):
    """
    SLURM HPC cluster deployment using CLI commands.

    **Workflow**:
    1. User: ssh login_node@hpc.example.com
    2. User: madengine run --tags model --additional-context '{"deploy": "slurm", ...}'
    3. madengine: Runs sbatch locally (no SSH needed)

    Uses subprocess to call SLURM CLI commands locally:
    - sbatch: Submit jobs to SLURM scheduler
    - squeue: Monitor job status
    - scancel: Cancel jobs
    - scontrol: Get cluster info

    No Python SLURM library required (zero dependencies).
    No SSH handling needed (user is already on login node).
    """

    DEPLOYMENT_TYPE = "slurm"
    REQUIRED_TOOLS = ["sbatch", "squeue", "scontrol"]  # Must be available locally
    GCM_ALLOWED_HEALTH_CHECKS = {"check-hca", "check-ibstat"}
    GCM_ALLOWED_COLLECTORS = {"slurm_job_monitor"}

    def __init__(self, config: DeploymentConfig):
        """
        Initialize SLURM deployment.

        Args:
            config: Deployment configuration
        """
        apply_deployment_config(config, ConfigLoader.load_slurm_config)
        super().__init__(config)

        # Parse SLURM configuration (now with defaults applied)
        self.slurm_config = config.additional_context.get("slurm", {})
        self.distributed_config = config.additional_context.get("distributed", {})
        self.cluster_config = config.additional_context.get("cluster", {})
        self.rdma_config = self.cluster_config.get("rdma", {})
        self.gcm_config = self.cluster_config.get("gcm", {})

        # SLURM parameters
        self.partition = self.slurm_config.get("partition", "gpu")
        self.nodes = self.slurm_config.get("nodes", 1)
        self.gpus_per_node = self.slurm_config.get("gpus_per_node", 8)
        self.time_limit = self.slurm_config.get("time", "24:00:00")
        self.output_dir = Path(self.slurm_config.get("output_dir", "./slurm_results"))

        # Setup Jinja2 template engine
        template_dir = Path(__file__).parent / "templates" / "slurm"
        self.jinja_env = create_jinja_env(template_dir)

        # Generated script path
        self.script_path = None

    def _gcm_enabled(self) -> bool:
        return bool(self.gcm_config.get("enabled", False))

    def _gcm_artifact_layout(self) -> Dict[str, str]:
        source = self.gcm_config.get("source", {})
        artifacts = self.gcm_config.get("artifacts", {})
        files = artifacts.get("files", {})
        return {
            "repo": source.get("repo", "https://github.com/coketaste/gcm"),
            "ref": source.get("ref", "9fed02cd0721d3937f8749672951185f31955bd4"),
            "dir": artifacts.get("dir", str(self.output_dir / "cluster_artifacts")),
            "health_summary_json": files.get(
                "health_summary_json", "gcm_health_summary.json"
            ),
            "health_raw_log": files.get("health_raw_log", "gcm_health_raw.log"),
            "collector_output": files.get("collector_output", "gcm_collector_output.log"),
            "provenance_json": files.get(
                "provenance_json", "gcm_source_provenance.json"
            ),
        }

    def _validate_gcm_allowlist(self) -> None:
        checks = self.gcm_config.get("health_checks", ["check-hca", "check-ibstat"])
        invalid_checks = [c for c in checks if c not in self.GCM_ALLOWED_HEALTH_CHECKS]
        if invalid_checks:
            raise ValueError(
                f"Invalid cluster.gcm.health_checks value(s): {invalid_checks}. "
                f"Allowed: {sorted(self.GCM_ALLOWED_HEALTH_CHECKS)}"
            )

        collector_cfg = self.gcm_config.get("collector", {})
        if not collector_cfg.get("enabled", False):
            return
        collector_cmd = collector_cfg.get("command", "slurm_job_monitor")
        if collector_cmd not in self.GCM_ALLOWED_COLLECTORS:
            raise ValueError(
                f"Invalid cluster.gcm.collector.command: {collector_cmd}. "
                f"Allowed: {sorted(self.GCM_ALLOWED_COLLECTORS)}"
            )

    def _resolve_gcm_provenance(self) -> Dict[str, str]:
        code = (
            "import importlib.metadata as m, json, pathlib; "
            "d=m.distribution('gcm'); "
            "u=d.read_text('direct_url.json') or '{}'; "
            "print(json.dumps({'version': d.version, 'direct_url': json.loads(u)}))"
        )
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {}
        direct_url = parsed.get("direct_url", {})
        vcs_info = direct_url.get("vcs_info", {})
        return {
            "version": str(parsed.get("version", "")),
            "url": str(direct_url.get("url", "")),
            "commit_id": str(vcs_info.get("commit_id", "")),
        }

    def _ensure_gcm_installation(self) -> Dict[str, Any]:
        layout = self._gcm_artifact_layout()
        expected_repo = layout["repo"]
        expected_ref = layout["ref"]
        strict = bool(self.gcm_config.get("strict", False))

        existing = self._resolve_gcm_provenance()
        has_expected = (
            existing.get("url", "").startswith(expected_repo)
            and existing.get("commit_id", "").startswith(expected_ref)
        )

        install_log = {"installed": False, "provenance": existing}
        if has_expected:
            return install_log

        install_target = f"git+{expected_repo}.git@{expected_ref}"
        pip_cmd = ["python3", "-m", "pip", "install", install_target]
        result = subprocess.run(
            pip_cmd, capture_output=True, text=True, timeout=300, check=False
        )
        install_log["installed"] = result.returncode == 0
        install_log["install_stdout"] = result.stdout[-4000:]
        install_log["install_stderr"] = result.stderr[-4000:]

        refreshed = self._resolve_gcm_provenance()
        install_log["provenance"] = refreshed
        now_expected = (
            refreshed.get("url", "").startswith(expected_repo)
            and refreshed.get("commit_id", "").startswith(expected_ref)
        )
        if strict and not now_expected:
            raise RuntimeError(
                "GCM strict mode: installed package provenance does not match "
                f"{expected_repo}@{expected_ref}"
            )
        return install_log

    def validate(self) -> bool:
        """Validate SLURM commands are available locally."""
        # Check required SLURM CLI tools
        for tool in self.REQUIRED_TOOLS:
            result = subprocess.run(
                ["which", tool], capture_output=True, timeout=5
            )
            if result.returncode != 0:
                self.console.print(
                    f"[red]✗ Required tool not found: {tool}[/red]\n"
                    f"[yellow]Make sure you are on a SLURM login node[/yellow]"
                )
                return False

        # Verify we can query SLURM cluster
        result = subprocess.run(["sinfo", "-h"], capture_output=True, timeout=10)
        if result.returncode != 0:
            self.console.print("[red]✗ Cannot query SLURM (sinfo failed)[/red]")
            return False

        # Validate configuration
        if self.nodes < 1:
            self.console.print(f"[red]✗ Invalid nodes: {self.nodes}[/red]")
            return False

        if self.gpus_per_node < 1:
            self.console.print(f"[red]✗ Invalid GPUs per node: {self.gpus_per_node}[/red]")
            return False

        self.console.print("[green]✓ SLURM environment validated[/green]")
        return True

    def _validate_cli_availability(self) -> bool:
        """
        Validate madengine is available before job submission.
        
        Compute nodes inherit the submission environment, so madengine
        must be available in PATH on the submission node.
        
        Returns:
            bool: True if madengine is available and functional
        """
        try:
            result = subprocess.run(
                ["madengine", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )
            if result.returncode == 0:
                version = result.stdout.strip() or "unknown"
                self.console.print(
                    f"[green]✓[/green] madengine available: [cyan]{version}[/cyan]"
                )
                
                # Show path for transparency
                which_result = subprocess.run(
                    ["which", "madengine"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if which_result.returncode == 0:
                    cli_path = which_result.stdout.strip()
                    self.console.print(f"  Path: [dim]{cli_path}[/dim]")
                
                return True
            else:
                self.console.print(
                    "[red]✗ madengine found but returned error[/red]"
                )
                if result.stderr:
                    self.console.print(f"  Error: {result.stderr.strip()}")
                return False
                
        except FileNotFoundError:
            self.console.print(
                "\n[red]✗ ERROR: madengine not found[/red]\n"
            )
            self.console.print(
                "[yellow]Compute nodes need madengine in PATH.[/yellow]\n"
                "\n[bold]To fix:[/bold]\n"
                "  1. Activate virtual environment: [cyan]source venv/bin/activate[/cyan]\n"
                "  2. Install madengine:\n"
                "     • Development: [cyan]pip install -e .[/cyan]\n"
                "     • Production:  [cyan]pip install madengine[/cyan]\n"
                "  3. Verify: [cyan]madengine --version[/cyan]\n"
            )
            return False
        except subprocess.TimeoutExpired:
            self.console.print("[red]✗ madengine command timed out[/red]")
            return False
        except Exception as e:
            self.console.print(f"[red]✗ Error checking madengine: {e}[/red]")
            return False

    def prepare(self) -> bool:
        """Generate sbatch script from template."""
        # Validate environment BEFORE generating job scripts
        self.console.print("\n[bold]Validating submission environment...[/bold]")
        if not self._validate_cli_availability():
            self.console.print(
                "\n[yellow]⚠ Tip: Compute nodes inherit your submission environment[/yellow]"
            )
            return False
        
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Get model info from manifest
            model_keys = list(self.manifest["built_models"].keys())
            if not model_keys:
                raise ValueError("No models in manifest")

            model_key = model_keys[0]
            model_info = self.manifest["built_models"][model_key]

            # Prepare template context
            context = self._prepare_template_context(model_info)

            # Render template
            template = self.jinja_env.get_template("job.sh.j2")
            script_content = template.render(**context)

            # Save script
            self.script_path = self.output_dir / f"madengine_{model_info['name']}.sh"
            self.script_path.write_text(script_content)
            self.script_path.chmod(0o755)

            self.console.print(
                f"[green]✓ Generated sbatch script: {self.script_path}[/green]"
            )
            return True

        except Exception as e:
            self.console.print(f"[red]✗ Failed to generate script: {e}[/red]")
            return False

    @staticmethod
    def _normalize_nodelist(nodelist: Optional[str]) -> Optional[str]:
        """Normalize nodelist to comma-separated without spaces for #SBATCH --nodelist."""
        if not nodelist or not nodelist.strip():
            return None
        return ",".join(n.strip() for n in nodelist.split(",") if n.strip())

    def _prepare_template_context(self, model_info: Dict) -> Dict[str, Any]:
        """Prepare context for Jinja2 template rendering."""
        # Use hierarchical GPU resolution: runtime > deployment > model > default
        additional_context = self.config.additional_context.copy()
        additional_context["slurm"] = self.slurm_config
        resolved_gpus_per_node = resolve_runtime_gpus(model_info, additional_context)
        
        # Extract launcher configuration
        launcher_type = self.distributed_config.get("launcher", "torchrun")  # Default to torchrun
        
        # Normalize launcher based on deployment type and validity
        launcher_type = normalize_launcher(launcher_type, "slurm")
        
        nnodes = self.distributed_config.get("nnodes", self.nodes)
        nproc_per_node = self.distributed_config.get("nproc_per_node", resolved_gpus_per_node)
        master_port = self.distributed_config.get("port", 29500)
        
        # Apply multi-node profiling logic if tools are configured
        tools = additional_context.get("tools", [])
        if nnodes > 1 and tools:
            # Configure multi-node profiling (handles rocprofv3 detection and tool upgrades)
            # Create a simple logger wrapper for configure_multi_node_profiling
            class ConsoleLogger:
                def __init__(self, console):
                    self.console = console
                def info(self, msg):
                    self.console.print(f"[cyan]{msg}[/cyan]")
                def warning(self, msg):
                    self.console.print(f"[yellow]{msg}[/yellow]")
                def debug(self, msg):
                    pass  # Skip debug messages in console
            
            profiling_config = configure_multi_node_profiling(
                nnodes=nnodes,
                tools_config=tools,
                logger=ConsoleLogger(self.console)
            )
            
            if profiling_config["enabled"]:
                tools = profiling_config["tools"]
            else:
                # rocprofv3 not available - skip profiling for multi-node
                tools = []
            
            # Update tools in additional_context
            additional_context["tools"] = tools
        
        # Generate launcher-specific command
        launcher_command = self._generate_launcher_command(
            launcher_type=launcher_type,
            nnodes=nnodes,
            nproc_per_node=nproc_per_node,
            master_port=master_port,
            model_name=model_info.get("name", "") or "",
        )
        
        return {
            "model_name": model_info["name"],
            "manifest_file": os.path.abspath(self.config.manifest_file),
            "partition": self.partition,
            "nodes": self.nodes,
            "gpus_per_node": resolved_gpus_per_node,  # Use resolved GPU count
            "time_limit": self.time_limit,
            "output_dir": str(self.output_dir),
            "master_port": master_port,
            "distributed_backend": self.distributed_config.get("backend", "nccl"),
            "network_interface": self.slurm_config.get("network_interface"),
            "exclusive": self.slurm_config.get("exclusive", True),
            "exclude": self.slurm_config.get("exclude"),
            "constraint": self.slurm_config.get("constraint"),
            "nodelist": self._normalize_nodelist(self.slurm_config.get("nodelist")),
            "qos": self.slurm_config.get("qos"),
            "account": self.slurm_config.get("account"),
            "modules": self.slurm_config.get("modules", []),
            "env_vars": self.config.additional_context.get("env_vars", {}),
            "shared_workspace": self.slurm_config.get("shared_workspace"),
            "shared_data": self.config.additional_context.get("shared_data"),
            "results_dir": self.slurm_config.get("results_dir"),
            "timeout": self.config.timeout,
            "live_output": self.config.additional_context.get("live_output", False),
            "tags": " ".join(model_info.get("tags", [])),
            "multiple_results": model_info.get("multiple_results"),
            "credential_file": "credential.json"
            if Path("credential.json").exists()
            else None,
            "data_file": "data.json" if Path("data.json").exists() else None,
            # Launcher configuration
            "launcher_type": launcher_type,
            "launcher_command": launcher_command,
            "nnodes": nnodes,
            "nproc_per_node": nproc_per_node,
            # Profiling tools (processed for multi-node compatibility)
            "tools": tools,
            # Cluster feature flags
            "rdma_enabled": self.rdma_config.get("enabled", False),
            "rdma_strict": self.rdma_config.get("strict", False),
            "rdma_mode": self.rdma_config.get("mode", "recommend"),
            "rdma_apply_env": self.rdma_config.get("apply_env", True),
            "rdma_artifact_name": self.rdma_config.get(
                "artifact_name", "rdma_recommendation.json"
            ),
        }

    def _generate_launcher_command(
        self,
        launcher_type: str,
        nnodes: int,
        nproc_per_node: int,
        master_port: int,
        model_name: str = "",
    ) -> str:
        """
        Generate launcher-specific command based on launcher type.
        
        Follows k8s pattern: different launchers have different command generation.
        
        Args:
            launcher_type: Type of launcher (torchrun, vllm, sglang, deepspeed, etc.)
            nnodes: Number of nodes
            nproc_per_node: GPUs per node
            master_port: Master communication port
            
        Returns:
            Launcher-specific environment setup and command string
        """
        if launcher_type == "torchrun":
            return self._generate_torchrun_command(nnodes, nproc_per_node, master_port)
        elif launcher_type == "vllm":
            return self._generate_vllm_command(nnodes, nproc_per_node, master_port)
        elif launcher_type == "sglang":
            return self._generate_sglang_command(nnodes, nproc_per_node, master_port)
        elif launcher_type == "sglang-disagg" or launcher_type == "sglang_disagg":
            return self._generate_sglang_disagg_command(nnodes, nproc_per_node, master_port)
        elif launcher_type == "deepspeed":
            return self._generate_deepspeed_command(nnodes, nproc_per_node, master_port)
        elif launcher_type == "megatron":
            return self._generate_megatron_command(nnodes, nproc_per_node, master_port)
        elif launcher_type == "torchtitan":
            return self._generate_torchtitan_command(nnodes, nproc_per_node, master_port)
        elif launcher_type == "primus":
            return self._generate_primus_command(
                nnodes, nproc_per_node, master_port, model_name=model_name
            )
        else:
            # For unknown launchers, provide basic environment variables
            # and let the model script handle launcher invocation
            self.console.print(
                f"[yellow]Warning: Unknown launcher type '{launcher_type}'. "
                f"Using basic environment setup.[/yellow]"
            )
            return self._generate_basic_env_command(nnodes, nproc_per_node, master_port)

    def _generate_torchrun_command(
        self, nnodes: int, nproc_per_node: int, master_port: int
    ) -> str:
        """
        Generate torchrun launcher command for SLURM.
        
        For single-node (nnodes=1): Uses standalone mode
        For multi-node (nnodes>1): Uses distributed mode with SLURM environment
        
        Args:
            nnodes: Number of nodes
            nproc_per_node: GPUs per node
            master_port: Master port
            
        Returns:
            MAD_MULTI_NODE_RUNNER environment variable setup
        """
        if nnodes == 1:
            return f'export MAD_MULTI_NODE_RUNNER="torchrun --standalone --nproc_per_node={nproc_per_node}"'
        else:
            # Multi-node: Build command with SLURM_PROCID for node_rank
            return f'''# Multi-node torchrun setup
export MAD_MULTI_NODE_RUNNER="torchrun --nnodes={nnodes} --nproc_per_node={nproc_per_node} --node_rank=${{NODE_RANK}} --master_addr=${{MASTER_ADDR}} --master_port={master_port}"'''

    def _generate_vllm_command(
        self, nnodes: int, nproc_per_node: int, master_port: int
    ) -> str:
        """
        Generate vLLM launcher environment variables.
        
        vLLM manages its own process spawning - no torchrun needed.
        Model script directly invokes vLLM with tensor/pipeline parallelism.
        
        Args:
            nnodes: Number of nodes
            nproc_per_node: GPUs per node
            master_port: Master port
            
        Returns:
            Environment variable setup for vLLM
        """
        if nnodes == 1:
            return f'''# vLLM single-node setup (Tensor Parallelism)
export VLLM_TENSOR_PARALLEL_SIZE={nproc_per_node}
export VLLM_PIPELINE_PARALLEL_SIZE=1
export VLLM_DISTRIBUTED_BACKEND="auto"
# vLLM handles its own process management - no MAD_MULTI_NODE_RUNNER needed'''
        else:
            # One vLLM serve per node (TP only on that node), no shared Ray = data parallelism
            return f'''# vLLM multi-node setup (data parallel: one serve per node, TP only)
export VLLM_TENSOR_PARALLEL_SIZE={nproc_per_node}
export VLLM_PIPELINE_PARALLEL_SIZE=1
export VLLM_DISTRIBUTED_BACKEND="none"
# vLLM handles its own process management - no MAD_MULTI_NODE_RUNNER needed'''

    def _generate_sglang_command(
        self, nnodes: int, nproc_per_node: int, master_port: int
    ) -> str:
        """
        Generate SGLang launcher environment variables.
        
        SGLang similar to vLLM - manages its own process spawning.
        
        Args:
            nnodes: Number of nodes
            nproc_per_node: GPUs per node
            master_port: Master port
            
        Returns:
            Environment variable setup for SGLang
        """
        if nnodes == 1:
            return f'''# SGLang single-node setup (Tensor Parallelism)
export SGLANG_TENSOR_PARALLEL_SIZE={nproc_per_node}
export SGLANG_PIPELINE_PARALLEL_SIZE=1
# SGLang handles its own process management - no MAD_MULTI_NODE_RUNNER needed'''
        else:
            # One SGLang serve per node (TP only on that node), no cross-node coordination = data parallel
            return f'''# SGLang multi-node setup (data parallel: one serve per node, TP only)
export SGLANG_TENSOR_PARALLEL_SIZE={nproc_per_node}
export SGLANG_PIPELINE_PARALLEL_SIZE=1
# SGLang handles its own process management - no MAD_MULTI_NODE_RUNNER needed'''

    def _generate_sglang_disagg_command(
        self, nnodes: int, nproc_per_node: int, master_port: int
    ) -> str:
        """
        Generate SGLang Disaggregated launcher environment for SLURM.
        
        SGLang Disaggregated Architecture:
        - Node 0: Proxy (load balancer)
        - Nodes 1 to xP: Prefill nodes
        - Nodes xP+1 to xP+yD: Decode nodes
        
        Minimum cluster: 3 nodes (1 proxy + 1 prefill + 1 decode)
        
        Args:
            nnodes: Total number of nodes (must be >= 3)
            nproc_per_node: GPUs per node (tensor parallel size)
            master_port: Master port for coordination
            
        Returns:
            Environment setup with node role assignment
            
        Raises:
            ValueError: If nnodes < 3 (minimum for disagg)
        """
        if nnodes < 3:
            raise ValueError(
                f"SGLang Disaggregated requires minimum 3 nodes "
                f"(1 proxy + 1 prefill + 1 decode), got {nnodes}"
            )
        
        # Check if custom split is specified in additional_context
        sglang_disagg_config = self.config.additional_context.get("distributed", {}).get("sglang_disagg", {})
        prefill_nodes = sglang_disagg_config.get("prefill_nodes")
        decode_nodes = sglang_disagg_config.get("decode_nodes")
        
        if prefill_nodes is not None and decode_nodes is not None:
            # User specified custom split - validate
            if prefill_nodes < 1 or decode_nodes < 1:
                raise ValueError(
                    f"SGLang Disaggregated requires at least 1 prefill and 1 decode node, "
                    f"got prefill={prefill_nodes}, decode={decode_nodes}"
                )
            if prefill_nodes + decode_nodes + 1 != nnodes:
                raise ValueError(
                    f"Custom split validation failed: "
                    f"prefill_nodes ({prefill_nodes}) + decode_nodes ({decode_nodes}) + 1 proxy "
                    f"must equal nnodes ({nnodes}), but got {prefill_nodes + decode_nodes + 1}"
                )
            xP = prefill_nodes
            yD = decode_nodes
        else:
            # Default split: use golden ratio for prefill/decode
            # For N total nodes: 1 proxy + ~40% prefill + ~60% decode
            xP = max(1, (nnodes - 1) * 2 // 5)  # ~40% of worker nodes
            yD = nnodes - 1 - xP  # remaining nodes
        
        return f'''# SGLang Disaggregated multi-node setup
# ============================================
# Cluster Configuration:
#   Total Nodes: {nnodes}
#   Proxy: 1 node (NODE_RANK=0)
#   Prefill: {xP} nodes (NODE_RANK=1 to {xP})
#   Decode: {yD} nodes (NODE_RANK={xP+1} to {nnodes-1})
# ============================================

# Export cluster topology
export SGLANG_DISAGG_MODE="enabled"
export SGLANG_DISAGG_PREFILL_NODES={xP}
export SGLANG_DISAGG_DECODE_NODES={yD}
export SGLANG_DISAGG_TOTAL_NODES={nnodes}
export SGLANG_TP_SIZE={nproc_per_node}

# Master coordination
export MASTER_PORT={master_port}

# Build node IP list from SLURM
SLURM_NODE_IPS=$(scontrol show hostname ${{SLURM_JOB_NODELIST}} | while read node; do
    getent hosts "$node" | awk '{{print $1}}'
done | tr '\\n' ',' | sed 's/,$//')

export SGLANG_NODE_IPS="$SLURM_NODE_IPS"
export SGLANG_NODE_RANK=${{SLURM_PROCID}}

echo "=========================================="
echo "SGLang Disaggregated Cluster Info"
echo "=========================================="
echo "Node Rank: $SGLANG_NODE_RANK"
echo "Node IPs: $SGLANG_NODE_IPS"
echo "Prefill Nodes: {xP}"
echo "Decode Nodes: {yD}"
echo "TP Size: {nproc_per_node}"
echo "=========================================="

# No MAD_MULTI_NODE_RUNNER - SGLang disagg handles process management
# Model script should detect SGLANG_DISAGG_MODE and launch appropriately'''

    def _generate_deepspeed_command(
        self, nnodes: int, nproc_per_node: int, master_port: int
    ) -> str:
        """
        Generate DeepSpeed launcher command.
        
        DeepSpeed has its own launcher similar to torchrun.
        
        Args:
            nnodes: Number of nodes
            nproc_per_node: GPUs per node
            master_port: Master port
            
        Returns:
            MAD_MULTI_NODE_RUNNER with deepspeed launcher
        """
        if nnodes == 1:
            return f'''# DeepSpeed single-node setup
export MAD_MULTI_NODE_RUNNER="deepspeed --num_gpus={nproc_per_node}"'''
        else:
            return f'''# DeepSpeed multi-node setup
# Generate hostfile dynamically from SLURM
cat > /tmp/deepspeed_hostfile_${{SLURM_JOB_ID}}.txt << EOF
$(scontrol show hostnames $SLURM_JOB_NODELIST | awk -v slots={nproc_per_node} '{{print $1" slots="slots}}')
EOF
export MAD_MULTI_NODE_RUNNER="deepspeed --hostfile=/tmp/deepspeed_hostfile_${{SLURM_JOB_ID}}.txt --master_addr=${{MASTER_ADDR}} --master_port={master_port}"'''

    def _generate_megatron_command(
        self, nnodes: int, nproc_per_node: int, master_port: int
    ) -> str:
        """
        Generate Megatron-LM launcher command.
        
        Megatron-LM typically uses torchrun but with specific environment variables.
        
        Args:
            nnodes: Number of nodes
            nproc_per_node: GPUs per node
            master_port: Master port
            
        Returns:
            MAD_MULTI_NODE_RUNNER with megatron-specific setup
        """
        # Megatron uses torchrun with Megatron-Core standard environment variables
        if nnodes == 1:
            return f'''# Megatron-LM single-node setup
export TENSOR_MODEL_PARALLEL_SIZE={min(nproc_per_node, 8)}
export PIPELINE_MODEL_PARALLEL_SIZE=1
export CONTEXT_PARALLEL_SIZE=1
export MAD_MULTI_NODE_RUNNER="torchrun --standalone --nproc_per_node={nproc_per_node}"'''
        else:
            return f'''# Megatron-LM multi-node setup
export TENSOR_MODEL_PARALLEL_SIZE={nproc_per_node}
export PIPELINE_MODEL_PARALLEL_SIZE={nnodes}
export CONTEXT_PARALLEL_SIZE=1
export MAD_MULTI_NODE_RUNNER="torchrun --nnodes={nnodes} --nproc_per_node={nproc_per_node} --node_rank=${{NODE_RANK}} --master_addr=${{MASTER_ADDR}} --master_port={master_port}"'''

    def _generate_torchtitan_command(
        self, nnodes: int, nproc_per_node: int, master_port: int
    ) -> str:
        """
        Generate TorchTitan launcher command for SLURM.
        
        TorchTitan is a PyTorch native platform for LLM pre-training that uses
        torchrun as its underlying launcher but requires additional configuration
        for multi-dimensional parallelism (FSDP2, Tensor Parallel, Pipeline Parallel).
        
        Key TorchTitan features:
        - Uses TOML configuration files for training setup
        - Supports FSDP2, Tensor Parallel, Pipeline Parallel, Context Parallel
        - Built on top of torchrun for distributed coordination
        
        For single-node (nnodes=1): Uses standalone torchrun mode
        For multi-node (nnodes>1): Uses distributed torchrun with SLURM environment
        
        Args:
            nnodes: Number of nodes
            nproc_per_node: GPUs per node
            master_port: Master port
            
        Returns:
            MAD_MULTI_NODE_RUNNER with torchtitan-specific setup
        """
        if nnodes == 1:
            return f'''# TorchTitan single-node setup
# TorchTitan uses torchrun as underlying launcher
export TORCHTITAN_TENSOR_PARALLEL_SIZE={nproc_per_node}
export TORCHTITAN_PIPELINE_PARALLEL_SIZE=1
export MAD_MULTI_NODE_RUNNER="torchrun --standalone --nproc_per_node={nproc_per_node}"'''
        else:
            # Multi-node: Use torchrun with SLURM coordination
            # TorchTitan will detect multi-node and enable appropriate parallelism
            return f'''# TorchTitan multi-node setup
# Configure multi-dimensional parallelism for TorchTitan
export TORCHTITAN_TENSOR_PARALLEL_SIZE={nproc_per_node}
export TORCHTITAN_PIPELINE_PARALLEL_SIZE={nnodes}
export TORCHTITAN_FSDP_ENABLED=1
export TORCHTITAN_CONTEXT_PARALLEL_SIZE=1

# Use torchrun as launcher (TorchTitan built on top of it)
export MAD_MULTI_NODE_RUNNER="torchrun --nnodes={nnodes} --nproc_per_node={nproc_per_node} --node_rank=${{NODE_RANK}} --master_addr=${{MASTER_ADDR}} --master_port={master_port}"'''

    def _generate_primus_command(
        self,
        nnodes: int,
        nproc_per_node: int,
        master_port: int,
        model_name: str = "",
    ) -> str:
        """
        Generate Primus launcher environment for SLURM.

        Primus (Megatron-LM, TorchTitan, Jax/MaxText) runs via model script that calls
        run_pretrain.sh; NNODES, NODE_RANK, MASTER_ADDR, etc. are set by the job script.
        We only export PRIMUS_CONFIG_PATH and optional PRIMUS_CLI_EXTRA. No MAD_MULTI_NODE_RUNNER.
        """
        primus_cfg = merged_primus_config(
            self.manifest if isinstance(getattr(self, "manifest", None), dict) else None,
            self.config.additional_context,
        )
        config_path = primus_cfg.get("config_path", "exp_pretrain.yaml")
        cli_extra = primus_cfg.get("cli_extra", "")
        # Safe shell quoting for config_path and cli_extra
        config_path_quoted = config_path.replace('"', '\\"')
        lines = [f'# Primus launcher (model script runs run_pretrain.sh)',
                 f'export PRIMUS_CONFIG_PATH="{config_path_quoted}"']
        if (cli_extra or "").strip():
            cli_extra_quoted = cli_extra.replace('"', '\\"')
            lines.append(f'export PRIMUS_CLI_EXTRA="{cli_extra_quoted}"')
        backend_override = (primus_cfg.get("backend") or "").strip()
        backend_effective = backend_override
        if not backend_effective and model_name:
            backend_effective = infer_primus_backend_from_model_name(model_name) or ""
        if backend_effective:
            backend_quoted = backend_effective.replace('"', '\\"')
            lines.append(f'export BACKEND="{backend_quoted}"')
        return "\n".join(lines)

    def _generate_basic_env_command(
        self, nnodes: int, nproc_per_node: int, master_port: int
    ) -> str:
        """
        Generate basic environment variables for unknown launchers.
        
        Provides standard distributed execution environment variables
        and lets the model script handle launcher invocation.
        
        Args:
            nnodes: Number of nodes
            nproc_per_node: GPUs per node
            master_port: Master port
            
        Returns:
            Basic environment variable setup
        """
        return f'''# Basic distributed environment (custom launcher)
export NNODES={nnodes}
export NPROC_PER_NODE={nproc_per_node}
export MASTER_PORT={master_port}
# Model script should handle launcher invocation'''

    def _run_gcm_health_preflight(self) -> None:
        if not self._gcm_enabled():
            return
        enabled_platforms = self.gcm_config.get("enabled_platforms", ["slurm"])
        if "slurm" not in enabled_platforms:
            self.console.print(
                "[dim]GCM configured but slurm not enabled in enabled_platforms; skipping.[/dim]"
            )
            return

        strict = bool(self.gcm_config.get("strict", False))
        layout = self._gcm_artifact_layout()
        artifact_dir = Path(layout["dir"])
        artifact_dir.mkdir(parents=True, exist_ok=True)
        summary_path = artifact_dir / layout["health_summary_json"]
        raw_log_path = artifact_dir / layout["health_raw_log"]
        provenance_path = artifact_dir / layout["provenance_json"]

        self._validate_gcm_allowlist()
        install_log = self._ensure_gcm_installation()
        provenance_path.write_text(
            json.dumps(
                {
                    "expected_repo": layout["repo"],
                    "expected_ref": layout["ref"],
                    "install": install_log,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        gcm_bin = shutil.which("gcm")
        health_bin = shutil.which("health_checks")
        missing = []
        if not gcm_bin:
            missing.append("gcm")
        if not health_bin:
            missing.append("health_checks")
        if missing:
            msg = f"Missing GCM binaries: {missing}"
            if strict:
                raise RuntimeError(msg)
            self.console.print(f"[yellow]{msg}; continuing (strict=false).[/yellow]")
            summary_path.write_text(
                json.dumps({"status": "skipped", "reason": msg}, indent=2),
                encoding="utf-8",
            )
            return

        checks = self.gcm_config.get("health_checks", ["check-hca", "check-ibstat"])
        per_check: List[Dict[str, Any]] = []
        raw_chunks: List[str] = []
        timeout_sec = int(self.gcm_config.get("timeout_sec", 60))
        for check in checks:
            cmd = [
                "srun",
                "--nodes",
                str(self.nodes),
                "--ntasks",
                str(self.nodes),
                "--ntasks-per-node",
                "1",
                "health_checks",
                check,
            ]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    check=False,
                )
                returncode = result.returncode
                stdout = result.stdout
                stderr = result.stderr
            except subprocess.TimeoutExpired as exc:
                returncode = 124
                stdout = exc.stdout or ""
                stderr = (exc.stderr or "") + "\nhealth check timeout"
            per_check.append(
                {
                    "check": check,
                    "returncode": returncode,
                    "passed": returncode == 0,
                }
            )
            raw_chunks.append(f"## {check}\n{stdout}\n{stderr}\n")

        raw_log_path.write_text("\n".join(raw_chunks), encoding="utf-8")
        failed = [c for c in per_check if not c["passed"]]
        summary = {
            "status": "failed" if failed else "passed",
            "strict": strict,
            "checks": per_check,
            "artifact_raw_log": str(raw_log_path),
            "artifact_provenance": str(provenance_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if failed and strict:
            raise RuntimeError(
                "GCM strict preflight failed: "
                + ", ".join(c["check"] for c in failed)
            )
        if failed:
            self.console.print(
                "[yellow]GCM preflight reported failures; continuing (strict=false).[/yellow]"
            )

    def deploy(self) -> DeploymentResult:
        """Submit sbatch script to SLURM scheduler (locally)."""
        if not self.script_path or not self.script_path.exists():
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id="",
                message="Script not generated. Run prepare() first.",
            )

        # ==================== PREFLIGHT NODE SELECTION ====================
        # For single- and multi-node jobs, check for clean nodes and exclude bad ones.
        # Single-node: we still run the check so bad nodes (e.g. Docker broken) get excluded;
        # we never gate submission for nodes==1 so behavior stays backward compatible.
        # Health-check srun invocations create SLURM jobs; we cancel them after preflight.
        enable_preflight = self.slurm_config.get("enable_node_check", True)
        auto_cleanup = self.slurm_config.get("auto_cleanup_nodes", False)
        allow_submit_without_clean = self.slurm_config.get("allow_submit_without_clean_nodes", False)
        clean_nodes: List[str] = []
        health_check_job_name: Optional[str] = None

        if enable_preflight and self.nodes >= 1 and not self.slurm_config.get("nodelist"):
            try:
                selector = SlurmNodeSelector(
                    console=self.console,
                    auto_cleanup=auto_cleanup,
                    verbose=self.slurm_config.get("verbose_node_check", False),
                )
                clean_nodes, updated_exclude = selector.select_nodes(
                    partition=self.partition,
                    nodes_needed=self.nodes,
                    exclude=self.slurm_config.get("exclude"),
                    constraint=self.slurm_config.get("constraint"),
                )
                health_check_job_name = getattr(selector, "_health_check_job_name", None)

                # Update exclude list if we found dirty/unreachable/unknown nodes
                if updated_exclude and updated_exclude != self.slurm_config.get("exclude", ""):
                    self.console.print(
                        f"[dim]Updated exclude list for sbatch: {updated_exclude}[/dim]\n"
                    )
                    self.slurm_config["exclude"] = updated_exclude
                    self.prepare()

                # Gate: do not submit if not enough clean nodes (multi-node only; single-node always allowed)
                if (
                    self.nodes > 1
                    and not allow_submit_without_clean
                    and len(clean_nodes) < self.nodes
                ):
                    SlurmNodeSelector.cancel_health_check_jobs(health_check_job_name, self.console)
                    return DeploymentResult(
                        status=DeploymentStatus.FAILED,
                        deployment_id="",
                        message=(
                            f"Not enough clean nodes: need {self.nodes}, found {len(clean_nodes)}. "
                            "Set slurm.allow_submit_without_clean_nodes=true to submit anyway."
                        ),
                    )

                # When we have enough clean nodes, pin the job to them via nodelist
                if len(clean_nodes) >= self.nodes:
                    nodelist_str = ",".join(clean_nodes[: self.nodes])
                    self.slurm_config["nodelist"] = nodelist_str
                    self.console.print(f"[dim]Using nodelist: {nodelist_str}[/dim]\n")
                    self.prepare()
            except Exception as e:
                self.console.print(
                    f"[yellow]⚠ Node health check failed: {e}[/yellow]"
                )
                self.console.print("[dim]Continuing with job submission[/dim]\n")
            finally:
                # Always cancel health-check jobs so they do not stay in the queue
                SlurmNodeSelector.cancel_health_check_jobs(health_check_job_name, self.console)
        # ==================== END PREFLIGHT ====================

        try:
            self._run_gcm_health_preflight()
        except Exception as e:
            strict = bool(self.gcm_config.get("strict", False))
            if strict:
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    deployment_id="",
                    message=f"GCM preflight failed: {e}",
                )
            self.console.print(
                f"[yellow]⚠ GCM preflight warning: {e} (strict=false, continuing)[/yellow]"
            )

        try:
            # Submit job to SLURM (runs locally on login node)
            result = subprocess.run(
                ["sbatch", str(self.script_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                # Parse job ID: "Submitted batch job 12345"
                job_id = result.stdout.strip().split()[-1]

                self.console.print(f"[green]✓ Submitted SLURM job: {job_id}[/green]")
                self.console.print(f"  Nodes: {self.nodes} x {self.gpus_per_node} GPUs")
                self.console.print(f"  Partition: {self.partition}")

                return DeploymentResult(
                    status=DeploymentStatus.SUCCESS,
                    deployment_id=job_id,
                    message=f"SLURM job {job_id} submitted successfully",
                    logs_path=str(self.output_dir),
                )
            else:
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    deployment_id="",
                    message=f"sbatch failed: {result.stderr}",
                )

        except subprocess.TimeoutExpired:
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id="",
                message="sbatch submission timed out",
            )
        except Exception as e:
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id="",
                message=f"Deployment error: {str(e)}",
            )

    def monitor(self, deployment_id: str) -> DeploymentResult:
        """Check SLURM job status (locally)."""
        try:
            # Query job status using squeue (runs locally)
            result = subprocess.run(
                ["squeue", "-j", deployment_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0 or not result.stdout.strip():
                # Job not found in queue - likely completed or failed
                return self._check_job_completion(deployment_id)

            status = result.stdout.strip().upper()
            
            # Check if live output is enabled
            live_output = self.config.additional_context.get("live_output", False)

            # Stream work node output if live_output is enabled and job is running
            if status == "RUNNING" and live_output:
                self._stream_job_output(deployment_id)

            if status in ["RUNNING", "PENDING", "CONFIGURING", "COMPLETING"]:
                # COMPLETING is a transient state before COMPLETED - treat as running
                return DeploymentResult(
                    status=DeploymentStatus.RUNNING,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} is {status.lower()}",
                )
            elif status in ["COMPLETED"]:
                # Show final output only if live_output is enabled
                if live_output:
                    self._stream_job_output(deployment_id, final=True)
                else:
                    self._show_log_summary(deployment_id, success=True)
                return DeploymentResult(
                    status=DeploymentStatus.SUCCESS,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} completed successfully",
                )
            else:  # FAILED, CANCELLED, TIMEOUT, NODE_FAIL, etc.
                # Show output on failure or show summary
                if live_output:
                    self._stream_job_output(deployment_id, final=True)
                else:
                    self._show_log_summary(deployment_id, success=False)
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    deployment_id=deployment_id,
                    message=f"Job {deployment_id} {status.lower()}",
                )

        except Exception as e:
            self.console.print(f"[red]Monitor exception for job {deployment_id}: {e}[/red]")
            import traceback
            self.console.print(f"[dim red]{traceback.format_exc()}[/dim red]")
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id=deployment_id,
                message=f"Monitor error: {str(e)}",
            )

    def _stream_job_output(self, job_id: str, final: bool = False):
        """Stream output from SLURM job output file."""
        # Track last position read from output file
        if not hasattr(self, '_output_positions'):
            self._output_positions = {}
        
        # Find output file
        output_dir = str(self.output_dir)
        output_pattern = f"{output_dir}/madengine-*_{job_id}_*.out"
        
        try:
            import glob
            output_files = glob.glob(output_pattern)
            
            if not output_files:
                return  # Output file not created yet
            
            output_file = output_files[0]  # Use first match
            
            # Read new content from file
            try:
                with open(output_file, 'r') as f:
                    # Seek to last position
                    last_pos = self._output_positions.get(job_id, 0)
                    f.seek(last_pos)
                    
                    # Read new lines
                    new_content = f.read()
                    
                    if new_content:
                        # Print new output with prefix
                        for line in new_content.splitlines():
                            if line.strip():  # Skip empty lines
                                self.console.print(f"[dim cyan]│[/dim cyan] {line}")
                    
                    # Update position
                    self._output_positions[job_id] = f.tell()
                    
            except FileNotFoundError:
                pass  # File not ready yet
                
        except Exception as e:
            # Silently ignore streaming errors to not disrupt monitoring
            if final:
                self.console.print(f"[dim yellow]Note: Could not stream output: {e}[/dim yellow]")

    def _show_log_summary(self, job_id: str, success: bool = True):
        """Show a summary with pointers to log files instead of streaming verbose output."""
        output_dir = str(self.output_dir)
        
        try:
            import glob
            # Find output and error files for this job
            output_files = glob.glob(f"{output_dir}/madengine-*_{job_id}_*.out")
            error_files = glob.glob(f"{output_dir}/madengine-*_{job_id}_*.err")
            
            if output_files or error_files:
                status_symbol = "✓" if success else "✗"
                status_color = "green" if success else "red"
                
                self.console.print(f"[{status_color}]{status_symbol}[/{status_color}] SLURM job {job_id} logs saved to:")
                
                for out_file in output_files:
                    self.console.print(f"  [cyan]→[/cyan] Output: {out_file}")
                    
                for err_file in error_files:
                    # Check if error file has content
                    if os.path.exists(err_file) and os.path.getsize(err_file) > 0:
                        self.console.print(f"  [yellow]→[/yellow] Errors: {err_file}")
                
                if not success and error_files:
                    # Show last few lines of error file for failed jobs
                    for err_file in error_files:
                        if os.path.exists(err_file) and os.path.getsize(err_file) > 0:
                            self.console.print(f"\n[yellow]Last 10 lines of error log:[/yellow]")
                            try:
                                with open(err_file, 'r') as f:
                                    lines = f.readlines()
                                    for line in lines[-10:]:
                                        if line.strip():
                                            self.console.print(f"  {line.rstrip()}")
                            except Exception:
                                pass
                            break  # Only show first error file
            else:
                self.console.print(f"[dim yellow]Note: Log files for job {job_id} not found in {output_dir}[/dim yellow]")
                
        except Exception as e:
            self.console.print(f"[dim yellow]Note: Could not locate log files: {e}[/dim yellow]")

    def _check_job_completion(self, job_id: str) -> DeploymentResult:
        """Check completed job status using sacct (locally)."""
        try:
            result = subprocess.run(
                ["sacct", "-j", job_id, "-n", "-X", "-o", "State"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                status = result.stdout.strip().upper()
                self.console.print(f"[dim]SLURM job {job_id} final status: {status}[/dim]")
                
                # Check if live output is enabled
                live_output = self.config.additional_context.get("live_output", False)
                
                if "COMPLETED" in status:
                    # Show final output or summary based on live_output flag
                    if live_output:
                        self._stream_job_output(job_id, final=True)
                    else:
                        self._show_log_summary(job_id, success=True)
                    return DeploymentResult(
                        status=DeploymentStatus.SUCCESS,
                        deployment_id=job_id,
                        message=f"Job {job_id} completed successfully",
                    )
                else:
                    # Show output on failure or summary
                    if live_output:
                        self._stream_job_output(job_id, final=True)
                    else:
                        self._show_log_summary(job_id, success=False)
                    return DeploymentResult(
                        status=DeploymentStatus.FAILED,
                        deployment_id=job_id,
                        message=f"Job {job_id} failed: {status}",
                    )

            # Fallback - assume completed
            self.console.print(f"[dim yellow]Warning: Could not get status for job {job_id}, assuming success[/dim yellow]")
            return DeploymentResult(
                status=DeploymentStatus.SUCCESS,
                deployment_id=job_id,
                message=f"Job {job_id} completed (assumed)",
            )

        except Exception as e:
            self.console.print(f"[dim yellow]Warning: Exception checking job {job_id}: {e}[/dim yellow]")
            return DeploymentResult(
                status=DeploymentStatus.SUCCESS,
                deployment_id=job_id,
                message=f"Job {job_id} completed (status unavailable)",
            )

    def _build_perf_entry_from_aggregated(
        self,
        aggregated_record: Dict[str, Any],
        model_info: Dict[str, Any],
        build_info: Dict[str, Any],
        deployment_id: str,
    ) -> Dict[str, Any]:
        """
        Build a full run-details dict (same shape as container_runner create_run_details_dict)
        from an aggregated multi-node record or single-node parsed record, for use with
        update_perf_csv and update_perf_super_*.
        """
        from madengine.reporting.update_perf_csv import flatten_tags
        from madengine.utils.config_parser import ConfigParser

        launcher_type = self.distributed_config.get("launcher", "torchrun")
        launcher = normalize_launcher(launcher_type, "slurm")

        run_details = {
            "model": model_info.get("name", aggregated_record.get("model", "")),
            "n_gpus": str(aggregated_record.get("n_gpus", self.nodes * self.gpus_per_node)),
            "nnodes": str(aggregated_record.get("nnodes", self.nodes)),
            "gpus_per_node": str(aggregated_record.get("gpus_per_node", self.gpus_per_node)),
            "training_precision": model_info.get("training_precision", ""),
            "pipeline": get_pipeline(),
            "args": model_info.get("args", ""),
            "tags": model_info.get("tags", ""),
            "docker_file": build_info.get("dockerfile", ""),
            "base_docker": build_info.get("base_docker", ""),
            "docker_sha": build_info.get("docker_sha", ""),
            "docker_image": build_info.get("docker_image", ""),
            "git_commit": "",
            "machine_name": deployment_id,
            "deployment_type": self.DEPLOYMENT_TYPE,
            "launcher": launcher,
            "gpu_architecture": aggregated_record.get("gpu_architecture", ""),
            "performance": str(aggregated_record.get("performance", "")),
            "metric": aggregated_record.get("metric", ""),
            "relative_change": "",
            "status": aggregated_record.get("status", "SUCCESS"),
            "build_duration": build_info.get("build_duration", ""),
            "test_duration": aggregated_record.get("test_duration", ""),
            "dataname": aggregated_record.get("data_name", ""),
            "data_provider_type": aggregated_record.get("data_provider", ""),
            "data_size": "",
            "data_download_duration": "",
            "build_number": get_build_number(),
            "additional_docker_run_options": model_info.get("additional_docker_run_options", ""),
        }
        flatten_tags(run_details)

        # Configs for perf_super (optional)
        try:
            scripts_path = model_info.get("scripts", "")
            scripts_base_dir = scripts_base_dir_from(scripts_path)
            config_parser = ConfigParser(scripts_base_dir=scripts_base_dir)
            run_details["configs"] = config_parser.parse_and_load(
                model_info.get("args", ""), scripts_path
            )
        except Exception:
            run_details["configs"] = None

        return run_details

    def _build_common_info_dict(
        self,
        model_info: Dict[str, Any],
        build_info: Dict[str, Any],
        deployment_id: str,
        gpu_architecture: str = "",
    ) -> Dict[str, Any]:
        """Build common_info dict for update_perf_csv/update_perf_super (multiple_results path)."""
        from madengine.reporting.update_perf_csv import flatten_tags

        launcher_type = self.distributed_config.get("launcher", "torchrun")
        launcher = normalize_launcher(launcher_type, "slurm")
        total_gpus = self.nodes * self.gpus_per_node
        result = {
            "n_gpus": str(total_gpus),
            "nnodes": str(self.nodes),
            "gpus_per_node": str(self.gpus_per_node),
            "training_precision": model_info.get("training_precision", ""),
            "pipeline": get_pipeline(),
            "args": model_info.get("args", ""),
            "tags": model_info.get("tags", ""),
            "docker_file": build_info.get("dockerfile", ""),
            "base_docker": build_info.get("base_docker", ""),
            "docker_sha": build_info.get("docker_sha", ""),
            "docker_image": build_info.get("docker_image", ""),
            "git_commit": "",
            "machine_name": deployment_id,
            "deployment_type": self.DEPLOYMENT_TYPE,
            "launcher": launcher,
            "gpu_architecture": gpu_architecture,
            "relative_change": "",
            "build_duration": build_info.get("build_duration", ""),
            "test_duration": "",
            "dataname": model_info.get("data", ""),
            "data_provider_type": "",
            "data_size": "",
            "data_download_duration": "",
            "build_number": get_build_number(),
            "additional_docker_run_options": model_info.get("additional_docker_run_options", ""),
        }
        flatten_tags(result)
        return result

    def collect_results(self, deployment_id: str) -> Dict[str, Any]:
        """Collect performance results from SLURM output files.

        Option (1): slurm_results holds only collected inputs; login node builds one run
        record and runs the same reporting pipeline as local (perf_entry -> update_perf_csv
        / update_perf_super_*) so project root has a single cumulative perf for both local
        and distributed runs.
        """
        from madengine.reporting.update_perf_csv import update_perf_csv
        from madengine.reporting.update_perf_super import (
            update_perf_super_csv,
            update_perf_super_json,
        )

        session_start_row = self.config.additional_context.get("session_start_row")
        results = {
            "job_id": deployment_id,
            "nodes": self.nodes,
            "gpus_per_node": self.gpus_per_node,
            "perf_files": [],
            "logs": [],
            "successful_runs": [],
            "failed_runs": [],
            "session_start_row": session_start_row,
            "cluster_features": {},
        }
        if self._gcm_enabled():
            layout = self._gcm_artifact_layout()
            health_summary = Path(layout["dir"]) / layout["health_summary_json"]
            if health_summary.exists():
                try:
                    results["cluster_features"]["gcm_health"] = json.loads(
                        health_summary.read_text(encoding="utf-8")
                    )
                except json.JSONDecodeError:
                    results["cluster_features"]["gcm_health"] = {
                        "status": "error",
                        "error": "Invalid health summary JSON",
                    }

        model_keys = list(self.manifest.get("built_models") or {})
        model_key = model_keys[0] if model_keys else None
        # Use logical model name for job_dir so it matches the task script (which uses model_info["name"]).
        # built_models is keyed by image name; value has "name" = logical model name.
        built_models_dict = self.manifest.get("built_models") or {}
        model_info_for_path = built_models_dict.get(model_key, {}) if model_key else {}
        model_name_for_path = model_info_for_path.get("name", model_key or "unknown")
        model_name = model_key or "unknown"  # image key for build_info / model_info_for_entry lookups

        build_info = {}
        built_images = self.manifest.get("built_images") or {}
        if built_images:
            # First image or one keyed by model key (image name)
            if model_name in built_images:
                build_info = built_images[model_name]
            else:
                build_info = next(iter(built_images.values()), {})

        job_dir = self.output_dir / model_name_for_path / deployment_id
        job_dir.mkdir(parents=True, exist_ok=True)

        rdma_cfg = self.cluster_config.get("rdma", {})
        if rdma_cfg.get("enabled", False):
            rdma_artifact_name = rdma_cfg.get(
                "artifact_name", "rdma_recommendation.json"
            )
            rdma_entries: List[Dict[str, Any]] = []
            for node_dir in sorted(job_dir.glob("node_*")):
                artifact = node_dir / rdma_artifact_name
                if not artifact.exists():
                    continue
                try:
                    payload = json.loads(artifact.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    payload = {"status": "invalid_json"}
                rdma_entries.append(
                    {
                        "node": node_dir.name,
                        "artifact": str(artifact),
                        "status": payload.get("status", "unknown"),
                        "recommended_env": payload.get("recommended_env", {}),
                    }
                )
            if rdma_entries:
                results["cluster_features"]["rdma"] = {
                    "nodes": rdma_entries,
                    "artifact_name": rdma_artifact_name,
                }

        # Gather log content per node: from job_dir/node_N/ (new) or flat output_dir .out files
        per_node_log_contents: List[tuple] = []
        flat_out_files = sorted(self.output_dir.glob(f"madengine-*_{deployment_id}_*.out"))
        # Multi-node: only use explicit node logs (_node_N.out) to avoid also picking up
        # SBATCH %t output (madengine-*_<jobid>_0.out, _1.out), which would duplicate metrics.
        if self.nodes > 1:
            flat_out_files = [f for f in flat_out_files if "_node_" in f.name]
        results["logs"] = [str(f) for f in flat_out_files]

        for i, out_path in enumerate(flat_out_files):
            content = out_path.read_text(encoding="utf-8", errors="ignore")
            per_node_log_contents.append((i, content))

        # If we have node subdirs (multi-node job script wrote them), prefer stdout.out there
        for node_dir in sorted(job_dir.glob("node_*")):
            stdout_path = node_dir / "stdout.out"
            if stdout_path.exists():
                try:
                    idx = int(node_dir.name.replace("node_", ""))
                    if idx >= self.nodes:
                        continue  # ignore stale node_N dirs from previous runs
                    content = stdout_path.read_text(encoding="utf-8", errors="ignore")
                    # Replace or append for this index
                    per_node_log_contents = [
                        (n, c) for n, c in per_node_log_contents if n != idx
                    ]
                    per_node_log_contents.append((idx, content))
                    per_node_log_contents.sort(key=lambda x: x[0])
                except (ValueError, OSError):
                    pass

        # Multi-node: keep only log entries for actual node indices [0, nodes-1]
        if self.nodes > 1:
            per_node_log_contents = [(n, c) for n, c in per_node_log_contents if n < self.nodes]

        # Copy flat logs into job_dir/node_<task>/ for consistency if not already there.
        # Only create dirs for indices in [0, nodes-1] so we never create extra node_2, etc.
        for idx, content in per_node_log_contents:
            if idx >= self.nodes:
                continue
            node_subdir = job_dir / f"node_{idx}"
            node_subdir.mkdir(parents=True, exist_ok=True)
            if not (node_subdir / "stdout.out").exists():
                (node_subdir / "stdout.out").write_text(content)

        # Parse performance from each node's log
        all_parsed: List[Dict[str, Any]] = []
        for idx, content in sorted(per_node_log_contents, key=lambda x: x[0]):
            perf_data = self._parse_performance_from_log(content, model_name)
            if perf_data:
                all_parsed.append(perf_data)

        # Deduplicate by node_id so each node is only counted once (avoids double-counting
        # when multiple log sources exist for the same node, e.g. SBATCH vs node_*.out).
        per_node_metrics: List[Dict[str, Any]] = []
        if self.nodes > 1 and all_parsed:
            seen_node_ids: set = set()
            for m in all_parsed:
                nid = m.get("node_id")
                if nid is not None and nid in seen_node_ids:
                    continue
                if nid is not None:
                    seen_node_ids.add(nid)
                per_node_metrics.append(m)
                self.console.print(
                    f"[dim]  Parsed node: {m.get('performance')} "
                    f"{m.get('metric', '')} (node_id={m.get('node_id')})[/dim]"
                )
        else:
            per_node_metrics = all_parsed
            for m in per_node_metrics:
                self.console.print(
                    f"[dim]  Parsed node: {m.get('performance')} "
                    f"{m.get('metric', '')} (node_id={m.get('node_id')})[/dim]"
                )

        run_details_dict: Optional[Dict[str, Any]] = None
        model_info_for_entry = (self.manifest.get("built_models") or {}).get(
            model_key, {}
        ) if model_key else {}

        # Multiple results path: resolve CSV from job_dir/node_*, then cwd/run_directory
        mult_res = model_info_for_entry.get("multiple_results")
        if mult_res:
            resolved_csv: Optional[Path] = None
            if (job_dir / mult_res).is_file():
                resolved_csv = job_dir / mult_res
            else:
                for i in range(self.nodes):
                    candidate = job_dir / f"node_{i}" / mult_res
                    if candidate.is_file():
                        resolved_csv = candidate
                        break
            if not resolved_csv and Path(mult_res).is_file():
                resolved_csv = Path(mult_res)
            if not resolved_csv and Path("run_directory", mult_res).is_file():
                resolved_csv = Path("run_directory", mult_res)
            if resolved_csv:
                self._ensure_perf_csv_exists()
                gpu_arch = ""
                if per_node_metrics:
                    gpu_arch = per_node_metrics[0].get("gpu_architecture", "") or ""
                common_info = self._build_common_info_dict(
                    model_info_for_entry, build_info, deployment_id, gpu_arch
                )
                common_info_path = Path("common_info.json")
                with open(common_info_path, "w", encoding="utf-8") as f:
                    json.dump(common_info, f, indent=2)
                update_perf_csv(
                    perf_csv="perf.csv",
                    multiple_results=str(resolved_csv),
                    common_info=str(common_info_path),
                    model_name=model_info_for_entry.get("name", model_name),
                )
                scripts_path = model_info_for_entry.get("scripts", "")
                scripts_base_dir = scripts_base_dir_from(scripts_path)
                num_entries = update_perf_super_json(
                    perf_super_json="perf_super.json",
                    multiple_results=str(resolved_csv),
                    common_info=str(common_info_path),
                    model_name=model_info_for_entry.get("name", model_name),
                    scripts_base_dir=scripts_base_dir,
                )
                update_perf_super_csv(
                    perf_super_json="perf_super.json",
                    perf_super_csv="perf_super.csv",
                    num_entries=num_entries,
                )
                results["perf_files"] = [str(Path("perf.csv").resolve())]
                import csv as _csv
                try:
                    with open(resolved_csv, "r", encoding="utf-8", errors="ignore") as f:
                        reader = _csv.DictReader(f)
                        for row in reader:
                            row = {k.strip(): v for k, v in row.items() if k}
                            if row.get("performance") and row.get("metric"):
                                results["successful_runs"].append({
                                    "model": model_info_for_entry.get("name", "") + "_" + row.get("model", ""),
                                    "status": "SUCCESS",
                                    "performance": str(row.get("performance", "")),
                                    "metric": row.get("metric", ""),
                                    "duration": row.get("test_duration", ""),
                                    "gpu_arch": gpu_arch,
                                    "deployment": "slurm",
                                    "machine": deployment_id,
                                })
                except Exception:
                    pass
                self.console.print(
                    f"[green]✓ Updated perf.csv, perf_super.* from multiple_results (Docker-compatible)[/green]"
                )
                return self._attach_gcm_collector_summary(results, deployment_id)
            # multiple_results set but CSV not found: fall through to single-result path (may write FAILURE)

        if self.nodes > 1 and per_node_metrics:
            launcher_type = self.distributed_config.get("launcher", "torchrun")
            aggregated = self._aggregate_node_metrics(
                per_node_metrics, self.nodes, launcher_type
            )
            if aggregated and model_info_for_entry:
                run_details_dict = self._build_perf_entry_from_aggregated(
                    aggregated, model_info_for_entry, build_info, deployment_id
                )
                self.console.print(
                    f"[green]✓ Aggregated from {len(per_node_metrics)} nodes "
                    f"({aggregated.get('aggregation_method', '')}): "
                    f"{aggregated.get('performance')} {aggregated.get('metric', '')}[/green]"
                )
        elif self.nodes == 1 and per_node_metrics and model_info_for_entry:
            single = per_node_metrics[0]
            single_record = {
                "model": single.get("model", model_name),
                "n_gpus": self.gpus_per_node,
                "nnodes": 1,
                "gpus_per_node": self.gpus_per_node,
                "performance": single.get("performance"),
                "metric": single.get("metric", ""),
                "status": "SUCCESS",
                "test_duration": single.get("test_duration", ""),
                "gpu_architecture": single.get("gpu_architecture", ""),
                "data_name": single.get("data_name", ""),
                "data_provider": single.get("data_provider", ""),
            }
            run_details_dict = self._build_perf_entry_from_aggregated(
                single_record, model_info_for_entry, build_info, deployment_id
            )
        elif self.nodes == 1:
            # Single-node but no parsed metric (log parse failed or run failed before metrics).
            # In-job run skips perf write (skip_perf_collection); write a FAILURE row so run appears in perf.
            if model_info_for_entry:
                single_record = {
                    "model": model_info_for_entry.get("name", model_name),
                    "n_gpus": self.gpus_per_node,
                    "nnodes": 1,
                    "gpus_per_node": self.gpus_per_node,
                    "performance": "",
                    "metric": "",
                    "status": "FAILURE",
                    "test_duration": "",
                    "gpu_architecture": "",
                    "data_name": "",
                    "data_provider": "",
                }
                run_details_dict = self._build_perf_entry_from_aggregated(
                    single_record, model_info_for_entry, build_info, deployment_id
                )
            else:
                workspace_perf = Path("perf.csv")
                if workspace_perf.exists():
                    results["perf_files"] = [str(workspace_perf)]
                self.console.print(
                    f"[green]✓ Collected results: {len(results['perf_files'])} perf files, "
                    f"{len(results['logs'])} log files[/green]"
                )
                self._collect_results_parse_perf_csv(results, session_start_row)
                return self._attach_gcm_collector_summary(results, deployment_id)
        else:
            # Multi-node but no metrics parsed - optional failure record
            if per_node_metrics and model_info_for_entry:
                launcher_type = self.distributed_config.get("launcher", "torchrun")
                aggregated = self._aggregate_node_metrics(
                    per_node_metrics, self.nodes, launcher_type
                )
                if aggregated:
                    aggregated["status"] = "FAILURE"
                    run_details_dict = self._build_perf_entry_from_aggregated(
                        aggregated, model_info_for_entry, build_info, deployment_id
                    )

        if run_details_dict is not None:
            perf_entry_path = Path("perf_entry.json")
            with open(perf_entry_path, "w") as f:
                json.dump(run_details_dict, f, indent=2)
            perf_csv_path = "perf.csv"
            self._ensure_perf_csv_exists()
            if run_details_dict.get("status") == "SUCCESS":
                update_perf_csv(perf_csv=perf_csv_path, single_result=str(perf_entry_path))
            else:
                update_perf_csv(perf_csv=perf_csv_path, exception_result=str(perf_entry_path))
            try:
                scripts_path = model_info_for_entry.get("scripts", "")
                scripts_base_dir = scripts_base_dir_from(scripts_path)
                if run_details_dict.get("status") == "SUCCESS":
                    num_entries = update_perf_super_json(
                        single_result=str(perf_entry_path),
                        perf_super_json="perf_super.json",
                        scripts_base_dir=scripts_base_dir,
                    )
                else:
                    num_entries = update_perf_super_json(
                        exception_result=str(perf_entry_path),
                        perf_super_json="perf_super.json",
                        scripts_base_dir=scripts_base_dir,
                    )
                update_perf_super_csv(
                    perf_super_json="perf_super.json",
                    perf_super_csv="perf_super.csv",
                    num_entries=num_entries,
                )
            except Exception as e:
                self.console.print(f"[yellow]⚠ Could not update perf_super: {e}[/yellow]")
            results["perf_files"] = [str(Path(perf_csv_path).resolve())]
            run_data = {
                "model": run_details_dict.get("model", ""),
                "status": run_details_dict.get("status", ""),
                "performance": str(run_details_dict.get("performance", "")),
                "metric": run_details_dict.get("metric", ""),
                "duration": run_details_dict.get("test_duration", ""),
                "gpu_arch": run_details_dict.get("gpu_architecture", ""),
                "deployment": run_details_dict.get("deployment_type", ""),
                "machine": run_details_dict.get("machine_name", ""),
            }
            if run_details_dict.get("status") == "SUCCESS":
                results["successful_runs"].append(run_data)
            else:
                results["failed_runs"].append(run_data)
            summary = {
                "job_id": deployment_id,
                "model": model_name,
                "nodes": self.nodes,
                "per_node_metrics": per_node_metrics,
                "final_metric": run_details_dict.get("metric", ""),
                "final_performance": str(run_details_dict.get("performance", "")),
                "perf_entry_path": str(perf_entry_path),
            }
            (job_dir / "results_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )

        if not results["perf_files"]:
            workspace_perf = Path("perf.csv")
            if workspace_perf.exists():
                results["perf_files"] = [str(workspace_perf)]
        # When we already appended the current run from run_details_dict, skip re-parsing
        # the whole perf.csv so Execution Results shows only the current run.
        if run_details_dict is None:
            self._collect_results_parse_perf_csv(results, session_start_row)
        self.console.print(
            f"[green]✓ Collected results: {len(results['perf_files'])} perf files, "
            f"{len(results['logs'])} log files[/green]"
        )
        return self._attach_gcm_collector_summary(results, deployment_id)

    def _collect_results_parse_perf_csv(
        self, results: Dict[str, Any], session_start_row: Optional[int]
    ) -> None:
        """Parse perf.csv to populate results['successful_runs'] and results['failed_runs']."""
        if not results.get("perf_files"):
            return
        import csv

        perf_file = Path(results["perf_files"][0])
        try:
            with open(perf_file, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if session_start_row is not None and session_start_row < len(rows):
                rows = rows[session_start_row:]
            elif session_start_row is not None and session_start_row >= len(rows):
                rows = []
            for row in rows:
                run_data = {
                    "model": row.get("model", ""),
                    "status": row.get("status", ""),
                    "performance": row.get("performance", ""),
                    "metric": row.get("metric", ""),
                    "duration": row.get("test_duration", ""),
                    "gpu_arch": row.get("gpu_architecture", ""),
                    "deployment": row.get("deployment_type", ""),
                    "machine": row.get("machine_name", ""),
                }
                if row.get("status") == "SUCCESS":
                    results["successful_runs"].append(run_data)
                else:
                    results["failed_runs"].append(run_data)
        except Exception as e:
            self.console.print(f"[yellow]⚠ Could not parse perf.csv: {e}[/yellow]")

    def _run_gcm_collector_snapshot(self, deployment_id: str) -> Dict[str, Any]:
        if not self._gcm_enabled():
            return {"status": "disabled"}

        collector_cfg = self.gcm_config.get("collector", {})
        if not collector_cfg.get("enabled", False):
            return {"status": "disabled"}

        self._validate_gcm_allowlist()
        layout = self._gcm_artifact_layout()
        artifact_dir = Path(layout["dir"]) / deployment_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        output_path = artifact_dir / layout["collector_output"]

        collector_cmd = collector_cfg.get("command", "slurm_job_monitor")
        timeout_sec = int(collector_cfg.get("timeout_sec", 120))
        max_retries = int(collector_cfg.get("max_retries", 1))
        best_effort = bool(collector_cfg.get("best_effort", True))
        sink = collector_cfg.get("sink", "file")
        once_enabled = bool(collector_cfg.get("once", True))

        if not shutil.which("gcm"):
            msg = "gcm binary not found; collector skipped"
            if best_effort:
                output_path.write_text(msg + "\n", encoding="utf-8")
                return {"status": "skipped", "reason": msg, "output": str(output_path)}
            raise RuntimeError(msg)

        args = ["gcm", collector_cmd]
        if once_enabled:
            args.append("--once")
        if sink in {"stdout", "file"}:
            args.extend(["--sink", sink])

        attempts = 0
        last_error = ""
        while attempts < max_retries:
            attempts += 1
            try:
                result = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    check=False,
                )
                returncode = result.returncode
                stdout = result.stdout
                stderr = result.stderr
            except subprocess.TimeoutExpired as exc:
                returncode = 124
                stdout = exc.stdout or ""
                stderr = (exc.stderr or "") + "\ncollector timeout"
            output_path.write_text(
                f"# attempt={attempts}\n{stdout}\n{stderr}\n",
                encoding="utf-8",
            )
            if returncode == 0:
                return {
                    "status": "success",
                    "attempts": attempts,
                    "output": str(output_path),
                    "command": " ".join(args),
                }
            last_error = f"returncode={returncode}"

        if best_effort:
            return {
                "status": "failed_best_effort",
                "attempts": attempts,
                "output": str(output_path),
                "error": last_error,
            }
        raise RuntimeError(f"GCM collector failed after {attempts} attempt(s): {last_error}")

    def _attach_gcm_collector_summary(
        self, results: Dict[str, Any], deployment_id: str
    ) -> Dict[str, Any]:
        results.setdefault("cluster_features", {})
        try:
            collector_summary = self._run_gcm_collector_snapshot(deployment_id)
            results["cluster_features"]["gcm_collector"] = collector_summary
        except Exception as e:
            results["cluster_features"]["gcm_collector"] = {
                "status": "error",
                "error": str(e),
            }
        return results

    def cleanup(self, deployment_id: str) -> bool:
        """Cancel SLURM job if still running (locally)."""
        try:
            subprocess.run(
                ["scancel", deployment_id], capture_output=True, timeout=10
            )
            self.console.print(f"[yellow]Cancelled SLURM job: {deployment_id}[/yellow]")
            return True

        except Exception as e:
            self.console.print(f"[yellow]⚠ Cleanup warning: {e}[/yellow]")
            return False

