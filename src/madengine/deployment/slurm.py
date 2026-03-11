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
import subprocess
from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader

from .base import BaseDeployment, DeploymentConfig, DeploymentResult, DeploymentStatus
from .config_loader import ConfigLoader
from .slurm_node_selector import SlurmNodeSelector
from madengine.utils.gpu_config import resolve_runtime_gpus
from typing import Optional


# Valid distributed launchers
VALID_LAUNCHERS = [
    "torchrun",
    "torchtitan",
    "deepspeed",
    "megatron-lm",
    "vllm",
    "sglang",
    "slurm_multi",
]

# slurm_multi launcher name variants (underscore and hyphen)
SLURM_MULTI_ALIASES = [
    "slurm_multi",
    "slurm-multi",
]


def normalize_launcher(launcher_type: Optional[str], deployment_type: str) -> str:
    """
    Normalize launcher field based on deployment type and launcher value.
    
    Logic:
    - If launcher is in VALID_LAUNCHERS: keep as-is
    - If launcher is None/empty/invalid:
        * local → "docker" (runs in Docker container)
        * slurm → "docker" (typically uses containers on compute nodes)
        * kubernetes → "native" (pod itself is the container)
    
    Args:
        launcher_type: Raw launcher type from config (may be None)
        deployment_type: "local", "slurm", or "kubernetes"
        
    Returns:
        Normalized launcher string
    """
    # If launcher is valid, keep it
    if launcher_type and launcher_type in VALID_LAUNCHERS:
        return launcher_type
    
    # Otherwise, default based on deployment type
    if deployment_type == "local":
        return "docker"
    elif deployment_type == "slurm":
        return "docker"
    elif deployment_type == "kubernetes":
        return "native"
    else:
        # Fallback for unknown deployment types
        return "docker"


def is_rocprofv3_available() -> bool:
    """
    Check if rocprofv3 is available on the system.
    
    rocprofv3 is required for multi-node profiling with MPI support.
    It's part of rocprofiler-sdk package in ROCm >= 6.4.1.
    
    Returns:
        True if rocprofv3 is available and executable, False otherwise
    """
    try:
        # Note: rocprofv3 doesn't support --version, use --help instead
        result = subprocess.run(
            ["rocprofv3", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def configure_multi_node_profiling(
    nnodes: int,
    tools_config: list,
    logger
) -> Dict[str, Any]:
    """
    Configure profiling for multi-node SLURM runs with rocprofv3 support.
    
    Industry best practice for multi-node profiling:
    - Profile ALL nodes to detect stragglers, load imbalances, and communication bottlenecks
    - Use rocprofv3 (MPI-aware) for distributed profiling
    - Collect per-node outputs for detailed analysis
    
    Logic:
    1. Single node (nnodes == 1): Use existing tool behavior
    2. Multi-node (nnodes > 1):
       a. Check if rocprofv3 is available
       b. If available: Enable per-node profiling, upgrade "rocprof" to "rocprofv3"
       c. If not available: Log warning and skip profiling
    
    Args:
        nnodes: Number of nodes in the SLURM deployment
        tools_config: List of tool configurations from user
        logger: Logger instance for messages
        
    Returns:
        Dictionary with profiling configuration:
        - enabled: bool - Whether profiling is enabled
        - mode: str - "single_node", "multi_node", or "multi_node_unsupported"
        - tools: list - Processed tool configurations
        - per_node_collection: bool - Whether to collect from all nodes
    """
    if nnodes == 1:
        # Single node - existing behavior works fine
        return {
            "enabled": True,
            "mode": "single_node",
            "tools": tools_config,
            "per_node_collection": False
        }
    
    # Multi-node case - check rocprofv3 availability
    if not is_rocprofv3_available():
        logger.warning(
            "╔════════════════════════════════════════════════════════════════════════════╗\n"
            "║ Multi-Node Profiling Requirements Not Met                                 ║\n"
            "╠════════════════════════════════════════════════════════════════════════════╣\n"
            "║ Multi-node profiling requires rocprofv3 (MPI-aware profiling support).    ║\n"
            "║                                                                            ║\n"
            "║ Current Status: rocprofv3 NOT FOUND on system                             ║\n"
            "║                                                                            ║\n"
            "║ Profiling will be SKIPPED for this multi-node run.                        ║\n"
            "║                                                                            ║\n"
            "║ To enable multi-node profiling:                                           ║\n"
            "║   • Install rocprofiler-sdk package (ROCm >= 6.4.1)                       ║\n"
            "║   • Command: apt install rocprofiler-sdk                                  ║\n"
            "║   • Or upgrade to ROCm 6.4.1 or later                                     ║\n"
            "║                                                                            ║\n"
            "║ Note: Single-node profiling uses rocprof (no rocprofv3 required)          ║\n"
            "╚════════════════════════════════════════════════════════════════════════════╝"
        )
        return {
            "enabled": False,
            "mode": "multi_node_unsupported",
            "tools": [],
            "per_node_collection": False
        }
    
    # rocprofv3 is available - enable full multi-node profiling
    logger.info(f"✓ Multi-node profiling enabled for {nnodes} nodes (rocprofv3 detected)")
    
    # Upgrade "rocprof" tools to "rocprofv3" for multi-node compatibility
    upgraded_tools = []
    rocprof_upgraded = False
    
    for tool in tools_config:
        tool_name = tool.get("name") if isinstance(tool, dict) else None
        
        if tool_name == "rocprof":
            # Upgrade to rocprofv3 for multi-node MPI support
            logger.info(
                f"  → Upgrading 'rocprof' to 'rocprofv3' for multi-node MPI compatibility"
            )
            upgraded_tool = tool.copy() if isinstance(tool, dict) else {"name": "rocprofv3"}
            upgraded_tool["name"] = "rocprofv3"
            upgraded_tools.append(upgraded_tool)
            rocprof_upgraded = True
        else:
            upgraded_tools.append(tool)
    
    # Log profiling tools being used
    if upgraded_tools:
        tool_names = [t.get("name") if isinstance(t, dict) else str(t) for t in upgraded_tools]
        logger.info(f"  → Multi-node profiling tools: {', '.join(filter(None, tool_names))}")
        
        # Highlight RCCL trace if present (critical for multi-node communication)
        if "rccl_trace" in tool_names:
            logger.info("  → ✓ rccl_trace enabled (critical for multi-node communication profiling)")
    
    return {
        "enabled": True,
        "mode": "multi_node",
        "tools": upgraded_tools,
        "per_node_collection": True,
        "profiler": "rocprofv3",
        "wrapper_mode": "launcher"
    }


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

    def __init__(self, config: DeploymentConfig):
        """
        Initialize SLURM deployment.

        Args:
            config: Deployment configuration
        """
        # Apply intelligent defaults using ConfigLoader
        # This merges built-in presets with user configuration
        full_config = ConfigLoader.load_slurm_config(config.additional_context)
        config.additional_context = full_config

        super().__init__(config)

        # Parse SLURM configuration (now with defaults applied)
        self.slurm_config = config.additional_context.get("slurm", {})
        self.distributed_config = config.additional_context.get("distributed", {})

        # SLURM parameters
        self.partition = self.slurm_config.get("partition", "gpu")
        self.nodes = self.slurm_config.get("nodes", 1)
        self.gpus_per_node = self.slurm_config.get("gpus_per_node", 8)
        self.time_limit = self.slurm_config.get("time", "24:00:00")
        self.output_dir = Path(self.slurm_config.get("output_dir", "./slurm_output"))
        self.reservation = self.slurm_config.get("reservation", None)

        # Setup Jinja2 template engine
        template_dir = Path(__file__).parent / "templates" / "slurm"
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        
        # Register custom Jinja2 filters
        self.jinja_env.filters['dirname'] = lambda path: str(Path(path).parent)
        self.jinja_env.filters['basename'] = lambda path: str(Path(path).name)

        # Generated script path
        self.script_path = None

        # ========== OPTION 2: Detect existing SLURM allocation ==========
        # If SLURM_JOB_ID exists, we're inside an salloc allocation
        self.inside_allocation = os.environ.get("SLURM_JOB_ID") is not None
        self.existing_job_id = os.environ.get("SLURM_JOB_ID", "")
        self.allocation_nodes = self._get_allocation_node_count()
        
        if self.inside_allocation:
            self.console.print(
                f"[cyan]✓ Detected existing SLURM allocation: Job {self.existing_job_id}[/cyan]"
            )
            self.console.print(
                f"  Allocation has {self.allocation_nodes} nodes available"
            )

    def _get_allocation_node_count(self) -> int:
        """
        Get number of nodes in current SLURM allocation.
        
        Note: SLURM_NNODES reflects the current job step, not the full allocation.
        We query the job directly using scontrol to get the actual node count.
        """
        if not self.inside_allocation:
            return 0
        
        job_id = self.existing_job_id
        
        # Query the actual job's node count using scontrol (most accurate)
        try:
            result = subprocess.run(
                ["scontrol", "show", "job", job_id],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # Parse NumNodes=X from output
                for line in result.stdout.split("\n"):
                    if "NumNodes=" in line:
                        # Format: "NumNodes=3 NumCPUs=..."
                        for part in line.split():
                            if part.startswith("NumNodes="):
                                try:
                                    return int(part.split("=")[1])
                                except (ValueError, IndexError):
                                    pass
        except Exception:
            pass
        
        # Fallback: Try SLURM_JOB_NUM_NODES (full job node count, if set)
        job_num_nodes = os.environ.get("SLURM_JOB_NUM_NODES")
        if job_num_nodes:
            try:
                return int(job_num_nodes)
            except ValueError:
                pass
        
        # Fallback: SLURM_NNODES (may be step-specific, not full allocation)
        nnodes = os.environ.get("SLURM_NNODES")
        if nnodes:
            try:
                return int(nnodes)
            except ValueError:
                pass
        
        # Last resort: count nodes in SLURM_NODELIST
        nodelist = os.environ.get("SLURM_NODELIST")
        if nodelist:
            try:
                result = subprocess.run(
                    ["scontrol", "show", "hostname", nodelist],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    return len(result.stdout.strip().split("\n"))
            except Exception:
                pass
        
        return 0

    def _validate_allocation_nodes(self) -> tuple[bool, str]:
        """
        Validate that existing allocation has enough nodes for the job.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.inside_allocation:
            return True, ""
        
        requested_nodes = self.nodes
        available_nodes = self.allocation_nodes
        
        if available_nodes < requested_nodes:
            return False, (
                f"Insufficient nodes in current allocation. "
                f"Requested: {requested_nodes}, Available: {available_nodes}. "
                f"Either reduce nodes in config or use a larger allocation."
            )
        
        if available_nodes > requested_nodes:
            self.console.print(
                f"[yellow]⚠ Note: Using {requested_nodes} of {available_nodes} "
                f"available nodes in allocation[/yellow]"
            )
        
        return True, ""

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
        
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Get model info from manifest
            model_keys = list(self.manifest["built_models"].keys())
            if not model_keys:
                raise ValueError("No models in manifest")

            model_key = model_keys[0]
            model_info = self.manifest["built_models"][model_key]

            # Check if this is a slurm_multi launcher (baremetal multi-node)
            # Priority: model_info.distributed.launcher > additional_context.distributed.launcher
            model_distributed = model_info.get("distributed", {})
            launcher_type = model_distributed.get("launcher") or self.distributed_config.get("launcher", "torchrun")
            launcher_normalized = launcher_type.lower().replace("_", "-")
            
            # Check against slurm_multi aliases (includes legacy sglang-disagg, vllm-disagg)
            slurm_multi_aliases_normalized = [a.lower().replace("_", "-") for a in SLURM_MULTI_ALIASES]
            if launcher_normalized in slurm_multi_aliases_normalized:
                # For slurm_multi launchers, generate simple wrapper script
                # that runs the model's .slurm script directly on baremetal
                self.console.print(f"[cyan]Detected slurm_multi launcher: {launcher_type}[/cyan]")
                # Pass model_key as docker_image_name (for manifests, the key IS the built image name)
                return self._prepare_baremetal_script(model_info, docker_image_name=model_key)
            
            # Standard flow: validate madengine availability for complex job template
            if not self._validate_cli_availability():
                self.console.print(
                    "\n[yellow]⚠ Tip: Compute nodes inherit your submission environment[/yellow]"
                )
                return False

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

    def _prepare_baremetal_script(self, model_info: Dict, docker_image_name: str = None) -> bool:
        """
        Generate a simple wrapper script for baremetal/slurm_multi launchers.
        
        These launchers (slurm_multi, sglang-disagg, vllm-disagg) run the model's 
        .slurm script directly on baremetal, which then manages Docker containers 
        via srun. No madengine wrapper needed.
        
        Args:
            model_info: Model configuration from manifest
            docker_image_name: The built Docker image name from manifest key
        """
        # Get the model's script path
        model_script = model_info.get("scripts", "")
        if not model_script:
            self.console.print("[red]✗ No scripts defined in model_info[/red]")
            return False
        
        # Get manifest directory (where the model script is relative to)
        manifest_dir = Path(self.config.manifest_file).parent.absolute()
        model_script_path = manifest_dir / model_script
        
        if not model_script_path.exists():
            self.console.print(f"[red]✗ Model script not found: {model_script_path}[/red]")
            return False
        
        # Get environment variables
        env_vars = {}
        
        # From model_info.env_vars
        if "env_vars" in model_info:
            env_vars.update(model_info["env_vars"])
        
        # From additional_context.env_vars
        if "env_vars" in self.config.additional_context:
            env_vars.update(self.config.additional_context["env_vars"])
        
        # From distributed config (model's distributed section)
        model_distributed = model_info.get("distributed", {})
        sglang_disagg_config = model_distributed.get("sglang_disagg", {}) or self.distributed_config.get("sglang_disagg", {})
        if sglang_disagg_config:
            env_vars["xP"] = str(sglang_disagg_config.get("prefill_nodes", 1))
            env_vars["yD"] = str(sglang_disagg_config.get("decode_nodes", 1))
        
        # Override DOCKER_IMAGE_NAME with the built image from manifest
        # This ensures the run uses the freshly built image, not the base image
        # Priority: docker_image_name param > model_info.docker_image > env_vars.DOCKER_IMAGE_NAME
        if docker_image_name and docker_image_name.startswith("ci-"):
            # The manifest key IS the built image name for madengine-built images
            self.console.print(f"[cyan]Using built Docker image: {docker_image_name}[/cyan]")
            env_vars["DOCKER_IMAGE_NAME"] = docker_image_name
        elif "docker_image" in model_info:
            built_image = model_info["docker_image"]
            self.console.print(f"[cyan]Using Docker image: {built_image}[/cyan]")
            env_vars["DOCKER_IMAGE_NAME"] = built_image
        elif "image" in model_info:
            # Fallback to 'image' field
            built_image = model_info["image"]
            self.console.print(f"[cyan]Using Docker image: {built_image}[/cyan]")
            env_vars["DOCKER_IMAGE_NAME"] = built_image
        
        # Get model args
        model_args = model_info.get("args", "")
        
        # Generate simple wrapper script
        # IMPORTANT: SBATCH directives MUST be at the top, right after #!/bin/bash
        script_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=madengine-{model_info['name']}",
            f"#SBATCH --output={self.output_dir}/madengine-{model_info['name']}_%j.out",
            f"#SBATCH --error={self.output_dir}/madengine-{model_info['name']}_%j.err",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --nodes={self.nodes}",
            f"#SBATCH --ntasks={self.nodes}",
            f"#SBATCH --gpus-per-node={self.gpus_per_node}",
            f"#SBATCH --time={self.time_limit}",
            "#SBATCH --exclusive",
        ]
        
        # Add reservation if specified
        if self.reservation:
            script_lines.append(f"#SBATCH --reservation={self.reservation}")
        
        # Add nodelist if specified (from model card or --additional-context)
        nodelist = self._normalize_nodelist(self.slurm_config.get("nodelist"))
        if nodelist:
            script_lines.append(f"#SBATCH --nodelist={nodelist}")
        
        script_lines.extend([
            "",
            f"# Baremetal launcher script for {model_info['name']}",
            f"# Generated by madengine for slurm_multi",
            "",
            "set -e",
            "",
            "# Environment variables",
        ])
        
        for key, value in env_vars.items():
            script_lines.append(f"export {key}=\"{value}\"")
        
        script_lines.append("")
        script_lines.extend([
            "echo '=========================================='",
            "echo 'Baremetal Launcher - slurm_multi'",
            "echo '=========================================='",
            f"echo 'Model: {model_info['name']}'",
            f"echo 'Script: {model_script_path}'",
            "echo 'SLURM_JOB_ID:' $SLURM_JOB_ID",
            "echo 'SLURM_NNODES:' $SLURM_NNODES",
            "echo 'SLURM_NODELIST:' $SLURM_NODELIST",
            "echo ''",
        ])
        
        # Check if image needs parallel pull on all nodes
        # Pull if: image is from registry (contains / or .) and not a local ci-* build
        docker_image = env_vars.get("DOCKER_IMAGE_NAME", "")
        is_registry_image = docker_image and not docker_image.startswith("ci-") and ("/" in docker_image or "." in docker_image)
        
        if is_registry_image:
            # Add parallel docker pull on all nodes
            # This ensures all nodes have the image before running
            script_lines.extend([
                "",
                "# Pull Docker image in parallel on all nodes",
                "echo '=========================================='",
                "echo 'Pulling Docker image on all nodes in parallel'",
                "echo '=========================================='",
                f"echo 'Image: {docker_image}'",
                "echo ''",
                "",
                f"srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES bash -c \"",
                f"    echo \\\"[\\$(hostname)] Pulling {docker_image}...\\\"",
                f"    docker pull {docker_image}",
                "    PULL_RC=\\$?",
                "    if [ \\$PULL_RC -eq 0 ]; then",
                "        echo \\\"[\\$(hostname)] Pull SUCCESS\\\"",
                "    else",
                "        echo \\\"[\\$(hostname)] Pull FAILED with exit code \\$PULL_RC\\\"",
                "    fi",
                "    exit \\$PULL_RC",
                "\"",
                "PULL_EXIT=$?",
                "",
                "if [ $PULL_EXIT -ne 0 ]; then",
                "    echo 'Docker pull failed on one or more nodes'",
                "    exit $PULL_EXIT",
                "fi",
                "",
                "echo ''",
                "echo 'Docker image pulled on all nodes'",
                "echo ''",
            ])
        
        # Create completion marker path for robust completion detection
        # Use absolute path since script will cd to different directory
        completion_marker = (self.output_dir / f"madengine_{model_info['name']}.complete").resolve()
        
        script_lines.extend([
            "",
            "# Change to script directory",
            f"cd {model_script_path.parent}",
            "",
            "# Run the model script directly on baremetal",
            f"echo 'Executing: bash {model_script_path.name} {model_args}'",
            f"bash {model_script_path.name} {model_args}",
            "SCRIPT_EXIT_CODE=$?",
            "",
            "echo ''",
            "echo 'Script completed.'",
            "",
            "# Write completion marker for madengine to detect",
            f"echo \"exit_code=$SCRIPT_EXIT_CODE\" > {completion_marker}",
            f"echo \"timestamp=$(date -Iseconds)\" >> {completion_marker}",
            f"echo 'Completion marker written: {completion_marker}'",
            "",
            "exit $SCRIPT_EXIT_CODE",
        ])
        
        # Store marker path for monitor to check
        self._completion_marker = completion_marker
        
        script_content = "\n".join(script_lines)
        
        # Save script
        self.script_path = self.output_dir / f"madengine_{model_info['name']}.sh"
        self.script_path.write_text(script_content)
        self.script_path.chmod(0o755)
        
        self.console.print(f"[green]✓ Generated baremetal script: {self.script_path}[/green]")
        self.console.print(f"  Model script: {model_script_path}")
        self.console.print(f"  Environment: {len(env_vars)} variables")
        
        return True

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
            master_port=master_port
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
        }

    def _generate_launcher_command(
        self, launcher_type: str, nnodes: int, nproc_per_node: int, master_port: int
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
        elif launcher_type.lower().replace("_", "-") in [a.lower().replace("_", "-") for a in SLURM_MULTI_ALIASES]:
            return self._generate_slurm_multi_command(nnodes, nproc_per_node, master_port)
        elif launcher_type == "deepspeed":
            return self._generate_deepspeed_command(nnodes, nproc_per_node, master_port)
        elif launcher_type == "megatron":
            return self._generate_megatron_command(nnodes, nproc_per_node, master_port)
        elif launcher_type == "torchtitan":
            return self._generate_torchtitan_command(nnodes, nproc_per_node, master_port)
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
            return f'''# vLLM multi-node setup (TP + PP with Ray)
export VLLM_TENSOR_PARALLEL_SIZE={nproc_per_node}
export VLLM_PIPELINE_PARALLEL_SIZE={nnodes}
export VLLM_DISTRIBUTED_BACKEND="ray"
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
            return f'''# SGLang multi-node setup (TP + PP with Ray)
export SGLANG_TENSOR_PARALLEL_SIZE={nproc_per_node}
export SGLANG_PIPELINE_PARALLEL_SIZE={nnodes}
# SGLang handles its own process management - no MAD_MULTI_NODE_RUNNER needed'''

    def _generate_slurm_multi_command(
        self, nnodes: int, nproc_per_node: int, master_port: int
    ) -> str:
        """
        Generate slurm_multi launcher environment for SLURM.
        
        slurm_multi Architecture (multi-node baremetal):
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
            ValueError: If nnodes < 3
        """
        if nnodes < 3:
            raise ValueError(
                f"slurm_multi requires minimum 3 nodes "
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

    def deploy(self) -> DeploymentResult:
        """
        Deploy to SLURM - either via sbatch (new job) or bash (existing allocation).
        
        If SLURM_JOB_ID is set (inside salloc), runs script directly with bash.
        Otherwise, submits a new job via sbatch.
        """
        if not self.script_path or not self.script_path.exists():
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id="",
                message="Script not generated. Run prepare() first.",
            )

        # ========== BRANCH: Inside allocation vs new job ==========
        if self.inside_allocation:
            return self._run_inside_existing_allocation()
        else:
            return self._submit_new_job()

    def _run_inside_existing_allocation(self) -> DeploymentResult:
        """
        Run script directly inside existing salloc allocation using bash.
        
        The script will use the nodes already allocated to the current job.
        SLURM environment variables (SLURM_NODELIST, etc.) are inherited.
        """
        # Validate node count before running
        is_valid, error_msg = self._validate_allocation_nodes()
        if not is_valid:
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id=self.existing_job_id,
                message=error_msg,
            )
        
        self.console.print(
            f"\n[bold cyan]Running inside existing SLURM allocation[/bold cyan]"
        )
        self.console.print(f"  Job ID: {self.existing_job_id}")
        self.console.print(f"  Using {self.nodes} of {self.allocation_nodes} allocated nodes")
        self.console.print(f"  GPUs per node: {self.gpus_per_node}")
        self.console.print(f"  Script: {self.script_path}")
        self.console.print(f"\n[dim]Executing: bash {self.script_path}[/dim]\n")
        
        try:
            # Run script directly with bash (synchronous, blocks until done)
            # Don't capture output - let it stream directly to console
            result = subprocess.run(
                ["bash", str(self.script_path)],
                timeout=self.config.timeout if self.config.timeout > 0 else None,
            )
            
            if result.returncode == 0:
                self.console.print(
                    f"\n[green]✓ Script completed successfully in allocation {self.existing_job_id}[/green]"
                )
                return DeploymentResult(
                    status=DeploymentStatus.SUCCESS,
                    deployment_id=self.existing_job_id,
                    message=f"Completed inside existing allocation {self.existing_job_id}",
                    logs_path=str(self.output_dir),
                    skip_monitoring=True,  # Already ran synchronously, no need to poll
                )
            else:
                self.console.print(
                    f"\n[red]✗ Script failed with exit code {result.returncode}[/red]"
                )
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    deployment_id=self.existing_job_id,
                    message=f"Script failed with exit code {result.returncode}",
                    logs_path=str(self.output_dir),
                    skip_monitoring=True,  # Already ran synchronously
                )
                
        except subprocess.TimeoutExpired:
            self.console.print(
                f"\n[red]✗ Script timed out after {self.config.timeout}s[/red]"
            )
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id=self.existing_job_id,
                message=f"Script timed out after {self.config.timeout}s",
            )
        except Exception as e:
            self.console.print(f"\n[red]✗ Execution error: {e}[/red]")
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id=self.existing_job_id,
                message=f"Execution error: {str(e)}",
            )

    def _submit_new_job(self) -> DeploymentResult:
        """Submit new SLURM job via sbatch (original behavior)."""
        # ==================== PREFLIGHT NODE SELECTION ====================
        # For multi-node jobs with Ray/vLLM, check for clean nodes first
        # to avoid OOM errors from stale processes
        enable_preflight = self.slurm_config.get("enable_node_check", True)
        auto_cleanup = self.slurm_config.get("auto_cleanup_nodes", False)
        
        if enable_preflight and self.nodes > 1 and not self.slurm_config.get("nodelist"):
            try:
                selector = SlurmNodeSelector(
                    console=self.console,
                    auto_cleanup=auto_cleanup,
                    verbose=self.slurm_config.get("verbose_node_check", False),
                    reservation=self.reservation,
                )
                
                # Select clean nodes and get updated exclude list
                clean_nodes, updated_exclude = selector.select_nodes(
                    partition=self.partition,
                    nodes_needed=self.nodes,
                    exclude=self.slurm_config.get("exclude"),
                    constraint=self.slurm_config.get("constraint"),
                )
                
                # Update exclude list if dirty nodes found
                if updated_exclude and updated_exclude != self.slurm_config.get("exclude", ""):
                    self.console.print(
                        f"[dim]Updated exclude list for sbatch: {updated_exclude}[/dim]\n"
                    )
                    # Re-generate script with updated exclude list
                    self.slurm_config["exclude"] = updated_exclude
                    self.prepare()  # Re-generate sbatch script
                    
            except Exception as e:
                # Don't fail deployment if preflight fails
                self.console.print(
                    f"[yellow]⚠ Node health check failed: {e}[/yellow]"
                )
                self.console.print("[dim]Continuing with job submission[/dim]\n")
        # ==================== END PREFLIGHT ====================

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
        # If we ran inside an existing allocation, script already completed synchronously
        # No need to poll - just return success (deploy() already handled the result)
        if self.inside_allocation:
            return DeploymentResult(
                status=DeploymentStatus.SUCCESS,
                deployment_id=deployment_id,
                message=f"Completed (ran inside existing allocation {deployment_id})",
            )
        
        # Check for completion marker (robust detection for interactive/salloc jobs)
        if hasattr(self, '_completion_marker') and self._completion_marker:
            marker_path = Path(self._completion_marker)
            if marker_path.exists():
                # Read exit code from marker
                try:
                    content = marker_path.read_text()
                    exit_code = 0
                    for line in content.splitlines():
                        if line.startswith("exit_code="):
                            exit_code = int(line.split("=")[1])
                            break
                    
                    self.console.print(f"[green]✓ Completion marker found: {marker_path}[/green]")
                    
                    if exit_code == 0:
                        return DeploymentResult(
                            status=DeploymentStatus.SUCCESS,
                            deployment_id=deployment_id,
                            message=f"Script completed successfully (exit code {exit_code})",
                        )
                    else:
                        return DeploymentResult(
                            status=DeploymentStatus.FAILED,
                            deployment_id=deployment_id,
                            message=f"Script failed with exit code {exit_code}",
                        )
                except Exception as e:
                    self.console.print(f"[yellow]Warning: Could not read completion marker: {e}[/yellow]")
        
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
        output_dir = self.slurm_config.get("output_dir", "./slurm_output")
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
        output_dir = self.slurm_config.get("output_dir", "./slurm_output")
        
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

    def collect_results(self, deployment_id: str) -> Dict[str, Any]:
        """Collect performance results from SLURM output files.
        
        NOTE: Current implementation works with single-node jobs where perf.csv
        is written to shared storage. For multi-node jobs with per-node metrics,
        this would need enhancement to:
        1. Read all node output files (madengine-*_jobid_noderank.out)
        2. Parse per-node metrics from each file
        3. Aggregate using _aggregate_node_metrics() (similar to kubernetes.py)
        4. Write aggregated result to perf.csv
        
        Args:
            deployment_id: SLURM job ID
        """
        # Get session_start_row from config (passed from orchestrator)
        session_start_row = self.config.additional_context.get("session_start_row")
        
        results = {
            "job_id": deployment_id,
            "nodes": self.nodes,
            "gpus_per_node": self.gpus_per_node,
            "perf_files": [],
            "logs": [],
            "successful_runs": [],
            "failed_runs": [],
            "session_start_row": session_start_row,  # Track for downstream filtering
        }

        try:
            # Find output files
            output_pattern = f"madengine-*_{deployment_id}_*.out"
            output_files = list(self.output_dir.glob(output_pattern))

            results["logs"] = [str(f) for f in output_files]

            # Find performance CSV files
            # Strategy 1: Check results_dir if configured
            if self.slurm_config.get("results_dir"):
                results_dir = Path(self.slurm_config["results_dir"])
                perf_pattern = f"perf_{deployment_id}_*.csv"
                perf_files = list(results_dir.glob(perf_pattern))
                results["perf_files"] = [str(f) for f in perf_files]
            
            # Strategy 2: Check shared workspace (NFS) for perf.csv
            # When using shared storage, perf.csv is written directly to workspace
            # Retry briefly to allow NFS propagation after job completion
            if not results["perf_files"]:
                workspace_perf = Path("perf.csv")
                for _attempt in range(6):
                    if workspace_perf.exists():
                        results["perf_files"] = [str(workspace_perf)]
                        self.console.print("[dim]Note: Using perf.csv from shared workspace[/dim]")
                        break
                    import time
                    time.sleep(5)
            
            # Parse perf.csv to populate successful_runs and failed_runs
            # Filter based on session_start_row passed as parameter (no external files!)
            if results["perf_files"]:
                perf_file = Path(results["perf_files"][0])
                try:
                    import csv
                    
                    with open(perf_file, 'r') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        
                        # Filter to only include rows from current session if session_start_row provided
                        if session_start_row is not None and session_start_row < len(rows):
                            rows = rows[session_start_row:]
                            self.console.print(f"[cyan]📊 Filtered to current session: {len(rows)} runs (from row {session_start_row} of {len(rows) + session_start_row} total)[/cyan]")
                        elif session_start_row is not None:
                            # Session start equals or exceeds current rows - no new runs yet
                            self.console.print(f"[yellow]⚠️  No new runs in this session (session started at row {session_start_row}, CSV has {len(rows)} rows)[/yellow]")
                            rows = []
                        else:
                            # No session info provided - show all rows (for backward compatibility)
                            self.console.print(f"[dim]Showing all {len(rows)} runs from perf.csv (no session filtering)[/dim]")
                        
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
                except Exception as parse_error:
                    import traceback
                    self.console.print(f"[red]ERROR parsing perf.csv: {parse_error}[/red]")
                    self.console.print(f"[dim]{traceback.format_exc()}[/dim]")

            self.console.print(
                f"[green]✓ Collected results: {len(results['perf_files'])} perf files, "
                f"{len(results['logs'])} log files[/green]"
            )

        except Exception as e:
            self.console.print(f"[yellow]⚠ Results collection incomplete: {e}[/yellow]")

        return results

    def cleanup(self, deployment_id: str) -> bool:
        """Cancel SLURM job if still running (locally)."""
        # CRITICAL: Never cancel an existing allocation we're running inside!
        # The user's salloc session should not be terminated by madengine
        if self.inside_allocation:
            self.console.print(
                f"[dim]Skipping cleanup - running inside existing allocation {deployment_id}[/dim]"
            )
            return True
        
        try:
            subprocess.run(
                ["scancel", deployment_id], capture_output=True, timeout=10
            )
            self.console.print(f"[yellow]Cancelled SLURM job: {deployment_id}[/yellow]")
            return True

        except Exception as e:
            self.console.print(f"[yellow]⚠ Cleanup warning: {e}[/yellow]")
            return False

