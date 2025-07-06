#!/usr/bin/env python3
"""
Command-line interface for MADEngine Distributed Orchestrator

This provides CLI commands for building and running models in distributed scenarios.
"""

import argparse
import sys
import os
import json
import logging
from typing import Dict, Any
from madengine.tools.distributed_orchestrator import (
    DistributedOrchestrator, 
    create_ansible_playbook, 
    create_kubernetes_manifests
)

# Constants
DEFAULT_MANIFEST_FILE = 'build_manifest.json'
DEFAULT_EXECUTION_CONFIG = 'execution_config.json'
DEFAULT_PERF_OUTPUT = 'perf.csv'
DEFAULT_DATA_CONFIG = 'data.json'
DEFAULT_TOOLS_CONFIG = './scripts/common/tools.json'
DEFAULT_ANSIBLE_OUTPUT = 'madengine_distributed.yml'
DEFAULT_K8S_NAMESPACE = 'madengine'
DEFAULT_TIMEOUT = -1

# Exit codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_BUILD_FAILURE = 2
EXIT_RUN_FAILURE = 3
EXIT_INVALID_ARGS = 4

# -----------------------------------------------------------------------------
# Sub-command functions
# -----------------------------------------------------------------------------
# Router of the command-line arguments to the corresponding functions

def build_models(args: argparse.Namespace) -> int:
    """Build Docker images for models in distributed scenarios.
    
    Args:
        args: The command-line arguments.
        
    Returns:
        int: Exit code (0 for success, 2 for build failure)
    """
    try:
        logging.info("Starting model build process")
        orchestrator = DistributedOrchestrator(args)
        
        # Mark this as separate build phase for log naming
        args._separate_phases = True
        
        build_summary = orchestrator.build_phase(
            registry=args.registry,
            clean_cache=args.clean_docker_cache,
            manifest_output=args.manifest_output
        )
        
        # Save build summary
        if args.summary_output:
            try:
                with open(args.summary_output, 'w') as f:
                    json.dump(build_summary, f, indent=2)
                logging.info(f"Build summary saved to: {args.summary_output}")
            except IOError as e:
                logging.error(f"Failed to save build summary: {e}")
                return EXIT_FAILURE
        
        failed_builds = len(build_summary.get("failed_builds", []))
        if failed_builds == 0:
            logging.info("All builds completed successfully")
            return EXIT_SUCCESS
        else:
            logging.error(f"Build failed for {failed_builds} models")
            return EXIT_BUILD_FAILURE
            
    except Exception as e:
        logging.error(f"Build process failed: {e}")
        return EXIT_FAILURE


def run_models(args: argparse.Namespace) -> int:
    """Run model containers in distributed scenarios.
    
    If manifest-file is provided and exists, runs only the execution phase.
    Registry information is auto-detected from the manifest when available.
    If manifest-file is not provided or doesn't exist, runs the complete workflow.
    
    Args:
        args: The command-line arguments.
        
    Returns:
        int: Exit code (0 for success, 2 for build failure, 3 for run failure)
    """
    try:
        # Input validation
        if args.timeout < -1:
            logging.error("Timeout must be -1 (default) or a positive integer")
            return EXIT_INVALID_ARGS
            
        orchestrator = DistributedOrchestrator(args)
        
        # Check if manifest file is provided and exists
        if hasattr(args, 'manifest_file') and args.manifest_file and os.path.exists(args.manifest_file):
            # Run only execution phase using existing manifest
            logging.info(f"Running models using existing manifest: {args.manifest_file}")
            
            # Mark this as separate run phase for log naming
            args._separate_phases = True
            
            try:
                execution_summary = orchestrator.run_phase(
                    manifest_file=args.manifest_file,
                    registry=args.registry,
                    timeout=args.timeout,
                    keep_alive=args.keep_alive
                )
                
                # Save execution summary
                if args.summary_output:
                    try:
                        with open(args.summary_output, 'w') as f:
                            json.dump(execution_summary, f, indent=2)
                        logging.info(f"Execution summary saved to: {args.summary_output}")
                    except IOError as e:
                        logging.error(f"Failed to save execution summary: {e}")
                        return EXIT_FAILURE
                
                failed_runs = len(execution_summary.get("failed_runs", []))
                if failed_runs == 0:
                    logging.info("All model executions completed successfully")
                    return EXIT_SUCCESS
                else:
                    logging.error(f"Execution failed for {failed_runs} models")
                    return EXIT_RUN_FAILURE
                    
            except Exception as e:
                logging.error(f"Model execution failed: {e}")
                return EXIT_RUN_FAILURE
        
        else:
            # Run complete workflow (build + run)
            if args.manifest_file:
                logging.warning(f"Manifest file {args.manifest_file} not found, running complete workflow")
            else:
                logging.info("No manifest file provided, running complete workflow (build + run)")
            
            try:
                # Always use separate log files for build and run phases
                args._separate_phases = True
                
                # Build phase
                build_summary = orchestrator.build_phase(
                    registry=args.registry,
                    clean_cache=getattr(args, 'clean_docker_cache', False),
                    manifest_output=getattr(args, 'manifest_output', DEFAULT_MANIFEST_FILE)
                )
                
                # Check build results
                failed_builds = len(build_summary.get("failed_builds", []))
                if failed_builds > 0:
                    logging.error(f"Build failed for {failed_builds} models, aborting workflow")
                    return EXIT_BUILD_FAILURE
                
                # Run phase  
                execution_summary = orchestrator.run_phase(
                    manifest_file=getattr(args, 'manifest_output', DEFAULT_MANIFEST_FILE),
                    registry=args.registry,
                    timeout=args.timeout,
                    keep_alive=args.keep_alive
                )
                
                # Combine summaries
                workflow_summary = {
                    "build_phase": build_summary,
                    "run_phase": execution_summary,
                    "overall_success": (
                        len(build_summary.get("failed_builds", [])) == 0 and
                        len(execution_summary.get("failed_runs", [])) == 0
                    )
                }
                
                # Save workflow summary
                if args.summary_output:
                    try:
                        with open(args.summary_output, 'w') as f:
                            json.dump(workflow_summary, f, indent=2)
                        logging.info(f"Workflow summary saved to: {args.summary_output}")
                    except IOError as e:
                        logging.error(f"Failed to save workflow summary: {e}")
                        return EXIT_FAILURE
                
                if workflow_summary["overall_success"]:
                    logging.info("Complete workflow finished successfully")
                    return EXIT_SUCCESS
                else:
                    failed_runs = len(execution_summary.get("failed_runs", []))
                    if failed_runs > 0:
                        logging.error(f"Workflow completed but {failed_runs} model executions failed")
                        return EXIT_RUN_FAILURE
                    else:
                        logging.error("Workflow failed for unknown reasons")
                        return EXIT_FAILURE
                        
            except Exception as e:
                logging.error(f"Complete workflow failed: {e}")
                return EXIT_FAILURE
                
    except Exception as e:
        logging.error(f"Run process failed: {e}")
        return EXIT_FAILURE


def generate_ansible(args: argparse.Namespace) -> int:
    """Generate Ansible playbook for distributed execution.
    
    Args:
        args: The command-line arguments.
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        logging.info("Generating Ansible playbook")
        
        # Validate input files exist if specified
        if hasattr(args, 'manifest_file') and args.manifest_file != DEFAULT_MANIFEST_FILE:
            if not os.path.exists(args.manifest_file):
                logging.warning(f"Manifest file {args.manifest_file} does not exist")
        
        if hasattr(args, 'execution_config') and args.execution_config != DEFAULT_EXECUTION_CONFIG:
            if not os.path.exists(args.execution_config):
                logging.warning(f"Execution config file {args.execution_config} does not exist")
        
        create_ansible_playbook(
            manifest_file=args.manifest_file,
            execution_config=args.execution_config,
            playbook_file=args.output
        )
        
        logging.info(f"Ansible playbook generated successfully: {args.output}")
        return EXIT_SUCCESS
        
    except Exception as e:
        logging.error(f"Failed to generate Ansible playbook: {e}")
        return EXIT_FAILURE


def generate_k8s(args: argparse.Namespace) -> int:
    """Generate Kubernetes manifests for distributed execution.
    
    Args:
        args: The command-line arguments.
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        logging.info("Generating Kubernetes manifests")
        
        # Validate input files exist if specified
        if hasattr(args, 'manifest_file') and args.manifest_file != DEFAULT_MANIFEST_FILE:
            if not os.path.exists(args.manifest_file):
                logging.warning(f"Manifest file {args.manifest_file} does not exist")
        
        if hasattr(args, 'execution_config') and args.execution_config != DEFAULT_EXECUTION_CONFIG:
            if not os.path.exists(args.execution_config):
                logging.warning(f"Execution config file {args.execution_config} does not exist")
        
        create_kubernetes_manifests(
            manifest_file=args.manifest_file,
            execution_config=args.execution_config,
            namespace=args.namespace
        )
        
        logging.info("Kubernetes manifests generated successfully")
        return EXIT_SUCCESS
        
    except Exception as e:
        logging.error(f"Failed to generate Kubernetes manifests: {e}")
        return EXIT_FAILURE


def export_config(args: argparse.Namespace) -> int:
    """Export execution configuration for external tools.
    
    Args:
        args: The command-line arguments.
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        logging.info("Exporting execution configuration")
        orchestrator = DistributedOrchestrator(args)
        
        # Discover models to get configuration
        from madengine.tools.discover_models import DiscoverModels
        discover_models = DiscoverModels(args=args)
        models = discover_models.run()
        
        if not models:
            logging.warning("No models discovered for configuration export")
        
        orchestrator.export_execution_config(models, args.output)
        logging.info(f"Execution configuration exported to: {args.output}")
        return EXIT_SUCCESS
        
    except Exception as e:
        logging.error(f"Failed to export configuration: {e}")
        return EXIT_FAILURE


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.
    
    Args:
        verbose: Enable verbose logging
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_common_args(args: argparse.Namespace) -> bool:
    """Validate common arguments across commands.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Validate timeout
    if hasattr(args, 'timeout') and args.timeout < -1:
        logging.error("Timeout must be -1 (default) or a positive integer")
        return False
    
    # Validate output directory exists for file outputs
    if hasattr(args, 'output') and args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            logging.error(f"Output directory does not exist: {output_dir}")
            return False
    
    return True


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def main() -> int:
    """Main function to parse the command-line arguments for distributed execution.
    
    Returns:
        int: Exit code
    """
    parser = argparse.ArgumentParser(
        description="MADEngine Distributed Orchestrator - Build and run models in distributed scenarios.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build models with specific tags and push to registry
  %(prog)s build --tags llama bert --registry localhost:5000 --clean-docker-cache
  
  # Run complete workflow (build + run) with specific tags and registry
  %(prog)s run --tags resnet --registry localhost:5000 --timeout 3600 --live-output
  
  # Run models using pre-built manifest (execution phase only - registry auto-detected)
  %(prog)s run --manifest-file build_manifest.json --timeout 3600
  
  # Run models using pre-built manifest with explicit registry override
  %(prog)s run --manifest-file build_manifest.json --registry custom-registry.com --timeout 3600
  
  # Generate Ansible playbook for distributed execution
  %(prog)s generate ansible --output madengine.yml
  
  # Generate Kubernetes manifests with custom namespace
  %(prog)s generate k8s --namespace madengine-prod
        """
    )
    
    subparsers = parser.add_subparsers(title="Commands", description="Available commands for distributed model execution.", dest="command")
    
    # Function to add common model arguments
    def add_model_arguments(parser):
        """Add common model selection and context arguments."""
        parser.add_argument('--tags', nargs='+', default=[], 
                           help="tags to run (can be multiple).")
        parser.add_argument('--ignore-deprecated-flag', action='store_true', 
                           help="Force run deprecated models even if marked deprecated.")
        parser.add_argument('--additional-context-file', default=None, 
                           help="additional context, as json file, to filter behavior of workloads. Overrides detected contexts.")
        parser.add_argument('--additional-context', default='{}', 
                           help="additional context, as string representation of python dict, to filter behavior of workloads. Overrides detected contexts and additional-context-file.")
        parser.add_argument('--data-config-file-name', default=DEFAULT_DATA_CONFIG, 
                           help="custom data configuration file.")
        parser.add_argument('--tools-json-file-name', default=DEFAULT_TOOLS_CONFIG, 
                           help="custom tools json configuration file.")
        parser.add_argument('--generate-sys-env-details', default=True, 
                           help='generate system config env details by default')
        parser.add_argument('--force-mirror-local', default=None, 
                           help="Path to force all relevant dataproviders to mirror data locally on.")
        parser.add_argument('--disable-skip-gpu-arch', action='store_true', 
                           help="disables skipping model based on gpu architecture")
        parser.add_argument('-v', '--verbose', action='store_true',
                           help="enable verbose logging")

    # Function to add build-specific arguments  
    def add_build_arguments(parser):
        """Add build-specific arguments."""
        parser.add_argument('--registry', type=str,
                           help='Docker registry to push images to')
        parser.add_argument('--clean-docker-cache', action='store_true', 
                           help="rebuild docker image without using cache")
        parser.add_argument('--manifest-output', type=str, default=DEFAULT_MANIFEST_FILE,
                           help='Output file for build manifest (default: build_manifest.json)')
        parser.add_argument('--summary-output', type=str,
                           help='Output file for build summary JSON')
        parser.add_argument('--live-output', action='store_true', 
                           help="prints output in real-time directly on STDOUT")
        parser.add_argument('-o', '--output', default=DEFAULT_PERF_OUTPUT, 
                           help='output file')

    # Function to add run-specific arguments
    def add_run_arguments(parser):
        """Add run-specific arguments."""
        parser.add_argument('--manifest-file', type=str, default='',
                           help='Build manifest file. If provided and exists, will run execution phase only. If not provided or file does not exist, will run complete workflow (build + run)')
        parser.add_argument('--registry', type=str,
                           help='Docker registry to push/pull images to/from (optional - can be auto-detected from manifest)')
        parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT, 
                           help="time out for model run in seconds; Overrides per-model timeout if specified or default timeout of 7200 (2 hrs). Timeout of 0 will never timeout.")
        parser.add_argument('--keep-alive', action='store_true', 
                           help="keep Docker container alive after run; will keep model directory after run")
        parser.add_argument('--keep-model-dir', action='store_true', 
                           help="keep model directory after run")
        parser.add_argument('--skip-model-run', action='store_true', 
                           help="skips running the model; will not keep model directory after run unless specified through keep-alive or keep-model-dir")
        parser.add_argument('--summary-output', type=str,
                           help='Output file for execution/workflow summary JSON')
        parser.add_argument('-o', '--output', default=DEFAULT_PERF_OUTPUT, 
                           help='output file')
        # Add build arguments for full workflow mode (no duplicates)
        parser.add_argument('--clean-docker-cache', action='store_true', 
                           help="rebuild docker image without using cache (used when running complete workflow)")
        parser.add_argument('--manifest-output', type=str, default=DEFAULT_MANIFEST_FILE,
                           help='Output file for build manifest when running complete workflow (default: build_manifest.json)')
        parser.add_argument('--live-output', action='store_true', 
                           help="prints output in real-time directly on STDOUT")

    # Build command
    parser_build = subparsers.add_parser('build', 
                                        description="Build Docker images for models in distributed scenarios", 
                                        help='Build Docker images for models')
    add_model_arguments(parser_build)
    add_build_arguments(parser_build)
    parser_build.set_defaults(func=build_models)

    # Run command
    parser_run = subparsers.add_parser('run', 
                                      description="Run model containers in distributed scenarios. If manifest-file is provided and exists, runs execution phase only (registry auto-detected from manifest). Otherwise runs complete workflow (build + run).", 
                                      help='Run model containers (with optional build phase)')
    add_model_arguments(parser_run)
    add_run_arguments(parser_run)
    parser_run.set_defaults(func=run_models)

    # Generate command group
    parser_generate = subparsers.add_parser('generate', help='Generate orchestration files')
    subparsers_generate = parser_generate.add_subparsers(title="Generate Commands", 
                                                        description="Available commands for generating orchestration files.", 
                                                        dest="generate_command")
    
    # Generate Ansible subcommand
    parser_generate_ansible = subparsers_generate.add_parser('ansible', 
                                                           description="Generate Ansible playbook for distributed execution", 
                                                           help='Generate Ansible playbook')
    parser_generate_ansible.add_argument('--manifest-file', type=str, default=DEFAULT_MANIFEST_FILE,
                                       help='Build manifest file (default: build_manifest.json)')
    parser_generate_ansible.add_argument('--execution-config', type=str, default=DEFAULT_EXECUTION_CONFIG,
                                       help='Execution config file (default: execution_config.json)')
    parser_generate_ansible.add_argument('--output', type=str, default=DEFAULT_ANSIBLE_OUTPUT,
                                       help='Output Ansible playbook file (default: madengine_distributed.yml)')
    parser_generate_ansible.set_defaults(func=generate_ansible)

    # Generate Kubernetes subcommand
    parser_generate_k8s = subparsers_generate.add_parser('k8s',
                                                        description="Generate Kubernetes manifests for distributed execution", 
                                                        help='Generate Kubernetes manifests')
    parser_generate_k8s.add_argument('--manifest-file', type=str, default=DEFAULT_MANIFEST_FILE,
                                   help='Build manifest file (default: build_manifest.json)')
    parser_generate_k8s.add_argument('--execution-config', type=str, default=DEFAULT_EXECUTION_CONFIG,
                                   help='Execution config file (default: execution_config.json)')
    parser_generate_k8s.add_argument('--namespace', type=str, default=DEFAULT_K8S_NAMESPACE,
                                   help='Kubernetes namespace (default: madengine)')
    parser_generate_k8s.set_defaults(func=generate_k8s)

    # Export config command
    parser_export = subparsers.add_parser('export-config',
                                         description="Export execution configuration for external tools", 
                                         help='Export execution configuration')
    add_model_arguments(parser_export)
    parser_export.add_argument('--output', type=str, default=DEFAULT_EXECUTION_CONFIG,
                              help='Output configuration file (default: execution_config.json)')
    parser_export.set_defaults(func=export_config)
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(getattr(args, 'verbose', False))
    
    if not args.command:
        parser.print_help()
        return EXIT_INVALID_ARGS
    
    # Validate common arguments
    if not validate_common_args(args):
        return EXIT_INVALID_ARGS
    
    try:
        logging.info(f"Starting {args.command} command")
        exit_code = args.func(args)
        
        if exit_code == EXIT_SUCCESS:
            logging.info(f"Command {args.command} completed successfully")
        else:
            logging.error(f"Command {args.command} failed with exit code {exit_code}")
            
        return exit_code
        
    except KeyboardInterrupt:
        logging.info("Operation cancelled by user")
        return EXIT_FAILURE
    except Exception as e:
        logging.error(f"Unexpected error in {args.command}: {e}")
        logging.debug("Exception details:", exc_info=True)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
