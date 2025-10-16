#!/usr/bin/env python3
"""SSH Multi-Node Runner for MAD Engine

This script orchestrates distributed training across multiple nodes using SSH.
It automatically configures and executes madengine commands on remote nodes
for PyTorch Megatron-LM training workloads.

Example Usage:
    python run.py --model pyt_megatron_lm_train_llama2_7b \
                   --nodes 192.168.1.1,192.168.1.2 \
                   --master-addr 192.168.0.1 \
                   --master-port 4000 \
                   --ssh-user username \
                   --ssh-key /path/to/ssh/key \
                   --shared-data-path /nfs/data

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

# Local imports
try:
    from config_manager import MultiNodeConfig, merge_config_file_with_args
    from ssh_client_manager import SSHClientManager
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, os.path.dirname(__file__))
    from config_manager import MultiNodeConfig, merge_config_file_with_args
    from ssh_client_manager import SSHClientManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ValidationError(Exception):
    """Exception raised when validation fails."""
    pass


class SSHMultiNodeRunner:
    """SSH-based multi-node runner for distributed training."""
    
    def __init__(self, config: MultiNodeConfig):
        """Initialize the SSH multi-node runner.
        
        Args:
            config: Complete configuration object
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create SSH client managers for each node
        self.ssh_managers = {}
        for node in config.cluster.nodes:
            self.ssh_managers[node] = SSHClientManager(
                hostname=node,
                username=config.ssh.user,
                password=config.ssh.password,
                key_filename=config.ssh.key_file,
                timeout=config.ssh.timeout,
                max_retries=config.ssh.max_retries
            )
    
    def _build_madengine_command(self, node_rank: int) -> str:
        """Build the madengine command for a specific node, using venv's madengine binary."""
        multi_node_args = {
            'RUNNER': 'torchrun',
            'MASTER_ADDR': self.config.cluster.master_addr,
            'MASTER_PORT': str(self.config.cluster.master_port),
            'NNODES': str(len(self.config.cluster.nodes)),
            'NODE_RANK': str(node_rank),
            'NCCL_SOCKET_IFNAME': self.config.training.nccl_interface,
            'GLOO_SOCKET_IFNAME': self.config.training.gloo_interface
        }
        additional_context = f"'{json.dumps({'multi_node_args': multi_node_args})}'"
        # Use venv/bin/madengine explicitly
        madengine_bin = os.path.join('venv', 'bin', 'madengine')
        cmd_parts = [
            madengine_bin,
            'run',
            '--tags', self.config.training.model,
            '--additional-context', additional_context
        ]
        if self.config.training.shared_data_path:
            cmd_parts.extend(['--force-mirror-local', self.config.training.shared_data_path])
        if self.config.training.additional_args:
            cmd_parts.append(self.config.training.additional_args)
        return ' '.join(cmd_parts)
    
    def _check_node_connectivity(self) -> List[str]:
        """Check connectivity to all nodes.
        
        Returns:
            List of nodes that are reachable
        """
        reachable_nodes = []
        
        self.logger.info("Checking connectivity to all nodes...")
        
        for hostname in self.config.cluster.nodes:
            ssh_manager = self.ssh_managers[hostname]
            if ssh_manager.test_connectivity():
                reachable_nodes.append(hostname)
                self.logger.info(f"✓ {hostname} is reachable")
            else:
                self.logger.error(f"✗ {hostname} is not reachable")
        
        return reachable_nodes
    
    def _validate_remote_prerequisites(self, hostname: str) -> Tuple[bool, str]:
        """Validate that remote node has required prerequisites.
        
        Args:
            hostname: The hostname/IP of the node to validate
            
        Returns:
            Tuple of (success, error_message)
        """
        ssh_manager = self.ssh_managers[hostname]
        
        try:

            # Check if MAD directory exists, clone if missing
            self.logger.info(f"Checking if {self.config.madengine.working_directory} directory exists on {hostname}...")
            exit_code, stdout, stderr = ssh_manager.execute_command(
                f'test -d {self.config.madengine.working_directory} && echo "exists" || echo "missing"'
            )
            if stdout.strip() == "missing":
                self.logger.info(f"{self.config.madengine.working_directory} not found on {hostname}, cloning MAD repo...")
                exit_code, stdout, stderr = ssh_manager.execute_command(
                    f'git clone https://github.com/ROCm/MAD.git {self.config.madengine.working_directory}'
                )
                if exit_code != 0:
                    return False, f"Failed to clone MAD repo to {self.config.madengine.working_directory} on {hostname}: {stderr}"
            elif exit_code != 0:
                return False, f"Failed to check {self.config.madengine.working_directory} on {hostname}: {stderr}"

            # Ensure venv exists, create if missing
            venv_path = os.path.join(self.config.madengine.working_directory, 'venv')
            venv_python = os.path.join(venv_path, 'bin', 'python3')
            venv_madengine = os.path.join(venv_path, 'bin', 'madengine')
            self.logger.info(f"Ensuring venv exists at {venv_path} on {hostname}...")
            exit_code, stdout, stderr = ssh_manager.execute_command(
                f'cd {self.config.madengine.working_directory} && [ -d venv ] || python3 -m venv venv'
            )
            if exit_code != 0:
                return False, f"Failed to create venv in {self.config.madengine.working_directory} on {hostname}: {stderr}"

            # Install madengine in venv
            self.logger.info(f"Installing madengine in venv on {hostname}...")
            exit_code, stdout, stderr = ssh_manager.execute_command(
                f'cd {self.config.madengine.working_directory} && source venv/bin/activate && pip install --upgrade pip && pip install git+https://github.com/ROCm/madengine.git@main -q'
            )
            if exit_code != 0:
                return False, f"Failed to install madengine in venv on {hostname}: {stderr}"

            # Check if madengine is accessible in venv
            madengine_check_cmd = f'cd {self.config.madengine.working_directory} && source venv/bin/activate && madengine --help > /dev/null 2>&1 && echo "found" || echo "missing"'
            self.logger.debug(f"Checking madengine installation in venv on {hostname} with command: {madengine_check_cmd}")
            exit_code, stdout, stderr = ssh_manager.execute_command(madengine_check_cmd)
            self.logger.error(f"[DEBUG] madengine check in venv on {hostname}: exit_code={exit_code}, stdout='{stdout}', stderr='{stderr}'")
            if stdout.strip() != "found":
                return False, f"madengine not found or not accessible in venv on {hostname} (exit_code={exit_code}, stdout='{stdout}', stderr='{stderr}')"
            
            # Check if we can access the working directory
            self.logger.debug(f"Checking access to {self.config.madengine.working_directory} directory on {hostname}...")
            exit_code, stdout, stderr = ssh_manager.execute_command(
                f'cd {self.config.madengine.working_directory} && pwd'
            )
            
            if stderr or not stdout.endswith(self.config.madengine.working_directory):
                return False, f"Cannot access {self.config.madengine.working_directory} directory on {hostname}"
            
            # Check shared data path if specified and not default
            if self.config.training.shared_data_path != '/nfs/data':
                self.logger.debug(f"Checking shared data path on {hostname}...")
                exit_code, stdout, stderr = ssh_manager.execute_command(
                    f'test -d "{self.config.training.shared_data_path}" && echo "exists" || echo "missing"'
                )
                
                if stdout != "exists":
                    return False, f"Shared data path '{self.config.training.shared_data_path}' not found on {hostname}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Error validating prerequisites on {hostname}: {str(e)}"
    
    def _check_all_prerequisites(self) -> bool:
        """Check prerequisites on all nodes.
        
        Returns:
            True if all nodes meet prerequisites, False otherwise
        """
        self.logger.info("Validating prerequisites on all nodes...")
        
        failed_nodes = []
        
        for hostname in self.config.cluster.nodes:
            success, error_msg = self._validate_remote_prerequisites(hostname)
            if not success:
                self.logger.error(f"❌ {hostname}: {error_msg}")
                failed_nodes.append((hostname, error_msg))
            else:
                self.logger.info(f"✅ {hostname}: All prerequisites met")
        
        if failed_nodes:
            self.logger.error(f"Prerequisites check failed for {len(failed_nodes)} node(s)")
            self._print_setup_instructions()
            return False
        
        self.logger.info("All nodes meet the prerequisites")
        return True
    
    def _print_setup_instructions(self) -> None:
        """Print setup instructions for remote nodes."""
        instructions = f"""
{"="*60}
🔧 REMOTE NODE SETUP INSTRUCTIONS
{"="*60}

To prepare your remote nodes for multi-node training:

1. {self.config.madengine.working_directory} Directory:
   • Create or ensure the {self.config.madengine.working_directory} directory exists
   • Command: mkdir -p ~/{self.config.madengine.working_directory}

2. MAD Engine Installation:
   • Install madengine on each remote node
   • Command: pip install git+https://github.com/ROCm/madengine.git@main
   • Verify with: {self.config.madengine.path} --help

3. Shared Data Path:
   • Ensure the shared data path '{self.config.training.shared_data_path}' exists
   • This should be a shared filesystem mounted on all nodes

4. SSH Access:
   • Ensure SSH key-based or password authentication is configured
   • Test SSH access manually before running this script

5. Network Configuration:
   • Ensure nodes can communicate on the specified interfaces
   • NCCL interface: {self.config.training.nccl_interface}
   • GLOO interface: {self.config.training.gloo_interface}
   • Master node {self.config.cluster.master_addr} should be accessible on port {self.config.cluster.master_port}

{"="*60}
        """
        print(instructions)
    
    def _execute_on_node(self, hostname: str, node_rank: int) -> Tuple[str, bool, str]:
        """Execute madengine command on a single node, ensuring venv usage."""
        ssh_manager = self.ssh_managers[hostname]
        try:
            # Build madengine command (uses venv/bin/madengine)
            command = self._build_madengine_command(node_rank)
            # Compose setup and run commands, always use venv's python/pip
            setup_commands = [
                f"cd {self.config.madengine.working_directory}",
                "venv/bin/python -m pip install --upgrade pip",
                "venv/bin/python -m pip install git+https://github.com/ROCm/madengine.git@main -q"
            ]
            full_command = f"{' && '.join(setup_commands)} && {command}"
            self.logger.info(f"🚀 Executing on {hostname} (rank {node_rank}): {full_command}")
            with ssh_manager.get_client() as client:
                stdin, stdout, stderr = client.exec_command(
                    full_command,
                    timeout=self.config.training.timeout
                )
                output_lines = []
                error_lines = []
                for line in stdout:
                    line = line.strip()
                    if line:
                        self.logger.info(f"[{hostname}:{node_rank}] {line}")
                        output_lines.append(line)
                for line in stderr:
                    line = line.strip()
                    if line:
                        self.logger.warning(f"[{hostname}:{node_rank}] ERROR: {line}")
                        error_lines.append(line)
                exit_code = stdout.channel.recv_exit_status()
                if exit_code == 0:
                    return hostname, True, '\n'.join(output_lines)
                else:
                    error_output = '\n'.join(error_lines) if error_lines else "Command failed with no error output"
                    return hostname, False, f"Exit code {exit_code}: {error_output}"
        except Exception as e:
            return hostname, False, str(e)
    
    def run(self) -> bool:
        """Run distributed training across all nodes.
        
        Returns:
            True if all nodes completed successfully, False otherwise
        """
        self.logger.info(f"Starting multi-node training on {len(self.config.cluster.nodes)} nodes")
        self.logger.info(f"Model: {self.config.training.model}")
        self.logger.info(f"Master: {self.config.cluster.master_addr}:{self.config.cluster.master_port}")
        self.logger.info(f"Shared data: {self.config.training.shared_data_path}")
        self.logger.info(f"Nodes: {', '.join(self.config.cluster.nodes)}")
        
        # Check connectivity to all nodes
        reachable_nodes = self._check_node_connectivity()
        
        if len(reachable_nodes) != len(self.config.cluster.nodes):
            unreachable = set(self.config.cluster.nodes) - set(reachable_nodes)
            self.logger.error(f"Some nodes are unreachable: {', '.join(unreachable)}")
            return False
        
        self.logger.info("All nodes are reachable")
        
        # Validate prerequisites on all nodes
        if not self._check_all_prerequisites():
            return False
        
        # Execute on all nodes concurrently
        results = []
        
        with ThreadPoolExecutor(max_workers=len(self.config.cluster.nodes)) as executor:
            # Submit jobs for all nodes
            futures = []
            for i, hostname in enumerate(self.config.cluster.nodes):
                future = executor.submit(self._execute_on_node, hostname, i)
                futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(futures):
                hostname, success, output = future.result()
                results.append((hostname, success, output))
                
                if success:
                    self.logger.info(f"✅ {hostname} completed successfully")
                else:
                    self.logger.error(f"❌ {hostname} failed: {output}")
        
        # Check overall success
        successful_nodes = [r[0] for r in results if r[1]]
        failed_nodes = [r[0] for r in results if not r[1]]
        
        self.logger.info(f"Training Results:")
        self.logger.info(f"✅ Successful nodes: {len(successful_nodes)}/{len(self.config.cluster.nodes)}")
        
        if failed_nodes:
            self.logger.error(f"❌ Failed nodes: {', '.join(failed_nodes)}")
            return False
        
        self.logger.info("🎉 Multi-node training completed successfully!")
        return True


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="SSH Multi-Node Runner for MAD Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with SSH key authentication
  python run.py --model pyt_megatron_lm_train_llama2_7b \\
                 --nodes 192.168.1.1,192.168.1.2 \\
                 --master-addr 192.168.0.1 \\
                 --ssh-user ubuntu \\
                 --ssh-key ~/.ssh/id_rsa
  
  # Run with password authentication
  python run.py --model pyt_megatron_lm_train_llama2_7b \\
                 --nodes node1,node2,node3 \\
                 --ssh-user root \\
                 --ssh-password mypassword \\
                 --shared-data-path /shared/data
                 
  # Run with configuration file
  python run.py --config config.ini
        """
    )
    
    # Configuration file option
    parser.add_argument(
        '--config',
        help='Path to configuration file (INI format)'
    )
    
    # Required arguments (unless provided in config)
    parser.add_argument(
        '--model', 
        help='Model tag to run (e.g., pyt_megatron_lm_train_llama2_7b)'
    )
    
    parser.add_argument(
        '--nodes',
        help='Comma-separated list of node hostnames/IPs'
    )
    
    parser.add_argument(
        '--ssh-user',
        help='SSH username for all nodes'
    )
    
    # SSH authentication (one required unless in config)
    parser.add_argument(
        '--ssh-password',
        help='SSH password for all nodes'
    )
    parser.add_argument(
        '--ssh-key',
        help='Path to SSH private key file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--master-addr',
        help='Master node address (defaults to first node)'
    )
    
    parser.add_argument(
        '--master-port',
        type=int,
        default=4000,
        help='Master node port (default: 4000)'
    )
    
    parser.add_argument(
        '--shared-data-path',
        default='/nfs/data',
        help='Path to shared data filesystem (default: /nfs/data)'
    )
    
    parser.add_argument(
        '--nccl-interface',
        default='ens14np0',
        help='NCCL socket interface (default: ens14np0)'
    )
    
    parser.add_argument(
        '--gloo-interface', 
        default='ens14np0',
        help='GLOO socket interface (default: ens14np0)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=3600,
        help='Execution timeout in seconds (default: 3600)'
    )
    
    parser.add_argument(
        '--madengine-path',
        default='madengine',
        help='Path to madengine executable (default: madengine)'
    )
    
    parser.add_argument(
        '--additional-args',
        help='Additional arguments to pass to madengine'
    )
    
    parser.add_argument(
        '--working-directory',
        default='MAD',
        help='Working directory on remote nodes (default: MAD)'
    )
    
    parser.add_argument(
        '--ssh-timeout',
        type=int,
        default=30,
        help='SSH connection timeout in seconds (default: 30)'
    )
    
    parser.add_argument(
        '--ssh-max-retries',
        type=int,
        default=3,
        help='Maximum SSH connection retry attempts (default: 3)'
    )
    
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    try:
        args = parse_args()
        
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # Create configuration
        if args.config:
            config = merge_config_file_with_args(args.config, args)
        else:
            config = MultiNodeConfig.from_args(args)
        
        # Validate configuration
        config.validate()
        
        # Create and run the runner
        runner = SSHMultiNodeRunner(config)
        success = runner.run()
        
        if success:
            print("🎯 All nodes completed successfully!")
            sys.exit(0)
        else:
            print("💥 Some nodes failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
