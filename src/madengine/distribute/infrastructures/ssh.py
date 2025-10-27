"""SSH infrastructure for direct node connections."""

import os
import json
import subprocess
import time
from typing import Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from madengine.distribute.infrastructures.base import BaseInfrastructure, ExecutionResult


class SSHInfrastructure(BaseInfrastructure):
    """
    SSH infrastructure for direct node connections.
    
    Generates launch scripts for each node and executes them via SSH.
    """
    
    @property
    def infrastructure_name(self) -> str:
        return "ssh"
    
    def generate_orchestration(self, launcher) -> Dict[str, Any]:
        """
        Generate SSH launch scripts for each node.
        
        Args:
            launcher: Launcher instance (torchrun, mpirun, etc.)
            
        Returns:
            Dict with script paths and metadata
        """
        self.log("Generating SSH orchestration files...")
        
        nodes = self.get_nodes()
        master_node = self.get_master_node()
        master_addr = master_node['address']
        total_nodes = launcher.get_nodes_required()
        
        # Limit nodes to what launcher requires
        nodes = nodes[:total_nodes]
        
        # Get global config
        global_config = self.get_global_config()
        shared_fs = global_config.get('shared_filesystem', '/nfs/data')
        mad_workspace = global_config.get('mad_workspace', '~/MAD')
        
        scripts = {}
        
        # Generate launch script for each node
        for node_rank, node in enumerate(nodes):
            # Get launch command from launcher
            launch_cmd = launcher.generate_launch_command(
                node_rank=node_rank,
                master_addr=master_addr,
                total_nodes=total_nodes,
                manifest_path='build_manifest.json'
            )
            
            # Get environment variables
            env_vars = launcher.get_required_env_vars(node_rank, master_addr)
            
            # Build script
            script_content = self._build_launch_script(
                node=node,
                node_rank=node_rank,
                launch_cmd=launch_cmd,
                env_vars=env_vars,
                mad_workspace=mad_workspace,
                shared_fs=shared_fs
            )
            
            # Write script
            script_name = f"launch_node_{node_rank}.sh"
            script_path = self.output_dir / script_name
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            script_path.chmod(0o755)
            
            scripts[node_rank] = {
                'script_path': str(script_path),
                'node': node,
                'node_rank': node_rank
            }
            
            self.log(f"Generated script for node {node_rank}: {node['hostname']}")
        
        # Generate parallel launcher
        parallel_script = self._generate_parallel_launcher(scripts)
        
        return {
            'type': 'ssh',
            'scripts': scripts,
            'parallel_launcher': parallel_script,
            'master_addr': master_addr,
            'total_nodes': total_nodes
        }
    
    def _build_launch_script(
        self,
        node: Dict[str, Any],
        node_rank: int,
        launch_cmd: str,
        env_vars: Dict[str, str],
        mad_workspace: str,
        shared_fs: str
    ) -> str:
        """Build launch script for a node."""
        
        # Environment variables section
        env_section = "\n".join([f"export {k}={v}" for k, v in env_vars.items()])
        
        script = f"""#!/bin/bash
set -e

echo "==============================================="
echo "Node {node_rank}: {node['hostname']}"
echo "Address: {node['address']}"
echo "==============================================="

# Environment variables
{env_section}

# Ensure MAD workspace exists
if [ ! -d "{mad_workspace}" ]; then
    echo "ERROR: MAD workspace not found: {mad_workspace}"
    echo "Please ensure MAD is cloned and setup on all nodes"
    exit 1
fi

cd {mad_workspace}

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "$HOME/venv/bin/activate" ]; then
    source $HOME/venv/bin/activate
else
    echo "WARNING: No virtual environment found, using system Python"
fi

# Verify madengine is available
if ! command -v madengine-cli &> /dev/null; then
    echo "ERROR: madengine-cli not found. Please install madengine."
    exit 1
fi

echo "Starting distributed execution on node {node_rank}..."
echo ""

# Execute launch command
{launch_cmd}

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "Node {node_rank} completed successfully"
else
    echo ""
    echo "Node {node_rank} failed with exit code: $exit_code"
fi

exit $exit_code
"""
        return script
    
    def _generate_parallel_launcher(self, scripts: Dict[int, Dict]) -> str:
        """Generate parallel SSH launcher script."""
        
        parallel_script_path = self.output_dir / "parallel_launch.sh"
        
        # Build SSH commands for each node
        ssh_commands = []
        for node_rank, info in scripts.items():
            node = info['node']
            script_path = info['script_path']
            
            username = node.get('username', os.environ.get('USER', 'ubuntu'))
            ssh_key = node.get('ssh_key_path', '~/.ssh/id_rsa')
            
            # Copy script to node and execute
            ssh_cmd = f"""
# Node {node_rank}: {node['hostname']}
echo "Launching node {node_rank}: {node['hostname']}"
scp -i {ssh_key} {script_path} {username}@{node['address']}:/tmp/launch_node_{node_rank}.sh
ssh -i {ssh_key} {username}@{node['address']} 'bash /tmp/launch_node_{node_rank}.sh' &
pids[{node_rank}]=$!
"""
            ssh_commands.append(ssh_cmd)
        
        script_content = f"""#!/bin/bash
set -e

echo "========================================="
echo "Parallel SSH Distributed Launch"
echo "Total nodes: {len(scripts)}"
echo "========================================="
echo ""

# Launch all nodes in parallel
declare -a pids
{''.join(ssh_commands)}

echo ""
echo "Waiting for all nodes to complete..."
echo ""

# Wait for all nodes
failed=0
for i in "${{!pids[@]}}"; do
    pid=${{pids[$i]}}
    if wait $pid; then
        echo "Node $i: SUCCESS"
    else
        echo "Node $i: FAILED"
        failed=$((failed + 1))
    fi
done

echo ""
echo "========================================="
if [ $failed -eq 0 ]; then
    echo "All nodes completed successfully!"
    echo "========================================="
    exit 0
else
    echo "$failed node(s) failed"
    echo "========================================="
    exit 1
fi
"""
        
        with open(parallel_script_path, 'w') as f:
            f.write(script_content)
        parallel_script_path.chmod(0o755)
        
        self.log(f"Generated parallel launcher: {parallel_script_path}")
        
        return str(parallel_script_path)
    
    def execute(self, orchestration: Dict[str, Any]) -> ExecutionResult:
        """
        Execute distributed workload via SSH.
        
        Connects to each node via SSH and executes the launch script.
        """
        self.log("Executing distributed workload via SSH...", force=True)
        
        result = ExecutionResult()
        scripts = orchestration['scripts']
        
        # Get parallelism config
        infra_config = self.get_infrastructure_config()
        parallel = infra_config.get('parallel_execution', True)
        
        start_time = time.time()
        
        if parallel:
            # Execute in parallel
            self.log(f"Executing on {len(scripts)} nodes in parallel...")
            with ThreadPoolExecutor(max_workers=len(scripts)) as executor:
                futures = {
                    executor.submit(self._execute_node, node_rank, info): node_rank
                    for node_rank, info in scripts.items()
                }
                
                for future in as_completed(futures):
                    node_rank = futures[future]
                    try:
                        node_result = future.result()
                        result.add_node_result(
                            f"node_{node_rank}",
                            node_result['success'],
                            node_result.get('output', ''),
                            node_result.get('error', '')
                        )
                    except Exception as e:
                        result.add_node_result(
                            f"node_{node_rank}",
                            False,
                            '',
                            str(e)
                        )
        else:
            # Execute sequentially
            self.log(f"Executing on {len(scripts)} nodes sequentially...")
            for node_rank, info in scripts.items():
                try:
                    node_result = self._execute_node(node_rank, info)
                    result.add_node_result(
                        f"node_{node_rank}",
                        node_result['success'],
                        node_result.get('output', ''),
                        node_result.get('error', '')
                    )
                except Exception as e:
                    result.add_node_result(
                        f"node_{node_rank}",
                        False,
                        '',
                        str(e)
                    )
        
        result.execution_time = time.time() - start_time
        
        self.log(f"Execution completed in {result.execution_time:.2f}s", force=True)
        self.log(f"Success: {result.successful_nodes}/{result.total_nodes} nodes", force=True)
        
        return result
    
    def _execute_node(self, node_rank: int, info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute launch script on a specific node via SSH."""
        
        node = info['node']
        script_path = info['script_path']
        
        hostname = node['hostname']
        address = node['address']
        username = node.get('username', os.environ.get('USER', 'ubuntu'))
        ssh_key = node.get('ssh_key_path', '~/.ssh/id_rsa')
        
        self.log(f"Executing on node {node_rank}: {hostname} ({address})")
        
        try:
            # Copy script to node
            scp_cmd = [
                'scp',
                '-i', os.path.expanduser(ssh_key),
                '-o', 'StrictHostKeyChecking=no',
                script_path,
                f"{username}@{address}:/tmp/launch_node_{node_rank}.sh"
            ]
            
            scp_result = subprocess.run(
                scp_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if scp_result.returncode != 0:
                return {
                    'success': False,
                    'error': f"SCP failed: {scp_result.stderr}"
                }
            
            # Execute script via SSH
            ssh_cmd = [
                'ssh',
                '-i', os.path.expanduser(ssh_key),
                '-o', 'StrictHostKeyChecking=no',
                f"{username}@{address}",
                f"bash /tmp/launch_node_{node_rank}.sh"
            ]
            
            ssh_result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            success = ssh_result.returncode == 0
            
            if success:
                self.log(f"Node {node_rank} ({hostname}) completed successfully")
            else:
                self.log(f"Node {node_rank} ({hostname}) failed")
            
            return {
                'success': success,
                'output': ssh_result.stdout,
                'error': ssh_result.stderr if not success else ''
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Execution timeout'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

