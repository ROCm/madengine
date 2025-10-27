"""Ansible infrastructure for orchestrated deployment."""

import os
import json
import subprocess
import time
from typing import Dict, Any
from pathlib import Path

from madengine.distribute.infrastructures.base import BaseInfrastructure, ExecutionResult


class AnsibleInfrastructure(BaseInfrastructure):
    """
    Ansible infrastructure for orchestrated distributed execution.
    
    Generates Ansible playbook and inventory, then executes via ansible-playbook.
    """
    
    @property
    def infrastructure_name(self) -> str:
        return "ansible"
    
    def generate_orchestration(self, launcher) -> Dict[str, Any]:
        """
        Generate Ansible playbook and inventory.
        
        Args:
            launcher: Launcher instance
            
        Returns:
            Dict with playbook and inventory paths
        """
        self.log("Generating Ansible orchestration files...")
        
        nodes = self.get_nodes()
        master_node = self.get_master_node()
        master_addr = master_node['address']
        total_nodes = launcher.get_nodes_required()
        
        # Limit nodes
        nodes = nodes[:total_nodes]
        
        # Get global config
        global_config = self.get_global_config()
        shared_fs = global_config.get('shared_filesystem', '/nfs/data')
        mad_workspace = global_config.get('mad_workspace', '~/MAD')
        
        # Generate inventory
        inventory_path = self._generate_inventory(nodes, launcher, master_addr)
        
        # Generate playbook
        playbook_path = self._generate_playbook(
            launcher=launcher,
            master_addr=master_addr,
            total_nodes=total_nodes,
            mad_workspace=mad_workspace,
            shared_fs=shared_fs
        )
        
        # Generate ansible.cfg
        config_path = self._generate_ansible_config()
        
        return {
            'type': 'ansible',
            'playbook': str(playbook_path),
            'inventory': str(inventory_path),
            'config': str(config_path),
            'master_addr': master_addr,
            'total_nodes': total_nodes
        }
    
    def _generate_inventory(
        self,
        nodes: list,
        launcher,
        master_addr: str
    ) -> Path:
        """Generate Ansible inventory file."""
        
        inventory_path = self.output_dir / "inventory.ini"
        
        inventory_content = "[training_nodes]\n"
        
        # Add each node with variables
        for node_rank, node in enumerate(nodes):
            hostname = node['hostname']
            address = node['address']
            username = node.get('username', 'ubuntu')
            ssh_key = node.get('ssh_key_path', '~/.ssh/id_rsa')
            
            # Build node-specific additional_context
            additional_context = self.manifest.get('additional_context', {}).copy()
            multi_node_args = additional_context.get('multi_node_args', {}).copy()
            
            # Add node-specific values
            multi_node_args['NODE_RANK'] = str(node_rank)
            multi_node_args['MASTER_ADDR'] = master_addr
            
            if 'RUNNER' not in multi_node_args:
                multi_node_args['RUNNER'] = 'torchrun'
            
            additional_context['multi_node_args'] = multi_node_args
            
            # Write inventory line
            inventory_content += f"{hostname} "
            inventory_content += f"ansible_host={address} "
            inventory_content += f"ansible_user={username} "
            inventory_content += f"ansible_ssh_private_key_file={ssh_key} "
            inventory_content += f"node_rank={node_rank} "
            inventory_content += f"additional_context='{json.dumps(additional_context)}'\n"
        
        with open(inventory_path, 'w') as f:
            f.write(inventory_content)
        
        self.log(f"Generated inventory: {inventory_path}")
        
        return inventory_path
    
    def _generate_playbook(
        self,
        launcher,
        master_addr: str,
        total_nodes: int,
        mad_workspace: str,
        shared_fs: str
    ) -> Path:
        """Generate Ansible playbook."""
        
        playbook_path = self.output_dir / "distributed_playbook.yml"
        
        # Get timeout from manifest
        timeout = self.manifest.get('timeout', 7200)
        
        playbook = f"""---
- name: Setup MAD Environment
  hosts: training_nodes
  gather_facts: yes
  tasks:
    - name: Ensure MAD repository exists
      stat:
        path: {mad_workspace}
      register: mad_dir
    
    - name: Clone MAD if not exists
      git:
        repo: https://github.com/ROCm/MAD.git
        dest: {mad_workspace}
        update: yes
      when: not mad_dir.stat.exists
    
    - name: Ensure Python venv exists
      stat:
        path: {mad_workspace}/venv
      register: venv_dir
    
    - name: Create virtual environment
      command: python3 -m venv venv
      args:
        chdir: {mad_workspace}
      when: not venv_dir.stat.exists
    
    - name: Install madengine
      pip:
        name: madengine
        virtualenv: {mad_workspace}/venv
        state: present
    
    - name: Copy build manifest
      copy:
        src: {self.manifest.get('_path', 'build_manifest.json')}
        dest: {mad_workspace}/build_manifest.json
    
    - name: Copy credentials if exists
      copy:
        src: credential.json
        dest: {mad_workspace}/credential.json
      ignore_errors: yes
    
    - name: Copy data config if exists
      copy:
        src: data.json
        dest: {mad_workspace}/data.json
      ignore_errors: yes

- name: Launch Distributed Training
  hosts: training_nodes
  tasks:
    - name: Launch training on each node (async)
      shell: |
        cd {mad_workspace}
        source venv/bin/activate
        
        madengine-cli run \\
          --manifest-file build_manifest.json \\
          --additional-context '{{{{ additional_context }}}}' \\
          --verbose
      async: {timeout + 600}
      poll: 0
      register: training_job
      environment:
        NCCL_DEBUG: INFO
    
    - name: Wait for all nodes to complete
      async_status:
        jid: "{{{{ training_job.ansible_job_id }}}}"
      register: job_result
      until: job_result.finished
      retries: {timeout // 30}
      delay: 30

- name: Collect Results
  hosts: training_nodes
  tasks:
    - name: Display completion status
      debug:
        msg: "Node {{{{ inventory_hostname }}}} (rank {{{{ node_rank }}}}) completed"
"""
        
        with open(playbook_path, 'w') as f:
            f.write(playbook)
        
        self.log(f"Generated playbook: {playbook_path}")
        
        return playbook_path
    
    def _generate_ansible_config(self) -> Path:
        """Generate ansible.cfg file."""
        
        config_path = self.output_dir / "ansible.cfg"
        
        infra_config = self.get_infrastructure_config()
        forks = infra_config.get('forks', 10)
        
        config = f"""[defaults]
inventory = inventory.ini
forks = {forks}
host_key_checking = False
timeout = 30
stdout_callback = yaml
"""
        
        with open(config_path, 'w') as f:
            f.write(config)
        
        self.log(f"Generated ansible.cfg: {config_path}")
        
        return config_path
    
    def execute(self, orchestration: Dict[str, Any]) -> ExecutionResult:
        """
        Execute distributed workload via Ansible.
        
        Runs ansible-playbook command.
        """
        self.log("Executing distributed workload via Ansible...", force=True)
        
        result = ExecutionResult()
        
        playbook = orchestration['playbook']
        inventory = orchestration['inventory']
        config = orchestration['config']
        
        start_time = time.time()
        
        try:
            # Set ansible.cfg path
            env = os.environ.copy()
            env['ANSIBLE_CONFIG'] = config
            
            # Run ansible-playbook
            cmd = [
                'ansible-playbook',
                '-i', inventory,
                playbook
            ]
            
            if self.verbose:
                cmd.append('-vv')
            
            self.log(f"Running: {' '.join(cmd)}")
            
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=7200
            )
            
            result.execution_time = time.time() - start_time
            
            if proc.returncode == 0:
                result.success = True
                result.successful_nodes = orchestration['total_nodes']
                result.total_nodes = orchestration['total_nodes']
                self.log(f"Ansible execution completed successfully", force=True)
            else:
                result.success = False
                result.failed_nodes = orchestration['total_nodes']
                result.total_nodes = orchestration['total_nodes']
                result.errors.append(proc.stderr)
                self.log(f"Ansible execution failed", force=True)
            
            # Store output
            result.node_results['ansible_output'] = {
                'success': result.success,
                'output': proc.stdout,
                'error': proc.stderr if not result.success else ''
            }
            
        except subprocess.TimeoutExpired:
            result.success = False
            result.errors.append("Ansible execution timeout")
            result.execution_time = time.time() - start_time
        except FileNotFoundError:
            result.success = False
            result.errors.append("ansible-playbook command not found. Is Ansible installed?")
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            result.execution_time = time.time() - start_time
        
        self.log(f"Execution completed in {result.execution_time:.2f}s", force=True)
        
        return result

