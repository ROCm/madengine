"""SLURM infrastructure for HPC clusters."""

import os
import json
import subprocess
import time
from typing import Dict, Any
from pathlib import Path

from madengine.distribute.infrastructures.base import BaseInfrastructure, ExecutionResult


class SlurmInfrastructure(BaseInfrastructure):
    """
    SLURM infrastructure for HPC cluster execution.
    
    Generates sbatch script and submits to SLURM scheduler.
    """
    
    @property
    def infrastructure_name(self) -> str:
        return "slurm"
    
    def generate_orchestration(self, launcher) -> Dict[str, Any]:
        """
        Generate SLURM sbatch script.
        
        Args:
            launcher: Launcher instance
            
        Returns:
            Dict with sbatch script path
        """
        self.log("Generating SLURM orchestration files...")
        
        # Get login node
        nodes = self.get_nodes()
        if not nodes or nodes[0].get('role') != 'login':
            raise ValueError("SLURM infrastructure requires login node in inventory with role='login'")
        
        login_node = nodes[0]
        
        # Get SLURM config
        infra_config = self.get_infrastructure_config()
        partition = infra_config.get('partition', 'gpu')
        account = infra_config.get('account')
        qos = infra_config.get('qos')
        time_limit = infra_config.get('time_limit', '24:00:00')
        
        # Get global config
        global_config = self.get_global_config()
        shared_fs = global_config.get('shared_filesystem', '/gpfs/data')
        mad_workspace = global_config.get('mad_workspace', '$HOME/MAD')
        
        # Generate sbatch script
        sbatch_path = self._generate_sbatch_script(
            launcher=launcher,
            partition=partition,
            account=account,
            qos=qos,
            time_limit=time_limit,
            mad_workspace=mad_workspace,
            shared_fs=shared_fs
        )
        
        return {
            'type': 'slurm',
            'sbatch_script': str(sbatch_path),
            'login_node': login_node,
            'partition': partition,
            'account': account
        }
    
    def _generate_sbatch_script(
        self,
        launcher,
        partition: str,
        account: str,
        qos: str,
        time_limit: str,
        mad_workspace: str,
        shared_fs: str
    ) -> Path:
        """Generate SLURM sbatch script."""
        
        sbatch_path = self.output_dir / "submit.sbatch"
        
        total_nodes = launcher.get_nodes_required()
        processes_per_node = launcher.get_processes_per_node()
        
        # Get timeout
        timeout = self.manifest.get('timeout', 7200)
        
        # Get multi_node_args from manifest
        additional_context = self.manifest.get('additional_context', {}).copy()
        multi_node_args = additional_context.get('multi_node_args', {}).copy()
        
        # SLURM-specific environment variables
        master_port = multi_node_args.get('MASTER_PORT', '29500')
        network_interface = multi_node_args.get('NCCL_SOCKET_IFNAME', 'ib0')
        
        # Get manifest filename (not full path)
        manifest_filename = os.path.basename(self.manifest.get('_path', 'build_manifest.json'))
        
        # Generate launch command template
        launch_cmd = launcher.generate_launch_command(
            node_rank=0,  # Placeholder, will use $SLURM_PROCID
            master_addr='$MASTER_ADDR',
            total_nodes=total_nodes,
            manifest_path='build_manifest.json'
        )
        
        # Build sbatch script
        account_line = f"#SBATCH --account={account}" if account else ""
        qos_line = f"#SBATCH --qos={qos}" if qos else ""
        
        script = f"""#!/bin/bash
#SBATCH --job-name=madengine_distributed
#SBATCH --nodes={total_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node={processes_per_node}
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
{account_line}
{qos_line}
#SBATCH --output=madengine_%j.out
#SBATCH --error=madengine_%j.err

echo "========================================="
echo "SLURM Distributed Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Node list: $SLURM_NODELIST"
echo "========================================="

# Get master node address
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT={master_port}

echo "Master node: $MASTER_ADDR:$MASTER_PORT"
echo ""

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME={network_interface}
export GLOO_SOCKET_IFNAME={network_interface}

# Setup MAD workspace
cd $SLURM_SUBMIT_DIR

# Ensure MAD is available
if [ ! -d "{mad_workspace}" ]; then
    echo "Setting up MAD workspace..."
    git clone https://github.com/ROCm/MAD.git {mad_workspace}
    cd {mad_workspace}
    python3 -m venv venv
    source venv/bin/activate
    pip install madengine
else
    cd {mad_workspace}
    source venv/bin/activate
fi

# Copy manifest
cp $SLURM_SUBMIT_DIR/{manifest_filename} ./build_manifest.json

echo "Launching distributed training across $SLURM_NNODES nodes..."
echo ""

# Launch with srun (each task runs on one node)
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c '
    export NODE_RANK=$SLURM_PROCID
    export WORLD_SIZE=$SLURM_NNODES
    
    echo "Node $NODE_RANK starting..."
    
    # Build additional_context with node-specific values using Python (more robust than sed)
    cat > /tmp/additional_context_$NODE_RANK.json <<EOF
{json.dumps(additional_context, indent=2)}
EOF
    
    # Update NODE_RANK and MASTER_ADDR using Python
    python3 <<PYTHON_EOF
import json
with open("/tmp/additional_context_$NODE_RANK.json", "r") as f:
    ctx = json.load(f)
if "multi_node_args" in ctx:
    ctx["multi_node_args"]["NODE_RANK"] = "$NODE_RANK"
    ctx["multi_node_args"]["MASTER_ADDR"] = "$MASTER_ADDR"
with open("/tmp/additional_context_$NODE_RANK.json", "w") as f:
    json.dump(ctx, f)
PYTHON_EOF
    
    # Run madengine
    madengine-cli run \\
      --manifest-file build_manifest.json \\
      --additional-context "$(cat /tmp/additional_context_$NODE_RANK.json)" \\
      --verbose
    
    exit_code=$?
    echo "Node $NODE_RANK completed with exit code: $exit_code"
    exit $exit_code
'

exit_code=$?

echo ""
echo "========================================="
if [ $exit_code -eq 0 ]; then
    echo "Distributed training completed successfully!"
else
    echo "Distributed training failed with exit code: $exit_code"
fi
echo "========================================="

exit $exit_code
"""
        
        with open(sbatch_path, 'w') as f:
            f.write(script)
        sbatch_path.chmod(0o755)
        
        self.log(f"Generated sbatch script: {sbatch_path}")
        
        return sbatch_path
    
    def execute(self, orchestration: Dict[str, Any]) -> ExecutionResult:
        """
        Execute distributed workload via SLURM.
        
        Submits job to SLURM scheduler and waits for completion.
        """
        self.log("Executing distributed workload via SLURM...", force=True)
        
        result = ExecutionResult()
        
        sbatch_script = orchestration['sbatch_script']
        login_node = orchestration['login_node']
        
        start_time = time.time()
        
        try:
            # SSH to login node and submit job
            username = login_node.get('username', os.environ.get('USER'))
            address = login_node['address']
            ssh_key = login_node.get('ssh_key_path', '~/.ssh/id_rsa')
            
            self.log(f"Connecting to login node: {address}")
            
            # Copy sbatch script to login node
            remote_script = f"/tmp/madengine_submit_{int(time.time())}.sbatch"
            
            scp_cmd = [
                'scp',
                '-i', os.path.expanduser(ssh_key),
                '-o', 'StrictHostKeyChecking=no',
                sbatch_script,
                f"{username}@{address}:{remote_script}"
            ]
            
            subprocess.run(scp_cmd, check=True, capture_output=True)
            
            # Copy manifest to login node
            manifest_path = self.manifest.get('_path', 'build_manifest.json')
            if os.path.exists(manifest_path):
                scp_manifest_cmd = [
                    'scp',
                    '-i', os.path.expanduser(ssh_key),
                    '-o', 'StrictHostKeyChecking=no',
                    manifest_path,
                    f"{username}@{address}:~/"
                ]
                subprocess.run(scp_manifest_cmd, check=True, capture_output=True)
            
            # Submit job via sbatch
            self.log("Submitting job to SLURM scheduler...")
            
            ssh_submit_cmd = [
                'ssh',
                '-i', os.path.expanduser(ssh_key),
                '-o', 'StrictHostKeyChecking=no',
                f"{username}@{address}",
                f"sbatch {remote_script}"
            ]
            
            submit_result = subprocess.run(
                ssh_submit_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse job ID from output (e.g., "Submitted batch job 12345")
            job_id = None
            for line in submit_result.stdout.split('\n'):
                if 'Submitted batch job' in line:
                    job_id = line.split()[-1]
                    break
            
            if not job_id:
                raise ValueError(f"Could not parse job ID from: {submit_result.stdout}")
            
            self.log(f"Job submitted: {job_id}", force=True)
            
            # Wait for job to complete (poll sacct)
            self.log("Waiting for job to complete...")
            
            while True:
                time.sleep(30)  # Poll every 30 seconds
                
                check_cmd = [
                    'ssh',
                    '-i', os.path.expanduser(ssh_key),
                    '-o', 'StrictHostKeyChecking=no',
                    f"{username}@{address}",
                    f"sacct -j {job_id} --format=State --noheader | head -n 1"
                ]
                
                check_result = subprocess.run(
                    check_cmd,
                    capture_output=True,
                    text=True
                )
                
                state = check_result.stdout.strip()
                
                if state in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT']:
                    break
                
                if self.verbose:
                    self.log(f"Job state: {state}")
            
            result.execution_time = time.time() - start_time
            
            # Check final state
            if state == 'COMPLETED':
                result.success = True
                result.successful_nodes = 1
                result.total_nodes = 1
                self.log(f"SLURM job completed successfully", force=True)
            else:
                result.success = False
                result.failed_nodes = 1
                result.total_nodes = 1
                result.errors.append(f"Job failed with state: {state}")
                self.log(f"SLURM job failed: {state}", force=True)
            
            # Retrieve output files
            self.log("Retrieving output files...")
            
            scp_out_cmd = [
                'scp',
                '-i', os.path.expanduser(ssh_key),
                '-o', 'StrictHostKeyChecking=no',
                f"{username}@{address}:~/madengine_{job_id}.out",
                str(self.output_dir / f"job_{job_id}.out")
            ]
            subprocess.run(scp_out_cmd, capture_output=True)
            
            scp_err_cmd = [
                'scp',
                '-i', os.path.expanduser(ssh_key),
                '-o', 'StrictHostKeyChecking=no',
                f"{username}@{address}:~/madengine_{job_id}.err",
                str(self.output_dir / f"job_{job_id}.err")
            ]
            subprocess.run(scp_err_cmd, capture_output=True)
            
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            result.execution_time = time.time() - start_time
            self.log(f"SLURM execution failed: {e}", force=True)
        
        self.log(f"Execution completed in {result.execution_time:.2f}s", force=True)
        
        return result

