#!/usr/bin/env python3
"""Launcher command string generators for Kubernetes (mixin)."""

# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

class KubernetesLauncherMixin:
    def _generate_torchrun_command(
        self, nnodes: int, nproc_per_node: int, master_port: int, model_script: str
    ) -> str:
        """
        Generate torchrun launcher command for K8s Indexed Jobs.
        
        For single-node (nnodes=1), generates standalone torchrun command.
        For multi-node (nnodes>1), generates distributed torchrun with headless
        service DNS for coordination.
        
        Uses K8s environment variables for distributed coordination:
        - JOB_COMPLETION_INDEX: Pod index (0, 1, 2, ...)
        - Headless service DNS for MASTER_ADDR
        
        CRITICAL FIX: For bash scripts that use ${BASH_SOURCE[0]}, we cd into the
        script directory first so relative paths resolve correctly. This fixes the
        issue where profiling tool wrappers prevent BASH_SOURCE from resolving.
        
        Args:
            nnodes: Number of nodes (pods). Must be >= 1.
            nproc_per_node: GPUs per node. Must be >= 1.
            master_port: Master communication port. Must be 1-65535.
            model_script: Path to model's run script. Cannot be empty.
        
        Returns:
            Complete torchrun command string
        
        Raises:
            ValueError: If any parameter is invalid
        """
        from pathlib import Path
        
        # Validate inputs (defensive programming)
        if not isinstance(nnodes, int) or nnodes < 1:
            raise ValueError(f"nnodes must be integer >= 1, got {nnodes}")
        if not isinstance(nproc_per_node, int) or nproc_per_node < 1:
            raise ValueError(f"nproc_per_node must be integer >= 1, got {nproc_per_node}")
        if not isinstance(master_port, int) or not (1 <= master_port <= 65535):
            raise ValueError(f"master_port must be 1-65535, got {master_port}")
        if not model_script or not isinstance(model_script, str):
            raise ValueError(f"model_script must be non-empty string, got {model_script}")
        
        # Check if model_script is a bash script
        # If so, execute it directly as it handles torchrun internally
        if model_script.endswith('.sh'):
            # For bash scripts, set environment variables and execute script
            # The script itself will invoke torchrun with the appropriate Python file
            # CRITICAL: cd to script directory first so BASH_SOURCE[0] resolves correctly
            script_dir = str(Path(model_script).parent)
            script_name = str(Path(model_script).name)
            if nnodes == 1:
                return f"""export MAD_MULTI_NODE_RUNNER="torchrun --standalone --nproc_per_node={nproc_per_node}"
export MAD_RUNTIME_NGPUS={nproc_per_node}
cd {script_dir} && bash {script_name}"""
            else:
                return f"""# Multi-node torchrun setup (Kubernetes Indexed Job)
export MASTER_ADDR="{self.job_name}-0.{self.job_name}.{self.namespace}.svc.cluster.local"
export MASTER_PORT={master_port}
export MAD_MULTI_NODE_RUNNER="torchrun --nnodes={nnodes} --nproc_per_node={nproc_per_node} --node_rank=${{JOB_COMPLETION_INDEX}} --master_addr=${{MASTER_ADDR}} --master_port={master_port}"
export MAD_RUNTIME_NGPUS={nproc_per_node}
cd {script_dir} && bash {script_name}"""
        
        # For Python scripts, invoke torchrun directly
        # For single-node, simpler standalone command
        if nnodes == 1:
            return f"""torchrun \\
    --standalone \\
    --nnodes=1 \\
    --nproc_per_node={nproc_per_node} \\
    {model_script}"""
        
        # Multi-node: Use headless service DNS and JOB_COMPLETION_INDEX
        return f"""# Multi-node torchrun setup (Kubernetes Indexed Job)
export MASTER_ADDR="{self.job_name}-0.{self.job_name}.{self.namespace}.svc.cluster.local"
export MASTER_PORT={master_port}
export RANK=${{JOB_COMPLETION_INDEX}}
export WORLD_SIZE={nnodes}
export LOCAL_RANK=0
export NNODES={nnodes}
export NPROC_PER_NODE={nproc_per_node}

echo "Torchrun Configuration:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  RANK: $RANK"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"

torchrun \\
    --nnodes={nnodes} \\
    --nproc_per_node={nproc_per_node} \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
    --rdzv_id={self.job_name} \\
    --role=worker \\
    --tee=3 \\
    {model_script}"""
    
    def _generate_deepspeed_command(
        self, nnodes: int, nproc_per_node: int, master_port: int, model_script: str
    ) -> str:
        """
        Generate DeepSpeed launcher command for K8s Indexed Jobs.
        
        DeepSpeed has its own launcher that handles:
        - ZeRO optimization stages (ZeRO-1, ZeRO-2, ZeRO-3)
        - Gradient accumulation
        - Mixed precision training
        - Pipeline parallelism
        - Hostfile management (handled by K8s in our case)
        
        For single-node (nnodes=1), uses localhost setup.
        For multi-node (nnodes>1), uses headless service DNS for coordination.
        
        Args:
            nnodes: Number of nodes (pods). Must be >= 1.
            nproc_per_node: GPUs per node. Must be >= 1.
            master_port: Master communication port. Must be 1-65535.
            model_script: Path to model's run script. Cannot be empty.
        
        Returns:
            Complete DeepSpeed launcher command string
        
        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate inputs
        if not isinstance(nnodes, int) or nnodes < 1:
            raise ValueError(f"nnodes must be integer >= 1, got {nnodes}")
        if not isinstance(nproc_per_node, int) or nproc_per_node < 1:
            raise ValueError(f"nproc_per_node must be integer >= 1, got {nproc_per_node}")
        if not isinstance(master_port, int) or not (1 <= master_port <= 65535):
            raise ValueError(f"master_port must be 1-65535, got {master_port}")
        if not model_script or not isinstance(model_script, str):
            raise ValueError(f"model_script must be non-empty string, got {model_script}")
        
        # For single-node
        if nnodes == 1:
            return f"""# DeepSpeed Single-Node Setup
export MASTER_ADDR=localhost
export MASTER_PORT={master_port}
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE={nproc_per_node}

echo "DeepSpeed Configuration:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  NUM_GPUS: {nproc_per_node}"

# DeepSpeed launcher (single-node)
deepspeed --num_gpus={nproc_per_node} \\
    --master_port={master_port} \\
    {model_script}"""
        
        # Multi-node: Use K8s headless service for coordination
        return f"""# Multi-node DeepSpeed setup (Kubernetes Indexed Job)
export MASTER_ADDR="{self.job_name}-0.{self.job_name}.{self.namespace}.svc.cluster.local"
export MASTER_PORT={master_port}
export RANK=${{JOB_COMPLETION_INDEX}}
export LOCAL_RANK=0
export WORLD_SIZE={nnodes * nproc_per_node}
export NNODES={nnodes}
export NPROC_PER_NODE={nproc_per_node}

echo "DeepSpeed Multi-Node Configuration:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  RANK (Node Rank): $RANK"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  NNODES: $NNODES"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"

# Create hostfile for DeepSpeed (K8s Indexed Job aware)
cat > /tmp/hostfile << EOF
{self.job_name}-0.{self.job_name}.{self.namespace}.svc.cluster.local slots={nproc_per_node}
EOF

# Add all nodes to hostfile
for i in $(seq 1 $((NNODES - 1))); do
    echo "{self.job_name}-$i.{self.job_name}.{self.namespace}.svc.cluster.local slots={nproc_per_node}" >> /tmp/hostfile
done

echo ""
echo "Generated hostfile:"
cat /tmp/hostfile
echo ""

# DeepSpeed launcher (multi-node with hostfile)
deepspeed --hostfile=/tmp/hostfile \\
    --master_addr=$MASTER_ADDR \\
    --master_port=$MASTER_PORT \\
    --num_nodes={nnodes} \\
    --num_gpus={nproc_per_node} \\
    {model_script}"""
    
    def _generate_bash_script_command(
        self, nnodes: int, nproc_per_node: int, master_port: int, model_script: str
    ) -> str:
        """
        Generate command to execute a bash script directly.
        
        This is used when the model script is a .sh file that handles
        launcher invocation internally (e.g., using torchrun inside the script).
        
        Sets up environment variables for distributed training that the bash
        script can use.
        
        Args:
            nnodes: Number of nodes (pods)
            nproc_per_node: GPUs per node
            master_port: Master communication port
            model_script: Path to the bash script
        
        Returns:
            Command to execute the bash script with environment setup
        """
        # For single-node
        if nnodes == 1:
            return f"""# Bash Script Execution (Single-Node)
# Setting up environment for script to use
export MASTER_ADDR=localhost
export MASTER_PORT={master_port}
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE={nproc_per_node}
export NNODES=1
export NPROC_PER_NODE={nproc_per_node}

echo "Bash Script Configuration:"
echo "  Script: {model_script}"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  NNODES: $NNODES"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"
echo ""

# Execute the bash script directly
bash {model_script}"""
        
        # Multi-node: Use K8s headless service for coordination
        return f"""# Bash Script Execution (Multi-Node)
# Setting up environment for script to use
export MASTER_ADDR="{self.job_name}-0.{self.job_name}.{self.namespace}.svc.cluster.local"
export MASTER_PORT={master_port}
export RANK=${{JOB_COMPLETION_INDEX}}
export LOCAL_RANK=0
export WORLD_SIZE={nnodes * nproc_per_node}
export NNODES={nnodes}
export NPROC_PER_NODE={nproc_per_node}

echo "Bash Script Multi-Node Configuration:"
echo "  Script: {model_script}"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  RANK (Node Rank): $RANK"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  NNODES: $NNODES"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"
echo ""

# Execute the bash script directly
bash {model_script}"""
    
    def _generate_torchtitan_command(
        self, nnodes: int, nproc_per_node: int, master_port: int, model_script: str
    ) -> str:
        """
        Generate TorchTitan launcher command for K8s Indexed Jobs.
        
        TorchTitan is a PyTorch native platform for large-scale LLM pre-training
        that supports multi-dimensional parallelism:
        - FSDP2 (Fully Sharded Data Parallel v2)
        - Tensor Parallel (TP)
        - Pipeline Parallel (PP)
        - Context Parallel (CP)
        
        TorchTitan uses torchrun as its underlying distributed launcher but
        requires additional configuration for its parallelism strategies.
        
        For single-node (nnodes=1): Uses standalone torchrun with TP
        For multi-node (nnodes>1): Uses distributed torchrun with TP+PP+FSDP2
        
        Uses K8s environment variables for distributed coordination:
        - JOB_COMPLETION_INDEX: Pod index (0, 1, 2, ...)
        - Headless service DNS for MASTER_ADDR
        
        Args:
            nnodes: Number of nodes (pods). Must be >= 1.
            nproc_per_node: GPUs per node. Must be >= 1.
            master_port: Master communication port. Must be 1-65535.
            model_script: Path to model's run script. Cannot be empty.
        
        Returns:
            Complete torchtitan launch command string with environment setup
        
        Raises:
            ValueError: If any parameter is invalid
        
        Example single-node output:
            export TORCHTITAN_TENSOR_PARALLEL_SIZE=8
            export TORCHTITAN_PIPELINE_PARALLEL_SIZE=1
            torchrun --standalone --nproc_per_node=8 train.py --config llama3_8b.toml
        
        Example multi-node output:
            export MASTER_ADDR="job-0.job.namespace.svc.cluster.local"
            export TORCHTITAN_TENSOR_PARALLEL_SIZE=8
            export TORCHTITAN_PIPELINE_PARALLEL_SIZE=4
            export TORCHTITAN_FSDP_ENABLED=1
            torchrun --nnodes=4 --nproc_per_node=8 ... train.py --config llama3_405b.toml
        """
        # Validate inputs
        if not isinstance(nnodes, int) or nnodes < 1:
            raise ValueError(f"nnodes must be integer >= 1, got {nnodes}")
        if not isinstance(nproc_per_node, int) or nproc_per_node < 1:
            raise ValueError(f"nproc_per_node must be integer >= 1, got {nproc_per_node}")
        if not isinstance(master_port, int) or not (1 <= master_port <= 65535):
            raise ValueError(f"master_port must be 1-65535, got {master_port}")
        if not model_script or not isinstance(model_script, str):
            raise ValueError(f"model_script must be non-empty string, got {model_script}")
        
        # For single-node, use standalone mode with Tensor Parallelism only
        if nnodes == 1:
            return f"""# TorchTitan single-node setup (Tensor Parallelism)
export TORCHTITAN_TENSOR_PARALLEL_SIZE={nproc_per_node}
export TORCHTITAN_PIPELINE_PARALLEL_SIZE=1
export TORCHTITAN_FSDP_ENABLED=0
export TORCHTITAN_CONTEXT_PARALLEL_SIZE=1

echo "TorchTitan Configuration (Single Node):"
echo "  Tensor Parallel Size: {nproc_per_node}"
echo "  Pipeline Parallel Size: 1"
echo "  Total GPUs: {nproc_per_node}"

torchrun \\
    --standalone \\
    --nnodes=1 \\
    --nproc_per_node={nproc_per_node} \\
    {model_script}"""
        
        # Multi-node: Use headless service DNS and enable all parallelism strategies
        return f"""# TorchTitan multi-node setup (K8s Indexed Job)
export MASTER_ADDR="{self.job_name}-0.{self.job_name}.{self.namespace}.svc.cluster.local"
export MASTER_PORT={master_port}
export RANK=${{JOB_COMPLETION_INDEX}}
export WORLD_SIZE={nnodes}
export LOCAL_RANK=0
export NNODES={nnodes}
export NPROC_PER_NODE={nproc_per_node}

# TorchTitan multi-dimensional parallelism configuration
# These can be overridden by TOML config file in model script
export TORCHTITAN_TENSOR_PARALLEL_SIZE={nproc_per_node}
export TORCHTITAN_PIPELINE_PARALLEL_SIZE={nnodes}
export TORCHTITAN_FSDP_ENABLED=1
export TORCHTITAN_CONTEXT_PARALLEL_SIZE=1

echo "TorchTitan Configuration (Multi-Node):"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  RANK: $RANK"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  Tensor Parallel Size: {nproc_per_node}"
echo "  Pipeline Parallel Size: {nnodes}"
echo "  FSDP: Enabled"
echo "  Total GPUs: {nnodes * nproc_per_node}"

torchrun \\
    --nnodes={nnodes} \\
    --nproc_per_node={nproc_per_node} \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
    --rdzv_id={self.job_name} \\
    --role=worker \\
    --tee=3 \\
    {model_script}"""
    
    def _generate_sglang_disagg_command(
        self, nnodes: int, nproc_per_node: int, master_port: int, model_script: str
    ) -> str:
        """
        Generate SGLang Disaggregated launcher command for K8s Indexed Jobs.
        
        SGLang Disaggregated uses separate node pools for:
        - Proxy (index 0): Load balancer and request router
        - Prefill (indices 1 to xP): Prompt processing
        - Decode (indices xP+1 to end): Token generation
        
        Communication via Mooncake framework for efficient KV cache transfer.
        
        Architecture:
        - Pod 0: Runs mini_lb (proxy/load balancer)
        - Pods 1-xP: Run prefill servers
        - Pods xP+1 to N-1: Run decode servers
        
        Args:
            nnodes: Total number of pods (must be >= 3)
            nproc_per_node: GPUs per pod
            master_port: Port for proxy service
            model_script: Path to model launch script
            
        Returns:
            Complete disaggregated launch setup
            
        Raises:
            ValueError: If nnodes < 3 or invalid parameters
        """
        # Validate
        if not isinstance(nnodes, int) or nnodes < 3:
            raise ValueError(
                f"SGLang Disaggregated requires minimum 3 nodes, got {nnodes}"
            )
        if not isinstance(nproc_per_node, int) or nproc_per_node < 1:
            raise ValueError(f"nproc_per_node must be >= 1, got {nproc_per_node}")
        if not model_script or not isinstance(model_script, str):
            raise ValueError(f"model_script must be non-empty string")
        
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
            # Default automatic split (can be customized via additional_context)
            xP = max(1, (nnodes - 1) * 2 // 5)  # ~40% prefill
            yD = nnodes - 1 - xP  # remaining decode
        
        # Build prefill and decode server lists
        prefill_servers = " ".join([
            f"http://{self.job_name}-{i}.{self.job_name}.{self.namespace}.svc.cluster.local:30000"
            for i in range(1, xP + 1)
        ])
        
        decode_servers = " ".join([
            f"http://{self.job_name}-{i}.{self.job_name}.{self.namespace}.svc.cluster.local:30000"
            for i in range(xP + 1, nnodes)
        ])
        
        return f"""# SGLang Disaggregated K8s Setup
# ============================================
# Cluster: {nnodes} pods total
#   Proxy: Pod 0
#   Prefill: Pods 1-{xP} ({xP} nodes)
#   Decode: Pods {xP+1}-{nnodes-1} ({yD} nodes)
# ============================================

export POD_INDEX=${{JOB_COMPLETION_INDEX:-0}}
export TOTAL_PODS={nnodes}
export PREFILL_COUNT={xP}
export DECODE_COUNT={yD}
export TP_SIZE={nproc_per_node}

# Get pod IP
export POD_IP=$(hostname -i | awk '{{print $1}}')

echo "=========================================="
echo "SGLang Disaggregated Pod Info"
echo "=========================================="
echo "Pod Index: $POD_INDEX"
echo "Pod IP: $POD_IP"
echo "Total Pods: $TOTAL_PODS"
echo "Prefill Pods: $PREFILL_COUNT"
echo "Decode Pods: $DECODE_COUNT"
echo "TP Size: $TP_SIZE"
echo "=========================================="

# Node role assignment based on pod index
if [ "$POD_INDEX" -eq 0 ]; then
    # Proxy Node (Load Balancer)
    echo "🔀 This pod is PROXY (Load Balancer)"
    
    python3 -m sglang.srt.disaggregation.mini_lb \\
        --prefill {prefill_servers} \\
        --decode {decode_servers} \\
        --host 0.0.0.0 \\
        --port {master_port}
    
elif [ "$POD_INDEX" -le "{xP}" ]; then
    # Prefill Nodes
    echo "⚡ This pod is PREFILL Node"
    
    python3 -m sglang.launch_server \\
        --model-path "$MODEL_PATH" \\
        --disaggregation-mode prefill \\
        --tp-size {nproc_per_node} \\
        --host $POD_IP \\
        --port 30000 \\
        --trust-remote-code \\
        --disaggregation-transfer-backend mooncake
    
else
    # Decode Nodes
    echo "🔤 This pod is DECODE Node"
    
    python3 -m sglang.launch_server \\
        --model-path "$MODEL_PATH" \\
        --disaggregation-mode decode \\
        --tp-size {nproc_per_node} \\
        --host $POD_IP \\
        --port 30000 \\
        --trust-remote-code \\
        --disaggregation-transfer-backend mooncake
fi

echo "SGLang Disaggregated setup complete"
"""
    
    def _generate_vllm_command(
        self,
        nnodes: int,
        nproc_per_node: int,
        master_port: int,
        model_script: str,
        model_args: str = "",
    ) -> str:
        """
        Generate vLLM launcher command for K8s Indexed Jobs.
        
        vLLM is an inference engine with its own process management via Ray.
        Unlike training frameworks, vLLM doesn't use torchrun.
        
        Architecture:
        - Single-node: Tensor Parallelism (TP) across GPUs, no Ray needed
        - Multi-node: Data Parallelism where each node runs independent vLLM replica
          * Each replica uses TP across its local GPUs
          * Ray coordinates resources on each node independently
          * Benefits: Simpler, more robust, better for inference serving
        
        For K8s multi-node:
        - Each pod runs its own independent vLLM instance
        - Uses Ray for local GPU coordination
        - NO shared Ray cluster across pods (Data Parallelism mode)
        
        Args:
            nnodes: Number of nodes (pods). Must be >= 1.
            nproc_per_node: GPUs per node. Must be >= 1.
            master_port: Master communication port (for Ray). Must be 1-65535.
            model_script: Path to model's run script. Cannot be empty.
            model_args: CLI args for the script (e.g. --model_repo openai/gpt-oss-20b).
        
        Returns:
            Complete vLLM launch setup with environment configuration
        
        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate inputs
        if not isinstance(nnodes, int) or nnodes < 1:
            raise ValueError(f"nnodes must be integer >= 1, got {nnodes}")
        if not isinstance(nproc_per_node, int) or nproc_per_node < 1:
            raise ValueError(f"nproc_per_node must be integer >= 1, got {nproc_per_node}")
        if not isinstance(master_port, int) or not (1 <= master_port <= 65535):
            raise ValueError(f"master_port must be 1-65535, got {master_port}")
        if not model_script or not isinstance(model_script, str):
            raise ValueError(f"model_script must be non-empty string, got {model_script}")

        # Run script from its directory so relative paths (run_vllm.py, configs/) resolve
        script_dir = str(Path(model_script).parent)
        script_name = Path(model_script).name
        run_cmd = f"cd /workspace/{script_dir} && bash {script_name} {model_args}".strip()
        
        # For single-node, simple TP setup (no Ray needed)
        if nnodes == 1:
            return f"""# vLLM single-node setup (Tensor Parallelism)
export VLLM_TENSOR_PARALLEL_SIZE={nproc_per_node}
export VLLM_PIPELINE_PARALLEL_SIZE=1
export VLLM_DISTRIBUTED_BACKEND="auto"
export NNODES=1
export NPROC_PER_NODE={nproc_per_node}
export NODE_RANK=0

echo "vLLM Configuration (Single Node):"
echo "  Tensor Parallel Size: {nproc_per_node}"
echo "  Pipeline Parallel Size: 1"
echo "  Distributed Backend: auto (no Ray)"
echo "  Total GPUs: {nproc_per_node}"

# vLLM handles process management - run script from its directory so run_vllm.py/configs resolve
{run_cmd}"""
        
        # Multi-node: Data Parallelism with independent Ray clusters per pod
        return f"""# vLLM multi-node setup (K8s Data Parallelism Mode)
export MASTER_ADDR="{self.job_name}-0.{self.job_name}.{self.namespace}.svc.cluster.local"
export MASTER_PORT={master_port}
export NODE_RANK=${{JOB_COMPLETION_INDEX}}
export NNODES={nnodes}
export NPROC_PER_NODE={nproc_per_node}

# vLLM Data Parallelism configuration
# Each pod runs INDEPENDENT vLLM replica (no shared Ray cluster)
export VLLM_TENSOR_PARALLEL_SIZE={nproc_per_node}
export VLLM_PIPELINE_PARALLEL_SIZE=1
export VLLM_DISTRIBUTED_BACKEND="ray"

# Get current pod IP for Ray
POD_IP=$(hostname -i | awk '{{print $1}}')
export VLLM_HOST_IP="$POD_IP"

echo "vLLM Configuration (Multi-Node Data Parallelism):"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  NODE_RANK: $NODE_RANK (Pod Index)"
echo "  NNODES: $NNODES"
echo "  Tensor Parallel Size: {nproc_per_node} (per pod)"
echo "  Data Parallel Size: {nnodes} (independent replicas)"
echo "  Pod IP: $POD_IP"
echo "  Total GPUs: {nnodes * nproc_per_node}"
echo ""
echo "Mode: Each pod runs independent vLLM replica with local Ray"

# Clean any existing Ray processes
ray stop --force 2>/dev/null || true
pkill -9 -f "ray::" 2>/dev/null || true
sleep 2

# Start independent Ray cluster on THIS pod only
echo "Starting Ray cluster on Pod $NODE_RANK..."
ray start --head --port=6379 --node-ip-address="$POD_IP" --num-gpus={nproc_per_node}
sleep 3

echo "Ray cluster ready:"
ray status

# Run vLLM inference script from its directory so run_vllm.py/configs resolve
{run_cmd}

# Cleanup Ray on exit
trap "ray stop --force 2>/dev/null || true" EXIT"""

    def _generate_sglang_command(
        self,
        nnodes: int,
        nproc_per_node: int,
        master_port: int,
        model_script: str,
        model_args: str = "",
    ) -> str:
        """
        Generate SGLang launcher command for K8s Indexed Jobs.
        
        SGLang is an inference engine with native launcher (sglang.launch_server).
        Similar to vLLM, it manages its own process spawning via Ray.
        
        Architecture:
        - Single-node: Tensor Parallelism (TP) across GPUs
        - Multi-node: Uses SGLang's native multi-node launcher with Ray
          * TP across GPUs within each node
          * Ray for distributed coordination
        
        For K8s:
        - Uses headless service for node discovery (similar to torchrun)
        - Each pod knows its rank via JOB_COMPLETION_INDEX
        - SGLang native launcher handles Ray cluster setup
        
        Args:
            nnodes: Number of nodes (pods). Must be >= 1.
            nproc_per_node: GPUs per node. Must be >= 1.
            master_port: Master communication port (for NCCL/Ray). Must be 1-65535.
            model_script: Path to model's run script. Cannot be empty.
            model_args: CLI args for the script (e.g. --model_repo ...).
        
        Returns:
            Complete SGLang launch setup with environment configuration
        
        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate inputs
        if not isinstance(nnodes, int) or nnodes < 1:
            raise ValueError(f"nnodes must be integer >= 1, got {nnodes}")
        if not isinstance(nproc_per_node, int) or nproc_per_node < 1:
            raise ValueError(f"nproc_per_node must be integer >= 1, got {nproc_per_node}")
        if not isinstance(master_port, int) or not (1 <= master_port <= 65535):
            raise ValueError(f"master_port must be 1-65535, got {master_port}")
        if not model_script or not isinstance(model_script, str):
            raise ValueError(f"model_script must be non-empty string, got {model_script}")

        # Run script from its directory so relative paths resolve; pass model args
        script_dir = str(Path(model_script).parent)
        script_name = Path(model_script).name
        run_cmd = f"cd /workspace/{script_dir} && bash {script_name} {model_args}".strip()

        # For single-node, simple TP setup
        if nnodes == 1:
            return f"""# SGLang single-node setup (Tensor Parallelism)
export SGLANG_TENSOR_PARALLEL_SIZE={nproc_per_node}
export SGLANG_PIPELINE_PARALLEL_SIZE=1
export NNODES=1
export NPROC_PER_NODE={nproc_per_node}
export NODE_RANK=0

echo "SGLang Configuration (Single Node):"
echo "  Tensor Parallel Size: {nproc_per_node}"
echo "  Total GPUs: {nproc_per_node}"

# SGLang native launcher handles everything
{run_cmd}"""

        # Multi-node: Use SGLang's native multi-node support
        return f"""# SGLang multi-node setup (K8s Indexed Job)
export MASTER_ADDR="{self.job_name}-0.{self.job_name}.{self.namespace}.svc.cluster.local"
export MASTER_PORT={master_port}
export NODE_RANK=${{JOB_COMPLETION_INDEX}}
export NNODES={nnodes}
export NPROC_PER_NODE={nproc_per_node}

# SGLang parallelism configuration
export SGLANG_TENSOR_PARALLEL_SIZE={nproc_per_node}
export SGLANG_PIPELINE_PARALLEL_SIZE=1

# Get current pod IP
POD_IP=$(hostname -i | awk '{{print $1}}')
export SGLANG_HOST_IP="$POD_IP"

echo "SGLang Configuration (Multi-Node):"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  NODE_RANK: $NODE_RANK (Pod Index)"
echo "  NNODES: $NNODES"
echo "  Tensor Parallel Size: {nproc_per_node}"
echo "  Pod IP: $POD_IP"
echo "  Total GPUs: {nnodes * nproc_per_node}"

# Clean any existing Ray processes
ray stop --force 2>/dev/null || true
pkill -9 -f "ray::" 2>/dev/null || true
sleep 2

# SGLang native launcher will handle Ray cluster coordination
# Pass NCCL init address for multi-node setup
export NCCL_INIT_ADDR="${{MASTER_ADDR}}:${{MASTER_PORT}}"

echo "Starting SGLang with native multi-node launcher..."
{run_cmd}

# Cleanup Ray on exit
trap "ray stop --force 2>/dev/null || true" EXIT"""

    def _generate_megatron_command(
        self, nnodes: int, nproc_per_node: int, master_port: int, model_script: str
    ) -> str:
        """
        Generate Megatron-LM launcher command for K8s Indexed Jobs.
        
        Megatron-LM is a training framework for large transformers with tensor and pipeline parallelism.
        It uses torchrun as the underlying launcher but with Megatron-specific environment variables.
        
        Architecture:
        - Single-node: Tensor Parallelism (TP) across GPUs
        - Multi-node: Tensor + Pipeline Parallelism
          * TP across GPUs within each node
          * PP across nodes
        
        For K8s:
        - Uses headless service for node discovery (like torchrun/deepspeed)
        - Each pod knows its rank via JOB_COMPLETION_INDEX
        - Sets TENSOR_MODEL_PARALLEL_SIZE and PIPELINE_MODEL_PARALLEL_SIZE (Megatron-Core standard)
        
        Args:
            nnodes: Number of nodes (pods). Must be >= 1.
            nproc_per_node: GPUs per node. Must be >= 1.
            master_port: Master communication port (for NCCL). Must be 1-65535.
            model_script: Path to model's run script. Cannot be empty.
        
        Returns:
            Complete Megatron-LM launch setup with environment configuration
        
        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate inputs
        if not isinstance(nnodes, int) or nnodes < 1:
            raise ValueError(f"nnodes must be integer >= 1, got {nnodes}")
        if not isinstance(nproc_per_node, int) or nproc_per_node < 1:
            raise ValueError(f"nproc_per_node must be integer >= 1, got {nproc_per_node}")
        if not isinstance(master_port, int) or not (1 <= master_port <= 65535):
            raise ValueError(f"master_port must be 1-65535, got {master_port}")
        if not model_script or not isinstance(model_script, str):
            raise ValueError(f"model_script must be non-empty string, got {model_script}")
        
        # For single-node, use TP only
        if nnodes == 1:
            return f"""# Megatron-LM single-node setup (Tensor Parallelism)
export TENSOR_MODEL_PARALLEL_SIZE={min(nproc_per_node, 8)}
export PIPELINE_MODEL_PARALLEL_SIZE=1
export CONTEXT_PARALLEL_SIZE=1
export NNODES=1
export NPROC_PER_NODE={nproc_per_node}
export MASTER_ADDR=localhost
export MASTER_PORT={master_port}
export NODE_RANK=0

echo "Megatron-LM Configuration (Single-Node):"
echo "  Tensor Model Parallel Size: {min(nproc_per_node, 8)}"
echo "  Pipeline Model Parallel Size: 1"
echo "  Total GPUs: {nproc_per_node}"

# Launch using torchrun with Megatron configuration
torchrun \\
    --standalone \\
    --nproc_per_node={nproc_per_node} \\
    {model_script}"""
        
        # Multi-node: TP + PP
        else:
            # Use headless service for node discovery (set by template)
            return f"""# Megatron-LM multi-node setup (Tensor + Pipeline Parallelism)
export TENSOR_MODEL_PARALLEL_SIZE={nproc_per_node}
export PIPELINE_MODEL_PARALLEL_SIZE={nnodes}
export CONTEXT_PARALLEL_SIZE=1
export NNODES={nnodes}
export NPROC_PER_NODE={nproc_per_node}
export NODE_RANK=${{JOB_COMPLETION_INDEX}}
export MASTER_ADDR=${{MASTER_ADDR}}
export MASTER_PORT={master_port}

echo "Megatron-LM Configuration (Multi-Node):"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  NODE_RANK: $NODE_RANK (Pod Index)"
echo "  NNODES: $NNODES"
echo "  Tensor Model Parallel Size: {nproc_per_node}"
echo "  Pipeline Model Parallel Size: {nnodes}"
echo "  Total GPUs: {nnodes * nproc_per_node}"

# Wait for all pods to be ready (K8s Indexed Job coordination)
echo "Waiting for all {nnodes} pods to be ready..."
sleep 5

# Launch using torchrun with Megatron multi-node configuration
torchrun \\
    --nnodes={nnodes} \\
    --nproc_per_node={nproc_per_node} \\
    --node_rank=${{NODE_RANK}} \\
    --master_addr=${{MASTER_ADDR}} \\
    --master_port={master_port} \\
    {model_script}"""
    
