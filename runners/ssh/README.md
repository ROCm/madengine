# SSH Multi-Node Runner for MAD Engine

This SSH runner automates the execution of PyTorch Megatron-LM training across multiple nodes using SSH connections.

## Features

- ✅ Automated SSH connection management
- ✅ Parallel execution across multiple nodes
- ✅ Real-time output streaming from all nodes
- ✅ Robust error handling and connectivity checking
- ✅ Support for both SSH key and password authentication
- ✅ Configurable network interfaces (NCCL/GLOO)
- ✅ Shared filesystem support

## Prerequisites

1. **Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or use the quick-start script:
   ```bash
   bash quick-start.sh
   ```

2. **SSH Access**: Ensure you have SSH access to all target nodes with either:
   - SSH key-based authentication (recommended)
   - Password-based authentication

3. **Shared Filesystem**: All nodes should have access to a shared filesystem for data (e.g., NFS mount)

4. **MAD Engine**: Ensure `madengine` is installed and accessible on all target nodes

## Usage

### Basic Usage with SSH Key

```bash
python run.py --model pyt_megatron_lm_train_llama2_7b \
              --nodes 192.168.1.1,192.168.1.2 \
              --master-addr 192.168.0.1 \
              --ssh-user ubuntu \
              --ssh-key ~/.ssh/id_rsa \
              --shared-data-path /nfs/data
```

### Usage with Password Authentication

```bash
python run.py --model pyt_megatron_lm_train_llama2_7b \
              --nodes node1.cluster.com,node2.cluster.com \
              --ssh-user root \
              --ssh-password mypassword \
              --shared-data-path /shared/data
```

### Advanced Configuration

```bash
python run.py --model pyt_megatron_lm_train_llama2_7b \
              --nodes 192.168.1.10,192.168.1.11,192.168.1.12 \
              --master-addr 192.168.1.10 \
              --master-port 5000 \
              --ssh-user mluser \
              --ssh-key /home/user/.ssh/cluster_key \
              --shared-data-path /mnt/nfs/datasets \
              --nccl-interface eth0 \
              --gloo-interface eth0 \
              --timeout 7200 \
              --additional-args "--some-extra-flag"
```

## Command Line Arguments

### Required Arguments

- `--model`: Model tag to run (e.g., `pyt_megatron_lm_train_llama2_7b`)
- `--nodes`: Comma-separated list of node hostnames/IPs
- `--ssh-user`: SSH username for all nodes

### Authentication (one required)

- `--ssh-password`: SSH password for all nodes
- `--ssh-key`: Path to SSH private key file

### Optional Arguments

- `--master-addr`: Master node address (defaults to first node)
- `--master-port`: Master node port (default: 4000)
- `--shared-data-path`: Path to shared data filesystem (default: /nfs/data)
- `--nccl-interface`: NCCL socket interface (default: ens14np0)
- `--gloo-interface`: GLOO socket interface (default: ens14np0)
- `--timeout`: Execution timeout in seconds (default: 3600)
- `--madengine-path`: Path to madengine executable (default: madengine)
- `--additional-args`: Additional arguments to pass to madengine

## How It Works

1. **Connectivity Check**: Verifies SSH connectivity to all nodes
2. **Command Generation**: Builds appropriate `madengine` commands for each node with correct `NODE_RANK`
3. **Parallel Execution**: Executes commands on all nodes simultaneously using threading
4. **Output Streaming**: Streams real-time output from all nodes with node identification
5. **Result Aggregation**: Collects and reports results from all nodes

## Example Output

```
🌐 Starting multi-node training on 2 nodes
📋 Model: pyt_megatron_lm_train_llama2_7b
🏠 Master: 192.168.0.1:4000
📁 Shared data: /nfs/data
🔗 Nodes: 192.168.1.1, 192.168.1.2

🔍 Checking connectivity to all nodes...
✓ 192.168.1.1 is reachable
✓ 192.168.1.2 is reachable
✅ All nodes are reachable

🚀 Executing on 192.168.1.1 (rank 0): madengine run --tags pyt_megatron_lm_train_llama2_7b ...
🚀 Executing on 192.168.1.2 (rank 1): madengine run --tags pyt_megatron_lm_train_llama2_7b ...

[192.168.1.1:0] Starting training...
[192.168.1.2:1] Starting training...
...
✅ 192.168.1.1 completed successfully
✅ 192.168.1.2 completed successfully

📊 Training Results:
✅ Successful nodes: 2/2
🎉 Multi-node training completed successfully!
```

## Network Configuration

For optimal performance, ensure:

1. **Network Interface**: Use the correct network interface names for `--nccl-interface` and `--gloo-interface`
   ```bash
   # Check available interfaces on your nodes
   ssh user@node "ip addr show"
   ```

2. **Firewall**: Ensure the master port is open between nodes
   ```bash
   # Example: Open port 4000 on Ubuntu/Debian
   sudo ufw allow 4000
   ```

3. **Shared Storage**: Verify shared filesystem is mounted on all nodes
   ```bash
   # Check if NFS mount is available
   ssh user@node "ls -la /nfs/data"
   ```

## Troubleshooting

### SSH Connection Issues

- Verify SSH key permissions: `chmod 600 ~/.ssh/id_rsa`
- Test manual SSH connection: `ssh -i ~/.ssh/id_rsa user@node`
- Check SSH agent: `ssh-add ~/.ssh/id_rsa`

### Network Communication Issues

- Verify nodes can reach each other on the master port
- Check firewall settings
- Ensure correct network interface names

### MAD Engine Issues

- Verify madengine is installed on all nodes: `ssh user@node "which madengine"`
- Check shared data path exists: `ssh user@node "ls -la /nfs/data"`
- Review madengine logs for specific errors

## Integration with MAD Engine

This SSH runner integrates seamlessly with the MAD Engine multi-node framework:

- Automatically configures `multi_node_args` for each node
- Sets appropriate `NODE_RANK` for each node (0, 1, 2, ...)
- Configures `NNODES` based on the number of nodes provided
- Uses `torchrun` as the distributed runner
- Handles network interface configuration for NCCL and GLOO

The generated command for each node follows this pattern:

```bash
madengine run --tags pyt_megatron_lm_train_llama2_7b \
              --additional-context "{'multi_node_args': {
                  'RUNNER': 'torchrun',                  
                  'MASTER_ADDR': '192.168.0.1',
                  'MASTER_PORT': '4000',
                  'NNODES': '2',
                  'NODE_RANK': '0',  # Different for each node
                  'NCCL_SOCKET_IFNAME': 'ens14np0',
                  'GLOO_SOCKET_IFNAME': 'ens14np0'
              }}" \
              --force-mirror-local /nfs/data
```
