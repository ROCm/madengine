# How to Run Mulit-Node

**NOTE: all of the commands/examples shown below are only showing the multi-node arguments - you will probably need to add the other arguments for your run on top of these.**

## Multi-Node Runners

There are two mulit-node `RUNNER`s in DLM/MAD, namely `torchrun` and `mpirun` (coming soon). Each of these `RUNNER`s are enabled in the model's bash script via the environment variable `MAD_MULTI_NODE_RUNNER`. For example in the `pyt_megatron_lm_train_llama2_7b` script, this feature is enabled with the following code

```bash
run_cmd="
        $MAD_MULTI_NODE_RUNNER \
        $TRAIN_SCRIPT \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        $EXTRA_ARGS \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH
"
```

Note the use of the `$MAD_MULTI_NODE_RUNNER` environment variable. This environment variable will be expanded into which ever `RUNNER` is chosen at DLM/MAD runtime.

### torchrun

Default `RUNNER` is `torchrun` , `MASTER_ADDR` is `localhost` , `NNODES` is 1 , `NODE_RANK` is 0, additional context `multi_node_args` is not necessary to run on single node 

```bash
madengine run --tags {model}
```

#### Two-Node Example

Using the `torchrun` `RUNNER` requires you to execute the DLM/MAD CLI command on each node manually. `NCCL_SOCKET_IFNAME` , `GLOO_SOCKET_IFNAME` needs to be set using `ifconfig` from `net-tools`

```bash
apt install net-tools
```

So let's assume the first node is our "master" node and has an IP=10.227.23.63

On first node, run the following:

```bash
madengine run --tags {model} --additional-context "{'multi_node_args': {'RUNNER': 'torchrun', 'MASTER_ADDR': '10.227.23.63', 'MASTER_PORT': '400', 'NNODES': '2', 'NODE_RANK': '0'}}"
```

On the second node, run the following:

```bash
madengine run --tags {model} --additional-context "{'multi_node_args':{'RUNNER': 'torchrun', 'MASTER_ADDR': '10.227.23.63', 'MASTER_PORT': '400', 'NNODES': '2', 'NODE_RANK': '1'}}"
```

### mpirun

Coming Soon!

## Sharing Data

DLM/MAD multi-node feature assumes the dataset is in a shared-file system for all participating nodes. For example, look at the following 2-node run of the Megatron-LM Llama2 workload.

On the first node (assumed to be master node), run the following:

```bash
madengine run --tags pyt_megatron_lm_train_llama2_7b --additional-context "{'multi_node_args': {'RUNNER': 'torchrun', 'MASTER_ADDR': '10.194.129.113', 'MASTER_PORT': '4000', 'NNODES': '2', 'NODE_RANK': '0', 'NCCL_SOCKET_IFNAME': 'ens14np0', 'GLOO_SOCKET_IFNAME': 'ens14np0'}}" --force-mirror-local /nfs/data
```

On the second node, run the following:

```bash
madengine run --tags pyt_megatron_lm_train_llama2_7b --additional-context "{'multi_node_args': {'RUNNER': 'torchrun', 'MASTER_ADDR': '10.194.129.113', 'MASTER_PORT': '4000', 'NNODES': '2', 'NODE_RANK': '1', 'NCCL_SOCKET_IFNAME': 'ens14np0', 'GLOO_SOCKET_IFNAME': 'ens14np0'}}" --force-mirror-local /nfs/data
```

You can see at the end of these commands, we are pointing DLM/MAD to the shared-file system where the data can be located.

**NOTE: The above commands assumes the shared-file system is mounted at `/nfs` in the commands above. If this is not the case and a user simply copies/pastes the above commands on two nodes, DLM/MAD will create a folder called `nfs` on each node and copy the data there, which is not desired behavior.**

## SLURM Cluster Integration

madengine now supports running workloads on SLURM clusters, allowing you to leverage job scheduling and resource management for multi-node training and inference.

### Overview

When `slurm_args` is provided in the `additional-context`, madengine will:
1. Parse the SLURM configuration parameters
2. Submit the job directly to the SLURM cluster using `sbatch`
3. Skip the standard Docker container build and run workflow
4. Execute the model-specific script (e.g., `scripts/sglang_disagg/run.sh`) which handles SLURM job submission

### SLURM Arguments

The following arguments can be specified in the `slurm_args` dictionary:

| Argument | Description | Required | Example |
|----------|-------------|----------|---------|
| `FRAMEWORK` | Framework to use for the job | Yes | `'sglang_disagg'` |
| `PREFILL_NODES` | Number of nodes for prefill phase | Yes | `'2'` |
| `DECODE_NODES` | Number of nodes for decode phase | Yes | `'2'` |
| `PARTITION` | SLURM partition/queue name | Yes | `'amd-rccl'` |
| `TIME` | Maximum job runtime (HH:MM:SS) | Yes | `'12:00:00'` |
| `DOCKER_IMAGE` | Docker image to use (optional) | No | `''` (uses default from run.sh) |

### Usage Examples

#### Basic SLURM Job Submission

To run a model on SLURM with default settings:

```bash
madengine run --tags sglang_disagg_pd_qwen3-32B \
  --additional-context "{'slurm_args': {
    'FRAMEWORK': 'sglang_disagg',
    'PREFILL_NODES': '2',
    'DECODE_NODES': '2',
    'PARTITION': 'amd-rccl',
    'TIME': '12:00:00',
    'DOCKER_IMAGE': ''
  }}"
```

#### Custom Docker Image

To specify a custom Docker image for the SLURM job:

```bash
madengine run --tags sglang_disagg_pd_qwen3-32B \
  --additional-context "{'slurm_args': {
    'FRAMEWORK': 'sglang_disagg',
    'PREFILL_NODES': '4',
    'DECODE_NODES': '4',
    'PARTITION': 'gpu-high-priority',
    'TIME': '24:00:00',
    'DOCKER_IMAGE': 'myregistry/custom-image:latest'
  }}"
```

#### Running Different Model Configurations

For DeepSeek-V2 model:

```bash
madengine run --tags sglang_disagg_pd_deepseek_v2 \
  --additional-context "{'slurm_args': {
    'FRAMEWORK': 'sglang_disagg',
    'PREFILL_NODES': '8',
    'DECODE_NODES': '8',
    'PARTITION': 'amd-mi300x',
    'TIME': '48:00:00',
    'DOCKER_IMAGE': ''
  }}"
```

### Model Configuration

Models configured for SLURM should include the model name in the `args` attribute of `models.json`. For example:

```json
{
  "name": "sglang_disagg_pd_qwen3-32B",
  "args": "--model Qwen3-32B",
  "tags": ["sglang_disagg"]
}
```

The model name (e.g., `Qwen/Qwen3-32B`) will be extracted and set as the `MODEL_NAME` environment variable for the SLURM job.

### Requirements

To use SLURM integration, ensure the following are available:

1. **SLURM Cluster Access**: Access to a SLURM cluster with proper credentials
2. **Model Scripts**: Framework-specific scripts (e.g., `scripts/sglang_disagg/run.sh`) that handle SLURM job submission

### How It Works

1. **Context Parsing**: madengine detects `slurm_args` in the additional context
2. **Model Selection**: Extracts model information from `models.json` based on the provided tags
3. **Environment Setup**: Prepares environment variables including `MODEL_NAME`, node counts, partition, etc.
4. **Job Submission**: Executes the framework-specific run script which submits the SLURM job using `sbatch`
5. **Job Monitoring**: The SLURM cluster manages job execution, resource allocation, and scheduling

## TODO

### RUNNER

- [x] torchrun
- [ ] mpirun (requires ansible integration)

### Job Scheduler

- [x] SLURM (via slurm_args integration)
- [ ] Kubernetes

### Design Consideration

- [x] SLURM integration using sbatch scripts for job submission
- [ ] Full Python workflow for multi-node (without bash script intermediaries)
- [ ] Kubernetes-native job scheduling integration
