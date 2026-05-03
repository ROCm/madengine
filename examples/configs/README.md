# YAML Config Examples (`--config`)

```
configs/
├── templates/   # Full reference — every field shown and annotated
│   ├── local.yaml
│   ├── slurm.yaml
│   └── k8s.yaml
└── demo/        # Minimal ready-to-run examples organised by target
    ├── local/
    ├── slurm/
    └── k8s/
```

## Workflow

**Starting from scratch** — copy a template, fill in your model tag and cluster
settings, then delete the sections you don't need:

```bash
cp examples/configs/templates/slurm.yaml my_job.yaml
# edit my_job.yaml …
madengine run --config my_job.yaml
```

**Starting from an example** — find the demo closest to your use case and
adapt it:

```bash
cp examples/configs/demo/slurm/multi-node-torchrun.yaml my_job.yaml
# tweak partition, node count, tags …
madengine run --config my_job.yaml
```

**Inline overrides** — any field can be overridden without editing the file:

```bash
madengine run --config my_job.yaml --config distributed.nnodes=4
madengine run --config my_job.yaml --config +env=nccl_debug
madengine run --config my_job.yaml --config +tools=rocprofv3_lightweight
```

> `--config` is mutually exclusive with `--additional-context` /
> `--additional-context-file`. See `docs/configuration.md` for the full
> field reference.

---

## `templates/`

| File | Target | Contents |
|------|--------|----------|
| `local.yaml` | Local Docker | All docker, model, tools, scripts, log-error, output fields |
| `slurm.yaml` | SLURM | All slurm, distributed, env_vars, tools, scripts fields |
| `k8s.yaml` | Kubernetes | All k8s, distributed, env_vars, tools, secrets, storage fields |

## `demo/local/`

| File | Model | Description |
|------|-------|-------------|
| `single-gpu.yaml` | `dummy` | Single GPU, no distribution |
| `multi-gpu-torchrun.yaml` | `dummy_torchrun` | Single node, 4 GPUs, torchrun |
| `deepspeed.yaml` | `dummy_deepspeed` | DeepSpeed ZeRO, single node |
| `vllm-inference.yaml` | `dummy_vllm` | vLLM tensor parallelism, 4 GPUs |
| `profiling.yaml` | `dummy` | ROCprofv3 + power + VRAM profiling |

## `demo/slurm/`

| File | Model | Description |
|------|-------|-------------|
| `single-node-single-gpu.yaml` | `dummy` | Single GPU job |
| `multi-node-torchrun.yaml` | `dummy_torchrun` | 2 nodes × 8 GPUs, Ethernet |
| `multi-node-torchrun-infiniband.yaml` | `dummy_torchrun` | 4 nodes × 8 GPUs, InfiniBand, account/QoS |
| `deepspeed.yaml` | `dummy_deepspeed` | DeepSpeed, single node |
| `megatron-lm.yaml` | `dummy_megatron_lm` | Megatron-LM, 4 nodes × 8 GPUs |
| `torchtitan.yaml` | `dummy_torchtitan` | TorchTitan TP+PP+FSDP2, 4 nodes × 8 GPUs |
| `vllm-inference.yaml` | `dummy_vllm` | vLLM data parallelism, 2 nodes × 4 GPUs |
| `sglang-inference.yaml` | `dummy_sglang` | SGLang, 2 nodes × 4 GPUs |
| `sglang-disagg.yaml` | `dummy_sglang_disagg` | SGLang disaggregated prefill/decode, 5 nodes |
| `profiling-multi-gpu.yaml` | `dummy_torchrun` | torchrun + RCCL + power + VRAM profiling |

## `demo/k8s/`

| File | Model | Description |
|------|-------|-------------|
| `single-gpu.yaml` | `dummy` | Single GPU pod |
| `multi-gpu-torchrun.yaml` | `dummy_torchrun` | 1 pod × 8 GPUs, torchrun |
| `multi-node-torchrun.yaml` | `dummy_torchrun` | 2 pods × 8 GPUs, node selector |
| `nvidia-gpu.yaml` | `dummy_torchrun` | NVIDIA A100/H100, `nvidia.com/gpu` |
| `deepspeed.yaml` | `dummy_deepspeed` | DeepSpeed, single pod |
| `megatron-lm.yaml` | `dummy_megatron_lm` | Megatron-LM, 4 pods × 8 GPUs |
| `torchtitan.yaml` | `dummy_torchtitan` | TorchTitan TP+PP+FSDP2, 4 pods × 8 GPUs |
| `vllm-inference.yaml` | `dummy_vllm` | vLLM data parallelism, 2 pods × 4 GPUs |
| `sglang-inference.yaml` | `dummy_sglang` | SGLang, 2 pods × 4 GPUs |
| `sglang-disagg.yaml` | `dummy_sglang_disagg` | SGLang disaggregated, 5 pods |
