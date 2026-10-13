# MI250 llm-d values

These files are the configuration that served `Qwen/Qwen3-0.6B` on `mad-mi250-k8s-cluster`. Charts used:

- `llm-d-infra` v1.4.0
- `inferencepool` v1.5.0, with the endpoint picker image overridden to `ghcr.io/llm-d/llm-d-router-endpoint-picker:v0.11.0`
- `llm-d-modelservice` v0.4.16

For the madengine-managed equivalent (card-driven `llm_d` context, generated Helm
values, reporting), see [docs/llm-d.md](../../../docs/llm-d.md) — especially
[MI250 / AMD path constraints](../../../docs/llm-d.md#mi250--amd-path-constraints).

The namespace `llm-d-bench` must already contain `dockerhub-rocm` and `llm-d-hf-token` (key `HF_TOKEN`). This cluster has only the `local-path` StorageClass, so model weights use an emptyDir (`hf://`) and are not put on a ReadWriteMany volume.

`modelservice-disagg-values.yaml` is two prefill pods and two decode pods, **one GCD per pod** (`gpu_count: 1`, TP=1), with pod anti-affinity across nodes. UCX must use `rocm_copy,rocm_ipc,tcp,self,sm`. `cuda_copy` is not in this image and NIXL then fails to register GPU memory. There is no InfiniBand device on the pods, so this is not an RDMA / MoRI / WideEP path.

`modelservice-aggregated-values.yaml` is four independent vLLM pods and no prefill/decode split. For that mode, install the InferencePool chart without the endpoint-picker override in `inferencepool-values.yaml`. The prefill/decode picker will try to send every prompt to a prefill pod, and there is none.

Same gateway, same prompts (`vllm bench serve`, random 256 in / 128 out, 160 prompts, concurrency 32, seed 1), four MI250 GCDs either way:

| | Aggregated (4 decode) | Disaggregated (2 prefill + 2 decode) |
|---|---:|---:|
| Successful / failed | 160 / 0 | 160 / 0 |
| Request throughput (req/s) | 8.39 | 5.05 |
| Output throughput (tok/s) | 1074 | 647 |
| Mean TTFT (ms) | 514 | 1648 |
| Mean E2EL (ms) | 3648 | 5964 |

Qwen3-0.6B at this prompt length does not fill one MI250. KV transfer on this cluster is UCX `rocm_copy` over TCP (about 29 MB/s, 1.75 MB for a short prompt) because the pods have no InfiniBand device. Disaggregation is working here; it is not faster on this workload. A longer prompt, or Qwen3-8B, is the comparison that can change that.

```bash
helm repo add llm-d-infra https://llm-d-incubation.github.io/llm-d-infra/
helm repo add llm-d-modelservice https://llm-d-incubation.github.io/llm-d-modelservice/
helm repo update

helm upgrade --reset-values --install llm-d llm-d-infra/llm-d-infra --version v1.4.0 \
  -n llm-d-bench -f deploy/llm-d/mi250/infra-values.yaml
helm upgrade --reset-values --install llm-d-gaie \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool --version v1.5.0 \
  -n llm-d-bench -f deploy/llm-d/mi250/inferencepool-values.yaml
helm upgrade --reset-values --install llm-d-ms llm-d-modelservice/llm-d-modelservice --version v0.4.16 \
  -n llm-d-bench -f deploy/llm-d/mi250/modelservice-disagg-values.yaml
```
