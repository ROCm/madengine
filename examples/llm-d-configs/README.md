# llm-d Configuration Examples

Ready-to-use `--additional-context` configs for benchmarking
[llm-d](https://github.com/llm-d/llm-d). See [docs/llm-d.md](../../docs/llm-d.md) for
the full reference.

| File | Mode | What it shows | Model repo |
|---|---|---|---|
| [01-attach-existing-stack.json](01-attach-existing-stack.json) | attach | Benchmark a stack you already run. Installs nothing, tears nothing down. | `Qwen/Qwen3-32B` |
| [02-managed-dry-run.json](02-managed-dry-run.json) | managed (dry run) | Render values + `helm template`, install nothing. **Start here** before a real managed run. | `Qwen/Qwen3-32B` |
| [03-managed-disaggregated.json](03-managed-disaggregated.json) | managed | Prefill/decode disaggregation on MI300X, TP=8. | `deepseek-ai/DeepSeek-R1-0528` |
| [04-managed-aggregated-keep-stack.json](04-managed-aggregated-keep-stack.json) | managed | Aggregated serving (`prefill.replicas: 0`), `teardown: false`, `extra_values`. | `meta-llama/Llama-3.1-8B-Instruct` |
| [05-managed-qwen3-0.6b-amd.json](05-managed-qwen3-0.6b-amd.json) | managed | Real end-to-end smoke test of the llm-d path on AMD: aggregated serving of the smallest current Qwen model. | `Qwen/Qwen3-0.6B` |
| [06-managed-qwen3-0.6b-nvidia.json](06-managed-qwen3-0.6b-nvidia.json) | managed | Same smoke test on NVIDIA — only `gpu_vendor` and `decode.image` differ from `05`. | `Qwen/Qwen3-0.6B` |

`05`/`06` are a different kind of example from `01`-`04`: instead of demonstrating a
config feature against a large production model, they are a real (non-fixture)
smoke test meant to answer one question cheaply — does this cluster's llm-d
installation (Gateway, GAIE, modelservice) work at all — before committing GPUs
to a larger model. Both set `decode.image` explicitly, because `dummy_llm_d`'s
own built image is a CPU-only benchmark client with no vLLM installed; the
usual role-image default (the `--tags`-selected model's own image) can't serve
here. `05` and `06` differ only in `gpu_vendor` and `decode.image` — the full
extent of what changes to move this test between GPU vendors. This is distinct
from the CPU-only `dummy_llm_d` contract check described below in "Validating
the client contract", which exercises no cluster or GPU at all.

Each model repo in `01`-`04` is one MAD already tracks for standalone vLLM benchmarking
(`scripts/vllm/models.json` — `pyt_vllm_qwen3-32b`, `pyt_vllm_deepseek-r1`,
`pyt_vllm_llama-3.1-8b`). Those models run vLLM serve-and-benchmark inside a single
container; these configs benchmark the *same* repos served instead through an
external llm-d gateway, with disaggregated prefill/decode where it matters.

`model.hf_repo` (02) is shorthand for `model.uri: "hf://<hf_repo>"` — simplest to write, but
re-downloads the model on every standup. `03` adds `model.cache_pvc`, since re-downloading
1.5TiB every run is impractical: madengine downloads the repo onto that PVC itself, in a Job
it runs before standup, and reuses it on later runs (the chart's own `pvc://`/`pvc+hf://`
schemes only mount an already-populated PVC — they have no download logic of their own).
`04` uses a literal `model.uri`, with the plain `hf://` scheme. See
[Model weights: `hf_repo` vs `uri`](../../docs/llm-d.md#model-weights-hf_repo-vs-uri).

## Usage

```bash
madengine run --tags my_llm_d_benchmark \
  --additional-context "$(cat examples/llm-d-configs/01-attach-existing-stack.json)"
```

## Before you run a managed config

1. **Replace every `<pin>`** with a real chart version. Managed mode refuses to run
   while any `charts.<name>.version` is null, because a floating chart version makes
   benchmark numbers non-reproducible. Find versions with `helm show chart <ref>`.
   `05`/`06` already have real versions pinned, but override `infra`/`modelservice`'s
   `ref` too: anonymous `helm pull`/`helm template` against their `oci://ghcr.io/...`
   default refs returns 401/403 as of writing (the `gaie` chart on `registry.k8s.io`
   is unaffected). Both charts are also published to a plain, genuinely anonymous
   GitHub Pages Helm repo — run once before installing:
   `helm repo add llm-d-infra https://llm-d-incubation.github.io/llm-d-infra/ && helm repo add llm-d-modelservice https://llm-d-incubation.github.io/llm-d-modelservice/ && helm repo update`.
2. **Create the namespace.** madengine never creates or deletes one.
3. **Create the HF token Secret** if the model is gated, and pass its *name* as
   `model.hf_token_secret`. The token itself never reaches a values file or a helm
   command line.
4. **Dry-run first** with `02-managed-dry-run.json` and read
   `./k8s_manifests/llm-d-*-values.yaml`.

## Attach vs managed

Attach mode is the safe default on a shared cluster: with `endpoint_url` set, madengine
cannot install or uninstall anything. Managed mode installs three helm releases
(`<prefix>-infra`, `<prefix>-gaie`, `<prefix>-modelservice`) and removes them on the way
out — including after a failure, and including after a *successful* run.

## Validating the client contract without a cluster

None of the models above can be run without a real llm-d stack — standing one up needs a
Kubernetes cluster with GPU nodes and the Gateway API/GAIE CRDs installed. To sanity-check
just the client side of the contract (the `MAD_LLM_D_*` env vars madengine injects), use the
`dummy_llm_d` fixture (`tests/fixtures/dummy/`, tags `dummy_llm_d`/`llm_d`). It needs no
GPU — point it at any OpenAI-compatible server, such as a local `vllm serve`:

```bash
MAD_LLM_D_ENDPOINT=http://localhost:8000 MAD_LLM_D_MODEL=facebook/opt-125m \
  bash tests/fixtures/dummy/scripts/dummy_llm_d/run.sh
```

This does not exercise helm standup, Gateway resolution, or GPU-backed serving — only that
the endpoint and model name reach the client and a real request round-trips.
