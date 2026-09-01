# llm-d Configuration Examples

Ready-to-use `--additional-context` configs for benchmarking
[llm-d](https://github.com/llm-d/llm-d). See [docs/llm-d.md](../../docs/llm-d.md) for
the full reference.

| File | Mode | What it shows |
|---|---|---|
| [01-attach-existing-stack.json](01-attach-existing-stack.json) | attach | Benchmark a stack you already run. Installs nothing, tears nothing down. |
| [02-managed-dry-run.json](02-managed-dry-run.json) | managed (dry run) | Render values + `helm template`, install nothing. **Start here** before a real managed run. |
| [03-managed-disaggregated.json](03-managed-disaggregated.json) | managed | Prefill/decode disaggregation on MI300X, TP=8. |
| [04-managed-aggregated-keep-stack.json](04-managed-aggregated-keep-stack.json) | managed | Aggregated serving (`prefill.replicas: 0`), `teardown: false`, `extra_values`. |

## Usage

```bash
madengine run --tags my_llm_d_benchmark \
  --additional-context "$(cat examples/llm-d-configs/01-attach-existing-stack.json)"
```

## Before you run a managed config

1. **Replace every `<pin>`** with a real chart version. Managed mode refuses to run
   while any `charts.<name>.version` is null, because a floating chart version makes
   benchmark numbers non-reproducible. Find versions with `helm show chart <ref>`.
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
