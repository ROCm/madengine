# madengine

madengine discovers named Models from a catalog and runs them on a local machine or a cluster.

## Language

### Catalog

**Model**:
A named catalog entry, whether written in a registry file or generated. Its identity is its name: path-qualified for an entry discovered under `scripts/`, and the name as written for an entry in the root `models.json`.
_Avoid_: model card, workload, image, weights, container

**Tag**:
A label carried by a Model. Many Models may carry the same Tag. A Model's name is its identity, separate from its Tags.
_Avoid_: selector, model name

**Selector**:
A string given to `--tags` that resolves to one or more Models. It may be a Model's name, a Tag, `all`, or a scoped form such as `scope/filter`. A colon suffix is arguments for that selection, separate from the Selector itself.
_Avoid_: tag

**Registry**:
The Models contributed by one directory, from either a `models.json` or a `get_models_json.py`.
_Avoid_: catalog

**Catalog**:
Every Registry found in one discovery of the working tree, including the root `models.json`.
_Avoid_: registry, model list

### Execution

**Invocation**:
One `madengine run` command. It may select many Models.
_Avoid_: session, run

**Run**:
The attempt to execute one Model during an Invocation. On SLURM and Kubernetes, that Model is the first one in the manifest.
_Avoid_: invocation, result, session

**Result**:
One row a Run appends to `perf.csv`. A Run that names a multiple-results file appends one Result per row of that file.
_Avoid_: run, performance

**Image**:
The runnable artifact a Build records for one Model. The Model's dockerfile is the recipe.
_Avoid_: model, container, dockerfile

**Build**:
The recording of one Image per selected Model.
_Avoid_: run, invocation

**Context**:
The settings for an Invocation, supplied by `--additional-context` or `--config` and merged with the host facts the engine detects. A Build stores that Context with its Images.
_Avoid_: additional context, config, deployment config

**Manifest**:
The file a Build writes. It records one Image for each selected Model and the Context from that Build.
_Avoid_: model card, catalog, config

**Launcher**:
The backend that starts a Model's training or inference process during a Run. `native` means that Run has no Launcher. A Launcher is one of three contracts: a templated launcher, `slurm_multi`, or `primus`.
_Avoid_: platform, scheduler, runtime, deployment

**Templated launcher**:
A Launcher whose start command the deployment templates render. Training names are `torchrun`, `deepspeed`, `megatron-lm`, and `torchtitan`. Inference names are `vllm`, `sglang`, `sglang-disagg`, and `vllm-disagg`. These run on SLURM, Spur, and Kubernetes. `vllm-disagg` and `sglang-disagg` need a cluster. Aggregated llm-d records `vllm`. Disaggregated llm-d records `vllm-disagg`.
_Avoid_: slurm_multi, primus

**Self-managed launcher**:
`slurm_multi`. The Model's own script starts its containers. It runs on SLURM or Spur.
_Avoid_: vllm-disagg, templated launcher

**Primus**:
The training Launcher for the Primus project. A local single-node Run. The model script starts Primus.
_Avoid_: torchrun, slurm_multi
