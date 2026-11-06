# madengine brief: speaker notes

Target: 15 minutes, 11 slides, audience of principal and senior engineers who have not used madengine or MAD. Lead with the design reasoning, and keep flags and syntax in the slides, not in your mouth. Total is about 14.5 minutes, which leaves a little for questions or overrun.

Key terms to define the first time you say them:
- **Card**: a JSON entry in MAD that describes one runnable model.
- **Target**: where a run executes (local Docker, conda, SLURM, Spur, Kubernetes, llm-d).
- **Launcher**: how a workload starts inside its container (torchrun, vllm, and so on).
- **Context**: the one dictionary that holds every setting for a run.

---

## 1. Cover (0:30)

- madengine is a CLI that takes a model from a catalog entry to a benchmark number.
- The promise: the same command shape works on one GPU, eight GPUs, many nodes, or behind a serving gateway.
- Set the agenda out loud: problem, what it is, how it works, what you get, then real examples and the config cookbook.
- Mention the full deck exists for anyone who wants depth afterwards.

## 2. The problem (1:30)

- A benchmark that works on a workstation rarely survives a move to a cluster. Everything around the model changes: image names, launchers, schedulers, result formats.
- Walk the list quickly: docker run flags, per-cluster sbatch scripts, Kubernetes manifests, helm and gateways, and a different output format from every harness.
- The real cost is trust. When the plumbing differs, the number you report stops meaning the same thing across machines.
- Point at the code block. Same model, same command. The only change is one block in the context: `slurm`, `k8s`, or `llm_d`.
- Land the line: same model card, same `perf.csv`, no code changes between targets.

## 3. What it is, by the numbers (1:45)

- Three pieces. MAD is the catalog (what to run). madengine is the runner (how and where). `perf.csv` is the contract (what came out).
- Show the card on the right. It is small: a name, tags, a Dockerfile, a script, arguments, a GPU count.
- The contract with madengine is tiny. The script prints `performance: <number> <metric>`, or writes a CSV for a sweep. That is all a model author has to learn.
- Numbers: 121 static cards plus about 470 generated from Primus and JAX configs, 6 targets, 10 launcher names, 30 profiler presets.
- If asked: most static cards are vLLM serving (42) and disaggregated prefill/decode (34). Counts are from a MAD checkout in Nov 2026 and move with the Primus revision.

## 4. Quick start (1:30)

- Four commands: install, clone MAD, `discover`, `run`. Say that the Hugging Face token is the only secret you need for this card.
- Explain what happens on the right: discover, build, run in a container with a watchdog as PID 1, and a row in `perf.csv`.
- Stress that failed runs are recorded as `FAILURE`. Every attempt shows up, so a missing number is never a silent miss.
- There are seven commands total. You only need three of them to get started.
- Prerequisites are Docker, Python 3.9+, and ROCm or CUDA drivers.

## 5. The pipeline: build once, run anywhere (1:15)

- Read the flow left to right: discover, build, manifest, run, `perf.csv`, report.
- The manifest is the key design choice. It records the image and the Context that built it, so build and run can happen on different machines.
- Typical split: CI builds and pushes to a registry, then any target runs from `build_manifest.json`.
- Two escape hatches: `--use-image` to skip the build, and `--require-pinned-image` to run from a digest and fail rather than run a tag that moved.
- Reliability line: a cancelled run should not hold a GPU. A PID 1 watchdog, signal handlers, and `madengine cleanup` cover it.

## 6. Layer structure (1:45)

- Seven layers, each with one job. Walk top to bottom, and name only the highlighted pieces.
- CLI parses. Orchestration decides the order and writes the manifest. Core holds shared services, mainly the Context.
- The split that matters most: **Execution** runs on this machine, **Deployment** submits to a cluster. Both end up calling a launcher inside the job.
- Reporting is separate, after the run.
- Right panel: there is no `deploy` flag. The target is inferred from the keys in the Context, and the first match wins.
- Why this is good: moving a job between clusters is a diff of one block. Conflicting blocks, such as `slurm` with `k8s`, are rejected before submission.
- If asked about extensibility: every cluster target implements the same lifecycle (validate, prepare, deploy, monitor, collect, cleanup).

## 7. Feature overview (1:15)

- Do not read all eight cards. Group them aloud: where it runs, how it parallelizes, how you analyze it, and how you trust it.
- Where it runs: container-native Docker, bare metal in a conda prefix, and the four cluster targets with presets.
- How you analyze: profilers (rocprof, rocprofv3, rocm-trace-lite, TraceLens) that can be stacked.
- How you trust it: pinned image digests, a log error scan so a quiet crash does not report SUCCESS, and one result schema.
- Mention the extras line (Hydra YAML, data providers, environment capture) as "also there" and move on.

## 8. Launchers (1:30)

- Definition first: a launcher is how a workload starts inside its own container. It is not the scheduler and not the platform.
- Three contracts. **Templated**: the deployment templates render the start command. **Self-managed** (`slurm_multi`): the card's own `.slurm` script starts its containers. **Primus**: local single-node, and the model script starts Primus.
- The matrix answers "will this launcher run on this target?" Validation rejects a launcher on a target with no arm for it, before any GPU time is spent.
- Disaggregated launchers need at least 2 nodes on SLURM and 3 on Kubernetes.
- llm-d is a deployment, not a launcher. Its result rows are stamped with `launcher=vllm`, or `vllm-disagg` when prefill replicas run.

## 9. Parallelism (1:15)

- Give the legend in one breath: TP tensor, PP pipeline, DP data, CP context, EP expert.
- **Auto** means madengine configures it. **Manual** means the launcher supports it but your script or config sets the degrees.
- Training: torchtitan and megatron-lm configure the most automatically. torchrun and deepspeed are lighter and more manual.
- Inference: vLLM handles tensor parallelism and replicas automatically. Disaggregated SGLang assigns parallelism by role.
- MoE models such as DeepSeek-V3 get expert parallelism and DP attention from the model's own disaggregated scripts, run through `slurm_multi`.
- Primus and plain SGLang are in the footnote. Say so if someone asks where they went.

## 10. Hands-on: four rungs (1:15)

- Same tool, increasing scale: 1 GPU, 8 GPUs on one node, 2 nodes, then N nodes and llm-d.
- Name one real model per rung. For example `pyt_vllm_llama-3.1-8b_fp8` at the bottom and `pyt_sglang_disagg_mori_dp_deepseek-v3` at the top.
- The table is the practical part. Each target is a different block added to the Context, plus a different way ranks start (`docker run`, `srun`, a job array, an Indexed Job, an HTTP client).
- Close the slide on the guarantee: validation before submission, the same `perf.csv`, and cleanup of only what madengine created.
- Point to the full deck for ten worked examples with exact commands.

## 11. Cookbook and next steps (1:00)

- This slide answers "I want to change X, what do I add?" Pick three rows: base image, env vars, and profilers.
- Show the base-image example. A newer base image with no Dockerfile edit, using `docker_build_arg`.
- Warn about profilers: pick one GPU profiler and keep a baseline run with no tools. Do not combine `rocprof*` with `rocm_trace_lite`.
- Tip: use a JSON file for anything with booleans or nesting. `--config` cannot be combined with `--additional-context`.
- Finish on the docs list and the full deck link. Invite questions.

---

## Likely questions

| Question | Short answer |
|---|---|
| Do I have to write my own cards? | Only for new models. Existing MAD cards run as they are. A card is a JSON entry plus a Dockerfile and a script. |
| Does it need Docker everywhere? | Local and Kubernetes runs use containers. Bare metal uses a conda prefix instead (`"platform": "bare_metal"`). |
| How is it different from running sbatch myself? | madengine renders the sbatch script, sets rank variables, collects results, and cleans up. You supply a few keys, not a script. |
| What happens on a failure? | The run is recorded as `FAILURE` in `perf.csv`. Exit codes (2 build, 3 run, 4 invalid args, 5 GPU wedged) let CI react. |
| Is the target really inferred? | Yes. The first matching key wins: `platform`, `llm_d`, `k8s`, `slurm` (Spur or SLURM), else local Docker. |
| What is new in 3.0? | The brief only marks `vllm-disagg` as 3.0. For the rest, check the release notes before answering. |
