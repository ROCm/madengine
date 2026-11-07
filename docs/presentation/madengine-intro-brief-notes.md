# madengine brief: speech notes

**Talk length:** 15 minutes. **Slides:** 11. **Audience:** principal and senior engineers who have not used madengine or MAD.

**How to use this file:** the text under each slide is what you say, in plain words. Lines in *italics* are cues for you, not words to read out. Speak slowly. Pause at the end of each slide. If you run late, shorten slides 7 and 11 first.

**Four words to explain the first time you use them:**
- **Card:** one JSON entry in MAD that describes one model you can run.
- **Target:** the place where a run happens, such as local Docker, SLURM or Kubernetes.
- **Launcher:** the tool that starts the job inside the container, such as torchrun or vLLM.
- **Context:** one dictionary that holds all the settings for a run.

---

## Slide 1. Cover (0:30)

Hi everyone. Today I will show you madengine. It is a command line tool that takes an AI model from a catalog entry to a benchmark number.

The big idea is simple. You use the same command on one GPU, on eight GPUs, or on many nodes. Only a small piece of config changes.

I will keep it high level. There is a longer deck if you want details later.

*Click to next slide.*

---

## Slide 2. The problem (1:30)

Let me start with a problem you may know.

You build a benchmark on your workstation. It works. Then you move it to a cluster, and it breaks. Not the model. Everything around the model.

On the left you see five things that change. First, you start the container by hand with `docker run`, and you set devices, mounts and environment. Second, on a cluster you write `sbatch` scripts, and each cluster is a little different. Third, on Kubernetes you write manifests for jobs, volumes and secrets. Fourth, for serving you use helm and gateways. And fifth, every test harness prints its result in a different format.

The real cost is trust. When the setup is different, the number you report no longer means the same thing.

Now look at the code on the right. It is the same model every time. First line: local Docker. Then I add a `slurm` block, and it runs on SLURM. I add a `k8s` block, and it runs on Kubernetes. I add an `llm_d` block, and it runs behind an llm-d gateway.

Same model card. Same result file, called `perf.csv`. No code changes between targets. That is what madengine gives you.

---

## Slide 3. What it is, by the numbers (1:45)

So what is madengine? Think of three parts.

The first part is MAD. MAD is the catalog. It tells you what to run. It has model cards, Dockerfiles and scripts.

The second part is madengine. This is the runner. It decides how and where to run. It builds the image, picks the target from your config, starts the job and watches it.

The third part is the result file, `perf.csv`. This is the contract. Every target writes the same format. Even failed runs are written.

On the bottom right you can see a model card. It is very small. It has a name, some tags, a Dockerfile, a script, the arguments and the number of GPUs. And the contract for a model author is also small. Your script prints one line, like `performance: 1234 tokens_per_second`. Or it writes a CSV file if you run a sweep. That is all you need to learn.

Now the numbers. MAD has 121 static model cards. It also generates about 470 more for Primus and JAX training. madengine supports 6 targets: Docker, conda, SLURM, Spur, Kubernetes and llm-d. It has 10 launcher names, and 30 profiler presets that you can stack.

*If asked: most static cards are vLLM serving, 42 of them, and disaggregated prefill and decode, 34 of them.*

---

## Slide 4. Quick start (1:30)

Now let me show how easy it is to start.

There are four steps. Install madengine with pip. Clone the MAD repo. Run `discover` to check that the model card exists. Then run the model. For this model you need a Hugging Face token. You set it as an environment variable.

On the right you see what happens. First, discover finds the card by name or tag. Second, build makes the Docker image from the card's Dockerfile. Third, run starts the container. A small watchdog runs as the first process, so the container can clean itself up. Fourth, the result goes into `perf.csv`.

One thing I like: failed runs are also written, with the status FAILURE. So every attempt shows up. A missing number is never a silent miss.

There are seven commands in total. You only need three to start: `discover`, `build` and `run`. The others make reports, upload to a database, clean up, and make a model picker page.

For prerequisites, you need Docker, Python 3.9 or newer, and ROCm or CUDA drivers.

---

## Slide 5. The pipeline: build once, run anywhere (1:15)

This slide shows the full flow, from left to right. Discover, build, manifest, run, `perf.csv`, and report.

The important part is the manifest. It is a file called `build_manifest.json`. It stores the image and the Context that built it. Because of this, build and run can happen in different places.

A common pattern is this. Your CI system builds the image and pushes it to a registry. Later, any target can run from the manifest. That can be a workstation, a SLURM cluster or Kubernetes.

There are two useful options. If you already have an image, use `--use-image` and skip the build. And if you want a safe comparison, use `--require-pinned-image`. It runs from an exact digest, and it fails if the tag has moved.

One more point. A cancelled run should not keep your GPU. The watchdog, signal handlers and the `cleanup` command take care of that.

---

## Slide 6. Layer structure (1:45)

Now let me show how madengine is built. There are seven layers, and each layer has one job.

At the top is the CLI. It reads your command. Next is orchestration. It decides the order of steps and writes the manifest. Then Core. It has shared services, and the main one is the Context.

Now look at the next two layers, because this is the key idea. Execution runs on this machine. Deployment sends the job to a cluster. Both end by starting a launcher inside the job. The last layer is reporting, and it runs after the job.

Now look at the table on the right. There is no flag that says "deploy here". madengine looks at the keys in your Context. If you wrote a `slurm` block, it uses SLURM. If you wrote `k8s`, it uses Kubernetes. If you wrote `llm_d`, it uses llm-d. If you wrote none of these, it runs in local Docker. The first match wins.

This is good because moving a job from one cluster to another is a change of one block. And if you write blocks that conflict, like `slurm` and `k8s`, madengine rejects them before it sends the job.

*If asked about adding a new target: every cluster target follows the same six steps: validate, prepare, deploy, monitor, collect results and clean up.*

---

## Slide 7. Feature overview (1:15)

Here are the main features. I will not read every card. Let me group them.

First, where it runs. It can run in a container, with ROCm and CUDA handled for you. It can also run on bare metal in a conda environment. And it supports four cluster targets: SLURM, Spur, Kubernetes and llm-d. You add a block, and presets fill in the rest.

Second, how it runs. There are ten launcher names, for both training and inference. I will explain them in the next two slides.

Third, how you analyze. There are profiler presets, such as rocprof and rocprofv3, and also Kineto traces with TraceLens, power and memory.

Fourth, how you can trust the result. Image digests are recorded. A log scan stops a quiet crash from looking like a success. And you get one result format everywhere.

At the bottom there are also extras: YAML config, data providers and environment capture. Good to know, but I will skip them today.

---

## Slide 8. Launchers (1:30)

Now launchers. A launcher is how a job starts inside its own container. It is not the scheduler, and it is not the platform.

There are three kinds, or three contracts.

The first kind is templated. These are torchrun, deepspeed, megatron-lm, torchtitan, vllm, sglang, and the two disaggregated ones. madengine fills in a template and writes the start command for you.

The second kind is self-managed. This is `slurm_multi`. The model card has its own `.slurm` script that starts its own containers. MAD uses this for disaggregated and communication benchmarks.

The third kind is Primus. It runs on one node only. The model script starts Primus, and a cluster block is not allowed.

The table tells you where each launcher can run. A tick means yes. For example, the disaggregated launchers need at least 2 nodes on SLURM and 3 on Kubernetes. If you pick a launcher that a target does not support, madengine stops you before it uses any GPU time.

One small note about llm-d. It is a deployment, not a launcher. So the result rows are stamped with vllm, or vllm-disagg when there are prefill replicas.

---

## Slide 9. Parallelism (1:15)

This slide shows which launcher handles which kind of parallelism.

Quick reminder of the short names. TP is tensor parallel. PP is pipeline parallel. DP is data parallel. CP is context parallel, which splits the sequence. EP is expert parallel, for mixture-of-experts models.

The colors matter. Auto means madengine sets it up for you. Manual means the launcher supports it, but you set the numbers in your script or config.

For training, torchtitan and megatron-lm do the most for you. torchrun and deepspeed are lighter, and you do more yourself.

For inference, vLLM handles tensor parallelism and replicas automatically. Disaggregated SGLang gives each role its own setting.

For big mixture-of-experts models like DeepSeek-V3, expert parallelism comes from the model's own scripts, and they run through `slurm_multi`.

*If someone asks about Primus or plain SGLang: they are in the footnote, to keep the table short.*

---

## Slide 10. Hands-on: four rungs (1:15)

Now let me show real MAD models. Think of a ladder with four rungs. It is the same tool on every rung.

Rung one: one GPU in local Docker. For example, a Llama 3.1 8B model in FP8.

Rung two: eight GPUs on one node. For example, the Kimi model, or a Primus training model.

Rung three: two nodes, on SLURM, Spur or Kubernetes. For example, a KV-cache transfer benchmark, or a disaggregated SGLang model.

Rung four: many nodes, and also llm-d. For example, DeepSeek-V3 disaggregated, or DeepSeek-R1 behind llm-d.

Now look at the table. It shows what you add for each target, and how the ranks start. Local Docker needs nothing. SLURM needs a `slurm` block, and ranks come from SLURM variables. Kubernetes needs a `k8s` block, and it uses an Indexed Job. llm-d runs an HTTP client against a gateway.

Whatever the target, you get three things: validation before submit, the same `perf.csv`, and cleanup of only what madengine created. The long deck has ten worked examples with exact commands.

---

## Slide 11. Cookbook and next steps (1:00)

This last slide answers one question: "I want to change something. What do I add?"

Let me pick three rows. Want a different base image? Add `docker_build_arg`, and you do not edit the Dockerfile. The example on the right shows a newer vLLM base image.

Want to set environment variables? Use `docker_env_vars` for local Docker, and `env_vars` on a cluster.

Want to profile a run? Add a `tools` list. You can stack tools. But a warning: pick one GPU profiler, and keep a baseline run with no tools. Do not mix `rocprof` tools with `rocm_trace_lite`.

A tip: for anything with nesting or true and false values, use a JSON file instead of an inline string.

To finish, the bottom of the slide lists the docs to read: usage, configuration, launchers, deployment, llm-d and profiling. Also look at the `examples` folder. The full deck has more detail.

That is all from me. Thank you. I am happy to take questions.

---

## Questions you may get

| Question | Simple answer |
|---|---|
| Do I need to write my own cards? | Only for new models. Existing MAD cards run as they are. |
| Does it need Docker everywhere? | Local and Kubernetes runs use containers. Bare metal uses a conda environment instead. |
| Why not write my own sbatch script? | madengine writes it for you. It also sets the rank variables, collects results and cleans up. You only give a few keys. |
| What happens when a run fails? | The failure is written to `perf.csv`. The exit code tells CI what went wrong: 2 build, 3 run, 4 bad arguments, 5 GPU stuck. |
| How does it know where to run? | It reads your config keys in a fixed order: bare metal, llm-d, Kubernetes, Spur or SLURM, and then local Docker. The first match wins. |
| What is new in 3.0? | In this deck, only `vllm-disagg` is marked as 3.0. For anything else, check the release notes before you answer. |
