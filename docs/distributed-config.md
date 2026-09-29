# Configuring distributed inference

Configuration for these workloads is split across five places, owned by four
teams. They are not five ways of doing one thing, and the numbering they used to
carry implied a competition that does not exist. What separates them is **scope**:

| scope | who decides it | where it lives |
|-------|----------------|----------------|
| **the run** | whoever launches this job | `madengine --config` (Hydra groups: `scheduler`, `launcher`, `+profile`, `+env`, `+tools`) |
| **the site** | whoever provisioned the cluster | `cluster.sh`, and Hydra `+profile` / `+env` |
| **the model** | whoever tuned it on that hardware | `scripts/<dir>/models.yaml`, `mad-config.yaml` |
| **the measurement** | whoever defines the benchmark | `scripts/<dir>/configs/*.yaml`, `mad-config.yaml` |

Read it as a question of lifetime. A run-level value is chosen fresh each time
someone launches; a site value is true of a cluster until it is re-provisioned; a
model value travels with the model wherever it runs. Putting a value in the wrong
scope is how it ends up correct in one place and stale in another.

Whatever the scope, it ends up as **environment variables** reaching the workload,
because that is the only thing that survives both the container boundary and the
choice of launcher.

## How the run level composes with the rest

`madengine --config` translates Hydra YAML into exactly the `additional_context`
that `--additional-context` produces -- it is a front end, not a separate
mechanism. So a run-level value lands at the TOP of the precedence chain:

```
model                    from mad-config.yaml
benchmark                from mad-config.yaml, by kind
model_info.env_vars      the card's own env_vars
additional_context       -e at submit time, AND everything --config composes  (highest)
```

That is the right way round: `--config +env=nccl_debug` should beat a model
default, and a model default should beat nothing at all. The two systems compose
without either knowing about the other.

One caveat worth knowing before switching a pipeline to `--config`: a plain user
YAML translates to a byte-identical context, but composing the same thing from a
config GROUP does not -- the group brings its own defaults. `scheduler=slurm`
adds `OMP_NUM_THREADS` and `MIOPEN_FIND_MODE` to `env_vars`, among others. A
benchmark whose environment changed because of how its config was spelled is
exactly what this document exists to prevent, so a pipeline that cares about
reproducing a number should pass a file rather than compose from groups.

---

## Any one of them is enough

These formats are **alternatives, not a stack**. They do not have to be active at
the same time, and none of them is required:

- a team with only a `cluster.sh` gets a working run;
- a team with only a `models.yaml` gets a working run;
- a team with only a `mad-config.yaml` gets a working run;
- a team with none of them gets a working run, because those formats are read inside
  the container by scripts madengine never parses.

This is a rule about madengine, not about the formats: **nothing madengine adds on
its own may become something that has to exist.** An optional input that is absent
is skipped with a line saying so — never a failure.

It is worth stating because it was once violated in the one place hardest to see.
`gather_system_env_details` appended a diagnostic script nobody had asked for,
named by a path relative to the working directory. When a model repo carried no
copy, the `cp` failed and ended the run — after the image was built and the
container was up, ten minutes into a two-node job. The fix was not to make the
file mandatory but to make its absence survivable.

`tests/unit/test_layered_config.py` asserts both halves: each format resolving
alone, and each optional input missing without consequence.

The one thing that IS refused rather than skipped is a `site:` block in
`mad-config.yaml`, and the distinction matters. Absence is a legitimate state, so
it is survivable. A `site:` block is a setting its author expects to take effect,
so ignoring it would leave them with a configuration that reads correctly and does
nothing &mdash; the failure mode this rule exists to prevent, arrived at from the
other direction.

---

## The three formats madengine does not parse

**It routes, validates and documents. It does not parse them.**

`models.yaml`, `configs/*.yaml` and `cluster.sh` are read by the scripts that own
them, inside the container — madengine never interprets those schemas. That keeps
each format owned by its team, and means a team changing its own file cannot break
madengine.

What madengine does do is make sure the pointers reach the workload, and warn about
configurations that are genuinely ambiguous.

### The `args` trap

`args` on a model card means **two different things** depending on who runs it:

| consumer | treats `args` as |
|---|---|
| madengine (`deployment/slurm.py`) | arguments to the model script — `bash <script> <args>` |
| a plain `sbatch` submission of the card script | arguments to `sbatch` |

So a distributed card with `args: "-N 2 -n 2"` hands those flags to the batch script
under madengine, and to `sbatch` when the card script is submitted with plain sbatch. madengine warns when it sees this:

```
⚠ pyt_large_ep_bench_2n: args='-N 2 -n 2' contains sbatch flag(s) ['-N', '-n'].
  madengine passes args to the model script, not to sbatch, so these reach the
  script as positional arguments. Node count for a distributed card comes from
  distributed.nnodes / slurm.nodes.
```

For a distributed card, put the node count in `distributed.nnodes` and `slurm.nodes`,
which both consumers read the same way. This is also why `mad-config.yaml` is pointed at by an
environment variable rather than by `args` — adding a third meaning to that field
would make the ambiguity worse.

---

## `mad-config.yaml`: a model's recipe and its measurement

The other three formats are owned by the teams that authored them and are read
inside the container by their own scripts. This one is madengine's, and it exists
for a narrower job than it originally had: **the two things that travel with a
model and that nothing else can express.**

- **`model:`** &mdash; how this model is served on this hardware, including the
  serve flags per role &times; mode. `configs/*.yaml` has no concept of a role,
  and a Hydra group has no concept of a model.
- **`benchmark:`** &mdash; what is measured, optionally per benchmark kind.

It stops there. Anything true of the *cluster* rather than the *model* belongs to
the run, and madengine composes that from Hydra groups or `cluster.sh`. A model
card is the wrong place to describe a machine: the value outlives the wrong thing
and goes stale where nobody is looking.

```yaml
version: 1

model:                      # decided by: whoever tuned this model
  id: moonshotai/Kimi-K3    # how configs/*.yaml names it
  local_name: Kimi-K3       # the on-disk directory, how models.yaml and cluster.sh name it
  env:
    REQUIRE_LOCAL_WEIGHTS: '1'   # a fact about the MODEL: 1.5T will not load over NFS in time
    TP_SIZE: '8'
    PP_SIZE: '2'
  serve:
    base: "--attention-backend aiter"
    modes:
      tp: "--tensor-parallel-size 8"
    roles:
      prefill:
        tp: "--disable-cuda-graph"
      decode:
        tp: "--disable-radix-cache --cuda-graph-bs 8 16 32 64 128 256 512"

benchmark:                  # decided by: whoever defines the measurement
  - env:                    # no 'kind' -> applies to every benchmark
      SEEDS: '3'
  - kind: niah
    env:
      NIAH_WORDS: '10000,50000,100000,200000'
```

### What is no longer in this file

Earlier versions carried a `site:` section. Hydra's `+profile` and `+env` groups
cover the same ground at the right scope &mdash; chosen per run, by whoever
launches &mdash; so it is gone, and a file that still has one is **refused with an
error naming where each kind of key should go**. It is not ignored. A setting that
looks applied and is not is the failure this whole document is about, and
silently dropping one would be an unusually poor way to make that point.

Where a `site:` key goes depends on what it actually described:

| it was really... | it now lives in | example |
|---|---|---|
| a property of the **model** | `model.env` here | `REQUIRE_LOCAL_WEIGHTS` &mdash; Kimi-K3 is 1.5T |
| a property of the **cluster** | `cluster.sh`, as `${VAR:-default}` | `NVME_ROOT`, `KV_IB_DEVICE`, the fabric archetype |
| a property of **this run** | a Hydra `+profile` / `+env` group | `NCCL_DEBUG`, `GPU_MAX_HW_QUEUES` |

The split is usually obvious once the question is asked, and where it is not, the
value was doing two jobs and should be two values.

### Where it is found

A sibling `mad-config.yaml` next to the card's script directory, or an explicit
path in `env_vars.MAD_CONFIG`. (The sibling-by-convention rule mirrors how the
accuracy work locates a sibling `acc.yaml` next to `--config`.)

No file means nothing changes. The other formats are the common case and are
untouched &mdash; this one is opt-in, per card.

Naming a file something other than `mad-config.yaml` and pointing at it with
`MAD_CONFIG` is deliberate and useful: the default name is picked up by
convention for **every** card in that directory, so a file meant for one card must
not use it. `scripts/vllm_multinode/mad-config.kimi-k3.yaml` is there for exactly
that reason.

### Precedence

Lowest first:

```
model                                   from mad-config.yaml
benchmark  (task-level, then by kind)   from mad-config.yaml
model_info.env_vars                     the card's own env_vars
additional_context.env_vars             -e, and everything --config composes (highest)
```

Layers **merge**, they do not replace: a key set only in `model` survives a
`benchmark` section that does not mention it.

The last entry is where Hydra lands, so a run-level `--config +env=...` outranks
anything a card declares. That ordering is deliberate: an operator pinning
something at submit time should win, which is the property every one of these
formats already relies on.

### Serve flags are a map, not a string

Within `serve`, a flag string is parsed into a map so a later layer can override a
*single* flag:

```yaml
base: "--attention-backend aiter --tp 1"    # -> {--attention-backend: aiter, --tp: '1'}
modes: {tp: "--tp 8"}                       # -> --tp becomes '8', aiter survives
```

This is the one thing `mad-config.yaml` does that `models.yaml` cannot. `models.yaml` stores flags as an
opaque string, so overriding one flag means string surgery — its own comments note
that the moriio path "strips any duplicate from the yaml `tp:` string". A map removes
the need for that.

Flags taking several values are kept whole: `--cuda-graph-bs 8 16 32` stays
`'8 16 32'`, not three separate tokens.

### Why adopting it is safe

`mad-config.yaml` resolves into **exactly the `env_vars` a card already carries**. It is a front
end, not a migration: no workload script changes, and no launcher needs to know the
file exists. `tests/unit/test_layered_config.py` asserts this directly — a way-4 file
and the equivalent hand-written `env_vars` block must resolve to identical
environments.

### Scope

`mad-config.yaml` is resolved by madengine, so it is available on the madengine path. Submitting
the card script with plain sbatch deliberately runs without installing madengine, so cards
run that way should use the other three formats.

---

---

## The adapter contract

madengine and a workload script speak different vocabularies. A templated
launcher exports madengine's names; the proven launchers read their own. Some
workload has to translate, and today each one hand-rolls its own shim --
`scripts/sglang_disagg/run.sh` on the mad-rccl branch is one.

What a templated launcher exports, and what the disagg launchers read:

| madengine exports | workload reads | meaning |
|---|---|---|
| `SGLANG_NODE_RANK` | `NODE_RANK` | this node's global 0-based rank |
| `SGLANG_DISAGG_PREFILL_NODES` | `xP` | prefill node count |
| `SGLANG_DISAGG_DECODE_NODES` | `yD` | decode node count |
| `SGLANG_NODE_IPS` | `IPADDRS` | rank-ordered node IPs, comma separated |
| `SGLANG_TP_SIZE` | `TP_SIZE` | tensor-parallel degree |
| `MASTER_PORT` | `MASTER_PORT` | already shared |

Two rules make an adapter safe:

**Environment wins.** Read every value as `${VAR:-<fallback>}` so a submit-time
`-e` or a card's `env_vars` overrides the launcher. This is the same rule
`cluster.sh` and the `mad.env` templates already follow, and it is what lets one
card run under either path.

**Do not require the adapter for topology.** The self-managed path now exports
`MAD_NODE_IPS`, `IPADDRS`, `SGLANG_NODE_IPS` and `MAD_NODE_RANK` directly from
the allocation, so a script does not have to rediscover them. Before that, the
only way to get the rank-ordered IP list was to rebuild it in-container --
`scripts/sglang_disagg/ip_rendezvous.py` does exactly that, with a stdlib TCP
rendezvous and a 1800s budget. That is now a fallback for what the scheduler
cannot answer, not the normal path.

### Where a value has to cross three boundaries

A manifest-driven run passes through three places, and a value has to survive
all of them to reach the workload:

| declared in | crosses | reaches |
|---|---|---|
| `deployment_config.env_vars` | sbatch | the SLURM job |
| `context.docker_env_vars` | `docker run -e` | the container |
| `mad.env` | the shell | the submitting process |

These used to require declaring the same variable in more than one of them.
`context.docker_env_vars` is now also applied on the SLURM path, so one
declaration is enough. That matters because a variable written twice drifts, and
a wrong NIC list does not fail: RCCL initialises zero NICs, falls back to TCP,
and the run still reports a number measured over the wrong transport.

## Choosing

Pick by **how long the value should outlive the run**:

- **Chosen fresh for this launch** (which scheduler, which profile, debug env)
  &rarr; `madengine --config`. Nothing else is re-decided per run.
- **True of the cluster until it is re-provisioned** (weight roots, NIC list,
  fabric archetype) &rarr; `cluster.sh`, or a Hydra `+profile` / `+env` group.
  Every value `${VAR:-default}` so the environment still wins.
- **Travels with the model wherever it runs** (serve flags per role &times; mode)
  &rarr; `models.yaml`; it owns the role &times; mode axis nothing else has.
- **Defines the measurement** (benchmark kind, knobs, `extra_args`)
  &rarr; `configs/*.yaml`.
- **Wanting a model's serve recipe and its measurement in one reviewable file, on
  the madengine path** &rarr; `mad-config.yaml`.

If a value seems to belong in two of these, it is usually two different values
that happen to share a name today -- and they will drift.

Related: [configuration.md](configuration.md), [launchers.md](launchers.md).
