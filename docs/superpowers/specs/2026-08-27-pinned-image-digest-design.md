# Design: pin registry pulls to the digest recorded at build time

## Problem

A client-perf-hub accuracy run failed because the image pulled at run time
did not match the image pushed at build time:

- build pushed `sha256:df36ef7e...`
- the CI runner later pulled `sha256:cbb7e5ed...`

Both shared the same registry tag. A concurrent build of the same model
pushed a newer image to that tag between the two events, so the run phase
silently executed a different image than the one the build phase produced.

It was assumed madengine already guards against this ("we use it to confirm
if the pulled image is identical to the one in the build manifest"). It does
not. Tracing the code:

- `build_info["docker_sha"]` (`docker_builder.py:303-308`) is the digest of
  the Dockerfile's `FROM` (base) image, not the image madengine builds and
  pushes. It is consumed only as a reporting column (`update_perf_csv.py`,
  `k8s_results.py`, `slurm.py`).
- `push_image()` (`docker_builder.py:395`) discards `docker push` stdout,
  which is where the pushed digest (`digest: sha256:...`) is printed. It is
  never captured anywhere.
- The run phase pulls by tag only (`container_runner.py:2854` →
  `pull_image()` at `container_runner.py:572`, a plain `docker pull <tag>`)
  and performs no comparison against anything in the manifest. The only
  image-identity checks in the codebase (`_local_image_id`,
  `BUILD_FINGERPRINT_LABEL`, `container_runner.py:2464-2503`) exist solely
  for cross-node consistency in multi-node SLURM *local-image* mode; they
  never touch registry digests.

This design adds the missing capability: capture the real pushed digest at
build time, and optionally enforce it at run time.

## Non-goals

- This does not prevent the tag race itself. Two builds pushing the same
  mutable tag (`f"{registry}:{model_name}"`, `build_orchestrator.py:1055`)
  will still clobber each other; the loser under the new flag fails fast
  with a clear error instead of silently running the wrong image. Actually
  eliminating the race (e.g. unique tags per build) is a client-perf-hub /
  upstream CI change, out of scope here.
- This does not rename or repurpose `build_info["docker_sha"]`. It is wired
  into four existing reporting sinks and column orders; changing its
  meaning is a separate, riskier change and is called out only as a
  possible future cleanup.
- This does not change default pull behavior for any existing user. See
  "Opt-in enforcement" below.

## Design

### 1. Capture the pushed digest at build time (always on, default mode included)

In `DockerBuilder.push_image()` (`docker_builder.py`), after `docker push`
succeeds, parse the digest from its output (`digest: sha256:...`, the same
line format already parsed for base-image SHA in `docker_builder.py:305`).

If the push output doesn't contain a parseable digest line (registry output
format variance, e.g. some mirrors), fall back to:
```
docker image inspect --format '{{index .RepoDigests 0}}' <registry_image>
```
and extract the digest from that.

If both fail to produce a digest, log a **debug/dim-level** note (not a
user-facing warning) and continue — this is a manifest-completeness gap to
leave a trail for later, not a failure. Push already succeeded; we don't
block on this.

Store the result as a new field: `build_info["image_digest"]` (e.g.
`"sha256:df36ef7e..."`), separate from `docker_sha`. This flows into
`build_manifest.json` through the existing `built_images` /
`export_build_manifest()` path with no other changes needed.

This capture step runs unconditionally — no flag gates it. It's the
recording half of the fix, and it's inert (never read by pull logic) unless
strict mode is on.

### 2. Opt-in enforcement at run time

New flag: `--require-pinned-image` on the `run` command, mirrored as an
`additional_context` key (e.g. `"require_pinned_image": true`) so CI
pipelines that drive madengine through `additional_context` rather than raw
CLI flags can set it the same way as other behavior-affecting keys
(`k8s`, `slurm`, `tools`, ...).

**Default (flag absent):** no behavior change whatsoever. Pull by tag,
exactly as today. `image_digest` rides along in the manifest unused.

**With the flag set**, for every `build_info` entry with a
`registry_image`:
- If `image_digest` is present, construct a pinned reference
  `repo@sha256:...` (stripping any existing tag) and pull that image
  instead of the tag. The registry itself now enforces identity: a moved
  tag surfaces as a normal "manifest unknown" pull failure rather than a
  silent wrong-image success.
- If `image_digest` is absent (older manifest, or capture failed at build
  time), fail immediately with a clear error naming the model/image and
  explaining that the manifest has no recorded digest — do not fall back to
  pulling by tag. Silent degradation defeats the purpose of asking for the
  guarantee.

One shared helper (e.g. `build_pinned_reference(registry_image, digest)`)
builds the `repo@sha256:...` string, used identically by all three
enforcement call sites so they can't drift:

| Path | Location | Change under the flag |
|---|---|---|
| Local Docker | `container_runner.py:2854` | `pull_image(pinned_ref)` |
| Kubernetes | `k8s_template_context.py:518` (`"image": image_info["registry_image"]`) | `"image": pinned_ref` |
| SLURM | `slurm.py:544` (parallel `srun` pull) | pinned ref substituted into the generated pull command |

### 3. Compatibility check: log filenames

`container_runner_helpers.py:256` already strips `@sha256:...` before
deriving log/tar filenames from an image reference (`ref_without_digest =
s.split("@", 1)[0]`). Pinned references flow through this unchanged — no
new filename collisions. Confirmed by reading the function; will be
covered by a test regardless.

## Testing plan

TDD; each behavior below is a separate test:

1. `docker push` output containing a `digest: sha256:...` line →
   `build_info["image_digest"]` set to that value.
2. `docker push` output without a parseable digest line → falls back to
   `docker image inspect --format '{{index .RepoDigests 0}}'`.
3. Both parse paths fail → `image_digest` absent, a debug-level log line is
   emitted, and the build/push otherwise succeeds unaffected.
4. Flag absent (default) → run pulls by tag regardless of whether
   `image_digest` is present in the manifest; no new log output.
5. Flag present, `image_digest` present → the pull command (or k8s image
   field / SLURM pull line) contains `repo@sha256:...`.
6. Flag present, `image_digest` absent → run fails immediately with a clear
   error, before any pull is attempted.
7. K8s pod spec and SLURM generated script both carry the pinned reference
   when the flag is set, and the untouched tag reference when it isn't.
8. Log/tar filename derivation given a pinned (`@sha256:...`) reference
   produces the same filename as today's digest-free reference.
9. Existing tests touching `docker_sha` / `build_manifest.json` structure
   continue to pass unmodified (the new field is additive).

## Rollout note (for the reply to Tej/Rahul)

- The verification described in the thread ("we use it to confirm...")
  does not currently exist in the code; this design builds it, gated
  behind `--require-pinned-image` / `require_pinned_image` context key.
- Turning the flag on stops the *symptom* (silently running the wrong
  image) by failing fast instead. It does not stop the *cause* (two builds
  racing to push the same mutable tag) — that needs a change on the
  client-perf-hub / CI side (e.g. unique tags per build).
- client-perf-hub must explicitly opt in for this guarantee to apply to its
  runs; it is not automatic on upgrade.
