# Pinned Image Digest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record the real digest of every image madengine pushes, and let users opt into pinning run-time pulls to that digest so a moved registry tag fails loudly instead of silently running the wrong image.

**Architecture:** A new leaf module `madengine/core/image_digest.py` holds four small pure functions (two parsers, one reference builder, one policy resolver that raises when enforcement is on but a digest is missing). `DockerBuilder.push_image()` records digests into a `self.pushed_digests` dict — mirroring the existing `self.built_images` / `self.built_models` pattern — and the two push call sites copy the digest into `build_info["image_digest"]`, which rides into `build_manifest.json` unchanged. At run time a `--require-pinned-image` flag (mirrored as the `require_pinned_image` additional-context key) flows through `RunOrchestrator.additional_context` into all three execution paths, each calling the same `resolve_pinned_image()` helper.

**Tech Stack:** Python 3, Typer (CLI), pytest + `unittest.mock`, Jinja2 (SLURM/K8s templates), Docker CLI.

---

## Background you need before starting

Read the design spec first: `docs/superpowers/specs/2026-08-27-pinned-image-digest-design.md`.

Facts about this codebase that the tasks below depend on:

- `Console.sh(command)` (`src/madengine/core/console.py:138`) runs a shell command and **returns its stdout as a stripped string**, raising `RuntimeError` on non-zero exit unless `canFail=True`. This is how we capture `docker push` output.
- `build_info` dicts are created in `DockerBuilder.build_image()` (`src/madengine/execution/docker_builder.py:311-322`) and serialized wholesale into `build_manifest.json` under `built_images` by `export_build_manifest()`. **Any new key added to `build_info` appears in the manifest automatically** — no serializer changes needed.
- There are exactly two places that call `push_image` and set `build_info["registry_image"]`: `docker_builder.py:769` (single-arch) and `docker_builder.py:977` (per-GPU-arch). Both must record the digest.
- `ContainerRunner.run_models_from_manifest()` merges the manifest's `context` dict over its own `additional_context` (`src/madengine/execution/container_runner.py:2799-2800`). `RunOrchestrator._load_and_merge_manifest()` writes selected runtime keys into `manifest["context"]` (`src/madengine/orchestration/run_orchestrator.py:499-503`). Adding our key to that merge list is what makes the flag survive into the nested `madengine run` that the standard SLURM job script executes on each compute node.
- `_docker_image_ref_for_log_naming()` (`src/madengine/execution/container_runner_helpers.py:232`) already strips `@sha256:...`, so pinned references produce the same log filenames as tags. Task 9 locks that in with a test.
- Test style in this repo: `pytest` classes named `TestX` with plain `assert`, `MagicMock` for `Context`/`Console`, `tmp_path` for manifests. Follow the surrounding file's style in each test file you touch.

### Deliberate deviations from the spec (do not "fix" these)

1. **SLURM enforcement point.** The spec's table says the pinned reference goes into the generated `srun docker pull` line in `slurm.py:544`. Implementing only that would pull the pinned image and then `docker run` the *tag* — the exact race the feature exists to close. Instead we set `env_vars["DOCKER_IMAGE_NAME"]` to the pinned reference (Task 8); the pull line interpolates that same variable, so both pull and run are pinned by one change.
2. **Digest capture is not added to the build-on-compute-node path** (`build_orchestrator._execute_build_on_compute` at `build_orchestrator.py:748`, which pushes from a generated bash script inside an sbatch job and writes `built_images` entries at `:1224`/`:1239` with no `registry_image` and no digest). Manifests from that path have no `image_digest`, so `--require-pinned-image` runs against them fail fast with the Task 2 error. This is documented in Task 10, not implemented.

---

## File Structure

**Create:**

| File | Responsibility |
|---|---|
| `src/madengine/core/image_digest.py` | Pure helpers: parse a digest out of `docker push` / `docker image inspect` output, build a `repo@sha256:...` reference, and apply the enforcement policy (return-as-is / pin / raise). No I/O, no Docker calls. |
| `tests/unit/test_image_digest.py` | Unit tests for all four functions in the module above. |

**Modify:**

| File | Change |
|---|---|
| `src/madengine/execution/docker_builder.py` | Capture the pushed digest in `push_image()`; copy it into `build_info["image_digest"]` at both push call sites. |
| `src/madengine/cli/commands/run.py` | Add the `--require-pinned-image` flag; pass it into both `create_args_namespace(...)` calls. |
| `src/madengine/orchestration/run_orchestrator.py` | Fold the flag into `additional_context`; persist it into `manifest["context"]`. |
| `src/madengine/execution/container_runner.py` | Resolve the pinned reference before the registry pull. |
| `src/madengine/deployment/k8s_template_context.py` | Resolve the pinned reference for the pod spec `image` field. |
| `src/madengine/deployment/slurm.py` | Resolve the pinned reference for `DOCKER_IMAGE_NAME` in the slurm_multi wrapper. |
| `docs/cli-reference.md`, `docs/configuration.md` | Document the flag and the context key. |

**Test files touched:** `tests/unit/test_image_digest.py` (new), `tests/unit/test_docker_builder.py`, `tests/unit/test_orchestration.py`, `tests/unit/test_container_runner.py`, `tests/unit/test_k8s.py`, `tests/unit/test_slurm_multi.py`, `tests/unit/test_execution.py`.

---

## Task 1: Digest parsing and pinned-reference construction

**Files:**
- Create: `src/madengine/core/image_digest.py`
- Test: `tests/unit/test_image_digest.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_image_digest.py`:

```python
"""Unit tests for madengine.core.image_digest.

Covers digest extraction from `docker push` / `docker image inspect` output and
construction of pinned `repo@sha256:...` references.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import pytest

from madengine.core.image_digest import (
    build_pinned_reference,
    parse_push_digest,
    parse_repo_digest,
)


DIGEST = "sha256:" + "df36ef7e" * 8  # 64 hex chars
OTHER_DIGEST = "sha256:" + "cbb7e5ed" * 8


class TestParsePushDigest:
    """parse_push_digest extracts the digest line emitted by `docker push`."""

    def test_typical_push_output(self):
        output = (
            "The push refers to repository [docker.io/myorg/ci]\n"
            "a1b2c3d4e5f6: Pushed\n"
            "9f8e7d6c5b4a: Layer already exists\n"
            f"mymodel: digest: {DIGEST} size: 4738\n"
        )
        assert parse_push_digest(output) == DIGEST

    def test_no_space_after_colon(self):
        assert parse_push_digest(f"mymodel: digest:{DIGEST} size: 12") == DIGEST

    def test_last_digest_wins_when_multiple(self):
        output = (
            f"tag-a: digest: {OTHER_DIGEST} size: 10\n"
            f"tag-b: digest: {DIGEST} size: 10\n"
        )
        assert parse_push_digest(output) == DIGEST

    def test_uppercase_hex_is_not_matched(self):
        assert parse_push_digest("mymodel: digest: sha256:ABC size: 1") is None

    def test_output_without_digest_line(self):
        assert parse_push_digest("The push refers to repository [x]\nlayer: Pushed\n") is None

    def test_empty_output(self):
        assert parse_push_digest("") is None

    def test_none_output(self):
        assert parse_push_digest(None) is None


class TestParseRepoDigest:
    """parse_repo_digest extracts the digest from a `repo@sha256:...` reference."""

    def test_repo_digest_reference(self):
        assert parse_repo_digest(f"myorg/ci@{DIGEST}") == DIGEST

    def test_registry_with_port(self):
        assert parse_repo_digest(f"localhost:5000/myorg/ci@{DIGEST}") == DIGEST

    def test_surrounding_whitespace_and_quotes(self):
        assert parse_repo_digest(f"  'myorg/ci@{DIGEST}'  \n") == DIGEST

    def test_empty_repodigests_placeholder(self):
        # `docker image inspect` prints this when RepoDigests is empty.
        assert parse_repo_digest("<no value>") is None

    def test_none_output(self):
        assert parse_repo_digest(None) is None


class TestBuildPinnedReference:
    """build_pinned_reference produces repo@sha256:... with any tag stripped."""

    def test_strips_tag(self):
        assert build_pinned_reference(f"myorg/ci:mymodel", DIGEST) == f"myorg/ci@{DIGEST}"

    def test_no_tag(self):
        assert build_pinned_reference("myorg/ci", DIGEST) == f"myorg/ci@{DIGEST}"

    def test_registry_port_is_not_mistaken_for_a_tag(self):
        out = build_pinned_reference("localhost:5000/myorg/ci:latest", DIGEST)
        assert out == f"localhost:5000/myorg/ci@{DIGEST}"

    def test_registry_port_without_tag(self):
        out = build_pinned_reference("localhost:5000/myorg/ci", DIGEST)
        assert out == f"localhost:5000/myorg/ci@{DIGEST}"

    def test_bare_name_with_tag(self):
        assert build_pinned_reference("ci-dummy:latest", DIGEST) == f"ci-dummy@{DIGEST}"

    def test_existing_digest_is_replaced(self):
        out = build_pinned_reference(f"myorg/ci@{OTHER_DIGEST}", DIGEST)
        assert out == f"myorg/ci@{DIGEST}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_image_digest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'madengine.core.image_digest'`

- [ ] **Step 3: Write the implementation**

Create `src/madengine/core/image_digest.py`:

```python
#!/usr/bin/env python3
"""
Registry image digest helpers for madengine.

Build pushes record the digest of the image they push; runs can optionally be
pinned to that digest so a registry tag that moved between build and run fails
loudly instead of silently resolving to a different image.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import re
import typing


# `docker push` prints e.g. "mytag: digest: sha256:<64 hex> size: 4738".
_PUSH_DIGEST_RE = re.compile(r"digest:\s*(sha256:[0-9a-f]{64})")

# `docker image inspect --format '{{index .RepoDigests 0}}'` prints "repo@sha256:<64 hex>".
_REPO_DIGEST_RE = re.compile(r"@(sha256:[0-9a-f]{64})")


def parse_push_digest(push_output: typing.Optional[str]) -> typing.Optional[str]:
    """Extract the pushed image digest from `docker push` output.

    Args:
        push_output: Combined stdout/stderr of the push command.

    Returns:
        The digest (``sha256:...``), or None if no digest line was present.
        When several digest lines appear the last one is returned, which is the
        digest of the reference the push command was invoked with.
    """
    if not push_output:
        return None
    matches = _PUSH_DIGEST_RE.findall(push_output)
    return matches[-1] if matches else None


def parse_repo_digest(inspect_output: typing.Optional[str]) -> typing.Optional[str]:
    """Extract the digest from a ``repo@sha256:...`` reference.

    Args:
        inspect_output: Output of
            ``docker image inspect --format '{{index .RepoDigests 0}}' <image>``.

    Returns:
        The digest (``sha256:...``), or None if the output held no digest
        (e.g. ``<no value>`` when the image has no RepoDigests entry).
    """
    if not inspect_output:
        return None
    match = _REPO_DIGEST_RE.search(inspect_output)
    return match.group(1) if match else None


def build_pinned_reference(registry_image: str, digest: str) -> str:
    """Build a digest-pinned image reference.

    Any existing tag or digest on ``registry_image`` is dropped. A port in the
    registry host (``localhost:5000/org/img``) is not mistaken for a tag because
    only the final path segment is inspected for ``:``.

    Args:
        registry_image: Image reference, with or without tag/digest.
        digest: Digest to pin to (``sha256:...``).

    Returns:
        A reference of the form ``repo@sha256:...``.
    """
    repo = registry_image.split("@", 1)[0]
    last_slash = repo.rfind("/")
    tail = repo[last_slash + 1 :]
    if ":" in tail:
        repo = repo[: last_slash + 1] + tail.split(":", 1)[0]
    return f"{repo}@{digest}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_image_digest.py -v`
Expected: PASS (18 tests)

- [ ] **Step 5: Commit**

```bash
git add src/madengine/core/image_digest.py tests/unit/test_image_digest.py
git commit -m "feat(image-digest): add digest parsing and pinned reference helpers"
```

---

## Task 2: Enforcement policy resolver

**Files:**
- Modify: `src/madengine/core/image_digest.py`
- Test: `tests/unit/test_image_digest.py`

`resolve_pinned_image()` is the single place the "pin, pass through, or fail" decision is made. All three execution paths (local Docker, K8s, SLURM) call it so they cannot drift.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_image_digest.py`:

```python
class TestResolvePinnedImage:
    """resolve_pinned_image applies the --require-pinned-image policy."""

    def test_disabled_returns_image_unchanged_without_digest(self):
        assert resolve_pinned_image("myorg/ci:mymodel", None, False) == "myorg/ci:mymodel"

    def test_disabled_returns_image_unchanged_even_with_digest(self):
        # Default behaviour must not change for existing users: the digest rides
        # along in the manifest but is never used unless enforcement is on.
        assert resolve_pinned_image("myorg/ci:mymodel", DIGEST, False) == "myorg/ci:mymodel"

    def test_enabled_with_digest_returns_pinned_reference(self):
        out = resolve_pinned_image("myorg/ci:mymodel", DIGEST, True)
        assert out == f"myorg/ci@{DIGEST}"

    def test_enabled_without_digest_raises(self):
        with pytest.raises(ConfigurationError):
            resolve_pinned_image("myorg/ci:mymodel", None, True, model_name="my_model")

    def test_enabled_with_empty_digest_raises(self):
        with pytest.raises(ConfigurationError):
            resolve_pinned_image("myorg/ci:mymodel", "", True, model_name="my_model")

    def test_error_message_names_model_and_image(self):
        with pytest.raises(ConfigurationError) as excinfo:
            resolve_pinned_image("myorg/ci:mymodel", None, True, model_name="my_model")
        message = str(excinfo.value)
        assert "my_model" in message
        assert "myorg/ci:mymodel" in message

    def test_error_carries_actionable_suggestions(self):
        with pytest.raises(ConfigurationError) as excinfo:
            resolve_pinned_image("myorg/ci:mymodel", None, True, model_name="my_model")
        assert excinfo.value.suggestions
```

And extend the imports at the top of the file:

```python
from madengine.core.errors import ConfigurationError
from madengine.core.image_digest import (
    build_pinned_reference,
    parse_push_digest,
    parse_repo_digest,
    resolve_pinned_image,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_image_digest.py -k ResolvePinned -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_pinned_image'`

- [ ] **Step 3: Write the implementation**

Add the import near the top of `src/madengine/core/image_digest.py`, below `import typing`:

```python
from madengine.core.errors import ConfigurationError, create_error_context
```

Append to the same file:

```python
def resolve_pinned_image(
    registry_image: str,
    image_digest: typing.Optional[str],
    require_pinned: bool,
    model_name: str = "",
) -> str:
    """Resolve the image reference to use for a run.

    Args:
        registry_image: Tagged registry reference from the build manifest.
        image_digest: Digest recorded at build time, if any.
        require_pinned: True when --require-pinned-image / require_pinned_image is set.
        model_name: Model name, used only to make the error message actionable.

    Returns:
        ``registry_image`` unchanged when enforcement is off, otherwise a
        digest-pinned reference.

    Raises:
        ConfigurationError: When enforcement is on but the manifest recorded no
            digest. Falling back to the tag is deliberately not done: silent
            degradation would defeat the guarantee the caller asked for.
    """
    if not require_pinned:
        return registry_image

    if not image_digest:
        raise ConfigurationError(
            f"--require-pinned-image is set but the build manifest records no "
            f"image digest for model '{model_name}' (image: {registry_image}). "
            f"Refusing to pull by tag.",
            context=create_error_context(
                operation="resolve_pinned_image",
                component="image_digest",
                additional_info={"model": model_name, "image": registry_image},
            ),
            suggestions=[
                "Rebuild with this version of madengine so the push digest is recorded",
                "Drop --require-pinned-image / require_pinned_image to pull by tag",
            ],
        )

    return build_pinned_reference(registry_image, image_digest)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_image_digest.py -v`
Expected: PASS (25 tests)

- [ ] **Step 5: Commit**

```bash
git add src/madengine/core/image_digest.py tests/unit/test_image_digest.py
git commit -m "feat(image-digest): add resolve_pinned_image enforcement policy"
```

---

## Task 3: Capture the pushed digest in `DockerBuilder.push_image()`

**Files:**
- Modify: `src/madengine/execution/docker_builder.py:51`, `:394-403`
- Test: `tests/unit/test_docker_builder.py`

Digest capture is **always on** and never fails a build. If neither the push output nor `docker image inspect` yields a digest, we log a dim note and move on — the push already succeeded.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_docker_builder.py`:

```python
DIGEST = "sha256:" + "df36ef7e" * 8


class TestPushImageRecordsDigest:
    """push_image records the pushed image digest in builder.pushed_digests."""

    def _builder(self, sh_side_effect):
        ctx = MagicMock()
        ctx.ctx = {}
        console = MagicMock()
        console.sh = MagicMock(side_effect=sh_side_effect)
        builder = DockerBuilder(ctx, console)
        builder.rich_console = MagicMock()
        return builder

    def test_digest_parsed_from_push_output(self):
        def sh(command, *args, **kwargs):
            if "docker push" in command:
                return f"mymodel: digest: {DIGEST} size: 4738"
            return ""

        builder = self._builder(sh)
        result = builder.push_image("ci-dummy", "localhost:5000", None, "localhost:5000/ci-dummy")

        assert result == "localhost:5000/ci-dummy"
        assert builder.pushed_digests["localhost:5000/ci-dummy"] == DIGEST

    def test_falls_back_to_image_inspect_when_push_output_has_no_digest(self):
        def sh(command, *args, **kwargs):
            if "docker push" in command:
                return "The push refers to repository [localhost:5000/ci-dummy]\nlayer: Pushed"
            if "docker image inspect" in command:
                return f"localhost:5000/ci-dummy@{DIGEST}"
            return ""

        builder = self._builder(sh)
        builder.push_image("ci-dummy", "localhost:5000", None, "localhost:5000/ci-dummy")

        assert builder.pushed_digests["localhost:5000/ci-dummy"] == DIGEST
        inspect_calls = [
            c for c in builder.console.sh.call_args_list if "docker image inspect" in c.args[0]
        ]
        assert len(inspect_calls) == 1
        assert "RepoDigests" in inspect_calls[0].args[0]

    def test_no_digest_anywhere_leaves_entry_absent_and_push_still_succeeds(self):
        def sh(command, *args, **kwargs):
            if "docker push" in command:
                return "layer: Pushed"
            if "docker image inspect" in command:
                return "<no value>"
            return ""

        builder = self._builder(sh)
        result = builder.push_image("ci-dummy", "localhost:5000", None, "localhost:5000/ci-dummy")

        assert result == "localhost:5000/ci-dummy"
        assert "localhost:5000/ci-dummy" not in builder.pushed_digests
        # The gap is noted at dim level, not as a user-facing warning.
        printed = " ".join(str(c) for c in builder.rich_console.print.call_args_list)
        assert "[dim]" in printed
        assert "no pushed digest" in printed.lower()

    def test_inspect_failure_is_swallowed(self):
        def sh(command, *args, **kwargs):
            if "docker push" in command:
                return "layer: Pushed"
            if "docker image inspect" in command:
                raise RuntimeError("no such image")
            return ""

        builder = self._builder(sh)
        result = builder.push_image("ci-dummy", "localhost:5000", None, "localhost:5000/ci-dummy")

        assert result == "localhost:5000/ci-dummy"
        assert builder.pushed_digests == {}

    def test_no_registry_records_nothing(self):
        builder = self._builder(lambda command, *a, **k: "")
        result = builder.push_image("ci-dummy")

        assert result == "ci-dummy"
        assert builder.pushed_digests == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_docker_builder.py -k PushImageRecordsDigest -v`
Expected: FAIL — `AttributeError: 'DockerBuilder' object has no attribute 'pushed_digests'`

- [ ] **Step 3: Write the implementation**

Add the import to `src/madengine/execution/docker_builder.py`, after the `from madengine.core.context import Context` line:

```python
from madengine.core.image_digest import parse_push_digest, parse_repo_digest
```

In `DockerBuilder.__init__`, immediately after `self.built_images = {}  # Track built images` (line 51), add:

```python
        self.pushed_digests = {}  # registry_image -> digest recorded at push time
```

Replace the push block in `push_image()` (lines 394-399) — from `# Push the image` through `self.console.sh(push_command)` — with:

```python
            # Push the image
            push_command = f"docker push {shlex.quote(registry_image)}"
            self.rich_console.print(f"\n[bold blue]🚀 Starting docker push to registry...[/bold blue]")
            print(f"📤 Registry: {registry}")
            print(f"🏷️  Image: {registry_image}")
            push_output = self.console.sh(push_command)

            self._record_pushed_digest(registry_image, push_output)
```

Add the new method immediately after `push_image()` (i.e. after its `raise` at line 408, before `def export_build_manifest`):

```python
    def _record_pushed_digest(self, registry_image: str, push_output: str) -> None:
        """Record the digest of the image just pushed, for digest-pinned runs.

        Best-effort by design: the push has already succeeded by the time this
        runs, so a missing digest is a manifest-completeness gap (noted at dim
        level), never a build failure. Runs only consult the recorded digest
        when --require-pinned-image is set.

        Args:
            registry_image: The reference that was pushed.
            push_output: stdout/stderr captured from the push command.
        """
        digest = parse_push_digest(push_output)

        if not digest:
            # Some registries/mirrors do not print the digest line; ask the
            # daemon for the RepoDigests entry it recorded for this push.
            try:
                inspect_output = self.console.sh(
                    "docker image inspect --format '{{index .RepoDigests 0}}' "
                    + shlex.quote(registry_image)
                )
                digest = parse_repo_digest(inspect_output)
            except Exception:
                digest = None

        if not digest:
            self.rich_console.print(
                f"[dim]No pushed digest recorded for {registry_image}; "
                f"--require-pinned-image runs will reject this manifest entry[/dim]"
            )
            return

        self.pushed_digests[registry_image] = digest
        self.rich_console.print(f"[dim]Pushed digest: {registry_image} -> {digest}[/dim]")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_docker_builder.py -v`
Expected: PASS (all, including the 2 pre-existing naming tests)

- [ ] **Step 5: Verify no existing push tests regressed**

Run: `pytest tests/integration/test_docker_integration.py -k push_image -v`
Expected: PASS (6 tests). These mock `Console.sh` with `return_value="Success"`, so the digest parse returns None, the inspect fallback also returns `"Success"` (no digest), and `push_image` still returns the tag unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/madengine/execution/docker_builder.py tests/unit/test_docker_builder.py
git commit -m "feat(build): capture pushed image digest during docker push"
```

---

## Task 4: Write `image_digest` into `build_info` at both push sites

**Files:**
- Modify: `src/madengine/execution/docker_builder.py:762-772`, `:971-980`
- Test: `tests/unit/test_docker_builder.py`

`build_info` is serialized wholesale into `build_manifest.json`, so setting the key here is all that is needed to get it into the manifest.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_docker_builder.py`:

```python
class TestBuildInfoCarriesImageDigest:
    """Both push call sites copy the recorded digest into build_info."""

    def _builder(self):
        ctx = MagicMock()
        ctx.ctx = {}
        builder = DockerBuilder(ctx, MagicMock())
        builder.rich_console = MagicMock()
        return builder

    def _run_single_arch(self, builder):
        """Drive _build_model_single_arch with everything below push_image stubbed."""
        return builder._build_model_single_arch(
            model_info={"name": "dummy", "dockerfile": "docker/dummy"},
            credentials={},
            clean_cache=False,
            registry="localhost:5000",
            phase_suffix="",
            batch_build_metadata=None,
        )

    def test_single_arch_push_sets_image_digest(self):
        builder = self._builder()

        def fake_push(docker_image, registry, credentials, explicit_registry_image):
            builder.pushed_digests[explicit_registry_image] = DIGEST
            return explicit_registry_image

        with patch.object(
            builder, "_get_dockerfiles_for_model", return_value=["docker/dummy.ubuntu"]
        ), patch.object(
            builder, "build_image", return_value={"docker_image": "ci-dummy", "model": "dummy"}
        ), patch.object(
            builder, "_get_effective_gpu_architecture", return_value=""
        ), patch.object(
            builder, "_create_registry_image_name", return_value="localhost:5000/ci-dummy"
        ), patch.object(
            builder, "push_image", side_effect=fake_push
        ):
            results = self._run_single_arch(builder)

        assert results[0]["registry_image"] == "localhost:5000/ci-dummy"
        assert results[0]["image_digest"] == DIGEST
        # A push failure must still be recorded the way it is today.
        assert "push_error" not in results[0]

    def test_single_arch_push_without_digest_omits_key(self):
        builder = self._builder()

        with patch.object(
            builder, "_get_dockerfiles_for_model", return_value=["docker/dummy.ubuntu"]
        ), patch.object(
            builder, "build_image", return_value={"docker_image": "ci-dummy", "model": "dummy"}
        ), patch.object(
            builder, "_get_effective_gpu_architecture", return_value=""
        ), patch.object(
            builder, "_create_registry_image_name", return_value="localhost:5000/ci-dummy"
        ), patch.object(
            builder, "push_image", return_value="localhost:5000/ci-dummy"
        ):
            results = self._run_single_arch(builder)

        assert results[0]["registry_image"] == "localhost:5000/ci-dummy"
        assert "image_digest" not in results[0]
```

Extend the import at the top of `tests/unit/test_docker_builder.py`:

```python
from unittest.mock import MagicMock, patch
```

> `_build_model_single_arch(self, model_info, credentials, clean_cache, registry, phase_suffix, batch_build_metadata)` is at `docker_builder.py:729`; the per-arch sibling is `_build_model_for_arch`, which takes an extra `arch` argument. Both are reached from `build_all_models` (`docker_builder.py:581`, `:617`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_docker_builder.py -k BuildInfoCarriesImageDigest -v`
Expected: FAIL — `KeyError: 'image_digest'`

- [ ] **Step 3: Write the implementation**

At the single-arch site (`docker_builder.py:769-770`), replace:

```python
                    self.push_image(build_info["docker_image"], registry, credentials, registry_image)
                    build_info["registry_image"] = registry_image
```

with:

```python
                    self.push_image(build_info["docker_image"], registry, credentials, registry_image)
                    build_info["registry_image"] = registry_image
                    # Recorded at push time; consumed only by --require-pinned-image runs.
                    pushed_digest = self.pushed_digests.get(registry_image)
                    if pushed_digest:
                        build_info["image_digest"] = pushed_digest
```

At the per-arch site (`docker_builder.py:977-978`), replace:

```python
                    self.push_image(arch_image_name, registry, credentials, registry_image)
                    build_info["registry_image"] = registry_image
```

with:

```python
                    self.push_image(arch_image_name, registry, credentials, registry_image)
                    build_info["registry_image"] = registry_image
                    # Recorded at push time; consumed only by --require-pinned-image runs.
                    pushed_digest = self.pushed_digests.get(registry_image)
                    if pushed_digest:
                        build_info["image_digest"] = pushed_digest
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_docker_builder.py -v`
Expected: PASS

- [ ] **Step 5: Confirm the manifest structure is unchanged for existing consumers**

Run: `pytest tests/unit/test_orchestration.py tests/unit/test_slurm_multi.py -v`
Expected: PASS — `image_digest` is purely additive.

- [ ] **Step 6: Commit**

```bash
git add src/madengine/execution/docker_builder.py tests/unit/test_docker_builder.py
git commit -m "feat(build): record image_digest in build manifest entries"
```

---

## Task 5: `--require-pinned-image` flag and context propagation

**Files:**
- Modify: `src/madengine/cli/commands/run.py:162-168` (flag), `:230-247` and `:318-341` (both `create_args_namespace` calls)
- Modify: `src/madengine/orchestration/run_orchestrator.py:85` (context merge), `:502` (manifest merge keys)
- Test: `tests/unit/test_orchestration.py`

Two entry points must both work: the CLI flag, and the `require_pinned_image` key inside `--additional-context` (which is how CI pipelines drive madengine). Persisting the key into `manifest["context"]` is what carries the setting to the nested `madengine run` that the SLURM job script executes on each compute node.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_orchestration.py`:

```python
class TestRequirePinnedImageContext:
    """--require-pinned-image and require_pinned_image both reach additional_context."""

    @patch("madengine.orchestration.run_orchestrator.Context")
    def test_cli_flag_sets_context_key(self, mock_context):
        args = create_args_namespace(
            additional_context=None,
            require_pinned_image=True,
            live_output=False,
        )
        orch = RunOrchestrator(args)
        assert orch.additional_context["require_pinned_image"] is True

    @patch("madengine.orchestration.run_orchestrator.Context")
    def test_flag_absent_leaves_key_unset(self, mock_context):
        args = create_args_namespace(
            additional_context=None,
            require_pinned_image=False,
            live_output=False,
        )
        orch = RunOrchestrator(args)
        assert "require_pinned_image" not in orch.additional_context

    @patch("madengine.orchestration.run_orchestrator.Context")
    def test_additional_context_key_alone_is_honoured(self, mock_context):
        args = create_args_namespace(
            additional_context="{'require_pinned_image': True}",
            live_output=False,
        )
        orch = RunOrchestrator(args)
        assert orch.additional_context["require_pinned_image"] is True

    @patch("madengine.orchestration.run_orchestrator.Context")
    def test_key_is_persisted_into_manifest_context(self, mock_context, tmp_path):
        manifest_path = tmp_path / "build_manifest.json"
        manifest_path.write_text(json.dumps({
            "built_images": {"img1": {"registry_image": "myorg/ci:m"}},
            "built_models": {"img1": {"name": "m"}},
            "context": {},
            "deployment_config": {},
        }))

        args = create_args_namespace(
            additional_context=None,
            require_pinned_image=True,
            live_output=False,
        )
        orch = RunOrchestrator(args)
        orch._load_and_merge_manifest(str(manifest_path))

        written = json.loads(manifest_path.read_text())
        assert written["context"]["require_pinned_image"] is True
```

Make sure `json`, `patch`, `create_args_namespace` and `RunOrchestrator` are imported at the top of `tests/unit/test_orchestration.py` — the existing `TestRunOrchestratorInit` and `TestCreateManifestFromLocalImage` classes already import most of these; add only what is missing:

```python
from madengine.cli.utils import create_args_namespace
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_orchestration.py -k RequirePinnedImage -v`
Expected: FAIL — `KeyError: 'require_pinned_image'`

- [ ] **Step 3: Write the implementation — orchestrator**

In `src/madengine/orchestration/run_orchestrator.py`, immediately after `self.additional_context = merged_context` (line 85), add:

```python
        # The CLI flag and the require_pinned_image context key are equivalent;
        # the key lets CI pipelines that drive madengine through
        # --additional-context opt in the same way as for k8s/slurm/tools.
        if getattr(args, "require_pinned_image", False):
            self.additional_context["require_pinned_image"] = True
```

In `_load_and_merge_manifest`, extend the merge key list (line 502) from:

```python
            merge_keys = ["tools", "pre_scripts", "post_scripts", "encapsulate_script"]
```

to:

```python
            merge_keys = [
                "tools",
                "pre_scripts",
                "post_scripts",
                "encapsulate_script",
                # Persisted so nested runs on SLURM compute nodes (which re-enter
                # `madengine run --manifest-file`) inherit the enforcement setting.
                "require_pinned_image",
            ]
```

- [ ] **Step 4: Write the implementation — CLI**

In `src/madengine/cli/commands/run.py`, add a new option immediately after the `skip_model_run` option block (which ends at line 108 with `] = False,`):

```python
    require_pinned_image: Annotated[
        bool,
        typer.Option(
            "--require-pinned-image",
            help=(
                "Pull registry images by the digest recorded in the build manifest "
                "instead of by tag. Fails immediately if the manifest has no digest "
                "for an image. Equivalent to the 'require_pinned_image' "
                "additional-context key."
            ),
        ),
    ] = False,
```

Then add `require_pinned_image=require_pinned_image,` to **both** `create_args_namespace(...)` calls — in the manifest-exists branch (next to `skip_model_run=skip_model_run,` around line 245) and in the full-workflow branch (next to `skip_model_run=skip_model_run,` around line 339).

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_orchestration.py -v`
Expected: PASS

- [ ] **Step 6: Verify the flag is wired into the CLI**

Run: `madengine run --help | grep -A 3 require-pinned-image`
Expected: the option and its help text are listed.

- [ ] **Step 7: Commit**

```bash
git add src/madengine/cli/commands/run.py src/madengine/orchestration/run_orchestrator.py tests/unit/test_orchestration.py
git commit -m "feat(run): add --require-pinned-image flag and context propagation"
```

---

## Task 6: Enforce pinned pulls in local Docker execution

**Files:**
- Modify: `src/madengine/execution/container_runner.py:2851-2859`
- Test: `tests/unit/test_container_runner.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_container_runner.py`:

```python
DIGEST = "sha256:" + "df36ef7e" * 8


class TestRequirePinnedImageLocalRun:
    """run_models_from_manifest honours require_pinned_image for registry pulls."""

    def _manifest(self, tmpdir, build_info):
        manifest_path = os.path.join(tmpdir, "build_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(
                {
                    "built_images": {"img1": build_info},
                    "built_models": {
                        "img1": {"name": "m", "tags": "t", "n_gpus": "1", "args": ""}
                    },
                },
                f,
            )
        return manifest_path

    def _runner(self):
        ctx = MagicMock()
        ctx.ctx = {"docker_env_vars": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx90a"}}
        ctx.ensure_runtime_context = MagicMock()
        console = MagicMock()
        console.sh.return_value = "testhost"
        runner = ContainerRunner(context=ctx, console=console)
        runner.set_credentials({})
        return runner

    @patch("madengine.execution.container_runner.update_perf_csv")
    def test_default_pulls_by_tag_even_when_digest_present(self, _mock_csv):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._manifest(
                tmpdir,
                {"registry_image": "myorg/ci:m", "image_digest": DIGEST},
            )
            runner = self._runner()
            runner.perf_csv_path = os.path.join(tmpdir, "perf.csv")

            with patch.object(runner, "pull_image") as mock_pull, patch.object(
                runner, "run_container", return_value={"status": "SUCCESS"}
            ):
                runner.run_models_from_manifest(manifest_file=manifest_path, timeout=60)

            mock_pull.assert_called_once_with("myorg/ci:m")

    @patch("madengine.execution.container_runner.update_perf_csv")
    def test_enabled_pulls_pinned_reference(self, _mock_csv):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._manifest(
                tmpdir,
                {"registry_image": "myorg/ci:m", "image_digest": DIGEST},
            )
            runner = self._runner()
            runner.perf_csv_path = os.path.join(tmpdir, "perf.csv")
            runner.additional_context = {"require_pinned_image": True}

            with patch.object(runner, "pull_image") as mock_pull, patch.object(
                runner, "run_container", return_value={"status": "SUCCESS"}
            ) as mock_run:
                runner.run_models_from_manifest(manifest_file=manifest_path, timeout=60)

            mock_pull.assert_called_once_with(f"myorg/ci@{DIGEST}")
            # The container must run the same pinned reference that was pulled.
            assert mock_run.call_args[1]["docker_image"] == f"myorg/ci@{DIGEST}"

    @patch("madengine.execution.container_runner.update_perf_csv")
    def test_enabled_without_digest_fails_before_pulling(self, _mock_csv):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._manifest(tmpdir, {"registry_image": "myorg/ci:m"})
            runner = self._runner()
            runner.perf_csv_path = os.path.join(tmpdir, "perf.csv")
            runner.additional_context = {"require_pinned_image": True}

            with patch.object(runner, "pull_image") as mock_pull, patch.object(
                runner, "run_container"
            ) as mock_run:
                result = runner.run_models_from_manifest(
                    manifest_file=manifest_path, timeout=60
                )

            mock_pull.assert_not_called()
            mock_run.assert_not_called()
            assert len(result["failed_runs"]) == 1
            assert "require-pinned-image" in result["failed_runs"][0]["error"]

    @patch("madengine.execution.container_runner.update_perf_csv")
    def test_manifest_context_key_enables_enforcement(self, _mock_csv):
        """A nested run on a SLURM compute node inherits the setting via manifest context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "build_manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(
                    {
                        "built_images": {
                            "img1": {"registry_image": "myorg/ci:m", "image_digest": DIGEST}
                        },
                        "built_models": {
                            "img1": {"name": "m", "tags": "t", "n_gpus": "1", "args": ""}
                        },
                        "context": {"require_pinned_image": True},
                    },
                    f,
                )
            runner = self._runner()
            runner.perf_csv_path = os.path.join(tmpdir, "perf.csv")

            with patch.object(runner, "pull_image") as mock_pull, patch.object(
                runner, "run_container", return_value={"status": "SUCCESS"}
            ):
                runner.run_models_from_manifest(manifest_file=manifest_path, timeout=60)

            mock_pull.assert_called_once_with(f"myorg/ci@{DIGEST}")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_container_runner.py -k RequirePinnedImageLocalRun -v`
Expected: FAIL — `test_enabled_pulls_pinned_reference` asserts the pinned ref but the tag is pulled.

- [ ] **Step 3: Write the implementation**

Add the import to `src/madengine/execution/container_runner.py`, after `from madengine.core.docker import Docker`:

```python
from madengine.core.image_digest import resolve_pinned_image
```

Replace the registry branch at `container_runner.py:2851-2859`:

```python
                elif build_info.get("registry_image"):
                    # Registry image: Pull from registry
                    try:
                        self.pull_image(build_info["registry_image"])
                        # Update docker_image to use registry image
                        run_image = build_info["registry_image"]
                    except Exception as pull_error:
                        self.rich_console.print(f"[yellow]Warning: Could not pull from registry, using local image[/yellow]")
                        run_image = image_name
```

with:

```python
                elif build_info.get("registry_image"):
                    # Registry image: Pull from registry. Under
                    # require_pinned_image this resolves to repo@sha256:... and
                    # raises (outside the pull try/except, so there is no tag
                    # fallback) when the manifest recorded no digest.
                    pull_target = resolve_pinned_image(
                        build_info["registry_image"],
                        build_info.get("image_digest"),
                        bool((self.additional_context or {}).get("require_pinned_image")),
                        model_name=model_info.get("name", ""),
                    )
                    try:
                        self.pull_image(pull_target)
                        # Update docker_image to use registry image
                        run_image = pull_target
                    except Exception as pull_error:
                        self.rich_console.print(f"[yellow]Warning: Could not pull from registry, using local image[/yellow]")
                        run_image = image_name
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_container_runner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/madengine/execution/container_runner.py tests/unit/test_container_runner.py
git commit -m "feat(run): pin local docker pulls to manifest digest when required"
```

---

## Task 7: Enforce pinned images in the Kubernetes pod spec

**Files:**
- Modify: `src/madengine/deployment/k8s_template_context.py:518`
- Test: `tests/unit/test_k8s.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_k8s.py`:

```python
class TestK8sRequirePinnedImage:
    """The generated pod spec image field honours require_pinned_image."""

    DIGEST = "sha256:" + "df36ef7e" * 8

    def _template_context(self, tmp_path, monkeypatch, require_pinned, image_digest):
        """Build a real template context, the way prepare() does.

        _prepare_template_context reads the manifest and the model's scripts
        directory from the current working directory, so the test runs inside
        tmp_path with a minimal model tree.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / "scripts" / "dummy").mkdir(parents=True)
        (tmp_path / "scripts" / "dummy" / "run.sh").write_text("#!/bin/bash\necho hi\n")

        image_info = {"registry_image": "myorg/ci:m"}
        if image_digest:
            image_info["image_digest"] = image_digest
        model_info = {
            "name": "m",
            "tags": ["t"],
            "n_gpus": "1",
            "args": "",
            "scripts": "scripts/dummy/run.sh",
            "dockerfile": "docker/dummy",
        }
        manifest = {
            "built_images": {"img1": image_info},
            "built_models": {"img1": model_info},
            "context": {},
        }
        (tmp_path / "build_manifest.json").write_text(json.dumps(manifest))

        additional_context = {
            "k8s": {"namespace": "default"},
            "gpu_vendor": "AMD",
            "guest_os": "UBUNTU",
        }
        if require_pinned:
            additional_context["require_pinned_image"] = True

        cfg = DeploymentConfig(
            target="k8s",
            manifest_file="build_manifest.json",
            additional_context=additional_context,
        )
        deployment = KubernetesDeployment(cfg)
        return deployment._prepare_template_context(model_info, image_info)

    def test_default_uses_tag(self, tmp_path, monkeypatch):
        ctx = self._template_context(
            tmp_path, monkeypatch, require_pinned=False, image_digest=self.DIGEST
        )
        assert ctx["image"] == "myorg/ci:m"

    def test_enabled_uses_pinned_reference(self, tmp_path, monkeypatch):
        ctx = self._template_context(
            tmp_path, monkeypatch, require_pinned=True, image_digest=self.DIGEST
        )
        assert ctx["image"] == f"myorg/ci@{self.DIGEST}"

    def test_enabled_without_digest_raises(self, tmp_path, monkeypatch):
        with pytest.raises(ConfigurationError):
            self._template_context(
                tmp_path, monkeypatch, require_pinned=True, image_digest=None
            )
```

Add whatever of these imports `tests/unit/test_k8s.py` is missing at the top (`pytest` is already there):

```python
import json

from madengine.core.errors import ConfigurationError
from madengine.deployment.base import DeploymentConfig
from madengine.deployment.kubernetes import KubernetesDeployment
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_k8s.py -k K8sRequirePinnedImage -v`
Expected: FAIL — `test_template_context_wires_the_resolver` fails; the resolver-behaviour tests pass already (they exercise Task 2 code directly, which is intentional: they document the K8s-facing contract).

- [ ] **Step 3: Write the implementation**

Add the import to `src/madengine/deployment/k8s_template_context.py`, next to the existing `from madengine.core.errors import ConfigurationError` (line 33):

```python
from madengine.core.image_digest import resolve_pinned_image
```

In `_prepare_template_context`, immediately before the `return {` statement that begins the context dict, add:

```python
        # Under require_pinned_image the pod pulls repo@sha256:... so a moved tag
        # surfaces as an ImagePullBackOff rather than a silent wrong-image run.
        resolved_image = resolve_pinned_image(
            image_info["registry_image"],
            image_info.get("image_digest"),
            bool(additional_context.get("require_pinned_image")),
            model_name=model_name,
        )
```

Then change the image entry (line 518) from:

```python
            "image": image_info["registry_image"],
```

to:

```python
            "image": resolved_image,
```

> `additional_context` and `model_name` are both already local variables in this method (`additional_context = self.config.additional_context.copy()` and `model_name = model_info["name"]` near the top).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_k8s.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/madengine/deployment/k8s_template_context.py tests/unit/test_k8s.py
git commit -m "feat(k8s): pin pod image to manifest digest when required"
```

---

## Task 8: Enforce pinned images in the SLURM slurm_multi wrapper

**Files:**
- Modify: `src/madengine/deployment/slurm.py:440-457`
- Test: `tests/unit/test_slurm_multi.py`

The standard SLURM template path needs no change — it re-enters `madengine run --manifest-file` on each compute node, which goes through Task 6's local-Docker enforcement using the `require_pinned_image` key that Task 5 persisted into `manifest["context"]`. Only the self-managed `slurm_multi` wrapper, which bypasses that nested run, needs its own resolution.

Setting `DOCKER_IMAGE_NAME` to the pinned reference pins both the parallel `srun docker pull` (which interpolates this variable) and the `docker run` inside the model's own script. See "Deliberate deviations" above.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_slurm_multi.py`:

```python
DIGEST = "sha256:" + "df36ef7e" * 8


class TestSlurmMultiRequirePinnedImage:
    """slurm_multi wrapper pins DOCKER_IMAGE_NAME (and thus the pull) when required."""

    IMAGE_KEY = "rocm/pytorch-private:sglang_disagg_mori_20260502"

    def _deployment(self, tmp_path, require_pinned, image_digest):
        script_rel = PR186_MODEL_ENTRY["scripts"]
        script_abs = tmp_path / script_rel
        script_abs.parent.mkdir(parents=True, exist_ok=True)
        script_abs.write_text("#!/bin/bash\n# placeholder\n")

        image_entry = {
            "image_name": self.IMAGE_KEY,
            "docker_image": self.IMAGE_KEY,
            "registry_image": self.IMAGE_KEY,
        }
        if image_digest:
            image_entry["image_digest"] = image_digest

        manifest = {
            "built_images": {self.IMAGE_KEY: image_entry},
            "built_models": {self.IMAGE_KEY: PR186_MODEL_ENTRY},
            "context": {
                "docker_env_vars": {},
                "docker_mounts": {},
                "docker_build_arg": {},
                "gpu_vendor": "AMD",
                "guest_os": "UBUNTU",
                "docker_gpus": "all",
            },
        }
        manifest_path = tmp_path / "build_manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        additional_context = {
            "deploy": "slurm",
            "gpu_vendor": "AMD",
            "guest_os": "UBUNTU",
            "slurm": dict(
                PR186_MODEL_ENTRY["slurm"], output_dir=str(tmp_path / "slurm_results")
            ),
            "distributed": PR186_MODEL_ENTRY["distributed"],
        }
        if require_pinned:
            additional_context["require_pinned_image"] = True

        cfg = DeploymentConfig(
            target="slurm",
            manifest_file=str(manifest_path),
            additional_context=additional_context,
        )
        return SlurmDeployment(cfg)

    def test_default_exports_tag(self, tmp_path):
        dep = self._deployment(tmp_path, require_pinned=False, image_digest=DIGEST)
        assert dep.prepare() is True
        script_text = Path(dep.script_path).read_text()
        assert f"export DOCKER_IMAGE_NAME={shlex.quote(self.IMAGE_KEY)}" in script_text
        assert DIGEST not in script_text

    def test_enabled_exports_pinned_reference(self, tmp_path):
        dep = self._deployment(tmp_path, require_pinned=True, image_digest=DIGEST)
        assert dep.prepare() is True
        script_text = Path(dep.script_path).read_text()

        pinned = f"rocm/pytorch-private@{DIGEST}"
        assert f"export DOCKER_IMAGE_NAME={shlex.quote(pinned)}" in script_text
        # The parallel pull interpolates the same value, so it is pinned too.
        assert f"docker pull {pinned}" in script_text

    def test_enabled_without_digest_does_not_silently_fall_through(self, tmp_path):
        """A missing digest must abort, not quietly take the standard template path.

        prepare()'s launcher peek wraps the slurm_multi dispatch in a bare
        `except Exception: pass`. Without the re-raise added in Step 3b, a
        ConfigurationError here would be swallowed and prepare() would generate
        an ordinary (unpinned) sbatch script instead — the exact silent
        degradation the flag exists to prevent.
        """
        dep = self._deployment(tmp_path, require_pinned=True, image_digest=None)
        with pytest.raises(ConfigurationError):
            dep.prepare()
```

Add whatever of these imports `tests/unit/test_slurm_multi.py` is missing at the top:

```python
import pytest

from madengine.core.errors import ConfigurationError
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_slurm_multi.py -k RequirePinnedImage -v`
Expected: FAIL — `test_enabled_exports_pinned_reference` finds the tag, not the pinned reference.

- [ ] **Step 3: Write the implementation**

Add the import to `src/madengine/deployment/slurm.py`, next to the other `madengine.*` imports (after `from madengine.utils.gpu_config import resolve_runtime_gpus`):

```python
from madengine.core.image_digest import resolve_pinned_image
```

In `_prepare_slurm_multi_script`, replace the `DOCKER_IMAGE_NAME` resolution block (lines 440-457):

```python
        # Override DOCKER_IMAGE_NAME with the built image from manifest
        # This ensures the run uses the freshly built image, not the base image
        # Priority: docker_image_name param > model_info.docker_image > env_vars.DOCKER_IMAGE_NAME
        if docker_image_name and docker_image_name.startswith("ci-"):
            # The manifest key IS the built image name for madengine-built images
            self.console.print(f"[cyan]Using built Docker image: {docker_image_name}[/cyan]")
            env_vars["DOCKER_IMAGE_NAME"] = docker_image_name
        elif "docker_image" in model_info:
            built_image = model_info["docker_image"]
            self.console.print(f"[cyan]Using Docker image: {built_image}[/cyan]")
            env_vars["DOCKER_IMAGE_NAME"] = built_image
        elif "image" in model_info:
            # Fallback to 'image' field
            built_image = model_info["image"]
            self.console.print(f"[cyan]Using Docker image: {built_image}[/cyan]")
            env_vars["DOCKER_IMAGE_NAME"] = built_image
```

with:

```python
        # Override DOCKER_IMAGE_NAME with the built image from manifest
        # This ensures the run uses the freshly built image, not the base image
        # Priority: docker_image_name param > model_info.docker_image > env_vars.DOCKER_IMAGE_NAME
        if docker_image_name and docker_image_name.startswith("ci-"):
            # The manifest key IS the built image name for madengine-built images
            self.console.print(f"[cyan]Using built Docker image: {docker_image_name}[/cyan]")
            env_vars["DOCKER_IMAGE_NAME"] = docker_image_name
        elif "docker_image" in model_info:
            built_image = model_info["docker_image"]
            self.console.print(f"[cyan]Using Docker image: {built_image}[/cyan]")
            env_vars["DOCKER_IMAGE_NAME"] = built_image
        elif "image" in model_info:
            # Fallback to 'image' field
            built_image = model_info["image"]
            self.console.print(f"[cyan]Using Docker image: {built_image}[/cyan]")
            env_vars["DOCKER_IMAGE_NAME"] = built_image

        # Under require_pinned_image, pin DOCKER_IMAGE_NAME to the digest recorded
        # at build time. slurm_multi runs the model's own script (no nested
        # `madengine run` on the compute nodes), so enforcement has to happen here.
        # Pinning the variable covers both the parallel `srun docker pull` below,
        # which interpolates it, and the `docker run` inside the model script.
        require_pinned = bool(
            self.config.additional_context.get("require_pinned_image")
        )
        if require_pinned and env_vars.get("DOCKER_IMAGE_NAME"):
            image_entry = (self.manifest.get("built_images") or {}).get(
                docker_image_name, {}
            )
            env_vars["DOCKER_IMAGE_NAME"] = resolve_pinned_image(
                env_vars["DOCKER_IMAGE_NAME"],
                image_entry.get("image_digest"),
                True,
                model_name=model_info.get("name", ""),
            )
            self.console.print(
                f"[cyan]Pinned Docker image: {env_vars['DOCKER_IMAGE_NAME']}[/cyan]"
            )
```

- [ ] **Step 3b: Stop `prepare()` from swallowing the enforcement error**

`prepare()` (`slurm.py:315-341`) wraps the whole slurm_multi dispatch — including the `_prepare_slurm_multi_script` call itself — in `except Exception: pass`, then falls through to the standard template path. Left as-is, a `ConfigurationError` from Step 3 would be silently discarded and an ordinary *unpinned* sbatch script would be generated instead.

Narrow the handler so enforcement errors propagate. Replace the `except` clause at `slurm.py:339-341`:

```python
        except Exception:
            # Fall through to develop's standard flow on any peek error
            pass
```

with:

```python
        except ConfigurationError:
            # Enforcement failures (e.g. --require-pinned-image with no recorded
            # digest) are deliberate aborts, not peek errors. Falling through to
            # the standard path here would silently generate an unpinned script.
            raise
        except Exception:
            # Fall through to develop's standard flow on any peek error
            pass
```

Add the import alongside the other `madengine.*` imports in `slurm.py`:

```python
from madengine.core.errors import ConfigurationError
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_slurm_multi.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/madengine/deployment/slurm.py tests/unit/test_slurm_multi.py
git commit -m "feat(slurm): pin slurm_multi image to manifest digest when required"
```

---

## Task 9: Lock in log-filename compatibility for pinned references

**Files:**
- Test: `tests/unit/test_execution.py`

`_docker_image_ref_for_log_naming()` already strips `@sha256:...`; this test prevents a future refactor from breaking it and silently changing log/tar filenames when pinning is on.

- [ ] **Step 1: Write the test**

Append to the existing `_docker_image_ref_for_log_naming` test class in `tests/unit/test_execution.py`:

```python
    def test_pinned_reference_names_same_as_untagged_reference(self):
        digest = "sha256:" + "df36ef7e" * 8
        assert (
            _docker_image_ref_for_log_naming(f"registry/ns/myimg@{digest}")
            == _docker_image_ref_for_log_naming("registry/ns/myimg")
        )

    def test_pinned_ci_reference_still_yields_tag(self):
        digest = "sha256:" + "df36ef7e" * 8
        assert (
            _docker_image_ref_for_log_naming(f"rocm/ns/img:ci-m_model_df@{digest}")
            == "ci-m_model_df"
        )
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/unit/test_execution.py -k log_naming -v`
Expected: PASS immediately — this is a characterization test of behaviour that already exists. If either assertion fails, stop and report it; that would mean pinned references change log filenames and the spec's compatibility claim is wrong.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_execution.py
git commit -m "test(execution): cover log naming for digest-pinned image references"
```

---

## Task 10: Documentation and full-suite verification

**Files:**
- Modify: `docs/cli-reference.md:232` (run options table)
- Modify: `docs/configuration.md`

- [ ] **Step 1: Add the CLI reference row**

In `docs/cli-reference.md`, insert a new row into the `run` options table immediately after the `--skip-model-run` row (line 232):

```markdown
| `--require-pinned-image` | | FLAG | `False` | Pull registry images by the `sha256` digest recorded in the build manifest (`repo@sha256:...`) instead of by tag, so a tag that moved between build and run fails loudly instead of silently running a different image. Fails immediately — with no tag fallback — if the manifest has no digest for an image. Equivalent to the `require_pinned_image` additional-context key. See [Configuration — Pinned image digests](configuration.md#pinned-image-digests). |
```

- [ ] **Step 2: Add the configuration section**

In `docs/configuration.md`, add a new section after the "Run phase: log error pattern scan" section (which ends before "## System environment collection (rocEnvTool)"):

```markdown
## Pinned image digests

Every build records the digest of the image it pushes as `image_digest` on each
`built_images` entry in `build_manifest.json`. This capture is always on and
costs nothing: by default the digest is carried along and never used.

Pass `--require-pinned-image` (or set `"require_pinned_image": true` in
`--additional-context`) to make the run phase pull `repo@sha256:...` instead of
the tag:

```bash
madengine run --manifest-file build_manifest.json --require-pinned-image

# Equivalent, for pipelines that drive madengine through additional context
madengine run --manifest-file build_manifest.json \
  --additional-context "{'require_pinned_image': True}"
```

| Behaviour | Flag absent (default) | Flag set |
|---|---|---|
| Registry pull | By tag | By digest (`repo@sha256:...`) |
| Manifest has no `image_digest` | Pull by tag | **Fails immediately**, no tag fallback |
| Tag moved since the build | Silently runs the newer image | Registry rejects the pull (`manifest unknown`) |

Applies to all three execution paths: local Docker, Kubernetes (the pod spec
`image` field), and SLURM. On SLURM the setting is written into the manifest's
`context` block so the nested `madengine run` on each compute node inherits it.

**Limitations**

- This does not prevent two concurrent builds from racing to push the same
  mutable tag. It converts the resulting silent wrong-image run into a fast,
  clear failure. Eliminating the race requires unique tags per build in the
  calling CI pipeline.
- Manifests produced by the build-on-compute-node path (SLURM batch builds,
  which push from inside a generated sbatch script) carry no `image_digest`.
  Runs against those manifests fail fast when the flag is set.
```

- [ ] **Step 3: Run the full unit suite**

Run: `pytest tests/unit -v`
Expected: PASS, no regressions.

- [ ] **Step 4: Run the integration suite**

Run: `pytest tests/integration -v -m "not slow"`
Expected: PASS. Pay particular attention to `tests/integration/test_docker_integration.py -k push_image`, which asserts the exact `docker tag` / `docker push` call shapes.

- [ ] **Step 5: Format and lint the changed files**

```bash
black src/madengine/core/image_digest.py src/madengine/execution/docker_builder.py \
      src/madengine/execution/container_runner.py src/madengine/deployment/slurm.py \
      src/madengine/deployment/k8s_template_context.py \
      src/madengine/orchestration/run_orchestrator.py src/madengine/cli/commands/run.py \
      tests/unit/test_image_digest.py
isort src/madengine/core/image_digest.py tests/unit/test_image_digest.py
mypy src/madengine/core/image_digest.py
```

Expected: `black`/`isort` reformat or report no changes; `mypy` reports no errors in the new module.

> Only run `black`/`isort` on the files you actually touched. Reformatting untouched files would bloat the diff.

- [ ] **Step 6: Commit**

```bash
git add docs/cli-reference.md docs/configuration.md
git commit -m "docs: document --require-pinned-image and image digest capture"
```

- [ ] **Step 7: Manual end-to-end smoke check (optional, requires a registry)**

```bash
# Build and push, then confirm the digest landed in the manifest
madengine build --tags dummy --registry localhost:5000
python -c "import json; m=json.load(open('build_manifest.json')); print({k: v.get('image_digest') for k, v in m['built_images'].items()})"

# Run with enforcement and confirm the pull is by digest
madengine run --manifest-file build_manifest.json --require-pinned-image --live-output 2>&1 | grep "docker pull"
```

Expected: the manifest prints a `sha256:...` per image, and the pull line contains `@sha256:`.

---

## Verification checklist against the spec's testing plan

| Spec test | Covered by |
|---|---|
| 1. Push output digest → `image_digest` | Task 3 `test_digest_parsed_from_push_output` + Task 4 `test_single_arch_push_sets_image_digest` |
| 2. Fallback to `docker image inspect` | Task 3 `test_falls_back_to_image_inspect_when_push_output_has_no_digest` |
| 3. Both paths fail → absent key, debug log, build unaffected | Task 3 `test_no_digest_anywhere_leaves_entry_absent_and_push_still_succeeds`, `test_inspect_failure_is_swallowed` |
| 4. Flag absent → pull by tag, no new output | Task 6 `test_default_pulls_by_tag_even_when_digest_present`, Task 7 `test_default_uses_tag`, Task 8 `test_default_exports_tag` |
| 5. Flag present + digest → pinned reference | Task 6 `test_enabled_pulls_pinned_reference`, Task 7 `test_enabled_uses_pinned_reference`, Task 8 `test_enabled_exports_pinned_reference` |
| 6. Flag present, no digest → fail before pull | Task 2 `test_enabled_without_digest_raises`, Task 6 `test_enabled_without_digest_fails_before_pulling` |
| 7. K8s pod spec and SLURM script carry pinned/tag reference | Task 7 + Task 8 |
| 8. Log filename derivation unchanged | Task 9 |
| 9. Existing `docker_sha` / manifest tests unmodified | Task 4 Step 5, Task 10 Steps 3-4 |
