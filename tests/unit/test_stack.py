"""Unit tests for madengine.stack (wheel indexes, locks, Dockerfile generation)."""

import json
from unittest.mock import patch

import pytest

from madengine.execution import image_spec
from madengine.execution.image_spec import ImageSpecError
from madengine.stack import dockerfile, lock, wheels

FW = "https://fw.example/whl/vllm"
VLLM_URL = f"{FW}/vllm/vllm-0.29.1.dev0%2Brocm10.1.0.g98dff2a81.d20261002-cp314-cp314-linux_x86_64.whl"
AITER_URL = f"{FW}/amd-aiter/amd_aiter-0.1.22.post2.dev0%2Bgb4d9154d1-cp314-cp314-linux_x86_64.whl"
OLD_AITER_URL = f"{FW}/amd-aiter/amd_aiter-0.1.20.post1-cp314-cp314-linux_x86_64.whl"

LOCK = {
    "name": "vllm",
    "description": "test",
    "platform": {"vendor": "AMD", "rocm": "10.1", "python": "3.14",
                 "gpu_arch": ["gfx942", "gfx950"]},
    "base": "ubuntu:24.04",
    "indexes": {"therock": "https://therock.example/whl/"},
    "artifacts": [
        {"name": "rocm", "index": "therock",
         "requirement": "rocm[libraries,devel,device-{gpu_arch}]==10.1.0"},
        {"name": "torch", "index": "therock",
         "requirement": "torch[device-{gpu_arch}]==2.13.0+rocm10.1.0"},
        {"name": "amd-aiter", "url": AITER_URL, "built_by": "frameworks/vllm"},
        {"name": "fp4-kernels", "url": f"{FW}/x/fp4_kernels-1.0-py3-none-any.whl",
         "gpu_arch": "gfx950"},
        {"name": "vllm", "url": VLLM_URL, "built_by": "frameworks/vllm"},
    ],
    "sources": [
        {"name": "nixl", "repo": "https://github.com/ai-dynamo/nixl.git",
         "commit": "778edd1d1a50936b12c264879e12ef465e629002",
         "cherry_pick": ["fe3244c875ee4a22da6b0140c94c334546e9b715"],
         "env": {"UCX_HOME": "/opt/ucx"},
         "steps": ["make GPU={gpu_arch}"]},
    ],
    "env": {"VLLM_ROCM_USE_AITER": "1"},
}

METADATA = {
    "vllm": (["amd-aiter==0.1.22.post2.dev0+gb4d9154d1", "transformers>=5.10",
              'pytest; extra == "test"'], "<3.15,>=3.10"),
    "amd-aiter": ([], ">=3.10"),
    "fp4-kernels": ([], None),
}


def fake_fetch(wheel):
    wheel.requires, wheel.requires_python = METADATA[wheel.name]
    return wheel


class TestWheels:
    def test_filename_and_local_version_provenance(self):
        w = wheels.parse_wheel_filename(VLLM_URL)
        assert w.name == "vllm"
        assert w.version == "0.29.1.dev0+rocm10.1.0.g98dff2a81.d20261002"
        assert w.python == "cp314"
        assert w.local == {"rocm": "10.1.0", "git": "98dff2a81", "date": "20261002"}

    def test_not_a_wheel(self):
        assert wheels.parse_wheel_filename("https://x/vllm-0.1.tar.gz") is None

    def test_walk_records_nested_stack_index(self):
        pages = {
            "https://fw.example/whl/": ["https://fw.example/whl/migraphx/",
                                        "https://fw.example/whl/vllm-cdna/"],
            "https://fw.example/whl/migraphx/": [
                "https://fw.example/whl/migraphx/migraphx-2.16.0%2Brocm7.14.1-cp312-none-manylinux_2_28_x86_64.whl"],
            "https://fw.example/whl/vllm-cdna/": ["https://fw.example/whl/vllm-cdna/amd-aiter/"],
            "https://fw.example/whl/vllm-cdna/amd-aiter/": [
                "https://fw.example/whl/vllm-cdna/amd-aiter/amd_aiter-0.1.13-cp314-cp314-linux_x86_64.whl"],
        }
        with patch.object(wheels, "list_links", side_effect=lambda u: pages[u]):
            found = wheels.walk_index("https://fw.example/whl")
        assert {(w.index, w.name) for w in found} == {
            ("", "migraphx"), ("vllm-cdna", "amd-aiter")}
        with patch.object(wheels, "list_links", side_effect=lambda u: pages[u]):
            only = wheels.walk_index("https://fw.example/whl", projects=["AMD_Aiter"])
        assert [w.name for w in only] == ["amd-aiter"]


class TestLock:
    def test_resolve_selects_arch_and_fills_placeholders(self):
        r = lock.resolve(LOCK, "gfx942")
        names = [a["name"] for a in r["artifacts"]]
        assert "fp4-kernels" not in names
        assert r["artifacts"][0]["requirement"] == "rocm[libraries,devel,device-gfx942]==10.1.0"
        assert r["sources"][0]["steps"] == ["make GPU=gfx942"]
        assert "fp4-kernels" in [a["name"] for a in lock.resolve(LOCK, "gfx950")["artifacts"]]

    def test_unknown_arch_uses_device_all(self):
        r = lock.resolve(LOCK, None)
        assert r["artifacts"][1]["requirement"] == "torch[device-all]==2.13.0+rocm10.1.0"

    def test_arch_outside_lock_is_rejected(self):
        with pytest.raises(lock.LockError, match="locked for gfx942, gfx950"):
            lock.resolve(LOCK, "gfx1100")

    def test_override_replaces_and_marks(self):
        r = lock.resolve(LOCK, "gfx942", {"amd-aiter": {"url": OLD_AITER_URL},
                                         "torch": {"requirement": "torch==2.14.0"}})
        aiter = next(a for a in r["artifacts"] if a["name"] == "amd-aiter")
        assert aiter["url"] == OLD_AITER_URL and aiter["overridden"]
        assert "built_by" not in aiter
        torch = next(a for a in r["artifacts"] if a["name"] == "torch")
        assert torch["requirement"] == "torch==2.14.0" and torch["index"] == "therock"

    def test_override_can_add_a_source(self):
        r = lock.resolve(LOCK, "gfx942", {"mori": {"repo": "https://x/mori.git",
                                                   "commit": "abc1234"}})
        assert [s["name"] for s in r["sources"]] == ["nixl", "mori"]

    def test_check_passes_and_ignores_extras(self):
        assert lock.check(lock.resolve(LOCK, "gfx942"), fetch=fake_fetch) == []

    def test_check_catches_cross_artifact_mismatch(self):
        r = lock.resolve(LOCK, "gfx942", {"amd-aiter": {"url": OLD_AITER_URL}})
        problems = lock.check(r, fetch=fake_fetch)
        assert len(problems) == 1
        assert "requires amd-aiter==0.1.22" in problems[0]
        assert "lock has 0.1.20.post1" in problems[0]

    def test_check_catches_python_mismatch(self):
        r = lock.resolve({**LOCK, "platform": {**LOCK["platform"], "python": "3.15"}},
                         "gfx942")
        assert any("requires python" in p for p in lock.check(r, fetch=fake_fetch))

    @pytest.mark.parametrize("bad, msg", [
        ({"artifacts": [{"name": "x"}]}, "exactly one of 'url' or 'requirement'"),
        ({"artifacts": [{"name": "x", "requirement": "x", "index": "nope"}]}, "names index"),
        ({"artifacts": [{"name": "x", "url": "u", "colour": 1}]}, "unknown key"),
        ({"sources": [{"name": "s", "repo": "r"}]}, "'repo' and 'commit'"),
        ({"artifacts": [{"name": "A_b", "url": "u"}, {"name": "a-b", "url": "v"}]},
         "duplicates"),
    ])
    def test_validate(self, bad, msg):
        with pytest.raises(lock.LockError, match=msg):
            lock.validate({"name": "t", "platform": {}, "indexes": {}, **bad})

    def test_diff_shows_only_differences(self):
        a = lock.resolve(LOCK, "gfx942")
        b = lock.resolve(LOCK, "gfx942", {"amd-aiter": {"url": OLD_AITER_URL}})
        assert lock.diff(a, b) == [(
            "amd-aiter", "0.1.22.post2.dev0+gb4d9154d1", "0.1.20.post1")]

    def test_load_lists_known_locks(self, tmp_path):
        (tmp_path / "compat" / "stacks").mkdir(parents=True)
        (tmp_path / "compat" / "stacks" / "vllm.json").write_text(json.dumps(LOCK))
        assert lock.load("vllm", str(tmp_path))["name"] == "vllm"
        with pytest.raises(lock.LockError, match="known: vllm"):
            lock.load("sglang", str(tmp_path))


class TestDockerfile:
    def test_layers_in_lock_order(self):
        text = dockerfile.generate(lock.resolve(LOCK, "gfx942"))
        assert text.startswith("# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}")
        order = [text.index(f"# artifact: {n}") for n in ("rocm", "torch", "amd-aiter", "vllm")]
        assert order == sorted(order)
        assert text.index("# environment") < text.index("# source: nixl")
        assert "uv venv --seed --python 3.14" in text
        assert f"--no-deps {VLLM_URL}" in text
        # the completion layer pins by version, so pip does not re-download wheels
        assert "vllm==0.29.1.dev0+rocm10.1.0.g98dff2a81.d20261002" in text
        assert "cherry-pick fe3244c875ee4a22da6b0140c94c334546e9b715" in text
        assert "export UCX_HOME=/opt/ucx" in text
        assert 'ENV VLLM_ROCM_USE_AITER="1"' in text

    def test_base_python_when_lock_names_none(self):
        text = dockerfile.generate(lock.resolve(
            {**LOCK, "platform": {**LOCK["platform"], "python": None}}, "gfx942"))
        assert "uv venv" not in text

    def test_sha256_is_verified(self):
        locked = dict(LOCK, artifacts=[{"name": "vllm", "url": VLLM_URL, "sha256": "ab" * 32}])
        text = dockerfile.generate(lock.resolve(locked, "gfx942"))
        assert "sha256sum -c -" in text


class TestImageSpecStack:
    @pytest.fixture
    def mad_root(self, tmp_path, monkeypatch):
        (tmp_path / "compat" / "stacks").mkdir(parents=True)
        (tmp_path / "compat" / "stacks" / "vllm.json").write_text(json.dumps(LOCK))
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def test_plan_records_stack_and_generates_dockerfile(self, mad_root):
        plan = image_spec.resolve({"stack": "vllm"}, "AMD", "gfx942")
        assert plan.dockerfile_prefix.startswith(".madengine/build/stack-vllm-")
        assert (mad_root / f"{plan.dockerfile_prefix}.ubuntu.amd.Dockerfile").exists()
        assert plan.resolved["stack"] == "vllm"
        assert plan.resolved["components"]["vllm"].startswith("0.29.1")
        recorded = json.loads(plan.build_args[image_spec.BUILD_PLAN_ARG])
        assert recorded["resolved"]["lock_digest"] == plan.resolved["lock_digest"]

    def test_bad_override_fails_before_build(self, mad_root):
        with patch.object(wheels, "fetch_metadata", side_effect=fake_fetch):
            with pytest.raises(ImageSpecError, match="requires amd-aiter"):
                image_spec.resolve({"stack": "vllm", "override": {
                    "amd-aiter": {"url": OLD_AITER_URL}}}, "AMD", "gfx942")

    def test_stack_cannot_mix_flat_keys(self):
        with pytest.raises(ImageSpecError, match="change artifacts with 'override'"):
            image_spec.effective_image_spec({"image_spec": {"stack": "vllm", "torch": "2.12"}}, {})

    def test_vendor_mismatch(self, mad_root):
        with pytest.raises(ImageSpecError, match="locked for AMD"):
            image_spec.resolve({"stack": "vllm"}, "NVIDIA", "gfx942")

    def test_dockerfile_prefix_for_stack(self, mad_root):
        prefix = image_spec.dockerfile_prefix({"stack": "vllm"}, "gfx942")
        assert prefix.startswith(".madengine/build/stack-vllm-")


class TestExclude:
    LOCKED = {**LOCK, "exclude": [{"name": "transformers", "reason": "test"}]}

    def test_exclude_needs_reason(self):
        with pytest.raises(lock.LockError, match="reason"):
            lock.validate({**LOCK, "exclude": [{"name": "x"}]})

    def test_completion_drops_members_excludes_and_extras(self):
        reqs = lock.completion_requirements(lock.resolve(self.LOCKED, "gfx942"),
                                            fetch=fake_fetch)
        # vllm's amd-aiter pin is a lock member, transformers is excluded, and
        # pytest is only an extra: nothing is left to resolve.
        assert reqs == []
        reqs = lock.completion_requirements(lock.resolve(LOCK, "gfx942"),
                                            fetch=fake_fetch)
        assert reqs == ["transformers>=5.10"]

    def test_generate_lists_dependencies_and_notes_exclusion(self):
        resolved = lock.resolve(self.LOCKED, "gfx942")
        with pytest.raises(ValueError, match="completion"):
            dockerfile.generate(resolved)
        text = dockerfile.generate(resolved, completion=["numpy>=2"])
        assert "# excluded: transformers (test)" in text
        completion = text[text.index("# dependencies"):]
        assert "'numpy>=2'" in completion
        assert "vllm==" not in completion  # wheel artifacts stay out of the resolution
        assert "'torch[device-gfx942]==2.13.0+rocm10.1.0'" in completion
