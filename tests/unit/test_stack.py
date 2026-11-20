"""Unit tests for madengine.stack (compose, Dockerfile generation, wheel indexes)."""

import re
from unittest.mock import patch

import pytest

from madengine.stack import compose, dockerfile, wheels

FW = "https://fw.example/whl/vllm"
VLLM_URL = (f"{FW}/vllm/vllm-0.29.1.dev0%2Brocm10.1.0.g98dff2a81.d20261002"
            "-cp314-cp314-linux_x86_64.whl")
AITER_URL = f"{FW}/amd-aiter/amd_aiter-0.1.22.post2.dev0%2Bgb4d9154d1-cp314-cp314-linux_x86_64.whl"
OLD_AITER_URL = f"{FW}/amd-aiter/amd_aiter-0.1.20.post1-cp314-cp314-linux_x86_64.whl"
AITER = "0.1.22.post2.dev0+gb4d9154d1"

MATRIX = {
    "bare_os_base": "ubuntu:24.04",
    "indexes": {"therock": "https://therock.example/whl/"},
    "releases": {"10.1": {"default_install": "therock", "index": "therock",
                          "pip_spec": ">=10.1,<10.2"},
                 "7.2": {"default_install": "legacy"}},
    "components": {
        "torch": {
            "2.13.0+rocm10.1.0": {"index": "therock",
                                  "requirement": "torch[device-{gpu_arch}]==2.13.0+rocm10.1.0",
                                  "requires": ["rocm>=10.1,<10.2"]},
            "2.12.0+rocm10.1.0": {"index": "therock",
                                  "requirement": "torch==2.12.0+rocm10.1.0",
                                  "requires": ["rocm>=10.1,<10.2"]},
        },
        "amd-aiter": {
            f"{AITER}@fw": {"url": AITER_URL, "built_by": "frameworks/vllm",
                            "requires": ["torch==2.13.0+rocm10.1.0"], "python": ["3.14"]},
            "0.1.20.post1@fw": {"url": OLD_AITER_URL, "requires": ["torch==2.13.0+rocm10.1.0"]},
        },
        "fp4-kernels": {"1.0@x": {"url": f"{FW}/x/fp4_kernels-1.0-py3-none-any.whl",
                                  "gpu_arch": ["gfx950"]}},
        "vllm": {
            "0.29.1@fw": {"url": VLLM_URL, "built_by": "frameworks/vllm",
                          "requires": ["torch==2.13.0+rocm10.1.0", f"amd-aiter=={AITER}"],
                          "exclude": [{"name": "amd-quark", "reason": "no 3.14 build"}],
                          "env": {"VLLM_ROCM_USE_AITER": "1"}},
        },
        "ucx": {"da3fac2a@rocm": {"source": {"repo": "https://x/ucx.git",
                                             "commit": "da3fac2a",
                                             "steps": ["./configure --gpu={gpu_arch}"]}}},
        "nixl": {"1.4.1+pr2191@kv": {
            "requires": ["ucx"],
            "source": {"repo": "https://x/nixl.git", "commit": "778edd1d",
                       "cherry_pick": ["fe3244c8"], "steps": ["make"]},
            "system_packages": ["meson"]}},
    },
}
VLLM_SPEC = {"rocm": "10.1", "python": "3.14", "components": {"vllm": "0.29.1@fw"}}

METADATA = {
    "vllm": ([f"amd-aiter=={AITER}", "transformers>=5.10", 'pytest; extra == "test"'],
             "<3.15,>=3.10"),
    "amd-aiter": ([], ">=3.10"),
}


def fake_fetch(wheel):
    wheel.requires, wheel.requires_python = METADATA[wheel.name]
    return wheel


class TestCompose:
    def test_selection_pulls_in_requirements_in_dependency_order(self):
        plan = compose.compose(VLLM_SPEC, MATRIX, "gfx942")
        assert [i["name"] for i in plan["items"]] == ["rocm", "torch", "amd-aiter", "vllm"]
        why = {i["name"]: i["why"] for i in plan["items"]}
        assert why == {"rocm": "rocm", "torch": "required by vllm",
                       "amd-aiter": "required by vllm", "vllm": "selected"}
        assert plan["items"][0]["entry"]["requirement"] == (
            "rocm[libraries,devel,device-gfx942]>=10.1,<10.2")
        assert plan["exclude"] == [{"name": "amd-quark", "reason": "no 3.14 build"}]
        assert plan["env"] == {"VLLM_ROCM_USE_AITER": "1"}

    def test_order_follows_requirements_not_spec_order(self):
        spec = {"base": "img", "components": {"nixl": "1.4.1+pr2191@kv"}}
        names = [i["name"] for i in compose.compose(spec, MATRIX)["items"]]
        assert names == ["ucx", "nixl"]

    def test_selected_build_must_satisfy_requirements(self):
        spec = {**VLLM_SPEC, "components": {"vllm": "0.29.1@fw",
                                            "torch": "2.12.0+rocm10.1.0"}}
        with pytest.raises(compose.ComposeError, match="requires torch==2.13.0"):
            compose.compose(spec, MATRIX)

    def test_unauthorized_selection_is_rejected(self):
        with pytest.raises(compose.ComposeError, match="not authorized.*authorized builds"):
            compose.compose({"components": {"vllm": "0.30@fw"}}, MATRIX)

    def test_unsatisfiable_requirement_is_rejected(self):
        matrix = {**MATRIX, "components": {"a": {"1@x": {"url": "u", "requires": ["b>=2"]}}}}
        with pytest.raises(compose.ComposeError, match="no authorized build"):
            compose.compose({"components": {"a": "1@x"}}, matrix)

    AMBIGUOUS = {"components": {
        "torch": {"2.13.0+rocm10.1.0": {"url": "t1"}, "2.13.0+rocm10.0.0": {"url": "t0"},
                  "2.13.0@fw": {"url": "t2", "built_by": "fw"}},
        "a": {"1@x": {"url": "u", "requires": ["torch==2.13.0"]}},
        "b": {"1@fw": {"url": "v", "built_by": "fw", "requires": ["torch==2.13.0"]}},
        "c": {"1@x": {"url": "w", "requires": ["torch==2.13.0+rocm10.0.0"]}},
    }}

    def test_requirement_matching_several_builds_is_rejected(self):
        with pytest.raises(compose.ComposeError,
                           match=r"several authorized builds.*2\.13\.0\+rocm10\.1\.0"):
            compose.compose({"components": {"a": "1@x"}}, self.AMBIGUOUS)

    def test_ambiguity_resolves_to_the_requesters_origin(self):
        plan = compose.compose({"components": {"b": "1@fw"}}, self.AMBIGUOUS)
        assert compose.components(plan)["torch"] == "2.13.0@fw"

    def test_ambiguity_resolved_by_spec_selection_or_exact_pin(self):
        spec = {"components": {"a": "1@x", "torch": "2.13.0+rocm10.0.0"}}
        assert compose.components(compose.compose(spec, self.AMBIGUOUS))["torch"] == (
            "2.13.0+rocm10.0.0")
        plan = compose.compose({"components": {"c": "1@x"}}, self.AMBIGUOUS)
        assert compose.components(plan)["torch"] == "2.13.0+rocm10.0.0"

    def test_cycle_is_rejected(self):
        matrix = {"components": {"a": {"1@x": {"url": "u", "requires": ["b"]}},
                                 "b": {"1@x": {"url": "v", "requires": ["a"]}}}}
        with pytest.raises(compose.ComposeError, match="cycle"):
            compose.compose({"components": {"a": "1@x", "b": "1@x"}}, matrix)

    def test_python_and_arch_limits(self):
        with pytest.raises(compose.ComposeError, match="built for python 3.14"):
            compose.compose({**VLLM_SPEC, "python": "3.12"}, MATRIX)
        with pytest.raises(compose.ComposeError, match="built for gfx950, not gfx942"):
            compose.compose({"components": {"fp4-kernels": "1.0@x"}}, MATRIX, "gfx942")

    def test_python_given_as_a_string(self):
        matrix = {"components": {"a": {"1@x": {"url": "u", "python": "3.14"}}}}
        plan = compose.compose({"python": "3.14", "components": {"a": "1@x"}}, matrix)
        assert [i["name"] for i in plan["items"]] == ["a"]
        with pytest.raises(compose.ComposeError, match="built for python 3.14"):
            compose.compose({"python": "3.1", "components": {"a": "1@x"}}, matrix)

    def test_python_given_as_a_yaml_float_is_rejected(self):
        matrix = {"components": {"a": {"1@x": {"url": "u", "python": 3.14}}}}
        with pytest.raises(compose.ComposeError, match="a 1@x: python must be a string"):
            compose.compose({"python": "3.14", "components": {"a": "1@x"}}, matrix)
        matrix["components"]["a"]["1@x"]["python"] = [3.14]
        with pytest.raises(compose.ComposeError, match="python must be a string"):
            compose.compose({"python": "3.14", "components": {"a": "1@x"}}, matrix)
        with pytest.raises(compose.ComposeError, match="spec python must be a string"):
            compose.compose({"python": 3.14}, MATRIX)

    def test_rocm_release_must_be_authorized_and_therock(self):
        with pytest.raises(compose.ComposeError, match="known: 10.1, 7.2"):
            compose.compose({"rocm": "9.9"}, MATRIX)
        with pytest.raises(compose.ComposeError, match="legacy ROCm"):
            compose.compose({"rocm": "7.2"}, MATRIX)

    def test_inline_entry_is_marked_unauthorized(self):
        spec = {**VLLM_SPEC, "components": {
            "vllm": "0.29.1@fw",
            "amd-aiter": {"version": "0.1.20.post1", "url": OLD_AITER_URL}}}
        with pytest.raises(compose.ComposeError,
                           match=re.escape(f"requires amd-aiter=={AITER}")):
            compose.compose(spec, MATRIX)
        spec["components"]["amd-aiter"] = {"version": AITER, "url": AITER_URL}
        plan = compose.compose(spec, MATRIX)
        aiter = next(i for i in plan["items"] if i["name"] == "amd-aiter")
        assert aiter["authorized"] is False

    def test_check_reads_wheel_metadata(self):
        plan = compose.compose(VLLM_SPEC, MATRIX)
        assert compose.check(plan, fetch=fake_fetch) == []
        with patch.dict(METADATA, {"amd-aiter": ([], "<3.14")}):
            assert any("requires python" in p for p in compose.check(plan, fetch=fake_fetch))

    def test_completion_requirements_drop_members_excludes_extras(self):
        plan = compose.compose(VLLM_SPEC, MATRIX)
        assert compose.completion_requirements(plan, fetch=fake_fetch) == ["transformers>=5.10"]

    def test_completion_requirements_url_not_a_wheel(self):
        plan = compose.compose(VLLM_SPEC, MATRIX)
        vllm = next(i for i in plan["items"] if i["name"] == "vllm")
        vllm["entry"]["url"] = "https://x/vllm-0.29.1.tar.gz"
        with pytest.raises(compose.ComposeError, match="vllm 0.29.1@fw: url is not a wheel"):
            compose.completion_requirements(plan, fetch=fake_fetch)

    def test_diff(self):
        a = compose.compose(VLLM_SPEC, MATRIX)
        b = compose.compose({"base": "img", "components": {"nixl": "1.4.1+pr2191@kv"}}, MATRIX)
        rows = dict((r[0], r[1:]) for r in compose.diff(a, b))
        assert rows["[base]"] == ("ubuntu:24.04", "img")
        assert rows["vllm"] == ("0.29.1@fw", "-")
        assert rows["ucx"] == ("-", "da3fac2a@rocm")

    def test_bad_entry_is_rejected(self):
        matrix = {"components": {"a": {"1@x": {"url": "u", "requirement": "a"}}}}
        with pytest.raises(compose.ComposeError, match="exactly one of"):
            compose.compose({"components": {"a": "1@x"}}, matrix)


class TestDockerfile:
    def test_layers_follow_plan_order(self):
        plan = compose.compose(VLLM_SPEC, MATRIX, "gfx942")
        text = dockerfile.generate(plan, "vllm", completion=["transformers>=5.10"])
        assert text.startswith("# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}")
        marks = [text.index(f"# {n} ") for n in ("rocm", "torch", "amd-aiter", "vllm")]
        assert marks == sorted(marks)
        assert "(required by vllm)" in text
        assert "uv venv --seed --python 3.14" in text
        assert "RUN set -eo pipefail" in text
        assert f"--no-deps {VLLM_URL}" in text
        completion = text[text.index("# dependencies"):]
        assert "'transformers>=5.10'" in completion
        assert "vllm==" not in completion  # excludes: url wheels stay out of resolution
        assert "# excluded: amd-quark (no 3.14 build)" in text
        assert 'ENV VLLM_ROCM_USE_AITER="1"' in text

    def test_build_info_is_written_without_copy(self):
        import json
        import shlex
        import subprocess
        import sys

        text = dockerfile.generate(compose.compose(VLLM_SPEC, MATRIX), "vllm",
                                   completion=[])
        assert "COPY" not in text
        probe = text[text.index("# record the installed stack"):]
        assert "> /opt/mad/build_info.json" in probe
        # The inlined probe is one shell word that runs as written.
        line = next(l for l in probe.splitlines() if "python3 -c" in l)
        code = shlex.split(line.split("python3 -c ", 1)[1])[0]
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, check=True).stdout
        assert "packages" in json.loads(out)

    def test_pip_check_is_fatal(self):
        text = dockerfile.generate(compose.compose(VLLM_SPEC, MATRIX), "vllm",
                                   completion=[])
        completion = text[text.index("# dependencies"):]
        assert "pip check" in completion
        assert "pip check ||" not in completion

    def test_excludes_need_completion(self):
        with pytest.raises(ValueError, match="completion"):
            dockerfile.generate(compose.compose(VLLM_SPEC, MATRIX), "vllm")

    def test_without_excludes_wheels_are_pinned_by_version(self):
        spec = {"rocm": "10.1", "python": "3.14", "components": {"amd-aiter": f"{AITER}@fw"}}
        text = dockerfile.generate(compose.compose(spec, MATRIX), "w")
        assert f"amd-aiter=={AITER}" in text[text.index("# dependencies"):]

    def test_sources_after_wheels_with_cherry_pick_and_arch(self):
        spec = {"base": "img", "components": {"nixl": "1.4.1+pr2191@kv"}}
        text = dockerfile.generate(compose.compose(spec, MATRIX, "gfx942"), "kv")
        assert "uv venv" not in text  # no python: use the base image's
        assert text.index("# ucx") < text.index("# nixl")
        assert "cherry-pick fe3244c8" in text
        assert "./configure --gpu=gfx942" in text
        assert "meson" in text.split("# system packages")[1].split("\n\n")[0]

    def test_write_names_file_with_digest(self, tmp_path):
        plan = compose.compose({"base": "img", "components": {"ucx": "da3fac2a@rocm"}}, MATRIX)
        prefix = dockerfile.write(plan, "kv/x", "abc", root=str(tmp_path))
        assert prefix == ".madengine/build/spec-kv_x-abc"
        assert (tmp_path / f"{prefix}.ubuntu.amd.Dockerfile").exists()


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
        base = "https://fw.example/whl/"
        pages = {
            base: [base + "migraphx/", base + "vllm-cdna/"],
            base + "migraphx/": [
                base + "migraphx/migraphx-2.16.0%2Brocm7.14.1-cp312-none-manylinux_2_28_x86_64.whl"],
            base + "vllm-cdna/": [base + "vllm-cdna/amd-aiter/"],
            base + "vllm-cdna/amd-aiter/": [
                base + "vllm-cdna/amd-aiter/amd_aiter-0.1.13-cp314-cp314-linux_x86_64.whl"],
        }
        with patch.object(wheels, "list_links", side_effect=lambda u: pages[u]):
            found = wheels.walk_index("https://fw.example/whl")
        assert {(w.index, w.name) for w in found} == {
            ("", "migraphx"), ("vllm-cdna", "amd-aiter")}
        with patch.object(wheels, "list_links", side_effect=lambda u: pages[u]):
            only = wheels.walk_index("https://fw.example/whl", projects=["AMD_Aiter"])
        assert [w.name for w in only] == ["amd-aiter"]


class _Resp:
    """Stand-in for an urlopen response."""

    def __init__(self, status, body=b"", headers=None):
        self.status, self.body, self.headers = status, body, headers or {}

    def read(self, n=-1):
        return self.body if n is None or n < 0 else self.body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestRangeFile:
    URL = "https://fw.example/whl/x-1.0-py3-none-any.whl"
    DATA = bytes(range(256)) * 4

    def _open(self, answer):
        def get(url, method="GET", headers=None):
            if method == "HEAD":
                return _Resp(200, headers={"Content-Length": str(len(self.DATA))})
            return answer(headers["Range"])
        with patch.object(wheels, "_get", side_effect=get):
            remote = wheels._RangeFile(self.URL)
            remote.seek(100)
            return remote, remote.read(10)

    @staticmethod
    def _span(header):
        start, end = header.split("=")[1].split("-")
        return int(start), int(end)

    def test_partial_content_is_read(self):
        def answer(header):
            start, end = self._span(header)
            return _Resp(206, self.DATA[start:end + 1])
        remote, data = self._open(answer)
        assert data == self.DATA[100:110] and remote.tell() == 110

    def test_server_ignoring_range_is_refused(self):
        with pytest.raises(OSError, match="HTTP 200, not 206"):
            self._open(lambda header: _Resp(200, self.DATA))

    def test_body_longer_than_requested_is_refused(self):
        with pytest.raises(OSError, match="longer than the 10 bytes"):
            self._open(lambda header: _Resp(206, self.DATA))


class TestOrder:
    SPEC = {**VLLM_SPEC, "components": {"vllm": "0.29.1@fw", "fp4-kernels": "1.0@x"}}

    def test_order_moves_components_relative_to_each_other(self):
        default = [i["name"] for i in compose.compose(self.SPEC, MATRIX)["items"]]
        assert default.index("rocm") < default.index("fp4-kernels")
        ordered = compose.compose({**self.SPEC, "order": ["fp4-kernels", "rocm"]}, MATRIX)
        assert [i["name"] for i in ordered["items"]][:2] == ["fp4-kernels", "rocm"]

    def test_order_cannot_contradict_requirements(self):
        with pytest.raises(compose.ComposeError, match="cycle.*order"):
            compose.compose({**VLLM_SPEC, "order": ["vllm", "torch"]}, MATRIX)

    def test_order_names_must_be_in_plan(self):
        with pytest.raises(compose.ComposeError, match="order names nope"):
            compose.compose({**VLLM_SPEC, "order": ["nope", "vllm"]}, MATRIX)


class TestInfer:
    IMAGE = {"python": "3.14.10", "rocm_version": "10.1.0", "packages": {
        "rocm": "10.1.0", "rocm-sdk-core": "10.1.0",
        "torch": "2.13.0+rocm10.1.0", "amd-aiter": AITER, "vllm": "0.29.1",
        "triton": "3.8.0+git669b31ac.rocm10.1.0", "numpy": "2.5.4", "pip": "24.0"}}
    BASE = {"python": "3.12.3", "packages": {"pip": "24.0"}}

    def test_draft_selects_authorized_builds_and_reports_the_rest(self):
        from madengine.stack import infer

        spec, findings = infer.infer(self.IMAGE, self.BASE, "ubuntu:24.04", MATRIX)
        assert spec == {"base": "ubuntu:24.04", "python": "3.14", "rocm": "10.1",
                        "components": {"torch": "2.13.0+rocm10.1.0",
                                       "amd-aiter": f"{AITER}@fw", "vllm": "0.29.1@fw"}}
        status = {f["package"]: f["status"] for f in findings}
        assert status["triton"] == "new"            # custom build compat does not list
        assert status["1 packages"] == "dependencies"  # numpy: pip resolves it

    def test_unauthorized_version_and_base_provided_rocm(self):
        from madengine.stack import infer

        image = {**self.IMAGE, "packages": {**self.IMAGE["packages"], "vllm": "0.30.0"}}
        base = {"python": "3.14.1", "rocm_version": "10.1.0", "packages": {}}
        spec, findings = infer.infer(image, base, "rocm/base", MATRIX)
        assert "rocm" not in spec and "python" not in spec  # the base provides them
        vllm = next(f for f in findings if f["package"] == "vllm")
        assert vllm["status"] == "unauthorized" and vllm["authorized"] == ["0.29.1"]

    def test_drafted_spec_composes(self):
        from madengine.stack import infer

        spec, _ = infer.infer(self.IMAGE, self.BASE, "ubuntu:24.04", MATRIX)
        names = [i["name"] for i in compose.compose(spec, MATRIX)["items"]]
        assert names == ["rocm", "torch", "amd-aiter", "vllm"]


class TestSourceBuildsForPinnedStacks:
    """What a hand-written stack Dockerfile carried, as data: patches, a base per GPU, files, finish."""

    PATCHED = {"components": {"vllm": {"0+kimi@di": {
        "source": {"repo": "https://x/vllm.git", "commit": "9a7ae4b8",
                   "patches": [{"file": "vllm/kimi-k3.patch", "sha256": "ab" * 32, "upstream": "n/a"}],
                   "steps": ["pip install ."]},
        "capabilities": {"gfx942": {"weights": ["bf16"]}}}}}}

    def test_a_patch_is_copied_checked_and_applied_after_checkout(self):
        plan = compose.compose({"base": "b", "components": {"vllm": "0+kimi@di"}}, self.PATCHED, "gfx942")
        text = dockerfile.generate(plan, "w")
        assert "COPY patches/vllm/kimi-k3.patch /opt/mad/patches/vllm/kimi-k3.patch" in text
        check = text.index(f"echo '{'ab' * 32}  /opt/mad/patches/vllm/kimi-k3.patch' | sha256sum -c -")
        assert text.index("checkout --detach 9a7ae4b8") < check < text.index("apply --binary") < text.index("pip install .")

    def test_patches_must_be_under_docker_patches_with_a_digest(self):
        for bad in ({"file": "x.patch"}, {"file": "../x.patch", "sha256": "a"}, {"file": "/x", "sha256": "a"}):
            m = {"components": {"c": {"1@o": {"source": {"repo": "r", "commit": "c", "patches": [bad]}}}}}
            with pytest.raises(compose.ComposeError):
                compose.compose({"components": {"c": "1@o"}}, m)

    def test_a_base_per_gpu(self):
        spec = {"base": {"gfx942": "img@sha256:a", "gfx950": "img@sha256:b"}, "components": {}}
        assert compose.compose(spec, MATRIX, "gfx950")["base"] == "img@sha256:b"
        with pytest.raises(compose.ComposeError, match="not gfx90a"):
            compose.compose(spec, MATRIX, "gfx90a")

    def test_files_and_finish_steps(self):
        spec = {"base": "b", "components": {},
                "copy": [{"from": "di/check.py", "to": "/opt/mad/check.py"}],
                "finish": ["python3 /opt/mad/check.py {gpu_arch}"]}
        text = dockerfile.generate(compose.compose(spec, MATRIX, "gfx942"), "w")
        assert "COPY di/check.py /opt/mad/check.py" in text
        assert text.index("python3 /opt/mad/check.py gfx942") < text.index("record the installed stack")
        with pytest.raises(compose.ComposeError):
            compose.compose({"copy": [{"from": "../etc/passwd", "to": "/x"}]}, MATRIX)

    def test_a_spec_varies_by_gpu(self):
        m = {"components": {"mori": {"1@o": {"source": {"repo": "r", "commit": "c"}}}}}
        spec = {"base": "b", "components": {}, "env": {"ARCH": "{gpu_arch}"},
                "variants": {"gfx942": {"components": {"mori": "1@o"}, "base": "b942"}}}
        p942, p950 = compose.compose(spec, m, "gfx942"), compose.compose(spec, m, "gfx950")
        assert [i["name"] for i in p942["items"]] == ["mori"] and p942["base"] == "b942"
        assert p950["items"] == [] and p950["base"] == "b"
        assert 'ARCH="gfx950"' in dockerfile.generate(p950, "w")

    def test_a_run_build_and_a_sparse_source(self):
        m = {"components": {
            "gtest": {"1.14@di": {"run": ["wget -q -O /tmp/g.tgz https://x/g.tgz", "make -C /tmp/g"]}},
            "rocshmem": {"9c43b232@di": {"source": {"repo": "https://x/rocm-systems.git", "commit": "9c43b232",
                                                     "sparse": ["projects/rocshmem"], "steps": ["build {gpu_arch}"]}}}}}
        plan = compose.compose({"base": "b", "components": {"gtest": "1.14@di", "rocshmem": "9c43b232@di"}}, m, "gfx942")
        text = dockerfile.generate(plan, "w")
        assert "wget -q -O /tmp/g.tgz" in text and "dependencies of the components above" not in text
        assert "--no-checkout --filter=blob:none" in text and "sparse-checkout set --cone projects/rocshmem" in text
        assert "submodule update" not in text.split("# rocshmem")[1] and "build gfx942" in text
        with pytest.raises(compose.ComposeError, match="exactly one"):
            compose.compose({"components": {"x": "1@o"}}, {"components": {"x": {"1@o": {"run": ["a"], "url": "u"}}}})
