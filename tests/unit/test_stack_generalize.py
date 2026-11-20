"""madengine.stack.generalize: working stacks from build manifests into the compatibility matrix."""

import textwrap

import yaml

from madengine.stack import generalize as gen
from madengine.stack.wheels import Wheel

MATRIX_TEXT = textwrap.dedent("""\
    # The matrix (comments are kept).
    indexes:
      fw: https://fw.example/whl/
    releases:
      "10.1": {default_install: therock, index: fw, pip_spec: ">=10.1,<10.2"}
    components:
      torch:
        "2.13.0":
          index: fw
          requirement: torch==2.13.0
      vllm:
        "0.29.1@fw":
          url: https://fw.example/whl/vllm-0.29.1-cp314-cp314-linux_x86_64.whl
          requires: ["torch==2.13.0"]

    # after components
    engines: {}
    """)
MATRIX = yaml.safe_load(MATRIX_TEXT)


def stack(**packages):
    return {"python": "3.14.1", "packages": packages}


def finder(served):
    """An index that serves *served*: {(name, version): requires}."""
    def find(name, version):
        if (name, version) not in served:
            return None
        return "fw", Wheel(name=name, version=version, python="cp314", platform="linux_x86_64",
                           url=f"https://fw.example/whl/{name}-{version}-cp314-cp314-linux_x86_64.whl",
                           requires=served[(name, version)])
    return find


def test_a_stack_compat_covers_changes_nothing():
    d = gen.generalize([("c", stack(vllm="0.29.1", torch="2.13.0"))], MATRIX, finder({}))
    assert d == {"add": {}, "unresolved": [], "contradicts": []}


def test_a_served_new_version_becomes_a_build_with_its_known_requirements():
    served = {("vllm", "0.30.0"): ["torch==2.13.0", "transformers>=5", 'pytest; extra == "test"']}
    d = gen.generalize([("c", stack(vllm="0.30.0", torch="2.13.0"))], MATRIX, finder(served))
    assert d["add"] == {"vllm": {"0.30.0@fw": {
        "url": "https://fw.example/whl/vllm-0.30.0-cp314-cp314-linux_x86_64.whl",
        "requires": ["torch==2.13.0"], "python": ["3.14"], "note": gen.NOTE}}}


def test_a_version_no_index_serves_is_listed_not_invented():
    d = gen.generalize([("c1", stack(vllm="0.30.0+g9a7ae4b8")), ("c2", stack(vllm="0.30.0+g9a7ae4b8"))],
                       MATRIX, finder({}))
    assert d["add"] == {}
    assert d["unresolved"] == [{"component": "vllm", "version": "0.30.0+g9a7ae4b8", "seen_in": ["c1", "c2"],
                                "why": d["unresolved"][0]["why"]}]


def test_a_requirement_a_working_stack_contradicts_is_listed():
    d = gen.generalize([("c", stack(vllm="0.29.1", torch="2.14.0"))], MATRIX, finder({}))
    assert d["contradicts"] == [{"component": "vllm", "build": "0.29.1@fw", "requires": "torch==2.13.0",
                                 "ran_with": "torch==2.14.0", "seen_in": ["c"]}]


def test_an_image_spec_image_is_its_own_spec_and_adds_nothing():
    spec, delta = gen.image_of({"build_plan": {"spec": {"rocm": "10.1", "components": {"vllm": "0.29.1@fw"}}}}, MATRIX)
    assert spec == {"rocm": "10.1", "components": {"vllm": "0.29.1@fw"}}
    assert delta == {"add": {}, "unresolved": [], "contradicts": []}


def test_a_recorded_stack_becomes_a_spec_and_the_facts_compat_lacks():
    build = {"base_docker": "rocm/base:1", "image_info": stack(vllm="0.30.0", torch="2.13.0")}
    spec, delta = gen.image_of(build, MATRIX, "c", finder({("vllm", "0.30.0"): ["torch==2.13.0"]}))
    assert spec["base"] == "rocm/base:1" and spec["components"] == {"torch": "2.13.0", "vllm": "0.30.0@fw"}
    assert list(delta["add"]["vllm"]) == ["0.30.0@fw"]


def test_an_image_without_a_recorded_stack_says_nothing():
    assert gen.image_of({"docker_image": "x"}, MATRIX) == ({}, {"add": {}, "unresolved": [], "contradicts": []})


def test_builds_are_merged_into_the_matrix_text_keeping_it():
    add = {"vllm": {"0.30.0@fw": {"url": "u"}}, "flash-attn": {"2.8.3@fw": {"url": "f"}}}
    out = gen.insert_builds(MATRIX_TEXT, add)
    data = yaml.safe_load(out)
    assert data["components"]["vllm"]["0.30.0@fw"] == {"url": "u"}
    assert data["components"]["flash-attn"]["2.8.3@fw"] == {"url": "f"}
    assert data["engines"] == {} and "# The matrix (comments are kept)." in out and "# after components" in out
    assert gen.insert_builds(out, add) == out          # idempotent


# A matrix with source builds, as the DI stacks have them.
SOURCE_MATRIX = {
    "indexes": {"fw": "https://fw.example/whl/"},
    "components": {
        "mori": {"fe12a11a@di": {"source": {"repo": "https://example.com/mori.git", "commit": "fe12a11a" + "0" * 32,
                                            "steps": ["pip install ."]}}},
        "vllm": {"9a7ae4b8+cd0d2656@di": {"source": {"repo": "https://example.com/vllm.git", "commit": "9a7ae4b8" + "1" * 32,
                                                     "patches": [{"file": "p.patch", "sha256": "x"}], "steps": ["pip install ."]}}},
    },
}
BASE = {"python": "3.12.3", "packages": {"torch": "2.9.0+rocm7.1.0", "triton": "3.5.0"}}


def test_a_source_build_is_matched_by_the_commit_in_its_version():
    image = {"python": "3.12.3", "packages": dict(BASE["packages"], mori="1.2.3.dev4+gfe12a11a")}
    spec, delta = gen.image_of({"base_docker": "b", "image_info": image, "base_info": BASE}, SOURCE_MATRIX, "c",
                               finder({}))
    assert spec["components"] == {"mori": "fe12a11a@di"}           # the base's own packages are not the image's
    assert delta == {"add": {}, "unresolved": [], "contradicts": []}


def test_a_vcs_install_is_matched_by_its_recorded_commit_and_its_patches_are_flagged():
    from madengine.stack.infer import infer
    image = {"packages": {"vllm": "0.11.1"}, "sources": {"vllm": "9a7ae4b8" + "1" * 32}}
    spec, findings = infer(image, None, "b", SOURCE_MATRIX)
    assert spec["components"] == {"vllm": "9a7ae4b8+cd0d2656@di"}
    assert [f["status"] for f in findings] == ["patched"]


def test_a_source_build_at_a_new_commit_is_drafted_from_the_matrixs_build_of_that_repo():
    image = {"python": "3.12.3", "packages": dict(BASE["packages"], mori="1.2.4.dev1+g0123abcd")}
    _, delta = gen.image_of({"base_docker": "b", "image_info": image, "base_info": BASE}, SOURCE_MATRIX, "c",
                            finder({}))
    assert delta["add"] == {}                                        # not merged: someone confirms it
    [item] = delta["unresolved"]
    assert (item["component"], item["commit"]) == ("mori", "0123abcd")
    [(key, entry)] = item["draft"].items()
    assert key == "0123abcd@di"
    assert entry["source"] == {"repo": "https://example.com/mori.git", "commit": "0123abcd", "steps": ["pip install ."]}


def test_a_new_wheels_requirement_the_working_stack_contradicts_is_reported_not_recorded():
    image = {"python": "3.14.1", "packages": {"torch": "2.13.0", "vllm": "0.30.0", "mori": "1.0"}}
    MATRIX_M = dict(MATRIX, components=dict(MATRIX["components"], mori={"1.0": {}}))
    _, delta = gen.image_of({"base_docker": "b", "image_info": image}, MATRIX_M, "c",
                            finder({("vllm", "0.30.0"): ["torch==2.13.0", "mori>=1.1"]}))
    assert delta["add"]["vllm"]["0.30.0@fw"]["requires"] == ["torch==2.13.0"]
    assert [(c["requires"], c["ran_with"]) for c in delta["contradicts"]] == [("mori>=1.1", "mori==1.0")]


IMAGES = {"stack": {"base": {"gfx942": "b:1@sha256:x"}, "components": {"mori": "fe12a11a@di", "ucx": "u@di"},
                    "variants": {"gfx950": {"components": {"mori": "m950@di"}}}}}


def test_a_spec_equal_to_a_named_stack_is_that_stack():
    assert gen.place(dict(IMAGES["stack"]), IMAGES) == {"extends": "${images.stack}"}


def test_an_image_is_placed_under_its_closest_stack_with_only_what_differs():
    spec = {"base": "b:1", "components": {"mori": "fe12a11a@di", "vllm": "0.30.0@fw"}}
    assert gen.place(spec, IMAGES) == {"extends": "${images.stack}", "components": {"vllm": "0.30.0@fw"}}


def test_an_image_sharing_no_build_keeps_its_own_spec():
    spec = {"base": "z", "components": {"vllm": "0.30.0@fw"}}
    assert gen.place(spec, IMAGES) is spec
