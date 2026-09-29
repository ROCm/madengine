"""Tests for layered distributed-inference configuration (the fourth way).

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import textwrap

import pytest

from madengine.deployment.layered_config import (
    DEFAULT_FILENAME,
    LayeredConfigError,
    _as_arg_map,
    find_config,
    load_config,
    resolve_env,
    resolve_for_model,
    resolve_serve_args,
    validate_card,
)


def write(tmp_path, body, name=DEFAULT_FILENAME):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return path


class TestArgMap:
    """Serve flags normalise to a map so one layer can override ONE flag."""

    def test_string_becomes_map(self):
        assert _as_arg_map("--disable-radix-cache --tp 8") == {
            "--disable-radix-cache": True,
            "--tp": "8",
        }

    def test_multi_value_flag_is_kept_whole(self):
        # --cuda-graph-bs takes a list; splitting it would lose the recipe.
        assert _as_arg_map("--cuda-graph-bs 8 16 32")["--cuda-graph-bs"] == "8 16 32"

    def test_map_passes_through(self):
        assert _as_arg_map({"--max-model-len": 16384}) == {"--max-model-len": 16384}

    def test_none_is_empty(self):
        assert _as_arg_map(None) == {}

    def test_non_flag_leading_token_is_an_error(self):
        with pytest.raises(LayeredConfigError):
            _as_arg_map("8 --tp")


class TestServeArgs:
    """base -> mode -> role -> role+mode, each overriding the last."""

    CONFIG = {
        "version": 1,
        "model": {
            "serve": {
                "base": "--attention-backend aiter --tp 1",
                "modes": {"tp": "--tp 8", "dp": "--tp 2"},
                "roles": {
                    "prefill": {"tp": "--disable-cuda-graph"},
                    "decode": {"tp": "--cuda-graph-bs 8 16"},
                },
            }
        },
    }

    def test_base_only(self):
        assert resolve_serve_args(self.CONFIG) == {
            "--attention-backend": "aiter",
            "--tp": "1",
        }

    def test_mode_overrides_base(self):
        assert resolve_serve_args(self.CONFIG, mode="tp")["--tp"] == "8"

    def test_role_and_mode_compose(self):
        decode = resolve_serve_args(self.CONFIG, mode="tp", role="decode")
        assert decode["--cuda-graph-bs"] == "8 16"
        assert decode["--tp"] == "8"
        assert "--disable-cuda-graph" not in decode

    def test_roles_do_not_leak_into_each_other(self):
        prefill = resolve_serve_args(self.CONFIG, mode="tp", role="prefill")
        assert "--disable-cuda-graph" in prefill
        assert "--cuda-graph-bs" not in prefill

    def test_missing_serve_section_is_empty(self):
        assert resolve_serve_args({"version": 1}) == {}


class TestEnvPrecedence:
    """model < benchmark < card env_vars < submit-time override.

    There is no site layer: site facts are a property of the run, and madengine
    composes those from Hydra groups or cluster.sh.
    """

    CONFIG = {
        "version": 1,
        "model": {"env": {"NVME_ROOT": "/mnt/model", "AITER": "1"}},
        "benchmark": [
            {"env": {"SEEDS": "3"}},
            {"kind": "niah", "env": {"AITER": "0", "NIAH_WORDS": "10000"}},
        ],
    }

    def test_task_level_benchmark_defaults_always_apply(self):
        assert resolve_env(self.CONFIG)["SEEDS"] == "3"

    def test_benchmark_kind_beats_model(self):
        assert resolve_env(self.CONFIG, benchmark="niah")["AITER"] == "0"

    def test_unselected_benchmark_kind_does_not_apply(self):
        assert "NIAH_WORDS" not in resolve_env(self.CONFIG)

    def test_card_env_beats_file(self):
        got = resolve_env(self.CONFIG, model_env={"AITER": "card"})
        assert got["AITER"] == "card"

    def test_runtime_override_wins_everything(self):
        got = resolve_env(
            self.CONFIG,
            model_env={"AITER": "card"},
            runtime_env={"AITER": "runtime"},
            benchmark="niah",
        )
        assert got["AITER"] == "runtime"

    def test_layers_merge_rather_than_replace(self):
        """A key set only in model survives a benchmark that never mentions it."""
        got = resolve_env(self.CONFIG, benchmark="niah")
        assert got["NVME_ROOT"] == "/mnt/model"

    def test_hydra_lands_at_the_top(self):
        """--config translates to additional_context, which is runtime_env.

        So a run-level group outranks a model default, which is the right way
        round and is what lets the two systems compose without either knowing
        about the other.
        """
        got = resolve_env(self.CONFIG, runtime_env={"NVME_ROOT": "/from/hydra"})
        assert got["NVME_ROOT"] == "/from/hydra"


class TestSiteLayerIsRefusedNotIgnored:
    """A dropped setting that looks applied is the failure this module prevents."""

    def test_a_site_block_raises_and_says_where_it_went(self):
        with pytest.raises(LayeredConfigError) as exc:
            resolve_env({"version": 1, "site": {"env": {"NVME_ROOT": "/mnt"}}})
        msg = str(exc.value)
        assert "no longer read" in msg
        assert "+profile" in msg and "cluster.sh" in msg

    def test_a_config_without_site_is_unaffected(self):
        got = resolve_env({"version": 1, "model": {"env": {"A": "1"}}})
        assert got == {"A": "1"}


class TestLoad:
    def test_version_must_be_supported(self, tmp_path):
        path = write(tmp_path, "version: 99\n")
        with pytest.raises(LayeredConfigError, match="not supported"):
            load_config(path)

    def test_malformed_yaml_names_the_file(self, tmp_path):
        path = write(tmp_path, "version: 1\nmodel: [unclosed\n")
        with pytest.raises(LayeredConfigError, match="not valid YAML"):
            load_config(path)

    def test_top_level_must_be_a_map(self, tmp_path):
        path = write(tmp_path, "- version: 1\n")
        with pytest.raises(LayeredConfigError, match="must be a map"):
            load_config(path)

    def test_empty_file_is_empty_config(self, tmp_path):
        assert load_config(write(tmp_path, "")) == {}


class TestFindConfig:
    def test_sibling_by_convention(self, tmp_path):
        write(tmp_path, "version: 1\n")
        assert find_config({}, tmp_path) is not None

    def test_absent_returns_none(self, tmp_path):
        assert find_config({}, tmp_path) is None

    def test_env_var_overrides_convention(self, tmp_path):
        write(tmp_path, "version: 1\n", name="other.yaml")
        found = find_config({"env_vars": {"MAD_CONFIG": "other.yaml"}}, tmp_path)
        assert found is not None and found.name == "other.yaml"


class TestValidateCard:
    """The one genuinely ambiguous thing: `args` means two different things."""

    def test_distributed_card_with_sbatch_args_warns(self):
        warnings = validate_card(
            {
                "name": "x",
                "distributed": {"launcher": "slurm_multi"},
                "args": "-N 2 -n 2",
            }
        )
        assert len(warnings) == 1
        assert "distributed.nnodes" in warnings[0]

    def test_script_style_args_do_not_warn(self):
        warnings = validate_card(
            {
                "name": "x",
                "distributed": {"launcher": "slurm_multi"},
                "args": "--model_repo m --config configs/default.yaml",
            }
        )
        assert warnings == []

    def test_non_distributed_card_is_not_checked(self):
        assert validate_card({"name": "x", "args": "-N 2 -n 2"}) == []


class TestResolveForModel:
    def test_no_file_means_no_change(self, tmp_path):
        # Ways 1-3 are the common case and must be untouched.
        env, warnings = resolve_for_model({"name": "x"}, tmp_path)
        assert env == {} and warnings == []

    def test_benchmark_selected_from_card_env(self, tmp_path):
        write(
            tmp_path,
            """
            version: 1
            benchmark:
              - kind: niah
                env: {NIAH_WORDS: '10000'}
            """,
        )
        env, _ = resolve_for_model(
            {"name": "x", "env_vars": {"BENCHMARK_SCRIPT": "niah"}}, tmp_path
        )
        assert env["NIAH_WORDS"] == "10000"

    def test_equivalence_with_hand_written_env_vars(self, tmp_path):
        """The property that makes way 4 safe to adopt.

        A way-4 file and the hand-written env_vars a card carries today must
        resolve to the same environment. If this holds, adopting way 4 changes
        nothing downstream -- no workload script has to be touched.
        """
        write(
            tmp_path,
            """
            version: 1
            model:
              id: moonshotai/Kimi-K3
              local_name: Kimi-K3
              env:
                MODEL_NAME: Kimi-K3
                TP_SIZE: '8'
                PP_SIZE: '2'
                REQUIRE_LOCAL_WEIGHTS: '1'
            benchmark:
              - kind: niah
                env:
                  BENCHMARK_SCRIPT: niah
                  NIAH_WORDS: '10000,50000,100000,200000'
            """,
        )
        way4, _ = resolve_for_model(
            {"name": "kimi", "env_vars": {"BENCHMARK_SCRIPT": "niah"}}, tmp_path
        )

        # What the same card expresses today, as a flat env_vars block (way 1-3).
        # NVME_ROOT and SHARED_MOUNT are deliberately absent: they are site facts,
        # and site facts are configured per RUN now -- a Hydra +profile/+env group
        # or cluster.sh -- not per model.
        way123 = {
            "MODEL_NAME": "Kimi-K3",
            "TP_SIZE": "8",
            "PP_SIZE": "2",
            "REQUIRE_LOCAL_WEIGHTS": "1",
            "BENCHMARK_SCRIPT": "niah",
            "NIAH_WORDS": "10000,50000,100000,200000",
        }
        assert way4 == way123

    def test_malformed_file_raises_rather_than_silently_ignoring(self, tmp_path):
        write(tmp_path, "version: 1\nbenchmark: 5\n")
        with pytest.raises(LayeredConfigError):
            resolve_for_model({"name": "x"}, tmp_path)


class TestDockerEnvVarsReachSlurm:
    """A manifest's context.docker_env_vars must reach the SLURM path.

    It used to reach only `docker run -e` on the local path, so manifests had to
    declare the same variable twice -- once in context.docker_env_vars and again
    in deployment_config.env_vars. Two copies of a NIC list drift, and a wrong
    one does not fail loudly: RCCL falls back to TCP and the benchmark still
    reports a number.
    """

    @staticmethod
    def _env(additional_context, model_info):
        """The layering _build_env_vars performs, in order."""
        env = {}
        if "docker_env_vars" in additional_context:
            env.update(additional_context["docker_env_vars"])
        if "env_vars" in model_info:
            env.update(model_info["env_vars"])
        if "env_vars" in additional_context:
            env.update(additional_context["env_vars"])
        return env

    def test_docker_env_vars_now_arrive(self):
        env = self._env({"docker_env_vars": {"NCCL_IB_HCA": "rdma0:1"}}, {})
        assert env["NCCL_IB_HCA"] == "rdma0:1"

    def test_declared_once_is_enough(self):
        # The whole point: no need to repeat it in deployment_config.env_vars.
        env = self._env(
            {"docker_env_vars": {"RDMAV_DRIVERS": "ionic", "RCCL_AINIC_ROCE": "1"}},
            {"env_vars": {"MODEL_NAME": "DeepSeek-R1"}},
        )
        assert env["RDMAV_DRIVERS"] == "ionic"
        assert env["RCCL_AINIC_ROCE"] == "1"
        assert env["MODEL_NAME"] == "DeepSeek-R1"

    def test_existing_precedence_is_unchanged(self):
        # Anything that already flowed keeps winning; this change is additive.
        env = self._env(
            {
                "docker_env_vars": {"NCCL_IB_GID_INDEX": "1"},
                "env_vars": {"NCCL_IB_GID_INDEX": "3"},
            },
            {"env_vars": {"NCCL_IB_GID_INDEX": "2"}},
        )
        assert env["NCCL_IB_GID_INDEX"] == "3"


class TestToolsOnSelfManagedPath:
    """slurm_multi used to drop `tools` entirely.

    prepare() early-dispatches self-managed launchers and returns, so the
    templated path's profiling block never ran for them. madengine cannot wrap a
    script it does not control, but silently ignoring a configured tool is worse
    than saying so.
    """

    @staticmethod
    def _resolve(tools, enabled, resolved):
        """The decision _build_env_vars now makes, isolated from SLURM."""
        env = {}
        if tools:
            picked = resolved if enabled else []
            if picked:
                env["MAD_TOOLS"] = ",".join(
                    t.get("name", str(t)) if isinstance(t, dict) else str(t)
                    for t in picked
                )
        return env

    def test_no_tools_configured_adds_nothing(self):
        assert self._resolve([], True, []) == {}

    def test_resolved_tools_reach_the_script(self):
        env = self._resolve([{"name": "rocprofv3"}], True, [{"name": "rocprofv3"}])
        assert env["MAD_TOOLS"] == "rocprofv3"

    def test_plain_string_tools_are_handled(self):
        assert (
            self._resolve(["rocprofv3"], True, ["rocprofv3"])["MAD_TOOLS"]
            == "rocprofv3"
        )

    def test_profiling_unavailable_sets_nothing(self):
        # rocprofv3 missing -> no MAD_TOOLS, and the caller warns.
        assert self._resolve([{"name": "rocprofv3"}], False, []) == {}


class TestSelfManagedResultsHonourTheCard:
    """The self-managed path ignored the card's declared multiple_results.

    The templated path reads it and scores candidates; this one searched a
    hardcoded list naming /shared_inference -- one site's NFS mount. On any other
    cluster the card's declaration was the only thing that could have worked.
    """

    @staticmethod
    def _declared(manifest):
        for m in (manifest.get("built_models") or {}).values():
            if m.get("multiple_results"):
                return m["multiple_results"]
        return None

    def test_declaration_is_found(self):
        manifest = {"built_models": {"img": {"multiple_results": "perf_Kimi-K3.csv"}}}
        assert self._declared(manifest) == "perf_Kimi-K3.csv"

    def test_absent_declaration_falls_back(self):
        assert self._declared({"built_models": {"img": {}}}) is None

    def test_empty_manifest_is_safe(self):
        assert self._declared({}) is None

    def test_first_declaring_model_wins(self):
        manifest = {
            "built_models": {
                "a": {},
                "b": {"multiple_results": "perf_b.csv"},
            }
        }
        assert self._declared(manifest) == "perf_b.csv"


class TestEachWayStandsAlone:
    """Any ONE configuration source is enough; none of them is required.

    The four ways are alternatives, not a stack that has to be assembled. A team
    that has only cluster.sh, or only a models.yaml, or only a way-4 file, must
    get a working run -- and a team that has none of them must too, because ways
    1-3 are read inside the container by scripts madengine never parses.

    These assert the absence half of that: nothing madengine adds on its own may
    become a thing that has to exist.
    """

    def test_no_way4_file_resolves_to_empty_not_error(self, tmp_path):
        env, warnings = resolve_for_model({"name": "m"}, tmp_path)
        assert env == {}
        assert warnings == []

    def test_no_scripts_dir_at_all(self):
        env, _ = resolve_for_model({"name": "m"}, None)
        assert env == {}

    def test_way4_alone_needs_no_card_env(self, tmp_path):
        (tmp_path / "mad-config.yaml").write_text(
            "version: 1\nmodel:\n  env:\n    NVME_ROOT: /mnt/nvme\n"
        )
        env, _ = resolve_for_model({"name": "m"}, tmp_path)
        assert env == {"NVME_ROOT": "/mnt/nvme"}

    def test_card_env_alone_needs_no_way4_file(self, tmp_path):
        card = {"name": "m", "env_vars": {"TP_SIZE": "8"}}
        env, _ = resolve_for_model(card, tmp_path)
        # No file: way 4 contributes nothing and the card is untouched.
        assert env == {}

    def test_missing_mad_config_pointer_is_not_fatal(self, tmp_path):
        card = {"name": "m", "env_vars": {"MAD_CONFIG": "nope.yaml"}}
        env, _ = resolve_for_model(card, tmp_path)
        assert env == {}


class TestOptionalDiagnosticsStayOptional:
    """A missing optional input must not end the run.

    gather_system_env_details is called on a default, not on a card's request, and
    it names its script by a CWD-relative path. When the model repo carries no
    copy the `cp` inside the container failed and killed a two-node job ten
    minutes in. Absence has to be a skip.
    """

    @staticmethod
    def _runner():
        from madengine.execution.container_runner import ContainerRunner

        return ContainerRunner.__new__(ContainerRunner)

    def test_existing_copy_is_used_untouched(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rel = "scripts/common/pre_scripts/run_rocenv_tool.sh"
        (tmp_path / "scripts/common/pre_scripts").mkdir(parents=True)
        (tmp_path / rel).write_text("# the repo's own copy\n")
        assert self._runner()._ensure_rocenv_script(rel) is True
        assert (tmp_path / rel).read_text() == "# the repo's own copy\n"

    def test_packaged_copy_is_staged_when_repo_has_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rel = "scripts/common/pre_scripts/run_rocenv_tool.sh"
        assert not (tmp_path / rel).exists()
        assert self._runner()._ensure_rocenv_script(rel) is True
        assert (tmp_path / rel).is_file()

    def test_staging_brings_the_rocenvtool_dependency(self, tmp_path, monkeypatch):
        """run_rocenv_tool.sh is not self-contained.

        It runs rocEnvTool/rocenv_tool.py, taken from beside itself or from
        ../scripts/common/pre_scripts. Staging only the .sh would satisfy the cp
        and then fail inside the container, which is a worse failure than the one
        being fixed: later, and further from the cause.
        """
        monkeypatch.chdir(tmp_path)
        rel = "scripts/common/pre_scripts/run_rocenv_tool.sh"
        assert self._runner()._ensure_rocenv_script(rel) is True
        deps = tmp_path / "scripts/common/pre_scripts/rocEnvTool"
        assert deps.is_dir()
        assert list(deps.glob("*.py")), "rocEnvTool staged without its python files"

    def test_partial_stage_is_completed_not_skipped(self, tmp_path, monkeypatch):
        """A tree carrying the .sh but not rocEnvTool/ still has to be repaired."""
        monkeypatch.chdir(tmp_path)
        rel = "scripts/common/pre_scripts/run_rocenv_tool.sh"
        (tmp_path / "scripts/common/pre_scripts").mkdir(parents=True)
        (tmp_path / rel).write_text("# repo copy, no rocEnvTool beside it\n")
        assert self._runner()._ensure_rocenv_script(rel) is True
        assert (tmp_path / "scripts/common/pre_scripts/rocEnvTool").is_dir()
        # The repo's own script is authoritative and must not be overwritten.
        assert (tmp_path / rel).read_text().startswith("# repo copy")

    def test_absent_everywhere_skips_instead_of_raising(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rel = "scripts/common/pre_scripts/does_not_exist.sh"
        assert self._runner()._ensure_rocenv_script(rel) is False

    def test_unwritable_cwd_skips_instead_of_raising(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rel = "scripts/common/pre_scripts/run_rocenv_tool.sh"
        monkeypatch.setattr(
            "madengine.execution.container_runner.os.makedirs",
            lambda *a, **k: (_ for _ in ()).throw(OSError("read-only file system")),
        )
        assert self._runner()._ensure_rocenv_script(rel) is False

    def test_skip_means_no_pre_script_is_appended(self, tmp_path, monkeypatch):
        """The whole point: a skip must leave pre_scripts empty, not half-built."""
        monkeypatch.chdir(tmp_path)
        runner = self._runner()
        monkeypatch.setattr(
            type(runner), "_ensure_rocenv_script", lambda self, rel: False
        )
        scripts = {"pre_scripts": []}
        runner.gather_system_env_details(scripts, "some-model")
        assert scripts["pre_scripts"] == []


class TestNamespacedModelNamesReachDisk:
    """A card's name is routinely namespaced; a filename is one segment.

    Discovery names a card found in a nested models.json after its directory, so
    `vllm_multinode/pyt_vllm_kimi-k3_mi300x_pp2xtp8` is the normal shape for MAD.
    Interpolated into a filename, that slash is a directory separator and the
    write fails on a parent nobody created. Runs died this way,
    on stock madengine as well as this branch.
    """

    @staticmethod
    def _dep():
        from madengine.deployment.slurm import SlurmDeployment

        return SlurmDeployment.__new__(SlurmDeployment)

    def test_namespaced_name_becomes_one_segment(self):
        got = self._dep()._safe_name(
            {"name": "vllm_multinode/pyt_vllm_kimi-k3_mi300x_pp2xtp8"}
        )
        assert "/" not in got
        assert got == "vllm_multinode_pyt_vllm_kimi-k3_mi300x_pp2xtp8"

    def test_flat_name_is_untouched(self):
        """The RCCL team's cards are flat and must keep the filenames they have."""
        name = "primus_pyt_megatron_lm_train_llama-3.1-70b"
        assert self._dep()._safe_name({"name": name}) == name

    def test_every_segment_of_a_deep_name_is_flattened(self):
        got = self._dep()._safe_name({"name": "a/b/c"})
        assert got == "a_b_c"

    def test_missing_name_does_not_raise(self):
        assert self._dep()._safe_name({}) == "model"

    def test_the_path_that_failed_in_build_62_now_resolves(self, tmp_path):
        """The exact ENOENT: writing under a parent that was never created."""
        dep = self._dep()
        model = {"name": "vllm_multinode/pyt_vllm_kimi-k3_mi300x_pp2xtp8"}
        out = tmp_path / "slurm_results"
        out.mkdir()

        naive = out / f"madengine_{model['name']}.sh"
        with pytest.raises(FileNotFoundError):
            naive.write_text("#!/bin/bash\n")

        safe = out / f"madengine_{dep._safe_name(model)}.sh"
        safe.write_text("#!/bin/bash\n")
        assert safe.is_file()
        assert safe.parent == out

    def test_no_raw_interpolation_of_a_name_into_a_filename_remains(self):
        """Guards the other four sites, which no unit test would otherwise reach."""
        import inspect

        from madengine.deployment import slurm as slurm_mod

        src = inspect.getsource(slurm_mod)
        assert "madengine_{model_info['name']}" not in src, (
            "a filename is being built from an unsanitised model name; "
            "use self._safe_name(model_info)"
        )


class TestInconclusiveHealthCheckGatesNothing:
    """A probe that sees nothing must not decide the outcome.

    On OCI amd-rccl every node comes back "Unreachable / srun failed" when probed
    from the login node -- 46 of 46 in one run. That is a fact about the probe.
    Standing down has to be complete: the exclude list, the submission gate and the
    nodelist pin all read the same empty result, and clearing only the first left
    that run failing on "Not enough clean nodes: need 2, found 0" from a check that
    had just announced it had nothing to say.
    """

    class _Sel:
        def __init__(self, inconclusive):
            if inconclusive:
                self.health_check_inconclusive = True

    @staticmethod
    def _gate(selector, nodes, clean, allow=False):
        """Mirrors the decision in slurm.py: gate, and whether to pin a nodelist."""
        inconclusive = getattr(selector, "health_check_inconclusive", False)
        blocked = nodes > 1 and not allow and not inconclusive and len(clean) < nodes
        pinned = (not inconclusive) and len(clean) >= nodes
        return blocked, pinned

    def test_inconclusive_check_does_not_block(self):
        blocked, _ = self._gate(self._Sel(True), 2, [])
        assert blocked is False

    def test_inconclusive_check_does_not_pin_a_nodelist(self):
        """Pinning off a list the probe could not verify is worse than not pinning."""
        _, pinned = self._gate(self._Sel(True), 2, ["a", "b", "c"])
        assert pinned is False

    def test_a_working_check_still_blocks_when_short(self):
        blocked, _ = self._gate(self._Sel(False), 2, ["a"])
        assert blocked is True

    def test_a_working_check_still_pins_when_satisfied(self):
        blocked, pinned = self._gate(self._Sel(False), 2, ["a", "b"])
        assert blocked is False and pinned is True

    def test_allow_submit_override_still_wins(self):
        blocked, _ = self._gate(self._Sel(False), 2, [], allow=True)
        assert blocked is False

    def test_single_node_is_never_gated(self):
        blocked, _ = self._gate(self._Sel(False), 1, [])
        assert blocked is False

    def test_selector_sets_the_flag_only_when_it_condemns_everything(self):
        from madengine.deployment.slurm_node_selector import SlurmNodeSelector

        sel = SlurmNodeSelector.__new__(SlurmNodeSelector)
        assert getattr(sel, "health_check_inconclusive", False) is False


class TestComputeNodesCanPullAPrivateImage:
    """The generated job must be able to fetch the image it was told to run.

    An sbatch wrapper outside madengine does docker login on every node before
    pulling. madengine's slurm_multi path pulled anonymously, so a
    private repository answered "not found" for an image that was plainly there.
    """

    @staticmethod
    def _emitted_login_block():
        """The staging block as the compute node will actually see it.

        Evaluated rather than scraped: these are f-strings, so a regex over the
        source reports {{ where the script gets {. Two harness bugs hid behind
        that while I was checking this block by hand.
        """
        import inspect

        from madengine.deployment import slurm as mod

        src = inspect.getsource(mod)
        i = src.index(
            '"# Image staging, in the shape plain sbatch of the card script already proves'
        )
        j = src.index('"",\n', src.index('"PULL_EXIT=$?",', i))
        lines = eval(  # noqa: S307 - our own source, one bound name
            "[\n" + src[i:j].rstrip().rstrip(",") + "\n]",
            {"docker_image": "rocm/mad-private:tag"},
        )
        return "\n".join(lines)

    def test_a_login_is_emitted_before_the_pull(self):
        block = self._emitted_login_block()
        assert "docker login" in block
        assert "--password-stdin" in block

    def test_it_accepts_both_credential_spellings(self):
        """madengine's own names, and the ones an sbatch wrapper outside madengine binds."""
        block = self._emitted_login_block()
        for name in (
            "MAD_DOCKERHUB_USER",
            "MAD_DOCKERHUB_PASSWORD",
            "MAD_DOCKER_USER",
            "MAD_DOCKER_TOKEN",
        ):
            assert name in block, name

    def test_only_names_are_written_never_values(self):
        """This script is archived as a build artifact."""
        block = self._emitted_login_block()
        assert "--password-stdin" in block
        assert "-p " not in block and "--password " not in block

    def test_the_srun_body_has_no_apostrophes(self):
        """One would close MAD_FETCH early and truncate what the node runs."""
        block = self._emitted_login_block()
        i = block.index("MAD_FETCH='") + len("MAD_FETCH='")
        j = block.index("\n'", i)
        assert block[i:j].count("'") == 0

    def test_it_fans_out_one_task_per_node(self):
        """Without this a node can be skipped, and then has no image at all."""
        assert "--ntasks-per-node=1" in self._emitted_login_block()

    def test_a_present_image_is_not_repulled(self):
        assert "docker image inspect" in self._emitted_login_block()

    def test_missing_credentials_are_not_an_error(self):
        """A public image must still pull when nothing was exported."""
        block = self._emitted_login_block()
        assert 'if [ -n "$_u" ] && [ -n "$_p" ]; then' in block

    def test_the_job_exports_its_environment(self):
        """Without this the credentials never reach the node to begin with."""
        import inspect

        from madengine.deployment import slurm as mod

        assert "#SBATCH --export=ALL" in inspect.getsource(mod)

    def test_sbatch_output_paths_are_one_path_segment(self):
        """--output spells it madengine-<name>, which the earlier sweep missed."""
        import inspect

        from madengine.deployment import slurm as mod

        src = inspect.getsource(mod)
        assert "madengine-{model_info['name']}" not in src


class TestNodeIPsAndLoginCannotBreakTheJob:
    """Two defects a run exposed in the generated slurm_multi script.

        │ Node IPs: 127.0.1.1,10.158.213.181
        │ Logging in to the registry on all nodes
        <job FAILED, log ends here>

    Rank 0's address was loopback, and the log stops dead on the login line.
    """

    @staticmethod
    def _emitted(start, end):
        import inspect
        import re

        from madengine.deployment import slurm as mod

        src = inspect.getsource(mod)
        i = src.index(start)
        j = src.index(end, i)
        out = []
        for m in re.finditer(
            r"^\s+r?(?:'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"),\s*$", src[i:j], re.M
        ):
            t = m.group(1) if m.group(1) is not None else m.group(2)
            out.append(t.replace('\\"', '"').replace("\\\\", "\\"))
        return "\n".join(out)

    def test_loopback_addresses_are_rejected(self):
        """getent on the batch node answers 127.0.1.1 for its OWN hostname."""
        blk = self._emitted('"# Reject loopback.', '"export MAD_NODE_IPS",')
        assert "grep -v '^127\\." in blk

    def test_a_loopback_answer_falls_back_to_asking_the_node(self):
        """hostname -I on the node cannot return someone else's loopback."""
        blk = self._emitted('"# Reject loopback.', '"export MAD_NODE_IPS",')
        assert "hostname -I" in blk
        assert "--nodelist=" in blk

    def test_the_fallback_takes_one_address_not_the_whole_line(self):
        """hostname -I prints every interface, docker bridges included."""
        blk = self._emitted('"# Reject loopback.', '"export MAD_NODE_IPS",')
        assert "awk '{print $1}'" in blk

    def test_the_login_cannot_end_the_job(self):
        """The script runs under set -e; only a failed PULL may stop it."""
        blk = TestComputeNodesCanPullAPrivateImage._emitted_login_block()
        assert "|| echo" in blk, "a failed login must not end the job"
        assert "exit 1" in blk, "a failed pull must end the job"

    def test_the_generated_script_still_sets_e(self):
        """If this stops being true the guard above is merely harmless."""
        import inspect

        from madengine.deployment import slurm as mod

        assert '"set -e",' in inspect.getsource(mod)


class TestFailuresAreVisibleInTheLog:
    """A run's log ended after the node IPs and said nothing else.

    Two independent reasons, both of which hide a failure rather than cause one.
    """

    @staticmethod
    def _src():
        import inspect

        from madengine.deployment import slurm as mod

        return inspect.getsource(mod)

    def test_the_pull_status_is_reachable_under_set_e(self):
        """`PULL_EXIT=$?` after a bare srun never runs while set -e is on."""
        src = self._src()
        i = src.index(
            'srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -c "$MAD_FETCH"'
        )
        before = src[max(0, i - 400) : i]
        assert '"set +e",' in before, "the srun is not wrapped in set +e"
        after = src[i : i + 300]
        assert '"PULL_EXIT=$?",' in after and '"set -e",' in after

    def test_stderr_is_streamed_not_only_stdout(self):
        src = self._src()
        assert "_{job_id}_*.err" in src, "the .err file is never read"

    def test_the_two_streams_are_distinguishable(self):
        """A reader has to be able to tell which stream a line came from."""
        src = self._src()
        i = src.index("_{job_id}_*.err")
        seg = src[max(0, i - 400) : i + 200]
        assert '"│"' in seg and '"┇"' in seg

    def test_positions_are_tracked_per_file(self):
        """Two files share one job id; a per-job position would interleave them."""
        src = self._src()
        assert "self._output_positions[output_file]" in src


class TestExclusiveDoesNotBreakEverySrun:
    """A run's stderr, visible for the first time, named the cause:

        │ Docker pull failed on one or more nodes
        ┇ srun: error: Invalid --exclusive specification

    sbatch --exclusive exports SLURM_EXCLUSIVE; srun re-parses it as a step
    request and rejects it. Every srun in the job fails, the pull included.
    Plain sbatch of the card script never hits this: it carries the card's own directives, which set
    -N, -n, --ntasks-per-node and --switches, but not --exclusive.
    """

    @staticmethod
    def _src():
        import inspect

        from madengine.deployment import slurm as mod

        return inspect.getsource(mod)

    def test_slurm_exclusive_is_unset_in_the_job(self):
        assert '"unset SLURM_EXCLUSIVE",' in self._src()

    def test_it_is_unset_before_any_srun_runs(self):
        src = self._src()
        start = src.index("def _prepare_slurm_multi_script")
        unset = src.index('"unset SLURM_EXCLUSIVE",', start)
        sruns = [
            i
            for i in range(start, len(src))
            if src.startswith("srun --", i)
            or src.startswith("'srun --", i)
            or src.startswith('"srun --', i)
        ]
        assert sruns, "no srun found to guard"
        assert all(i > unset for i in sruns), "an srun precedes the unset"

    def test_the_allocation_is_still_requested_exclusive(self):
        """Unsetting the step variable must not give up the job-level allocation."""
        src = self._src()
        assert '"#SBATCH --exclusive"' in src


class TestTheHealthProbeCanActuallyReachANode:
    """A run died on the condition this check exists to prevent:

        RuntimeError: The memory capacity is unbalanced.
                      Some GPUs may be occupied by other processes.

    The check had reported every node UNREACHABLE for five builds and stood
    down each time, because its own srun failed:

        srun: error: Invalid specification

    It ran without --partition, and the OCI login node has no default one.
    """

    @staticmethod
    def _src():
        import inspect

        from madengine.deployment import slurm_node_selector as mod

        return inspect.getsource(mod)

    def test_the_probe_names_the_partition(self):
        src = self._src()
        assert 'srun_cmd.append(f"--partition={self.partition}")' in src

    def test_select_nodes_records_the_partition_for_the_probe(self):
        src = self._src()
        i = src.index("def select_nodes(")
        j = src.index("def ", i + 10)
        assert "self.partition = partition" in src[i:j]

    def test_the_probe_still_overlaps_a_running_job(self):
        """Without --overlap it would queue behind the job it is inspecting."""
        assert '"--overlap"' in self._src()

    def test_a_selector_with_no_partition_yet_emits_no_flag(self):
        from madengine.deployment.slurm_node_selector import SlurmNodeSelector

        sel = SlurmNodeSelector.__new__(SlurmNodeSelector)
        assert getattr(sel, "partition", None) is None


class TestCleanupCanActuallyCleanTheseNodes:
    """Why a node was occupied with no SLURM job on it to explain it.

    These workloads run in docker. `docker run` is attached here, but the
    container belongs to the daemon, not to the job: scancel kills the job's
    shell and the container keeps its GPU memory. The next allocation then finds
    busy GPUs, which is what a run reported.
    """

    @staticmethod
    def _src():
        import inspect

        from madengine.deployment import slurm_node_selector as mod

        return inspect.getsource(mod)

    def test_cleanup_stops_containers(self):
        """Killing host processes alone never touches a running container."""
        assert "docker stop" in self._src()

    def test_cleanup_covers_sglang_not_only_vllm(self):
        src = self._src()
        assert 'pkill -9 -f "sglang"' in src

    def test_cleanup_waits_for_gpu_memory_to_drain(self):
        """A container stopped a moment ago still shows its memory as used."""
        src = self._src()
        i = src.index("CLEANUP_OK")
        assert "sleep 5" in src[max(0, i - 500) : i]

    def test_cleanup_names_the_partition_too(self):
        """The probe was fixed for this; the cleanup had the same omission."""
        src = self._src()
        i = src.index("cleanup_script")
        seg = src[i : i + 2500]
        assert 'srun_cmd.append(f"--partition={self.partition}")' in seg


class TestALocalImageCannotRunMultinode:
    """A ci-* image exists only on the machine that built it.

    One node can run it; the others have nothing to run. One run built for 35
    minutes, failed to push, carried on with the local name, and the job then
    failed on nodes that had never seen it.

    This is the backstop, not the fix -- when a push does happen the run must use
    the pushed name (see TestAPushedImageIsTheOneThatRuns). It catches the case
    where no usable image exists at all, before an allocation is spent on it.
    """

    def test_a_local_image_on_a_multinode_run_fails_fast(self):
        from madengine.core.errors import ConfigurationError

        image, nodes = "ci-vllm_multinode_pyt_vllm_kimi-k3", 2
        with pytest.raises(ConfigurationError) as exc:
            if nodes > 1 and image.startswith("ci-"):
                raise ConfigurationError(
                    f"Image '{image}' is local to the build machine, and this is a "
                    f"{nodes}-node run: the other nodes cannot pull it."
                )
        assert "cannot pull it" in str(exc.value)

    def test_the_guard_is_present_in_the_source(self):
        import inspect

        from madengine.deployment import slurm as mod

        src = inspect.getsource(mod)
        assert 'self.nodes > 1 and docker_image.startswith("ci-")' in src

    def test_a_single_node_run_is_not_blocked(self):
        """A local image is perfectly usable when only one node runs it."""
        image, nodes = "ci-kimi", 1
        assert not (nodes > 1 and image.startswith("ci-"))


class TestAPushedImageIsTheOneThatRuns:
    """Runs pushed an image and then ran a name the nodes could not pull.

        Successfully pushed: rocm/mad-private:ci-vllm_multinode_..._kimi_k3...
        Using built Docker image: ci-vllm_multinode_..._kimi_k3...
        docker: Error response from daemon: pull access denied

    build_info["docker_image"] is set before the push and never updated;
    build_info["registry_image"] is what the push records. docker_builder already
    propagates the latter into built_models env_vars "for parallel pull in
    slurm_multi" -- and this block was overwriting it with the local manifest key.
    """

    @staticmethod
    def _usable(ref):
        return bool(ref) and not ref.startswith("ci-") and not ref.startswith("<")

    @classmethod
    def _pick(cls, entry, already, recorded, key):
        """Mirrors the resolution order in _prepare_slurm_multi_script."""
        from_push = entry.get("registry_image") or ""
        if cls._usable(from_push):
            return from_push
        if cls._usable(already):
            return already
        if cls._usable(recorded):
            return recorded
        if key and key.startswith("ci-"):
            return key
        return ""

    def test_the_pushed_image_wins(self):
        pushed = "rocm/mad-private:ci-kimi"
        got = self._pick({"registry_image": pushed}, pushed, "ci-kimi", "ci-kimi")
        assert got == pushed

    def test_a_local_build_with_no_push_uses_the_key(self):
        """Legitimate for single-node; the multi-node guard refuses it."""
        got = self._pick({}, "<supply-your-image>", "ci-kimi", "ci-kimi")
        assert got == "ci-kimi"

    def test_a_use_image_override_is_honoured(self):
        ref = "rocm/mad-private:pyt_vllm_kimi_k3_mi300x-db276a66649a"
        assert self._pick({}, ref, ref, "ci-kimi") == ref

    def test_a_card_placeholder_is_never_selected(self):
        """'<supply-your-image>' must not reach a compute node."""
        assert self._pick({}, "<supply-your-image>", "", "ci-kimi") == "ci-kimi"

    def test_the_source_reads_registry_image_before_the_key(self):
        import inspect

        from madengine.deployment import slurm as mod

        src = inspect.getsource(mod)
        i = src.index('_entry.get("registry_image")')
        j = src.index('docker_image_name.startswith("ci-")', i)
        assert i < j, "the local manifest key is still checked first"


class TestHydraConfigParityForThisPipeline:
    """Whether --config can replace --additional-context for this pipeline.

    Inert until ROCm/madengine#121 merges; active from the moment it does.

    The answer is conditional, and the condition is the point. A plain user YAML
    carrying our keys translates to a byte-identical context. Composing the same
    thing from the `scheduler=slurm` GROUP does not: the group carries its own
    defaults, so the run silently gains OMP_NUM_THREADS and MIOPEN_FIND_MODE plus
    a dozen slurm keys nobody asked for. A benchmark whose environment changed
    because of how its config was spelled is exactly the failure this project
    exists to stop, so the pipeline stays on --additional-context and any future
    move must use a plain YAML file, not groups.
    """

    OURS = {
        "slurm": {
            "partition": "amd-rccl",
            "nodes": 2,
            "gpus_per_node": 8,
            "time": "04:00:00",
            "exclusive": True,
        },
        "env_vars": {"TP_SIZE": "8"},
    }

    def test_a_plain_yaml_translates_to_an_identical_context(self):
        translator = pytest.importorskip("madengine.config.translator")
        omegaconf = pytest.importorskip("omegaconf")

        cfg = omegaconf.OmegaConf.create(dict(self.OURS))
        ctx, _meta = translator.ConfigTranslator.to_additional_context(cfg)
        assert ctx == self.OURS

    def test_a_scheduler_group_does_not(self):
        """Not a defect in the group -- a reason not to compose from one here."""
        translator = pytest.importorskip("madengine.config.translator")
        omegaconf = pytest.importorskip("omegaconf")

        group = omegaconf.OmegaConf.create(
            {
                "slurm": {"partition": "amd-rccl", "nodes": 1, "exclusive": True},
                "env_vars": {"OMP_NUM_THREADS": "8", "MIOPEN_FIND_MODE": "1"},
            }
        )
        merged = omegaconf.OmegaConf.merge(
            group, omegaconf.OmegaConf.create({"slurm": {"nodes": 2}})
        )
        ctx, _meta = translator.ConfigTranslator.to_additional_context(merged)
        assert ctx != self.OURS
        assert "OMP_NUM_THREADS" in ctx["env_vars"]


class TestTheProbeWaitsLongEnoughToLearnSomething:
    """A run's health check reported every node "Unreachable / Timeout".

    The --exclusive leak was fixed, so the probe finally ran -- and then hit a
    30-second ceiling. That ceiling is not a command runtime: the probe srun has
    to WAIT for the scheduler before it executes anything, and with a dozen jobs
    queued it never got a slot. The check stood down having learned nothing,
    which is the same blindness as before by a different route.
    """

    @staticmethod
    def _src(mod_name):
        import importlib
        import inspect

        return inspect.getsource(importlib.import_module(mod_name))

    def test_the_default_covers_a_busy_queue(self):
        src = self._src("madengine.deployment.slurm_node_selector")
        assert "timeout: int = 120," in src

    def test_it_is_configurable_per_cluster(self):
        """No single number fits both a quiet cluster and a full one."""
        src = self._src("madengine.deployment.slurm")
        assert 'self.slurm_config.get("node_check_timeout", 120)' in src

    def test_the_selector_actually_uses_it_for_both_probe_and_cleanup(self):
        src = self._src("madengine.deployment.slurm_node_selector")
        assert src.count("timeout=self.timeout") >= 2


class TestTheTemplatedPathAlsoNeedsOnePathSegment:
    """A run reported COMPLETED in 33 seconds and produced nothing.

        ✓ Submitted SLURM job: 123456
        SLURM job 123456 final status: COMPLETED
        │ 1 │ ❌ Failed │ sglang/pyt_sglang_kimi-k3 │

    job.sh.j2 interpolates model_name straight into #SBATCH --output, and the
    card's name is namespaced. sbatch could not write to
    slurm_results/madengine-sglang/... so the job ended immediately with no log.

    The identical fix went into the self-managed path's directives earlier. This
    path spells it {{ model_name }} in a template, so the same sweep missed it --
    the third time this bug has been found in a different spelling.
    """

    @staticmethod
    def _src():
        import inspect

        from madengine.deployment import slurm as mod

        return inspect.getsource(mod)

    def test_the_template_context_is_sanitised(self):
        assert '"model_name": self._safe_name(model_info),' in self._src()

    def test_the_raw_name_is_still_available_as_a_label(self):
        """Losing it entirely would make the generated script harder to read."""
        assert '"model_display_name": model_info["name"],' in self._src()

    def test_no_template_context_key_carries_a_raw_namespaced_name_into_a_path(self):
        src = self._src()
        i = src.index('"model_name": self._safe_name(model_info),')
        j = src.index('"manifest_file"', i)
        assert 'model_info["name"]' in src[i:j]  # only as model_display_name

    def test_the_template_builds_paths_from_model_name(self):
        """If the template stopped using it for paths, sanitising it would be moot."""
        import pathlib

        from madengine.deployment import slurm as mod

        tpl = (
            pathlib.Path(mod.__file__).parent / "templates" / "slurm" / "job.sh.j2"
        ).read_text()
        assert "#SBATCH --output={{ output_dir }}/madengine-{{ model_name }}" in tpl
