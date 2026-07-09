"""Unit tests for the bare-metal (conda) execution backend."""

import os
from unittest import mock

import pytest

from madengine.execution.conda_env import (
    CondaEnvManager,
    bootstrap_micromamba,
    resolve_conda_env_name,
    resolve_environment_file,
    resolve_rocm_index_url,
)
from madengine.execution.run_reporting import (
    determine_status,
    extract_performance_from_log,
)
from madengine.orchestration.run_orchestrator import RunOrchestrator

# ---- Deployment target inference ----


class TestBareMetalInference:
    """_infer_deployment_target routes bare_metal before slurm/k8s/local."""

    def _make_orch(self):
        args = mock.MagicMock()
        args.additional_context = "{}"
        args.output = "perf.csv"
        return RunOrchestrator(args)

    def test_bare_metal_key_selects_bare_metal(self):
        orch = self._make_orch()
        assert orch._infer_deployment_target({"bare_metal": {}}) == "bare_metal"

    def test_bare_metal_wins_over_slurm(self):
        orch = self._make_orch()
        assert (
            orch._infer_deployment_target({"bare_metal": {}, "slurm": {}})
            == "bare_metal"
        )

    def test_no_key_is_local(self):
        orch = self._make_orch()
        assert orch._infer_deployment_target({}) == "local"


# ---- Conda env name resolution ----


class TestResolveCondaEnvName:
    def test_config_overrides_model(self):
        assert resolve_conda_env_name({"conda_env": "m"}, {"conda_env": "c"}) == "c"

    def test_model_used_when_no_config(self):
        assert resolve_conda_env_name({"conda_env": "m"}, {}) == "m"

    def test_derived_from_name(self):
        assert resolve_conda_env_name({"name": "foo/bar"}, {}) == "mad_foo_bar"


# ---- CondaEnvManager command construction ----


class TestCondaEnvManager:
    def _mgr(self, bm_config=None):
        console = mock.MagicMock()
        mgr = CondaEnvManager(console=console, bm_config=bm_config or {})
        mgr._conda_bin = "/opt/conda/bin/conda"  # bypass PATH detection
        return mgr, console

    def test_conda_run_prefix(self):
        mgr, _ = self._mgr()
        prefix = mgr.conda_run_prefix("myenv")
        assert "run -n myenv" in prefix
        assert "--no-capture-output" in prefix

    def test_env_exists_matches_name(self):
        mgr, console = self._mgr()
        console.sh.return_value = (
            "# conda envs\nbase   *  /opt/conda\nmyenv     /opt/conda/envs/myenv"
        )
        assert mgr.env_exists("myenv") is True
        assert mgr.env_exists("absent") is False

    def test_create_with_python_version(self):
        mgr, console = self._mgr()
        # env does not exist
        console.sh.return_value = "# conda envs\nbase * /opt/conda"
        name = mgr.create_or_update(
            {"name": "m", "conda_env": "e", "python_version": "3.11"}
        )
        assert name == "e"
        # Find the create command among the calls.
        create_calls = [
            c.args[0] for c in console.sh.call_args_list if "create" in c.args[0]
        ]
        assert any("python=3.11" in cmd and "-n e" in cmd for cmd in create_calls)

    def test_reuse_existing_env_skips_create(self):
        mgr, console = self._mgr(bm_config={"reuse_env": True})
        console.sh.return_value = "myenv  /opt/conda/envs/myenv"
        mgr.create_or_update({"name": "m", "conda_env": "myenv"})
        create_calls = [
            c.args[0]
            for c in console.sh.call_args_list
            if "conda create" in c.args[0] or "env create" in c.args[0]
        ]
        assert create_calls == []


# ---- Vendor-aware dependency file resolution ----


class TestResolveEnvironmentFile:
    def test_prefers_amd_variant(self, tmp_path):
        base = tmp_path / "environment.yml"
        base.write_text("name: base\n")
        amd = tmp_path / "environment.amd.yml"
        amd.write_text("name: amd\n")
        assert resolve_environment_file(str(base), "AMD") == str(amd)

    def test_prefers_nvidia_variant(self, tmp_path):
        base = tmp_path / "environment.yml"
        base.write_text("name: base\n")
        nvidia = tmp_path / "environment.nvidia.yml"
        nvidia.write_text("name: nvidia\n")
        assert resolve_environment_file(str(base), "nvidia") == str(nvidia)

    def test_falls_back_to_base_when_no_variant(self, tmp_path):
        base = tmp_path / "environment.yml"
        base.write_text("name: base\n")
        assert resolve_environment_file(str(base), "AMD") == str(base)

    def test_unknown_vendor_returns_base(self, tmp_path):
        base = tmp_path / "environment.yml"
        base.write_text("name: base\n")
        # Even if an amd variant exists, an unknown vendor keeps the base.
        (tmp_path / "environment.amd.yml").write_text("name: amd\n")
        assert resolve_environment_file(str(base), "") == str(base)

    def test_setup_script_suffix(self, tmp_path):
        base = tmp_path / "setup_script.sh"
        base.write_text("echo base\n")
        amd = tmp_path / "setup_script.amd.sh"
        amd.write_text("echo amd\n")
        assert resolve_environment_file(str(base), "AMD") == str(amd)

    def test_create_or_update_uses_vendor_variant(self, tmp_path):
        base = tmp_path / "environment.yml"
        base.write_text("name: base\n")
        amd = tmp_path / "environment.amd.yml"
        amd.write_text("name: amd\n")
        console = mock.MagicMock()
        console.sh.return_value = "# conda envs\nbase * /opt/conda"
        mgr = CondaEnvManager(
            console=console,
            bm_config={"conda_env": "e", "environment_file": str(base)},
            gpu_vendor="AMD",
        )
        mgr._conda_bin = "/opt/conda/bin/conda"
        mgr._dep_hash_path = lambda env_name: str(tmp_path / f"hash_{env_name}")
        mgr.create_or_update({"name": "m"})
        env_calls = [
            c.args[0] for c in console.sh.call_args_list if "env create" in c.args[0]
        ]
        assert any(str(amd) in cmd for cmd in env_calls)


# ---- ROCm index URL resolution ----


class TestResolveRocmIndexUrl:
    def test_explicit_url_returned_unchanged(self):
        assert (
            resolve_rocm_index_url("https://example.com/wheels/", "gfx942")
            == "https://example.com/wheels/"
        )

    def test_auto_resolves_from_gfx_arch(self):
        assert (
            resolve_rocm_index_url("auto", "gfx942")
            == "https://rocm.nightlies.amd.com/v2/gfx942/"
        )

    def test_empty_resolves_from_gfx_arch(self):
        assert (
            resolve_rocm_index_url("", "gfx90a")
            == "https://rocm.nightlies.amd.com/v2/gfx90a/"
        )

    def test_auto_without_arch_raises(self):
        with pytest.raises(RuntimeError):
            resolve_rocm_index_url("auto", "")


# ---- ROCm wheel install ----


class TestInstallRocmWheels:
    def _mgr(self, bm_config=None, gpu_arch="gfx942"):
        console = mock.MagicMock()
        mgr = CondaEnvManager(
            console=console, bm_config=bm_config or {}, gpu_arch=gpu_arch
        )
        mgr._conda_bin = "/opt/conda/bin/conda"
        return mgr, console

    def test_installs_default_packages_from_auto_index(self):
        mgr, console = self._mgr()
        mgr._install_rocm_wheels("e", {"enabled": True}, timeout=None)
        cmds = [c.args[0] for c in console.sh.call_args_list]
        assert any(
            "pip install --index-url" in c
            and "rocm[libraries,devel]" in c
            and "gfx942" in c
            for c in cmds
        )
        # torch not requested -> no torch install
        assert not any("torch torchvision" in c for c in cmds)

    def test_installs_torch_when_requested(self):
        mgr, console = self._mgr()
        mgr._install_rocm_wheels("e", {"enabled": True, "torch": True}, timeout=None)
        cmds = [c.args[0] for c in console.sh.call_args_list]
        assert any("torch torchvision" in c and "gfx942" in c for c in cmds)

    def test_custom_packages_and_explicit_index(self):
        mgr, console = self._mgr(gpu_arch=None)
        mgr._install_rocm_wheels(
            "e",
            {"enabled": True, "index_url": "https://x/whl/", "packages": ["rocm"]},
            timeout=None,
        )
        cmds = [c.args[0] for c in console.sh.call_args_list]
        assert any("https://x/whl/" in c and " rocm" in c for c in cmds)

    def test_create_or_update_installs_rocm_on_fresh_env(self):
        mgr, console = self._mgr(
            bm_config={"conda_env": "e", "rocm": {"enabled": True}}
        )
        console.sh.return_value = "# conda envs\nbase * /opt/conda"
        mgr.create_or_update({"name": "m"})
        cmds = [c.args[0] for c in console.sh.call_args_list]
        assert any("pip install --index-url" in c and "gfx942" in c for c in cmds)

    def test_create_or_update_skips_rocm_on_reuse(self):
        mgr, console = self._mgr(
            bm_config={"conda_env": "e", "rocm": {"enabled": True}, "reuse_env": True}
        )
        console.sh.return_value = "e  /opt/conda/envs/e"
        mgr.create_or_update({"name": "m"})
        cmds = [c.args[0] for c in console.sh.call_args_list]
        assert not any("pip install --index-url" in c for c in cmds)


# ---- requirements_file install ----


class TestRequirementsFile:
    def _mgr(self, bm_config):
        console = mock.MagicMock()
        console.sh.return_value = "# conda envs\nbase * /opt/conda"
        mgr = CondaEnvManager(console=console, bm_config=bm_config)
        mgr._conda_bin = "/opt/conda/bin/conda"
        return mgr, console

    def test_pip_install_requirements(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("numpy\n")
        mgr, console = self._mgr({"conda_env": "e", "requirements_file": str(req)})
        with mock.patch.object(mgr, "_dep_hash_path", return_value=str(tmp_path / "h")):
            mgr.create_or_update({"name": "m"})
        cmds = [c.args[0] for c in console.sh.call_args_list]
        assert any(f"pip install -r {str(req)}" in c for c in cmds)

    def test_missing_requirements_file_raises(self):
        mgr, _ = self._mgr(
            {"conda_env": "e", "requirements_file": "/nonexistent/requirements.txt"}
        )
        with pytest.raises(RuntimeError):
            mgr.create_or_update({"name": "m"})


# ---- Dependency-hash reuse invalidation ----


class TestDepHashInvalidation:
    def _mgr(self, bm_config, tmp_path):
        console = mock.MagicMock()
        mgr = CondaEnvManager(console=console, bm_config=bm_config)
        mgr._conda_bin = "/opt/conda/bin/conda"
        # Route the stamp file into tmp so tests don't touch the real cache.
        mgr._dep_hash_path = lambda env_name: str(tmp_path / f"hash_{env_name}")
        return mgr, console

    def test_env_update_on_changed_environment_file(self, tmp_path):
        envf = tmp_path / "environment.yml"
        envf.write_text("name: base\n")
        mgr, console = self._mgr(
            {"conda_env": "e", "environment_file": str(envf), "reuse_env": True},
            tmp_path,
        )
        # First run: env does not exist -> creates and stamps hash.
        console.sh.return_value = "# conda envs\nbase * /opt/conda"
        mgr.create_or_update({"name": "m"})

        # Second run: env now exists; unchanged file -> reuse (no env command).
        console.sh.reset_mock()
        console.sh.return_value = "e  /opt/conda/envs/e"
        mgr.create_or_update({"name": "m"})
        cmds = [c.args[0] for c in console.sh.call_args_list]
        assert not any("env update" in c or "env create" in c for c in cmds)

        # Third run: change file content -> env update forced despite reuse.
        envf.write_text("name: base\ndependencies: [numpy]\n")
        console.sh.reset_mock()
        console.sh.return_value = "e  /opt/conda/envs/e"
        mgr.create_or_update({"name": "m"})
        cmds = [c.args[0] for c in console.sh.call_args_list]
        assert any("env update" in c for c in cmds)


# ---- micromamba bootstrap ----


class TestBootstrapMicromamba:
    def test_returns_cached_binary_without_download(self, tmp_path):
        cache = tmp_path / ".cache" / "madengine" / "micromamba"
        cache.mkdir(parents=True)
        binary = cache / "micromamba"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        with mock.patch(
            "madengine.execution.conda_env.os.path.expanduser",
            return_value=str(tmp_path),
        ), mock.patch("madengine.execution.conda_env.urllib.request.urlretrieve") as dl:
            result = bootstrap_micromamba()
        assert result == str(binary)
        dl.assert_not_called()

    def test_detect_conda_bin_bootstraps_when_nothing_found(self, tmp_path):
        console = mock.MagicMock()
        mgr = CondaEnvManager(console=console, bm_config={})
        with mock.patch(
            "madengine.execution.conda_env.shutil.which", return_value=None
        ), mock.patch(
            "madengine.execution.conda_env.bootstrap_micromamba",
            return_value="/cache/micromamba",
        ) as boot:
            assert mgr.detect_conda_bin() == "/cache/micromamba"
        boot.assert_called_once()


# ---- Opt-in env teardown ----


class TestCleanupEnv:
    def _runner(self, bm_config):
        from madengine.execution.bare_metal_runner import BareMetalRunner

        runner = BareMetalRunner.__new__(BareMetalRunner)
        runner.additional_context = {"bare_metal": bm_config}
        runner.bm_config = bm_config
        runner.perf_csv_path = "perf.csv"
        runner.rich_console = mock.MagicMock()
        runner.conda = mock.MagicMock()
        runner.ensure_perf_csv_exists = mock.MagicMock()
        runner._create_run_details = mock.MagicMock(return_value={})
        return runner

    def test_cleanup_env_removes_env(self):
        runner = self._runner({"conda_env": "e", "cleanup_env": True})
        with mock.patch("madengine.execution.bare_metal_runner.write_perf_records"):
            runner._record({"name": "m"}, {}, {"status": "SUCCESS"}, 1)
        runner.conda.remove.assert_called_once_with("e")

    def test_no_cleanup_by_default(self):
        runner = self._runner({"conda_env": "e"})
        with mock.patch("madengine.execution.bare_metal_runner.write_perf_records"):
            runner._record({"name": "m"}, {}, {"status": "SUCCESS"}, 1)
        runner.conda.remove.assert_not_called()


# ---- Preflight GPU validation in bare-metal build ----


class TestBareMetalBuildPreflight:
    def test_gpu_validation_failure_surfaces_as_build_error(self):
        from madengine.core.errors import BuildError
        from madengine.orchestration.build_orchestrator import BuildOrchestrator
        from madengine.utils.gpu_validator import (
            GPUInstallationError,
            GPUValidationResult,
            GPUVendor,
        )

        orch = BuildOrchestrator.__new__(BuildOrchestrator)
        orch.additional_context = {"bare_metal": {}}
        orch.args = mock.MagicMock()
        orch.console = mock.MagicMock()
        orch.rich_console = mock.MagicMock()
        orch.context = mock.MagicMock()
        orch.context.init_gpu_context = mock.MagicMock()
        orch.context.ctx = {"gpu_vendor": "AMD"}
        orch.context._rocm_path = "/bogus/rocm"

        bad = GPUValidationResult(is_valid=False, vendor=GPUVendor.AMD)
        bad.issues.append("ROCm not found")
        with mock.patch(
            "madengine.utils.gpu_validator.validate_gpu_installation",
            side_effect=GPUInstallationError(bad),
        ):
            with pytest.raises(BuildError):
                orch._execute_bare_metal_build()


# ---- Performance extraction ----


class TestExtractPerformance:
    def test_canonical_pattern(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text("some output\nperformance: 123.5 samples_per_second\n")
        perf, metric = extract_performance_from_log(str(log))
        assert perf == "123.5"
        assert metric == "samples_per_second"

    def test_missing_file(self):
        perf, metric = extract_performance_from_log("/nonexistent/x.log")
        assert perf is None and metric is None


class TestDetermineStatus:
    def test_success_with_performance(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text("performance: 5 samples_per_second\n")
        assert determine_status(str(log), "5", {}, {}) == "SUCCESS"

    def test_failure_no_performance(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text("nothing useful\n")
        assert determine_status(str(log), None, {}, {}) == "FAILURE"

    def test_failure_on_error_pattern(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text("performance: 5 x\nTraceback (most recent call last)\n")
        assert determine_status(str(log), "5", {}, {}) == "FAILURE"
