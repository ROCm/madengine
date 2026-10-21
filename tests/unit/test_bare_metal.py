"""Unit tests for the bare-metal (conda) execution backend."""

import json
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
    """platform selects the conda backend. A bare_metal key does not."""

    def _make_orch(self):
        args = mock.MagicMock()
        args.additional_context = "{}"
        args.output = "perf.csv"
        return RunOrchestrator(args)

    def test_platform_bare_metal_selects_conda(self):
        orch = self._make_orch()
        assert (
            orch._infer_deployment_target({"platform": "bare_metal"})
            == "bare_metal"
        )

    def test_platform_bare_metal_reads_conda_block(self):
        orch = self._make_orch()
        assert (
            orch._infer_deployment_target(
                {
                    "platform": "bare_metal",
                    "conda": {"conda_bin": "/opt/conda/bin/conda"},
                }
            )
            == "bare_metal"
        )

    def test_omitted_platform_is_local_docker(self):
        orch = self._make_orch()
        assert orch._infer_deployment_target({}) == "local"

    def test_explicit_docker_with_slurm_stays_slurm(self):
        orch = self._make_orch()
        assert (
            orch._infer_deployment_target(
                {"platform": "docker", "slurm": {"nodes": 1}}
            )
            == "slurm"
        )

    def test_bare_metal_key_alone_is_rejected(self):
        from madengine.core.errors import ConfigurationError

        orch = self._make_orch()
        with pytest.raises(ConfigurationError, match="platform"):
            orch._infer_deployment_target({"bare_metal": {}})

    def test_platform_bare_metal_with_slurm_is_rejected(self):
        from madengine.core.errors import ConfigurationError

        orch = self._make_orch()
        with pytest.raises(ConfigurationError, match="slurm"):
            orch._infer_deployment_target(
                {"platform": "bare_metal", "slurm": {}}
            )

    def test_conda_block_without_bare_metal_platform_is_rejected(self):
        from madengine.core.errors import ConfigurationError

        orch = self._make_orch()
        with pytest.raises(ConfigurationError, match="conda"):
            orch._infer_deployment_target({"conda": {"conda_env": "e"}})

    def test_unknown_platform_is_rejected(self):
        from madengine.core.errors import ConfigurationError

        orch = self._make_orch()
        with pytest.raises(ConfigurationError, match="podman"):
            orch._infer_deployment_target({"platform": "podman"})


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

    def test_conda_run_prefix(self, tmp_path):
        mgr, _ = self._mgr(bm_config={"env_root": str(tmp_path)})
        prefix = mgr.conda_run_prefix("myenv")
        assert f"run -p {tmp_path / 'myenv'}" in prefix
        assert "--no-capture-output" in prefix
        assert "docker" not in prefix
        assert " -n " not in prefix

    def test_env_exists_matches_prefix(self, tmp_path):
        mgr, console = self._mgr(bm_config={"env_root": str(tmp_path)})
        assert mgr.env_exists("myenv") is False
        os.makedirs(os.path.join(mgr.env_prefix("myenv"), "conda-meta"))
        assert mgr.env_exists("myenv") is True
        assert mgr.env_exists("absent") is False
        console.sh.assert_not_called()

    def test_create_with_python_version(self, tmp_path):
        mgr, console = self._mgr(bm_config={"env_root": str(tmp_path)})
        name = mgr.create_or_update(
            {"name": "m", "conda_env": "e", "python_version": "3.11"}
        )
        assert name == "e"
        create_calls = [
            c.args[0] for c in console.sh.call_args_list if "create" in c.args[0]
        ]
        prefix = str(tmp_path / "e")
        assert any(
            "python=3.11" in cmd
            and f"-p {prefix}" in cmd
            and "-c conda-forge" in cmd
            and "-n " not in cmd
            for cmd in create_calls
        )

    def test_reuse_existing_env_skips_create(self, tmp_path):
        mgr, console = self._mgr(
            bm_config={"reuse_env": True, "env_root": str(tmp_path)}
        )
        os.makedirs(os.path.join(mgr.env_prefix("myenv"), "conda-meta"))
        mgr.create_or_update({"name": "m", "conda_env": "myenv"})
        create_calls = [
            c.args[0]
            for c in console.sh.call_args_list
            if "create" in c.args[0] or "env update" in c.args[0]
        ]
        assert create_calls == []

    def test_reuse_false_removes_then_creates(self, tmp_path):
        mgr, console = self._mgr(
            bm_config={"reuse_env": False, "env_root": str(tmp_path)}
        )
        os.makedirs(os.path.join(mgr.env_prefix("e"), "conda-meta"))
        mgr.create_or_update({"name": "m", "conda_env": "e", "python_version": "3.11"})
        cmds = [c.args[0] for c in console.sh.call_args_list]
        assert any(
            "env remove" in cmd and f"-p {tmp_path / 'e'}" in cmd for cmd in cmds
        )
        assert any(
            "create -y -c conda-forge -p" in cmd and "python=3.11" in cmd
            for cmd in cmds
        )
        assert cmds.index(next(c for c in cmds if "env remove" in c)) < cmds.index(
            next(c for c in cmds if "create -y -c conda-forge -p" in c)
        )
        assert not any("create -y -n" in cmd or "create -n" in cmd for cmd in cmds)


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

    def test_create_or_update_installs_rocm_on_fresh_env(self, tmp_path):
        mgr, console = self._mgr(
            bm_config={
                "conda_env": "e",
                "rocm": {"enabled": True},
                "env_root": str(tmp_path / "envs"),
            }
        )
        mgr._dep_hash_path = lambda env_name: str(tmp_path / f"hash_{env_name}")
        mgr.create_or_update({"name": "m"})
        cmds = [c.args[0] for c in console.sh.call_args_list]
        assert any("pip install --index-url" in c and "gfx942" in c for c in cmds)

    def test_create_or_update_skips_rocm_when_fingerprint_matches(self, tmp_path):
        rocm = {"enabled": True}
        mgr, console = self._mgr(
            bm_config={
                "conda_env": "e",
                "rocm": rocm,
                "reuse_env": True,
                "env_root": str(tmp_path / "envs"),
            }
        )
        mgr._dep_hash_path = lambda env_name: str(tmp_path / f"hash_{env_name}")
        os.makedirs(os.path.join(mgr.env_prefix("e"), "conda-meta"))
        mgr._write_dep_hash(
            "e", mgr._compute_dep_hash([], extra=mgr._rocm_fingerprint(rocm))
        )
        mgr.create_or_update({"name": "m"})
        cmds = [c.args[0] for c in console.sh.call_args_list]
        assert not any("pip install --index-url" in c for c in cmds)

    def test_create_or_update_reinstalls_rocm_when_index_changes(self, tmp_path):
        mgr, console = self._mgr(
            bm_config={
                "conda_env": "e",
                "rocm": {"enabled": True, "index_url": "https://old/whl/"},
                "reuse_env": True,
                "env_root": str(tmp_path / "envs"),
            }
        )
        mgr._dep_hash_path = lambda env_name: str(tmp_path / f"hash_{env_name}")
        os.makedirs(os.path.join(mgr.env_prefix("e"), "conda-meta"))
        mgr._write_dep_hash(
            "e",
            mgr._compute_dep_hash(
                [],
                extra=mgr._rocm_fingerprint(
                    {"enabled": True, "index_url": "https://old/whl/"}
                ),
            ),
        )
        mgr.bm_config["rocm"] = {"enabled": True, "index_url": "https://new/whl/"}
        mgr.create_or_update({"name": "m"})
        cmds = [c.args[0] for c in console.sh.call_args_list]
        assert any("https://new/whl/" in c for c in cmds)


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
        mgr, console = self._mgr(
            {
                "conda_env": "e",
                "requirements_file": str(req),
                "env_root": str(tmp_path / "envs"),
            }
        )
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
            {
                "conda_env": "e",
                "environment_file": str(envf),
                "reuse_env": True,
                "env_root": str(tmp_path / "envs"),
            },
            tmp_path,
        )
        # First run: env does not exist -> creates and stamps hash.
        console.sh.return_value = "# conda envs\nbase * /opt/conda"
        mgr.create_or_update({"name": "m"})

        # Second run: prefix exists; unchanged file -> reuse (no env command).
        os.makedirs(os.path.join(mgr.env_prefix("e"), "conda-meta"))
        console.sh.reset_mock()
        mgr.create_or_update({"name": "m"})
        cmds = [c.args[0] for c in console.sh.call_args_list]
        assert not any("env update" in c or "env create" in c for c in cmds)

        # Third run: change file content -> env update forced despite reuse.
        envf.write_text("name: base\ndependencies: [numpy]\n")
        console.sh.reset_mock()
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

    def test_detect_conda_bin_does_not_download(self):
        console = mock.MagicMock()
        mgr = CondaEnvManager(console=console, bm_config={})
        with mock.patch(
            "madengine.execution.conda_env.shutil.which", return_value=None
        ), mock.patch("madengine.execution.conda_env.bootstrap_micromamba") as boot:
            with pytest.raises(RuntimeError, match="does not download"):
                mgr.detect_conda_bin()
        boot.assert_not_called()

    def test_rejects_non_executable_conda_bin(self, tmp_path):
        path = tmp_path / "conda"
        path.write_text("#!/bin/sh\n")
        path.chmod(0o644)
        mgr = CondaEnvManager(
            console=mock.MagicMock(), bm_config={"conda_bin": str(path)}
        )
        with pytest.raises(RuntimeError, match="not executable"):
            mgr.detect_conda_bin()

    def test_resolves_conda_bin_command_name(self):
        mgr = CondaEnvManager(
            console=mock.MagicMock(), bm_config={"conda_bin": "conda"}
        )
        with mock.patch(
            "madengine.execution.conda_env.shutil.which", return_value="/usr/bin/conda"
        ):
            assert mgr.detect_conda_bin() == "/usr/bin/conda"

    def test_dep_hash_path_cannot_escape_cache(self, tmp_path):
        mgr = CondaEnvManager(console=mock.MagicMock(), bm_config={})
        with mock.patch(
            "madengine.execution.conda_env.os.path.expanduser",
            return_value=str(tmp_path),
        ):
            path = mgr._dep_hash_path("../../etc/passwd")
        cache = os.path.join(str(tmp_path), ".cache", "madengine", "envhash")
        assert os.path.commonpath([cache, path]) == cache
        assert os.sep not in os.path.basename(path)


# ---- Opt-in env teardown ----


class TestCleanupEnv:
    def _runner(self, bm_config):
        from madengine.execution.bare_metal_runner import BareMetalRunner

        runner = BareMetalRunner.__new__(BareMetalRunner)
        runner.additional_context = {"platform": "bare_metal", "conda": bm_config}
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

    def test_record_passes_script_directory(self):
        runner = self._runner({"conda_env": "e"})
        with mock.patch(
            "madengine.execution.bare_metal_runner.write_perf_records"
        ) as write:
            runner._record(
                {"name": "m"},
                {},
                {"status": "SUCCESS"},
                1,
                model_dir="/models/dummy_bare_metal",
            )
        assert write.call_args.kwargs["model_dir"] == "/models/dummy_bare_metal"


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
        orch.additional_context = {"platform": "bare_metal"}
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


class TestRemoveDepHash:
    def test_remove_deletes_stamp_so_rebuild_is_not_reused(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("numpy\n")
        console = mock.MagicMock()
        mgr = CondaEnvManager(
            console=console,
            bm_config={
                "conda_env": "e",
                "env_root": str(tmp_path / "envs"),
                "requirements_file": str(req),
                "reuse_env": True,
            },
        )
        mgr._conda_bin = "/opt/conda/bin/conda"
        mgr._dep_hash_path = lambda env_name: str(tmp_path / f"hash_{env_name}")
        os.makedirs(os.path.join(mgr.env_prefix("e"), "conda-meta"))
        mgr._write_dep_hash("e", mgr._compute_dep_hash([str(req)]))

        mgr.remove("e")
        assert not os.path.exists(tmp_path / "hash_e")

        # Simulate a prefix that was recreated outside this process.
        os.makedirs(os.path.join(mgr.env_prefix("e"), "conda-meta"), exist_ok=True)
        console.sh.reset_mock()
        mgr.create_or_update({"name": "m"})
        cmds = [c.args[0] for c in console.sh.call_args_list]
        assert any("pip install -r" in cmd for cmd in cmds)


class TestGpuVisibility:
    def _runner(self, vendor):
        from madengine.execution.bare_metal_runner import BareMetalRunner

        runner = BareMetalRunner.__new__(BareMetalRunner)
        runner.context = mock.Mock()
        runner.context.ctx = {"gpu_vendor": vendor}
        return runner

    def test_all_gpus_keeps_host_mask(self, monkeypatch):
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2")
        runner = self._runner("AMD")
        assert runner._gpu_env(-1) == {}

    def test_slices_existing_amd_mask(self, monkeypatch):
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,3,5")
        monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        runner = self._runner("AMD")
        env = runner._gpu_env(1)
        assert env == {
            "HIP_VISIBLE_DEVICES": "2",
            "ROCR_VISIBLE_DEVICES": "2",
        }

    def test_no_mask_uses_first_n_devices(self, monkeypatch):
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
        monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        runner = self._runner("AMD")
        assert runner._gpu_env(2) == {
            "HIP_VISIBLE_DEVICES": "0,1",
            "ROCR_VISIBLE_DEVICES": "0,1",
        }

    def test_nvidia_slices_cuda_mask(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
        runner = self._runner("NVIDIA")
        assert runner._gpu_env(2)["CUDA_VISIBLE_DEVICES"] == "4,5"


class TestCredentialsAndCommand:
    def test_model_cred_is_exported(self):
        from madengine.execution.bare_metal_runner import BareMetalRunner

        runner = BareMetalRunner.__new__(BareMetalRunner)
        runner.context = mock.Mock()
        runner.context.ctx = {}
        runner.additional_context = {}
        runner.credentials = {"hf": {"token": "secret"}}
        env = runner._build_run_env({"name": "m", "cred": "hf"}, -1)
        assert env["hf_TOKEN"] == "secret"

    def test_model_command_is_conda_run_not_docker(self, tmp_path):
        from madengine.execution.bare_metal_runner import BareMetalRunner

        runner = BareMetalRunner.__new__(BareMetalRunner)
        runner.context = mock.Mock()
        runner.context.ctx = {
            "gpu_vendor": "AMD",
            "pre_scripts": [],
            "post_scripts": [],
            "docker_env_vars": {},
        }
        runner.additional_context = {
            "platform": "bare_metal",
            "conda": {"conda_env": "e"},
        }
        runner.bm_config = runner.additional_context["conda"]
        runner.console = mock.Mock()
        runner.console.sh.return_value = "host"
        runner.rich_console = mock.Mock()
        runner.live_output = True
        runner.credentials = None
        runner.conda = mock.Mock()
        runner.conda.conda_run_prefix.return_value = (
            "/opt/conda/bin/conda run -p /tmp/envs/e --no-capture-output"
        )
        runner._record = mock.Mock()
        script = tmp_path / "run.sh"
        script.write_text("#!/bin/bash\necho performance: 1 samples_per_second\n")
        captured = {}

        def sh_in_dir(cmd, env, working_dir, timeout, can_fail):
            captured["cmd"] = cmd
            return "performance: 1 samples_per_second\n"

        runner._sh_in_dir = sh_in_dir
        log_path = tmp_path / "run.log"
        with mock.patch(
            "madengine.execution.bare_metal_runner.make_run_log_file_path",
            return_value=str(log_path),
        ), mock.patch(
            "madengine.execution.bare_metal_runner.resolve_runtime_gpus",
            return_value=1,
        ):
            runner.run_model(
                {"name": "m", "scripts": str(script), "n_gpus": "1", "args": ""},
                build_info={},
                timeout=30,
            )
        assert captured["cmd"].startswith("/opt/conda/bin/conda run -p /tmp/envs/e")
        assert "docker" not in captured["cmd"]


class TestManifestBareMetalOverride:
    def test_runtime_bare_metal_wins_over_manifest(self, tmp_path):
        from madengine.execution.bare_metal_runner import BareMetalRunner

        manifest = {
            "built_images": {"m": {}},
            "built_models": {"m": {"name": "m", "scripts": "run.sh"}},
            "context": {
                "gpu_vendor": "AMD",
                "platform": "bare_metal",
                "conda": {
                    "conda_env": "from_build",
                    "cleanup_env": False,
                    "env_root": "/from/build",
                },
            },
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))
        runner = BareMetalRunner.__new__(BareMetalRunner)
        runner.additional_context = {
            "platform": "bare_metal",
            "conda": {"cleanup_env": True, "conda_env": "from_run"},
        }
        runner.bm_config = dict(runner.additional_context["conda"])
        runner.conda = mock.Mock()
        runner.rich_console = mock.Mock()
        runner.run_model = mock.Mock(
            return_value={"status": "SUCCESS", "performance": "1", "test_duration": 1}
        )
        runner.run_models_from_manifest(str(path))
        assert runner.bm_config["cleanup_env"] is True
        assert runner.bm_config["conda_env"] == "from_run"
        assert runner.bm_config["env_root"] == "/from/build"
        assert runner.conda.bm_config is runner.bm_config


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
