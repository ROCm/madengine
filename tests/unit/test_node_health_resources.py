"""The SLURM node health check judges what a job actually needs from a node.

It used to collect `amd-smi list`, never read it, and estimate GPU memory as
45 GB per process whose name matched ray/vllm. It passed one node four
times while its docker disk was full ("no space left on device" during the
pull), and another while other processes held GPU memory ("The memory
capacity is unbalanced").
"""

import io
import json
import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console as RichConsole

from madengine.core.console import manifest_safe_context, redact_secrets
from madengine.deployment.slurm_node_selector import (
    NodeHealth,
    SlurmNodeSelector,
    parse_node_resources,
)

GIB = 1 << 30
RUN = "madengine.deployment.slurm_node_selector.subprocess.run"


def _selector(**kw):
    sel = SlurmNodeSelector(console=RichConsole(file=io.StringIO()), timeout=5, **kw)
    sel.partition = "amd-rccl"
    return sel


def _probe_output(resources, processes="NO_PROCESSES"):
    return (
        "===GPU_INFO===\nGPU 0\n===END_GPU_INFO===\n"
        f"===PROCESSES===\n{processes}\n===END_PROCESSES===\n"
        f"===RESOURCES===\n{resources}\n===END_RESOURCES===\n"
    )


def _health(sel, resources, processes="NO_PROCESSES"):
    with patch(RUN) as run:
        run.return_value = MagicMock(returncode=0, stdout=_probe_output(resources, processes), stderr="")
        return sel.check_node_health("n1"), run.call_args[0][0][-1]


def _vram(*used_gb, total_gb=192):
    return "\n".join(f"VRAM {int(u * GIB)} {int(total_gb * GIB)}" for u in used_gb)


class TestParse:
    def test_reads_every_field(self):
        r = parse_node_resources("VRAM 1 2\nVRAM 3 4\nDOCKER_AVAIL 99\nIMAGE_PRESENT\n")
        assert r == {"vram": [(1, 2), (3, 4)], "docker_avail": 99, "image_present": True}

    def test_ignores_noise_and_missing_sections(self):
        assert parse_node_resources("VRAM x y\nDOCKER_AVAIL \ngarbage") == {
            "vram": [], "docker_avail": None, "image_present": False}
        assert parse_node_resources(None)["vram"] == []


class TestGpuMemory:
    def test_idle_gpus_are_clean(self):
        status, _ = _health(_selector(), _vram(0.3, 0.3, 0.3, 0.3))
        assert status.health == NodeHealth.CLEAN
        assert status.error_message is None

    def test_one_occupied_gpu_is_enough(self):
        status, _ = _health(_selector(), _vram(0.3, 120, 0.3, 0.3))
        assert status.health == NodeHealth.DIRTY
        assert "GPU1 120 GB" in status.error_message

    def test_occupied_by_a_process_the_name_filter_misses(self):
        # SGLang held the memory; no ray/vllm process was running.
        status, _ = _health(_selector(), _vram(80, 80, 80, 80), processes="NO_PROCESSES")
        assert status.health == NodeHealth.DIRTY

    def test_without_sysfs_the_old_estimate_still_applies(self):
        status, _ = _health(_selector(), "", processes="a vllm\nb vllm")
        assert status.health == NodeHealth.DIRTY
        status, _ = _health(_selector(), "")
        assert status.health == NodeHealth.CLEAN


class TestDockerDisk:
    def test_too_little_space_for_the_image_is_dirty(self):
        sel = _selector(image="rocm/mad-private:x", image_size_bytes=40 * GIB)
        status, _ = _health(sel, _vram(0.3) + f"\nDOCKER_AVAIL {2 * GIB}")
        assert status.health == NodeHealth.DIRTY
        assert "docker has 2 GB free, the image needs 40 GB" in status.error_message

    def test_image_already_on_the_node_needs_no_space(self):
        sel = _selector(image="rocm/mad-private:x", image_size_bytes=40 * GIB)
        status, _ = _health(sel, _vram(0.3) + f"\nDOCKER_AVAIL {2 * GIB}\nIMAGE_PRESENT")
        assert status.health == NodeHealth.CLEAN

    def test_enough_space_is_clean(self):
        sel = _selector(image="rocm/mad-private:x", image_size_bytes=40 * GIB)
        status, _ = _health(sel, _vram(0.3) + f"\nDOCKER_AVAIL {500 * GIB}")
        assert status.health == NodeHealth.CLEAN
        assert status.docker_avail_gb == pytest.approx(500)

    def test_unknown_size_or_unknown_space_skips_the_disk_check(self):
        status, _ = _health(_selector(), _vram(0.3) + f"\nDOCKER_AVAIL {1 * GIB}")
        assert status.health == NodeHealth.CLEAN
        sel = _selector(image="rocm/mad-private:x", image_size_bytes=40 * GIB)
        status, _ = _health(sel, _vram(0.3))
        assert status.health == NodeHealth.CLEAN

    def test_image_name_is_put_into_the_probe_only_when_plain(self):
        _, script = _health(_selector(image="rocm/mad-private:ci-a_b.c-1", image_size_bytes=1), "")
        assert 'docker image inspect "rocm/mad-private:ci-a_b.c-1"' in script
        _, script = _health(_selector(image="x; rm -rf /", image_size_bytes=1), "")
        assert "rm -rf" not in script


class TestProbeScriptRuns:
    def test_resources_section_reads_amdgpu_sysfs(self, tmp_path):
        for i, (used, total) in enumerate([(300 << 20, 192 * GIB), (130 * GIB, 192 * GIB)]):
            d = tmp_path / f"card{i}" / "device"
            d.mkdir(parents=True)
            (d / "mem_info_vram_used").write_text(f"{used}\n")
            (d / "mem_info_vram_total").write_text(f"{total}\n")
        (tmp_path / "card9" / "device").mkdir(parents=True)  # a non-amdgpu card
        _, script = _health(_selector(), "")
        script = script.replace("/sys/class/drm/card*/device", f"{tmp_path}/card*/device")
        script = script.replace('GPU_INFO=$(amd-smi list 2>/dev/null || echo "GPU_CHECK_FAILED")', "GPU_INFO=x")
        env = {"PATH": f"{tmp_path}/bin:/usr/bin:/bin"}
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "amd-smi").write_text("#!/bin/sh\necho gpu\n")
        (tmp_path / "bin" / "amd-smi").chmod(0o755)
        out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env).stdout
        section = out.split("===RESOURCES===")[1].split("===END_RESOURCES===")[0]
        assert parse_node_resources(section)["vram"] == [(300 << 20, 192 * GIB), (130 * GIB, 192 * GIB)]


class TestManifestCarriesNoSecrets:
    def test_secret_keys_are_dropped_everything_else_kept(self):
        ctx = {
            "docker_env_vars": {"MAD_SECRETS_HFTOKEN": "hf_abcdefghijklmnopqrstuvwxyz", "NCCL_DEBUG": "WARN"},
            "docker_build_arg": {"MAD_SECRETS_HFTOKEN": "hf_abcdefghijklmnopqrstuvwxyz", "ARCH": "gfx942"},
            "gpu_vendor": "AMD",
        }
        safe = manifest_safe_context(ctx)
        assert safe["docker_env_vars"] == {"NCCL_DEBUG": "WARN"}
        assert safe["docker_build_arg"] == {"ARCH": "gfx942"}
        assert safe["gpu_vendor"] == "AMD"
        assert "hf_" not in json.dumps(safe)
        # The live context is untouched: the build itself still needs the value.
        assert ctx["docker_build_arg"]["MAD_SECRETS_HFTOKEN"].startswith("hf_")

    def test_build_command_is_stored_masked(self):
        cmd = "docker build --build-arg MAD_SECRETS_HFTOKEN=hf_abcdefghijklmnopqrstuvwxyz ./docker"
        assert "hf_abc" not in redact_secrets(cmd)
        assert "MAD_SECRETS_HFTOKEN=" in redact_secrets(cmd)


class TestLargestPullImage:
    def test_picks_the_largest_by_the_name_nodes_pull(self):
        from madengine.deployment.slurm import SlurmDeployment

        dep = SlurmDeployment.__new__(SlurmDeployment)
        dep.manifest = {"built_images": {
            "a": {"docker_image": "a", "registry_image": "rocm/mad-private:a", "image_size_bytes": 10},
            "b": {"docker_image": "b", "image_size_bytes": 30},
            "c": {"docker_image": "c"},
        }}
        assert dep._largest_pull_image() == ("b", 30)
        dep.manifest = {"built_images": {"c": {"docker_image": "c"}}}
        assert dep._largest_pull_image() == (None, None)


class TestTableShowsTheEvidence:
    """A run passed a node with an empty Notes column; the log could
    not say whether its disk was fine, the image was already there, or the free
    space was never read."""

    def _render(self, sel, statuses):
        buf = io.StringIO()
        sel.console = RichConsole(file=buf, width=200)
        sel._display_status_table(statuses)
        return buf.getvalue()

    def test_free_space_image_presence_and_what_it_was_judged_against(self):
        sel = _selector(image="rocm/mad-private:x", image_size_bytes=27 * GIB)
        ok, _ = _health(sel, _vram(0.3) + f"\nDOCKER_AVAIL {300 * GIB}")
        present, _ = _health(sel, _vram(0.3) + f"\nDOCKER_AVAIL {1 * GIB}\nIMAGE_PRESENT")
        unread, _ = _health(sel, _vram(0.3))
        out = self._render(sel, [ok, present, unread])
        assert "300 GB" in out
        assert "1 GB (image present)" in out
        assert "?" in out
        assert "Disk judged against rocm/mad-private:x (27 GB)" in out

    def test_says_when_the_disk_was_not_checked(self):
        sel = _selector()
        st, _ = _health(sel, _vram(0.3) + f"\nDOCKER_AVAIL {5 * GIB}")
        assert "Disk not checked: the manifest records no image size" in self._render(sel, [st])
