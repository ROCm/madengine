"""Unit tests for cross-node local-image reconciliation and content staleness.

Covers two correctness fixes in ContainerRunner._ensure_local_image_available:

* Bug 1 (cross-node identity): a worker that already holds a stale image under
  the same tag must reload rank 0's tar so every node runs the identical image.
* Bug 2 (content staleness): a tag reused for a different build (changed
  ``mad.build_fingerprint``) must be rebuilt instead of silently reused.
"""

import os
from unittest.mock import MagicMock

import pytest

from madengine.execution.container_runner import ContainerRunner


def _make_runner(context=None):
    return ContainerRunner(context=context, console=MagicMock())


def _stub_io(runner):
    """Replace the docker-touching helpers with mocks for orchestration tests."""
    runner._build_or_pull_local_image = MagicMock(name="build_or_pull")
    runner._load_local_image_from_tar = MagicMock(name="load_tar")
    runner._save_local_image_to_tar = MagicMock(name="save_tar")
    runner.rich_console = MagicMock()
    return runner


class TestParseGoImageId:
    def test_bare_go_line_returns_none_id(self):
        matched, image_id = ContainerRunner._parse_go_image_id("GO TOK 3", "GO TOK 3")
        assert matched is True
        assert image_id is None

    def test_dash_payload_maps_to_none(self):
        matched, image_id = ContainerRunner._parse_go_image_id("GO TOK 3 -", "GO TOK 3")
        assert matched is True
        assert image_id is None

    def test_image_id_extracted(self):
        matched, image_id = ContainerRunner._parse_go_image_id(
            "GO TOK 3 sha256:abc", "GO TOK 3"
        )
        assert matched is True
        assert image_id == "sha256:abc"

    def test_wrong_rank_does_not_match(self):
        matched, image_id = ContainerRunner._parse_go_image_id("GO TOK 30", "GO TOK 3")
        assert matched is False
        assert image_id is None

    def test_unrelated_line_does_not_match(self):
        matched, image_id = ContainerRunner._parse_go_image_id("nope", "GO TOK 3")
        assert matched is False


class TestBuildFingerprint:
    def test_empty_when_no_dockerfile(self):
        runner = _make_runner()
        assert runner._build_fingerprint({}) == ""
        assert runner._build_fingerprint({"dockerfile": "N/A (local image mode)"}) == ""

    def test_changes_with_dockerfile_content(self, tmp_path):
        runner = _make_runner()
        df = tmp_path / "Dockerfile"
        df.write_text("FROM base\nARG RCCL_COMMIT=aaaa\n")
        fp1 = runner._build_fingerprint({"dockerfile": str(df)})
        df.write_text("FROM base\nARG RCCL_COMMIT=bbbb\n")
        fp2 = runner._build_fingerprint({"dockerfile": str(df)})
        assert fp1 and fp2 and fp1 != fp2

    def test_changes_with_build_args(self, tmp_path):
        df = tmp_path / "Dockerfile"
        df.write_text("FROM base\n")
        ctx_a = MagicMock()
        ctx_a.ctx = {"docker_build_arg": {"RCCL_COMMIT": "aaaa"}}
        ctx_b = MagicMock()
        ctx_b.ctx = {"docker_build_arg": {"RCCL_COMMIT": "bbbb"}}
        fp_a = _make_runner(ctx_a)._build_fingerprint({"dockerfile": str(df)})
        fp_b = _make_runner(ctx_b)._build_fingerprint({"dockerfile": str(df)})
        assert fp_a != fp_b


class TestImageIsStale:
    def test_no_fingerprint_never_stale(self):
        runner = _make_runner()
        runner._image_label = MagicMock(return_value="anything")
        assert runner._image_is_stale("img", "") is False

    def test_missing_label_not_stale(self):
        runner = _make_runner()
        runner._image_label = MagicMock(return_value=None)
        assert runner._image_is_stale("img", "fp1") is False

    def test_matching_label_not_stale(self):
        runner = _make_runner()
        runner._image_label = MagicMock(return_value="fp1")
        assert runner._image_is_stale("img", "fp1") is False

    def test_differing_label_is_stale(self):
        runner = _make_runner()
        runner._image_label = MagicMock(return_value="old")
        assert runner._image_is_stale("img", "fp1") is True


class TestPrimaryEnsure:
    def test_builds_and_saves_tar_when_missing(self, monkeypatch):
        monkeypatch.setenv("NODE_RANK", "0")
        runner = _stub_io(_make_runner())
        runner._get_local_image_tar_path = MagicMock(return_value="/shared/img.tar")
        runner._build_fingerprint = MagicMock(return_value="fp1")
        runner._local_image_exists = MagicMock(return_value=False)
        runner._local_image_id = MagicMock(return_value="sha256:master")
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        sync = MagicMock(side_effect=lambda run_image, master_image_id: master_image_id)
        runner._sync_after_local_image_ready = sync

        runner._ensure_local_image_available("img", {"dockerfile": "D"}, {})

        runner._build_or_pull_local_image.assert_called_once()
        runner._save_local_image_to_tar.assert_called_once_with("img", "/shared/img.tar")
        # The ensured image ID is what gets broadcast to workers.
        assert sync.call_args.kwargs["master_image_id"] == "sha256:master"

    def test_stale_image_is_rebuilt_and_tar_refreshed(self, monkeypatch):
        monkeypatch.setenv("NODE_RANK", "0")
        runner = _stub_io(_make_runner())
        runner._get_local_image_tar_path = MagicMock(return_value="/shared/img.tar")
        runner._build_fingerprint = MagicMock(return_value="fp_new")
        runner._local_image_exists = MagicMock(return_value=True)
        runner._image_is_stale = MagicMock(return_value=True)
        runner._local_image_id = MagicMock(return_value="sha256:new")
        monkeypatch.setattr(os.path, "exists", lambda p: True)  # stale tar present
        runner._sync_after_local_image_ready = MagicMock(
            side_effect=lambda run_image, master_image_id: master_image_id
        )

        runner._ensure_local_image_available("img", {"dockerfile": "D"}, {})

        runner._build_or_pull_local_image.assert_called_once()
        # Existing (stale) tar must be overwritten with the rebuilt image.
        runner._save_local_image_to_tar.assert_called_once_with("img", "/shared/img.tar")

    def test_fresh_image_not_rebuilt_and_tar_kept(self, monkeypatch):
        monkeypatch.setenv("NODE_RANK", "0")
        runner = _stub_io(_make_runner())
        runner._get_local_image_tar_path = MagicMock(return_value="/shared/img.tar")
        runner._build_fingerprint = MagicMock(return_value="fp1")
        runner._local_image_exists = MagicMock(return_value=True)
        runner._image_is_stale = MagicMock(return_value=False)
        runner._local_image_id = MagicMock(return_value="sha256:keep")
        monkeypatch.setattr(os.path, "exists", lambda p: True)  # tar already present
        runner._sync_after_local_image_ready = MagicMock(
            side_effect=lambda run_image, master_image_id: master_image_id
        )

        runner._ensure_local_image_available("img", {"dockerfile": "D"}, {})

        runner._build_or_pull_local_image.assert_not_called()
        runner._save_local_image_to_tar.assert_not_called()


class TestWorkerReconcile:
    def _worker(self, monkeypatch, tar_path, master_id, local_ids, tar_on_disk=True):
        monkeypatch.setenv("NODE_RANK", "1")
        runner = _stub_io(_make_runner())
        runner._get_local_image_tar_path = MagicMock(return_value=tar_path)
        runner._build_fingerprint = MagicMock(return_value="fp1")
        runner._local_image_id = MagicMock(side_effect=list(local_ids))
        runner._sync_after_local_image_ready = MagicMock(return_value=master_id)
        monkeypatch.setattr(os.path, "exists", lambda p: tar_on_disk)
        return runner

    def test_mismatch_triggers_tar_reload(self, monkeypatch):
        # local stale before reload, equals master after reload
        runner = self._worker(
            monkeypatch, "/shared/img.tar", "sha256:master",
            local_ids=["sha256:old", "sha256:master"],
        )
        runner._ensure_local_image_available("img", {"dockerfile": "D"}, {})
        runner._load_local_image_from_tar.assert_called_once_with("img", "/shared/img.tar")

    def test_match_skips_reload(self, monkeypatch):
        runner = self._worker(
            monkeypatch, "/shared/img.tar", "sha256:master",
            local_ids=["sha256:master"],
        )
        runner._ensure_local_image_available("img", {"dockerfile": "D"}, {})
        runner._load_local_image_from_tar.assert_not_called()
        runner._build_or_pull_local_image.assert_not_called()

    def test_persistent_mismatch_raises(self, monkeypatch):
        runner = self._worker(
            monkeypatch, "/shared/img.tar", "sha256:master",
            local_ids=["sha256:old", "sha256:still_old"],
        )
        with pytest.raises(RuntimeError, match="mismatch persists"):
            runner._ensure_local_image_available("img", {"dockerfile": "D"}, {})

    def test_missing_tar_when_reload_needed_raises(self, monkeypatch):
        runner = self._worker(
            monkeypatch, "/shared/img.tar", "sha256:master",
            local_ids=["sha256:old"], tar_on_disk=False,
        )
        with pytest.raises(RuntimeError, match="did not produce image tar"):
            runner._ensure_local_image_available("img", {"dockerfile": "D"}, {})

    def test_no_tar_missing_image_builds(self, monkeypatch):
        runner = self._worker(
            monkeypatch, None, "sha256:master", local_ids=[None],
        )
        runner._image_is_stale = MagicMock(return_value=False)
        runner._ensure_local_image_available("img", {"dockerfile": "D"}, {})
        runner._build_or_pull_local_image.assert_called_once()

    def test_no_tar_stale_image_rebuilds(self, monkeypatch):
        runner = self._worker(
            monkeypatch, None, "sha256:master", local_ids=["sha256:x"],
        )
        runner._image_is_stale = MagicMock(return_value=True)
        runner._ensure_local_image_available("img", {"dockerfile": "D"}, {})
        runner._build_or_pull_local_image.assert_called_once()

    def test_no_tar_fresh_image_kept(self, monkeypatch):
        runner = self._worker(
            monkeypatch, None, "sha256:master", local_ids=["sha256:x"],
        )
        runner._image_is_stale = MagicMock(return_value=False)
        runner._ensure_local_image_available("img", {"dockerfile": "D"}, {})
        runner._build_or_pull_local_image.assert_not_called()
