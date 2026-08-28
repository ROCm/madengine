"""Unit tests for madengine.core.image_digest.

Covers digest extraction from `docker push` / `docker image inspect` output and
construction of pinned `repo@sha256:...` references.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import pytest

from madengine.core.errors import ConfigurationError
from madengine.core.image_digest import (
    build_pinned_reference,
    parse_push_digest,
    parse_repo_digest,
    resolve_pinned_image,
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
        assert (
            parse_push_digest("The push refers to repository [x]\nlayer: Pushed\n")
            is None
        )

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
        assert (
            build_pinned_reference(f"myorg/ci:mymodel", DIGEST) == f"myorg/ci@{DIGEST}"
        )

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


class TestResolvePinnedImage:
    """resolve_pinned_image applies the --require-pinned-image policy."""

    def test_disabled_returns_image_unchanged_without_digest(self):
        assert (
            resolve_pinned_image("myorg/ci:mymodel", None, False) == "myorg/ci:mymodel"
        )

    def test_disabled_returns_image_unchanged_even_with_digest(self):
        # Default behaviour must not change for existing users: the digest rides
        # along in the manifest but is never used unless enforcement is on.
        assert (
            resolve_pinned_image("myorg/ci:mymodel", DIGEST, False)
            == "myorg/ci:mymodel"
        )

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
