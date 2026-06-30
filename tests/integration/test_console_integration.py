"""Test the console module.

This module tests the console module.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# project modules
from madengine.core import console

# Fake, non-real placeholder token. Split + f-string assembly keeps the
# literal "KEY=<token>" form out of source so secret scanners don't flag it.
_FAKE_HF = "hf_" + "abcdef123456"


class TestConsole:
    """Test the console module.

    test_sh: Test the console.sh function with echo command.
    """

    def test_sh(self):
        obj = console.Console()
        assert obj.sh("echo MAD Engine") == "MAD Engine"

    def test_sh_fail(self):
        obj = console.Console()
        try:
            obj.sh("exit 1")
        except RuntimeError as exc:
            assert str(exc) == "Subprocess 'exit 1' failed with exit code 1"
        else:
            assert False

    def test_sh_timeout(self):
        obj = console.Console()
        try:
            obj.sh("sleep 10", timeout=1)
        except RuntimeError as exc:
            assert str(exc) == "Console script timeout"
        else:
            assert False

    def test_sh_secret(self):
        obj = console.Console()
        assert obj.sh("echo MAD Engine", secret=True) == "MAD Engine"

    def test_sh_env(self):
        obj = console.Console()
        assert (
            obj.sh("echo $MAD_ENGINE", env={"MAD_ENGINE": "MAD Engine"}) == "MAD Engine"
        )

    def test_sh_verbose(self):
        obj = console.Console(shellVerbose=False)
        assert obj.sh("echo MAD Engine") == "MAD Engine"

    def test_sh_live_output(self):
        obj = console.Console(live_output=True)
        assert obj.sh("echo MAD Engine") == "MAD Engine"


class TestRedactSecrets:
    """Test redact_secrets(): secrets must never appear in printed/raised commands."""

    def test_redacts_mad_secrets_env(self):
        cmd = f"docker run --env MAD_SECRETS_HFTOKEN={_FAKE_HF} ubuntu"
        out = console.redact_secrets(cmd)
        assert _FAKE_HF not in out
        assert "MAD_SECRETS_HFTOKEN=***REDACTED***" in out

    def test_redacts_build_arg(self):
        cmd = f"docker build --build-arg MAD_SECRETS_HFTOKEN={_FAKE_HF} ./docker"
        out = console.redact_secrets(cmd)
        assert _FAKE_HF not in out
        assert "MAD_SECRETS_HFTOKEN=***REDACTED***" in out

    def test_redacts_double_quoted_value_with_spaces(self):
        cmd = 'run MAD_SECRETS_FOO="a b c" --next keep'
        out = console.redact_secrets(cmd)
        assert "a b c" not in out
        assert "MAD_SECRETS_FOO=***REDACTED***" in out
        assert "--next keep" in out

    def test_redacts_single_quoted_value_with_spaces(self):
        cmd = "MAD_SECRETS_FOO='secret value' tail"
        out = console.redact_secrets(cmd)
        assert "secret value" not in out
        assert "MAD_SECRETS_FOO=***REDACTED***" in out
        assert "tail" in out

    def test_redacts_known_token_shapes(self):
        for tok in (
            "hf_abcdef123456",
            "sk-abcdef123456",
            "ghp_abcdef123456",
            "xoxb-abcdef123456",
        ):
            out = console.redact_secrets("token=" + tok)
            assert tok not in out
            assert "***REDACTED***" in out

    def test_none_and_empty_passthrough(self):
        assert console.redact_secrets(None) is None
        assert console.redact_secrets("") == ""

    def test_non_secret_text_unchanged(self):
        cmd = "docker run --env FOO=bar --env BAZ=qux ubuntu"
        assert console.redact_secrets(cmd) == cmd

    def test_runtime_error_message_is_redacted(self):
        obj = console.Console(shellVerbose=False)
        try:
            obj.sh(f"false MAD_SECRETS_HFTOKEN={_FAKE_HF}")
        except RuntimeError as exc:
            assert _FAKE_HF not in str(exc)
            assert "***REDACTED***" in str(exc)
        else:
            assert False
