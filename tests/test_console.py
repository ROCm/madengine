"""Test the console module.

This module tests the console module.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import subprocess
import typing
# third-party modules
import pytest
import typing_extensions
# project modules
from madengine.core import console


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
        assert obj.sh("echo $MAD_ENGINE", env={"MAD_ENGINE": "MAD Engine"}) == "MAD Engine"

    def test_sh_verbose(self):
        obj = console.Console(shellVerbose=False)
        assert obj.sh("echo MAD Engine") == "MAD Engine"

    def test_sh_ignore_stderr(self):
        obj = console.Console(shellVerbose=False)
        assert obj.sh("echo fail 1>&2 | xargs echo success", ignore_stderr=True) == "success"
