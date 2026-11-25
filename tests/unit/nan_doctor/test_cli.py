"""`madengine nan ...`: the command group and its dispatcher."""

import pytest
from typer.testing import CliRunner

from madengine.cli.app import app
from madengine.nan_doctor import cli

runner = CliRunner()


def test_every_dispatcher_command_is_registered_in_the_cli():
    result = runner.invoke(app, ["nan", "--help"])
    assert result.exit_code == 0
    for name in cli.COMMANDS:
        assert name in result.output


def test_options_pass_through_to_the_command_parser():
    result = runner.invoke(app, ["nan", "intake", "--schema"])
    assert result.exit_code == 0 and "preference | values" in result.output


def test_command_help_comes_from_the_commands_own_parser():
    result = runner.invoke(app, ["nan", "replay", "--help"])
    assert (
        result.exit_code == 0
        and "--out-dir" in result.output
        and "Replay one NaN capsule" in result.output
    )


def test_exit_codes_propagate():
    assert runner.invoke(app, ["nan", "gpu-check"]).exit_code in (0, 4, 5)
    assert runner.invoke(app, ["nan", "intake", "/no/such/file.json"]).exit_code == 2


def test_dispatcher_rejects_unknown_commands(capsys):
    assert cli.main(["bogus"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_dispatcher_without_arguments_prints_usage(capsys):
    assert cli.main([]) == 2
    assert "usage: madengine nan" in capsys.readouterr().err


@pytest.mark.parametrize("name", sorted(cli.COMMANDS))
def test_every_command_module_imports_without_torch(name):
    import subprocess
    import sys

    code = "import sys, madengine.nan_doctor.%s; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code % cli.COMMANDS[name][0]], check=True)
