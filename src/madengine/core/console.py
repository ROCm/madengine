#!/usr/bin/env python3
"""Module to run console commands.

This module provides a class to run console commands.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import subprocess
import typing
import re


# Mask secret values (e.g. MAD_SECRETS_HFTOKEN) before printing/raising commands,
# so they don't leak into SLURM/run logs. The executed command is unchanged.
_REDACTED = "***REDACTED***"

# MAD_SECRETS*=value in any form (-e / --env / --build-arg / bare); key kept, value masked.
# The value may be unquoted, single-/double-quoted (possibly containing spaces),
# or empty; optional whitespace around '=' is also tolerated (e.g. "FOO= value",
# "FOO =  value"); the whole value (including surrounding quotes) is masked.
_SECRET_ASSIGN_RE = re.compile(
    r"""(MAD_SECRETS[A-Za-z0-9_]*\s*=\s*)("[^"]*"|'[^']*'|\S*)"""
)

# Fallback: known credential token shapes.
_TOKEN_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{6,}"),
    re.compile(r"sk-[A-Za-z0-9._-]{6,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{6,}"),
    re.compile(r"xox[abprs]-[A-Za-z0-9-]{6,}"),
)


def redact_secrets(text: typing.Optional[str]) -> typing.Optional[str]:
    """Mask secret values in a command/message string for safe logging.

    Args:
        text (Optional[str]): The text to scrub (may be None or empty).

    Returns:
        Optional[str]: The text with secret values replaced by a redaction
            marker, or the original value unchanged if it is None/empty.
    """
    if not text:
        return text
    text = _SECRET_ASSIGN_RE.sub(lambda m: m.group(1) + _REDACTED, text)
    for pattern in _TOKEN_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    return text


class Console:
    """Class to run console commands.

    Attributes:
        shellVerbose (bool): The shell verbose flag.
        live_output (bool): The live output flag.
    """

    def __init__(self, shellVerbose: bool = True, live_output: bool = False) -> None:
        """Constructor of the Console class.

        Args:
            shellVerbose (bool): The shell verbose flag.
            live_output (bool): The live output flag.
        """
        self.shellVerbose = shellVerbose
        self.live_output = live_output

    def _highlight_docker_operations(self, command: str) -> str:
        """Highlight docker push/pull/build/run operations for better visibility.

        Args:
            command (str): The command to potentially highlight.

        Returns:
            str: The highlighted command if it's a docker operation.
        """
        # Check if this is a docker operation
        docker_push_pattern = r"^docker\s+push\s+"
        docker_pull_pattern = r"^docker\s+pull\s+"
        docker_build_pattern = r"^docker\s+build\s+"
        docker_run_pattern = r"^docker\s+run\s+"

        if re.match(docker_push_pattern, command, re.IGNORECASE):
            return f"\n{'='*80}\n🚀 DOCKER PUSH OPERATION: {command}\n{'='*80}"
        elif re.match(docker_pull_pattern, command, re.IGNORECASE):
            return f"\n{'='*80}\n📥 DOCKER PULL OPERATION: {command}\n{'='*80}"
        elif re.match(docker_build_pattern, command, re.IGNORECASE):
            return f"\n{'='*80}\n🔨 DOCKER BUILD OPERATION: {command}\n{'='*80}"
        elif re.match(docker_run_pattern, command, re.IGNORECASE):
            return f"\n{'='*80}\n🏃 DOCKER RUN OPERATION: {command}\n{'='*80}"

        return command

    def _show_docker_completion(self, command: str, success: bool = True) -> None:
        """Show completion message for docker operations.

        Args:
            command (str): The command that was executed.
            success (bool): Whether the operation was successful.
        """
        docker_push_pattern = r"^docker\s+push\s+"
        docker_pull_pattern = r"^docker\s+pull\s+"
        docker_build_pattern = r"^docker\s+build\s+"
        docker_run_pattern = r"^docker\s+run\s+"

        if re.match(docker_push_pattern, command, re.IGNORECASE):
            if success:
                print(f"✅ DOCKER PUSH COMPLETED SUCCESSFULLY")
                print(f"{'='*80}\n")
            else:
                print(f"❌ DOCKER PUSH FAILED")
                print(f"{'='*80}\n")
        elif re.match(docker_pull_pattern, command, re.IGNORECASE):
            if success:
                print(f"✅ DOCKER PULL COMPLETED SUCCESSFULLY")
                print(f"{'='*80}\n")
            else:
                print(f"❌ DOCKER PULL FAILED")
                print(f"{'='*80}\n")
        elif re.match(docker_build_pattern, command, re.IGNORECASE):
            if success:
                print(f"✅ DOCKER BUILD COMPLETED SUCCESSFULLY")
                print(f"{'='*80}\n")
            else:
                print(f"❌ DOCKER BUILD FAILED")
                print(f"{'='*80}\n")
        elif re.match(docker_run_pattern, command, re.IGNORECASE):
            if success:
                print(f"✅ DOCKER RUN COMPLETED SUCCESSFULLY")
                print(f"{'='*80}\n")
            else:
                print(f"❌ DOCKER RUN FAILED")
                print(f"{'='*80}\n")

    def sh(
        self,
        command: str,
        canFail: bool = False,
        timeout: int = 60,
        secret: bool = False,
        prefix: str = "",
        env: typing.Optional[typing.Dict[str, str]] = None,
    ) -> str:
        """Run shell command.

        Args:
            command (str): The shell command.
            canFail (bool): The flag to allow failure.
            timeout (int): The timeout in seconds.
            secret (bool): The flag to hide the command.
            prefix (str): The prefix of the output.
            env (typing.Optional[typing.Dict[str, str]]): The environment variables.

        Returns:
            str: The output of the shell command.

        Raises:
            RuntimeError: If the shell command fails.
        """
        # Print the command if shellVerbose is True
        if self.shellVerbose and not secret:
            highlighted_command = self._highlight_docker_operations(command)
            print("> " + redact_secrets(highlighted_command), flush=True)

        # Run the shell command
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            universal_newlines=True,
            bufsize=1,
            env=env,
        )

        # Get the output of the shell command, and check for failure, and return the output.
        try:
            if not self.live_output:
                outs, errs = proc.communicate(timeout=timeout)
            else:
                try:
                    outs = []
                    for stdout_line in iter(
                        lambda: proc.stdout.readline()
                        .encode("utf-8", errors="replace")
                        .decode("utf-8", errors="replace"),
                        "",
                    ):
                        print(prefix + stdout_line, end="")
                        outs.append(stdout_line)
                    outs = "".join(outs)
                finally:
                    # Ensure all pipes are properly closed
                    if proc.stdout and not proc.stdout.closed:
                        proc.stdout.close()
                    if proc.stdin and not proc.stdin.closed:
                        proc.stdin.close()
                proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            # Wait for process to finish after kill and clean up pipes
            try:
                proc.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                # Force terminate if still not dead
                proc.terminate()
                proc.communicate()
            raise RuntimeError("Console script timeout") from exc
        finally:
            # Final cleanup: ensure all pipes are closed regardless of success/failure
            # This prevents ResourceWarning about unclosed files
            try:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
            except (OSError, ValueError):
                # Expected errors during cleanup - stdin may already be closed
                pass
            try:
                if proc.stdout and not proc.stdout.closed:
                    proc.stdout.close()
            except (OSError, ValueError):
                # Expected errors during cleanup - stdout may already be closed
                pass

        # Check for failure
        success = proc.returncode == 0

        # When output is captured rather than streamed it is discarded on
        # failure, and the RuntimeError below carries only the command and the
        # exit code. Echo it so the log records why the command actually failed.
        if not success and not canFail and not secret and not self.live_output and outs:
            print(redact_secrets(outs), flush=True)

        # Show docker operation completion status
        if not secret:
            self._show_docker_completion(command, success)

        if proc.returncode != 0:
            if not canFail:
                if not secret:
                    raise RuntimeError(
                        "Subprocess '"
                        + redact_secrets(command)
                        + "' failed with exit code "
                        + str(proc.returncode)
                    )
                else:
                    raise RuntimeError(
                        "Subprocess '***HIDDEN COMMAND***' failed with exit code "
                        + str(proc.returncode)
                    )

        # Return the output
        return outs.strip()
