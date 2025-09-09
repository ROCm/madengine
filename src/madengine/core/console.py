#!/usr/bin/env python3
"""Module to run console commands.

This module provides a class to run console commands.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import subprocess
import typing
# third-party modules
import typing_extensions


class Console:
    """Class to run console commands.
    
    Attributes:
        shellVerbose (bool): The shell verbose flag.
        live_output (bool): The live output flag.
    """
    def __init__(
            self, 
            shellVerbose: bool=True, 
            live_output: bool=False
        ) -> None:
        """Constructor of the Console class.
        
        Args:
            shellVerbose (bool): The shell verbose flag.
            live_output (bool): The live output flag.
        """
        self.shellVerbose = shellVerbose
        self.live_output = live_output

    def sh(
            self, 
            command: str, 
            canFail: bool=False, 
            timeout: int=60, 
            secret: bool=False, 
            prefix: str="", 
            env: typing.Optional[typing.Dict[str, str]]=None,
            ignore_stderr: bool=False
        ) -> str:
        """Run shell command.
        
        Args:
            command (str): The shell command.
            canFail (bool): The flag to allow failure.
            timeout (int): The timeout in seconds.
            secret (bool): The flag to hide the command.
            prefix (str): The prefix of the output.
            env (typing_extensions.TypedDict): The environment variables.
            ignore_stderr (bool): Don't include stderr in command's output.
        
        Returns:
            str: The output of the shell command.

        Raises:
            RuntimeError: If the shell command fails.
        """
        # Print the command if shellVerbose is True
        if self.shellVerbose and not secret:
            print("> " + command, flush=True)

        # Run the shell command
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None if ignore_stderr else subprocess.STDOUT,
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
                outs = []
                for stdout_line in iter(lambda: proc.stdout.readline().encode('utf-8', errors='replace').decode('utf-8', errors='replace'), ""):
                    print(prefix + stdout_line, end="")
                    outs.append(stdout_line)
                outs = "".join(outs)
                proc.stdout.close()
                proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            raise RuntimeError("Console script timeout") from exc
        
        # Check for failure
        if proc.returncode != 0:
            if not canFail:
                if not secret:
                    raise RuntimeError(
                        "Subprocess '"
                        + command
                        + "' failed with exit code "
                        + str(proc.returncode)
                    )
                else:
                    raise RuntimeError(
                        "Subprocess '"
                        + secret
                        + "' failed with exit code "
                        + str(proc.returncode)
                    )
                
        # Return the output
        return outs.strip()
