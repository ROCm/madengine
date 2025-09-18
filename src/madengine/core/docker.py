#!/usr/bin/env python3
"""Module to run docker commands.

This module provides a class to run commands inside docker.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import os
import typing
import subprocess

# user-defined modules
from madengine.core.console import Console


class Docker:
    """Class to run commands inside docker.

    Attributes:
        docker_sha (str): The docker sha.
        keep_alive (bool): The keep alive flag.
        console (Console): The console object.
        userid (str): The user id.
        groupid (str): The group id.
    """

    @staticmethod
    def is_valid_cmd(cmd) -> bool:
        """
        Check if the given command is a valid container runtime by running '<cmd> container ls'.
        This is necessary because a container runtime might be installed but not properly configured.
        """

        try:
            result = subprocess.run(
                [cmd, 'container', 'ls'],
                capture_output=True,
                text=True,
                timeout=10
            )

            return result.returncode == 0

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            return False

    @staticmethod
    def get_container_cmd() -> str:
        """
        Check which container runtime is available, docker or podman.
        Use docker by default if both are available.

        Returns:
            str: The available container command ('docker' or 'podman').

        Raises:
            RuntimeError: If neither docker nor podman is available.
        """

        if Docker.is_valid_cmd('docker'):
            return 'docker'

        if Docker.is_valid_cmd('podman'):
            return 'podman'

        raise RuntimeError("Neither docker nor podman is available on this system.")

    def __init__(
        self,
        image: str,
        container_name: str,
        dockerOpts: str,
        mounts: typing.Optional[typing.List] = None,
        envVars: typing.Optional[typing.Dict] = None,
        keep_alive: bool = False,
        console: Console = Console(),
    ) -> None:
        """Constructor of the Docker class.

        Args:
            image (str): The docker image.
            container_name (str): The container name.
            dockerOpts (str): The docker options.
            mounts (list): The list of mounts.
            envVars (dict): The dictionary of environment variables.
            keep_alive (bool): The keep alive flag.
            console (Console): The console object.

        Raises:
            RuntimeError: If the container name already exists or no container runtime available.
        """
        container_cmd = self.get_container_cmd()

        # initialize variables
        self.docker_sha = None
        self.keep_alive = keep_alive
        cwd = os.getcwd()
        self.console = console
        self.userid = self.console.sh("id -u")
        self.groupid = self.console.sh("id -g")

        # check if container name exists
        container_name_exists = self.console.sh(
            f"{container_cmd} container ps -a | grep " + container_name + " | wc -l"
        )
        # if container name exists, raise error.
        if container_name_exists != "0":
            raise RuntimeError(
                "Container with name, "
                + container_name
                + " already exists. "
                + f"Please stop ({container_cmd} stop --time=1 SHA) and remove this ({container_cmd} rm -f SHA) to proceed..."
            )

        # run docker command
        command = (
            f"{container_cmd} run -t -d -u "
            + self.userid
            + ":"
            + self.groupid
            + " "
            + dockerOpts
            + " "
        )

        # add mounts
        if mounts is not None:
            for mount in mounts:
                command += "-v " + mount + ":" + mount + " "

        # add current working directory
        command += "-v " + cwd + ":/myworkspace/ "

        # add envVars
        if envVars is not None:
            for evar in envVars.keys():
                command += "-e " + evar + "=" + envVars[evar] + " "

        command += "--workdir /myworkspace/ "
        command += "--name " + container_name + " "
        command += image + " "

        # hack to keep docker open
        command += "cat "
        self.console.sh(command)

        # find container sha
        self.docker_sha = self.console.sh(
            f"{container_cmd} ps -aqf 'name=" + container_name + "' "
        )

    def sh(
            self,
            command: str,
            timeout: int=60,
            secret: bool=False
        ) -> str:
        """Run shell command inside docker.

        Args:
            command (str): The shell command.
            timeout (int): The timeout in seconds.
            secret (bool): The flag to hide the command.

        Returns:
            str: The output of the shell command.
        """
        container_cmd = self.get_container_cmd()
        # run as root!
        return self.console.sh(
            f"{container_cmd} exec " + self.docker_sha + ' bash -c "' + command + '"',
            timeout=timeout,
            secret=secret,
        )

    def __del__(self):
        """Destructor of the Docker class."""
        container_cmd = self.get_container_cmd()
        # stop and remove docker container, if not keep_alive and docker sha exists, else print docker sha.
        if not self.keep_alive and self.docker_sha:
            self.console.sh(f"{container_cmd} stop --time=1 " + self.docker_sha)
            self.console.sh(f"{container_cmd} rm -f " + self.docker_sha)
            return

        # print docker sha
        if self.docker_sha:
            print("==========================================")
            print(f"Keeping {container_cmd} alive, sha :", self.docker_sha)
            print(
                "Open a bash session in container : ",
                f"{container_cmd} exec -it " + self.docker_sha + " bash",
            )
            print("Stop container : ", f"{container_cmd} stop --time=1 " + self.docker_sha)
            print("Remove container : ", f"{container_cmd} rm -f " + self.docker_sha)
            print("==========================================")
