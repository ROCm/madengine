#!/usr/bin/env python3
"""Module to run docker commands.

This module provides a class to run commands inside docker.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import os
import typing
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
        """
        # initialize variables
        self.docker_sha = None
        self.keep_alive = keep_alive
        cwd = os.getcwd()
        self.console = console
        self.userid = self.console.sh("id -u")
        self.groupid = self.console.sh("id -g")

        # check if container name exists
        container_name_exists = self.console.sh(
            "docker container ps -a | grep " + container_name + " | wc -l"
        )
        # If container name exists, print message then stop/remove the container.
        if container_name_exists != "0":
            msg = (
                "Container with name, "
                + container_name
                + " already exists. "
                 + "Going to stop (docker stop --time=1 SHA) and remove this (docker rm -f SHA) to proceed..."
            )
            print(msg)

            # Stop/remove by SHA only (ignore failures in case it's already stopped/removed).
            existing_shas = self.console.sh(
                'docker ps -aqf "name=' + container_name + '"',
                canFail=True,
            )
            if existing_shas:
                for sha in existing_shas.splitlines():
                    self.console.sh("docker stop --timeout=1 " + sha, canFail=True)
                    self.console.sh("docker rm -f " + sha, canFail=True)

        # run docker command
        command = (
            "docker run -t -d -u "
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
            "docker ps -aqf 'name=" + container_name + "' "
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
        # run as root!
        return self.console.sh(
            "docker exec " + self.docker_sha + ' bash -c "' + command + '"',
            timeout=timeout,
            secret=secret,
        )

    def __del__(self):
        """Destructor of the Docker class."""
        # stop and remove docker container, if not keep_alive and docker sha exists, else print docker sha.
        if not self.keep_alive and self.docker_sha:
            self.console.sh("docker stop --time=1 " + self.docker_sha)
            self.console.sh("docker rm -f " + self.docker_sha)
            return

        # print docker sha
        if self.docker_sha:
            print("==========================================")
            print("Keeping docker alive, sha :", self.docker_sha)
            print(
                "Open a bash session in container : ",
                "docker exec -it " + self.docker_sha + " bash",
            )
            print("Stop container : ", "docker stop --time=1 " + self.docker_sha)
            print("Remove container : ", "docker rm -f " + self.docker_sha)
            print("==========================================")
