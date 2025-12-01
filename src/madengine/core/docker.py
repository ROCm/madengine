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

        Raises:
            RuntimeError: If the container name already exists.
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
        # if container name exists, raise error.
        if container_name_exists != "0":
            raise RuntimeError(
                "Container with name, "
                + container_name
                + " already exists. "
                + "Please stop (docker stop --time=1 SHA) and remove this (docker rm -f SHA) to proceed..."
            )

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

    @staticmethod
    def save_image_to_tar(image_name: str, tar_path: str, console: Console = None) -> bool:
        """Save Docker image to tar file.

        Args:
            image_name: Name of the docker image to save
            tar_path: Output path for tar file
            console: Console object for shell operations (optional)

        Returns:
            bool: True if successful, False otherwise
        """
        if console is None:
            console = Console()

        try:
            # Ensure parent directory exists
            tar_dir = os.path.dirname(tar_path)
            if tar_dir and not os.path.exists(tar_dir):
                os.makedirs(tar_dir, exist_ok=True)

            # Save image to tar
            print(f"Saving Docker image {image_name} to {tar_path}...")
            console.sh(f'docker save "{image_name}" -o "{tar_path}"', timeout=600)

            # Verify tar file was created
            if not os.path.exists(tar_path):
                print(f"Error: Failed to create tar file at {tar_path}")
                return False

            file_size = os.path.getsize(tar_path)
            print(f"Docker image saved successfully ({file_size / (1024**2):.2f} MB)")
            return True

        except Exception as e:
            print(f"Error saving Docker image to tar: {e}")
            return False

    @staticmethod
    def load_image_from_tar(tar_path: str, console: Console = None) -> typing.Optional[str]:
        """Load Docker image from tar file.

        Args:
            tar_path: Path to tar file
            console: Console object for shell operations (optional)

        Returns:
            str: Loaded image name if successful, None otherwise
        """
        if console is None:
            console = Console()

        try:
            # Verify tar file exists
            if not os.path.exists(tar_path):
                print(f"Error: Tar file not found at {tar_path}")
                return None

            # Load image from tar
            print(f"Loading Docker image from {tar_path}...")
            output = console.sh(f'docker load -i "{tar_path}"', timeout=600)

            # Parse output to get image name
            # Output format: "Loaded image: <image_name>"
            image_name = None
            for line in output.split('\n'):
                if 'Loaded image:' in line:
                    image_name = line.split('Loaded image:')[-1].strip()
                    break

            if image_name:
                print(f"Docker image loaded successfully: {image_name}")
                return image_name
            else:
                print("Warning: Could not parse image name from docker load output")
                return None

        except Exception as e:
            print(f"Error loading Docker image from tar: {e}")
            return None

    @staticmethod
    def image_exists(image_name: str, console: Console = None) -> bool:
        """Check if Docker image exists locally.

        Args:
            image_name: Name of the docker image to check
            console: Console object for shell operations (optional)

        Returns:
            bool: True if image exists, False otherwise
        """
        if console is None:
            console = Console()

        try:
            output = console.sh(f'docker images -q "{image_name}"')
            return bool(output.strip())
        except Exception:
            return False

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
