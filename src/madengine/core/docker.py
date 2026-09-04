#!/usr/bin/env python3
"""Module to run docker commands.

This module provides a class to run commands inside docker.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import os
import re
import shlex
import typing

# user-defined modules
from madengine.core import lifecycle
from madengine.core.console import Console


class Docker:
    """Class to run commands inside docker.

    The container's PID 1 is a watchdog that stops the container as soon as
    madengine stops refreshing a heartbeat file in the bind-mounted workspace.
    That is what makes a container outliving the run impossible rather than
    merely unlikely: a cancelled CI job kills madengine's whole process tree
    without any of our cleanup code running, and a detached container would
    otherwise keep the GPU busy until someone rebooted the host.

    ``keep_alive`` deliberately opts out of the watchdog, since the point of
    that flag is to leave the container behind for manual inspection.

    Attributes:
        docker_sha (str): The docker sha.
        container_name (str): The container name.
        keep_alive (bool): The keep alive flag.
        console (Console): The console object.
        userid (str): The user id.
        groupid (str): The group id.
    """

    # Class-level defaults so close() stays safe on a half-built instance --
    # __del__ reaches objects whose __init__ raised part-way through, and that
    # is exactly the case where a container may already be running.
    docker_sha = None
    container_name = None
    heartbeat_path = None
    keep_alive = False
    _closed = False

    def __init__(
        self,
        image: str,
        container_name: str,
        dockerOpts: str,
        mounts: typing.Optional[typing.List] = None,
        envVars: typing.Optional[typing.Dict] = None,
        keep_alive: bool = False,
        console: Console = None,
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
            RuntimeError: If the container cannot be started.
        """
        # initialize variables
        self.docker_sha = None
        self.container_name = container_name
        self.keep_alive = keep_alive
        self._closed = False
        cwd = os.getcwd()
        self.heartbeat_path = os.path.join(cwd, lifecycle.HEARTBEAT_FILENAME)
        self.console = console if console is not None else Console()
        self.userid = self.console.sh("id -u")
        self.groupid = self.console.sh("id -g")

        # check if container name exists — use an exact-match filter so names
        # containing regex metacharacters (e.g. ".", "[") cannot produce false
        # positives, and substring matches are avoided entirely.
        container_name_regex = shlex.quote(f"^/{re.escape(container_name)}$")
        container_name_exists = self.console.sh(
            f"docker container ps -aq --filter name={container_name_regex}"
        )
        # if container name exists, clean it up automatically
        if container_name_exists:
            print(
                f"⚠️  Container '{container_name}' already exists. Cleaning up..."
            )
            lifecycle.reap(container_name)
            print(f"✓ Cleaned up existing container '{container_name}'")

        # run docker command
        command = "docker run -t -d "
        # Conditionally add -u flag if not already present in dockerOpts
        if "-u " not in dockerOpts:
            command += f"-u {self.userid}:{self.groupid} "
        command += dockerOpts + " "

        # add mounts
        if mounts is not None:
            for mount in mounts:
                quoted_mount = shlex.quote(mount)
                command += "-v " + quoted_mount + ":" + quoted_mount + " "

        # add current working directory
        command += "-v " + shlex.quote(cwd) + ":/myworkspace/ "

        # add envVars
        _env_key_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        if envVars is not None:
            for evar in envVars.keys():
                if not _env_key_re.match(evar):
                    raise ValueError(f"Invalid environment variable name: {evar!r}")
                command += "-e " + evar + "=" + shlex.quote(str(envVars[evar])) + " "

        # Label the container so `madengine cleanup` can find it later even if
        # this process died before it could record anything.
        command += (
            f"--label {lifecycle.LABEL_SESSION}={shlex.quote(lifecycle.session_id())} "
            f"--label {lifecycle.LABEL_OWNER_PID}={os.getpid()} "
        )
        command += "--workdir /myworkspace/ "
        command += "--name " + shlex.quote(container_name) + " "
        command += shlex.quote(image) + " "

        if keep_alive:
            # No watchdog: the container is meant to outlive this process.
            self.console.sh(command + "cat ")
        else:
            self._start_with_watchdog(command)

        # find container sha
        self.docker_sha = self._resolve_container_id(container_name)
        lifecycle.register(self.docker_sha, container_name)

    def _start_with_watchdog(self, command: str) -> None:
        """Start the container under the heartbeat watchdog.

        Falls back to the historical ``cat`` if the watchdog cannot run, so an
        image without a POSIX shell still works -- with a warning, since such
        a container is once again able to outlive madengine.

        Args:
            command (str): The ``docker run`` line, up to but excluding the
                container command.

        Raises:
            RuntimeError: If the container cannot be started at all.
        """
        # The watchdog exits immediately if the heartbeat file is missing, so
        # the first beat has to land before the container starts.
        lifecycle.start_heartbeat(self.heartbeat_path)
        self.console.sh(command + lifecycle.container_watchdog_command())

        if lifecycle.container_state(self.container_name) == "running":
            return

        print(
            f"⚠️  Container '{self.container_name}' did not stay up under the "
            f"heartbeat watchdog; the image may lack a POSIX shell. Falling "
            f"back to an unwatched container — it will NOT stop by itself if "
            f"this run is cancelled.",
            flush=True,
        )
        lifecycle.stop_heartbeat(self.heartbeat_path)
        self.heartbeat_path = None
        lifecycle.reap(self.container_name)
        self.console.sh(command + "cat ")

        state = lifecycle.container_state(self.container_name)
        if state != "running":
            raise RuntimeError(
                f"Container '{self.container_name}' failed to start "
                f"(state: {state or 'gone'})"
            )

    def _resolve_container_id(self, container_name: str) -> str:
        """Return the full id of the container.

        Args:
            container_name (str): The container name.

        Returns:
            str: The full container id.

        Raises:
            RuntimeError: If the id cannot be resolved.
        """
        code, out = lifecycle.docker(
            ["inspect", "-f", "{{.Id}}", container_name], lifecycle.INSPECT_TIMEOUT
        )
        if code != 0 or not out:
            raise RuntimeError(
                f"Could not resolve id of container '{container_name}': {out}"
            )
        return out.strip()

    def sh(self, command: str, timeout: int = 60, secret: bool = False) -> str:
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
            "docker exec " + self.docker_sha + " bash -c " + shlex.quote(command),
            timeout=timeout,
            secret=secret,
        )

    def close(self) -> None:
        """Stop and remove the container, unless it is being kept alive."""
        if self._closed:
            return
        self._closed = True

        if self.keep_alive:
            if self.docker_sha:
                lifecycle.unregister(self.docker_sha)
                print("==========================================")
                print("Keeping docker alive, sha :", self.docker_sha)
                print(
                    "Open a bash session in container : ",
                    "docker exec -it " + self.docker_sha + " bash",
                )
                print("Stop container : ", "docker stop -t 1 " + self.docker_sha)
                print("Remove container : ", "docker rm -f " + self.docker_sha)
                print("==========================================")
            return

        # Dropping the heartbeat is what stops the container; the reap below
        # then removes it and reports if it refused to die.
        if self.heartbeat_path:
            lifecycle.stop_heartbeat(self.heartbeat_path)
        if self.docker_sha:
            lifecycle.reap(self.container_name or self.docker_sha, self.docker_sha)
            lifecycle.unregister(self.docker_sha)

    def __enter__(self) -> "Docker":
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context manager, tearing the container down."""
        self.close()

    def __del__(self):
        """Destructor of the Docker class."""
        try:
            self.close()
        except Exception:
            # Destructors must not raise; close() is called explicitly on
            # every path that matters.
            pass
