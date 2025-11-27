"""Module to SSH into the database.

This module provides the functions to SSH into the database.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import os
import socket

# third-party modules
import paramiko


class SFTPClient(paramiko.SFTPClient):
    """Class to create an SFTP client for the database."""

    def __init__(self, *args, **kwargs):
        """Initialize the SFTPClient class."""
        super().__init__(*args, **kwargs)

    def put_dir(self, source: str, target: str) -> None:
        """Uploads the contents of the source directory to the target path. The
        target directory needs to exists. All subdirectories in source are
        created under target.

        Args:
            source (str): The source directory to upload.
            target (str): The target directory to upload to.

        Returns:
            None

        Raises:
            IOError: An error occurred uploading the directory.
        """
        for item in os.listdir(source):
            if os.path.isfile(os.path.join(source, item)):
                self.put(os.path.join(source, item), "%s/%s" % (target, item))
            else:
                self.mkdir("%s/%s" % (target, item), ignore_existing=True)
                self.put_dir(os.path.join(source, item), "%s/%s" % (target, item))

    def mkdir(self, path: str, mode: int = 511, ignore_existing: bool = False) -> None:
        """Augments mkdir by adding an option to not fail if the folder exists

        Args:
            path (str): The path to create.
            mode (int): The mode to create the path with.
            ignore_existing (bool): Whether to ignore if the path already exists.

        Returns:
            None

        Raises:
            IOError: An error occurred creating the directory.
        """
        try:
            super(SFTPClient, self).mkdir(path, mode)
        except IOError:
            if ignore_existing:
                pass
            else:
                raise


def print_ssh_out(client_output: tuple) -> None:
    """Print the output from the SSH client.

    Args:
        client_output (tuple): The output from the SSH client.

    Returns:
        None
    """
    ssh_stdin, ssh_stdout, ssh_stderr = client_output
    ssh_stdin.close()
    for line in ssh_stdout.read().splitlines():
        print("{}".format(line))
    for line in ssh_stderr.read().splitlines():
        print("{}".format(line))
