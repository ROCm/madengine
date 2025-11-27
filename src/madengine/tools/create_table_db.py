#!/usr/bin/env python
"""Module to create tables in the database.

This module provides the functions to create tables in the database.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import os
import argparse
import subprocess
import typing

# third-party modules
import paramiko
import socket

# mad-engine modules
from madengine.utils.ssh_to_db import SFTPClient, print_ssh_out
from madengine.db.logger import setup_logger
from madengine.db.utils import get_env_vars

# Create the logger
LOGGER = setup_logger()
# Get the environment variables
ENV_VARS = get_env_vars()


class CreateTable:
    """Class to create tables in the database.

    This class provides the functions to create tables in the database.
    """

    def __init__(self, args: argparse.Namespace):
        """Initialize the CreateTable class.

        Args:
            args (argparse.Namespace): The arguments passed to the script.
        """
        self.args = args
        self.db_name = ENV_VARS["db_name"]
        self.db_hostname = ENV_VARS["db_hostname"]
        self.db_port = ENV_VARS["db_port"]
        self.user_name = ENV_VARS["user_name"]
        self.user_password = ENV_VARS["user_password"]
        self.ssh_user = ENV_VARS["ssh_user"]
        self.ssh_password = ENV_VARS["ssh_password"]
        self.ssh_hostname = ENV_VARS["ssh_hostname"]
        self.ssh_port = ENV_VARS["ssh_port"]

        # get the db folder
        self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../db")
        LOGGER.info(f"DB path: {self.db_path}")
        self.status = False

    def run(self, table_name: str = "dlm_table") -> None:
        """Create an empty table in the database.

        Args:
            table_name (str): The name of the table to create.

        Returns:
            None

        Raises:
            Exception: An error occurred creating the table.
        """
        print(f"Creating table {table_name} in the database")

        if "localhost" in self.ssh_hostname or "127.0.0.1" in self.ssh_hostname:
            try:
                self.local_db()
                self.status = True
                return self.status
            except Exception as error:
                LOGGER.error(f"Error creating table in local database: {error}")
                return self.status
        else:
            try:
                self.remote_db()
                self.status = True
                return self.status
            except Exception as error:
                LOGGER.error(f"Error creating table in remote database: {error}")
                return self.status

    def local_db(self) -> None:
        """Create a table in the local database.

        Returns:
            None

        Raises:
            Exception: An error occurred creating the table in the local database.
        """
        print("Creating table in local database")

        # copy the db folder from the db_path to the current working directory
        cmd_list = ["cp", "-r", self.db_path, "."]

        try:
            ret = subprocess.Popen(
                cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            out, err = ret.communicate()
            if ret.returncode == 0:
                if out:
                    LOGGER.info(out.decode("utf-8"))
                print("Copied scripts to current work path")
            else:
                if err:
                    LOGGER.error(err.decode("utf-8"))
        except Exception as e:
            LOGGER.error(f"An error occurred: {e}")

        # run upload_csv_to_db.py in the db folder with environment variables using subprocess Popen
        cmd_list = ["python3", "./db/upload_csv_to_db.py"]

        # Ensure ENV_VARS is a dictionary
        env_vars = dict(ENV_VARS)
        print(f"ENV_VARS: {env_vars}")

        try:
            ret = subprocess.Popen(
                cmd_list, env=env_vars, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            out, err = ret.communicate()

            if ret.returncode == 0:
                if out:
                    LOGGER.info(out.decode("utf-8"))
            else:
                if err:
                    LOGGER.error(err.decode("utf-8"))
                raise Exception(
                    f"Error updating table in the local database: {err.decode('utf-8')}"
                )
        except Exception as e:
            LOGGER.error(f"An error occurred: {e}")

        print("Script execution completed")

    def remote_db(self) -> None:
        """Create a table in the remote database.

        Returns:
            None

        Raises:
            socket.error: An error occurred connecting to the database.
        """
        print("Creating table in remote database")

        # create an ssh client
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.load_system_host_keys()

        # connect to the host of database
        try:
            ssh_client.connect(
                hostname=self.ssh_hostname,
                port=self.ssh_port,
                username=self.ssh_user,
                password=self.ssh_password,
                timeout=10,
            )
        except paramiko.ssh_exception.AuthenticationException as error:
            print(f"Authentication failed: {error}")
            return
        except paramiko.ssh_exception.SSHException as error:
            print(f"SSH error: {error}")
            return
        except socket.error as error:
            print(f"Socket error: {error}")
            return

        print("SSH client created, connected to the host of database")

        # print remote dir layout
        print_ssh_out(ssh_client.exec_command("pwd"))
        print_ssh_out(ssh_client.exec_command("ls -l"))

        # get remote path for files
        upload_script_path_remote = os.path.basename(self.db_path)
        print(upload_script_path_remote)

        # clean up previous uploads
        print_ssh_out(
            ssh_client.exec_command("rm -rf {}".format(upload_script_path_remote))
        )
        print_ssh_out(ssh_client.exec_command("ls -l"))

        # upload file
        sftp_client = SFTPClient.from_transport(ssh_client.get_transport())
        sftp_client.mkdir(upload_script_path_remote, ignore_existing=True)
        sftp_client.put_dir(self.db_path, upload_script_path_remote)

        # close the sftp client
        sftp_client.close()

        # run script on remote node
        main_script = os.path.join(upload_script_path_remote, "upload_csv_to_db.py")
        print_ssh_out(
            ssh_client.exec_command(
                "TUNA_DB_USER_NAME={} TUNA_DB_USER_PASSWORD={} TUNA_DB_NAME={} TUNA_DB_HOSTNAME={} python3 {}".format(
                    self.user_name,
                    self.user_password,
                    self.db_name,
                    self.db_hostname,
                    main_script,
                )
            )
        )

        # print remote dir after upload
        print_ssh_out(ssh_client.exec_command("ls -l"))

        # close the ssh client
        ssh_client.close()
