#!/usr/bin/env python3
"""Data Provider module for MADEngine

This module provides data to the models. It can provide data from different sources like local, NAS, AWS, etc.

This module provides the following classes:
    - DataSourceException: Raised when data source not valid
    - DataProvider: Parent class for data providers
    - CustomDataProvider: Data provided through custom scripts in scripts/common/data
    - LocalDataProvider: Data provided from local path
    - NASDataProvider: Data provided from NAS
    - MinioDataProvider : Data provided from Minio
    - AWSDataProvider: Data provided from AWS
    - Data: Class to provide data

This module provides the following functions:
    - DataProviderFactory: Factory Method for DataProviders

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in python modules
import json
import os
import time
import typing

# MADEngine modules
from madengine.core.console import Console
from madengine.core.context import Context
from madengine.core.docker import Docker
from madengine.core.constants import NAS_NODES, MAD_AWS_S3, MAD_MINIO


class DataSourceException(Exception):
    """Raised when data source not valid"""

    pass


class DataProvider:
    """DataProvider parent class

    Parent class for data providers

    Attributes:
        provider_type (str): Type of data provider
        provider_count (int): Count of data providers
        dataname (str): Name of the data
        config (dict): Configuration of the data provider
        provider_instance_index (int): Index of the data provider instance

    Methods:
        check_source: Check if the data source is valid
        get_mountpath: Get the mount path of the data
        get_env: Get the environment variables
        prepare_data: Prepare the data
    """

    # class variables
    provider_type = None
    provider_count = 0

    def __init__(self, dataname: str, config: typing.Dict) -> None:
        """Constructor of the DataProvider class.

        Args:
            dataname (str): Name of the data
            config (dict): Configuration of the data provider

        Raises:
            DataSourceException: Raised when data source not valid
        """
        # instance variables, initialized with default values.
        # increment the provider_count by 1, each time a new data provider is created.
        DataProvider.provider_count += 1
        self.dataname = dataname
        self.config = config
        self.duration = 0
        self.size = ""
        # set the index of the data provider instance, to the current count of data providers, minus 1.
        self.provider_instance_index = DataProvider.provider_count - 1

        # check if the data source is valid, if not raise DataSourceException.
        if not self.check_source(config):
            raise DataSourceException()

    def check_source(self, config: typing.Dict) -> bool:
        """Check if the data source is valid

        Args:
            config (dict): Configuration of the data provider

        Returns:
            bool: The status of the data source
        """
        pass

    def get_mountpath(self):
        """Get the mount path of the data"""
        pass

    def get_env(self):
        """Get the environment variables

        Returns:
            dict: The environment variables
        """
        # set the data home environment variable, with the name of the data, and the provider_instance_index.
        datahome = "/data_dlm"
        if "home" in self.config:
            datahome = self.config["home"]
        # append the provider_instance_index to the datahome.
        datahome += "_" + str(self.provider_instance_index)
        return {"MAD_DATAHOME": datahome}

    def prepare_data(self, model_docker: Docker) -> bool:
        """Prepare the data

        Args:
            model_docker: The model docker object

        Returns:
            bool: The status of preparing the data
        """
        pass


########################### DATA PROVIDERS ###########################


class CustomDataProvider(DataProvider):
    """CustomDataProvider class
    Data provided through custom scripts in scripts/common/data.
    """

    provider_type = "custom"

    def __init__(self, dataname: str, config: typing.Dict) -> None:
        """Constructor of the CustomDataProvider class."""
        super().__init__(dataname, config)

    def check_source(self, config: typing.Dict) -> bool:
        """Check if the data source is valid

        Args:
            config (dict): Configuration of the data provider

        Returns:
            bool: The status of the data source

        Raises:
            RuntimeError: Raised when the mirrorlocal path is a non-existent path
        """
        # check if the mirrorlocal path is a non-existent path, if so raise RuntimeError.
        if "mirrorlocal" in self.config:
            # check if the mirrorlocal path exists.
            if not os.path.exists(self.config["mirrorlocal"]):
                raise RuntimeError("mirrorlocal is a non-existent path.")
            # create the mirrorlocal path if it does not exist.
            if not os.path.exists(self.config["mirrorlocal"] + "/" + self.dataname):
                os.makedirs(
                    self.config["mirrorlocal"] + "/" + self.dataname, exist_ok=True
                )

        # get the base directory of the current file.
        BASE_DIR = os.path.dirname(os.path.realpath(__file__))
        print("DEBUG - BASE_DIR::", BASE_DIR)
        print("DEBUG - self.config[path]::", self.config["path"])

        # check if the path exists in the base directory.
        # if os.path.exists(BASE_DIR + "/../" + self.config["path"]):
        if os.path.exists(self.config["path"]):
            return True
        return False

    def get_mountpath(self):
        if "mirrorlocal" in self.config:
            return {
                "path": self.config["mirrorlocal"] + "/" + self.dataname,
                "home": self.get_env()["MAD_DATAHOME"],
                "readwrite": "true",
            }
        return False

    def prepare_data(self, model_docker):

        datahome = self.get_env()["MAD_DATAHOME"]

        args = ""
        if "args" in self.config:
            args = self.config["args"]

        cmd = (
            "mkdir -p {datahome}; cd {datahome}; bash /myworkspace/{scriptpath} {args} "
        )
        cmd = cmd.format(scriptpath=self.config["path"], datahome=datahome, args=args)
        print(model_docker.sh(cmd, timeout=1200))  # 20 min timeout
        return True


class LocalDataProvider(DataProvider):
    """LocalDataProvider class"""

    provider_type = "local"

    def __init__(self, dataname, config):
        super().__init__(dataname, config)

    def check_source(self, config):
        if "mirrorlocal" in self.config:
            raise RuntimeError("mirrorlocal cannot be specified in LocalDataProvider")
        if os.path.exists(self.config["path"]):
            return True
        return False

    def get_mountpath(self):
        cfg = self.config.copy()
        if "home" not in cfg:
            cfg["home"] = self.get_env()["MAD_DATAHOME"]
        return cfg

    def prepare_data(self, model_docker):
        return True


class NASDataProvider(DataProvider):
    """NASDataProvider class"""

    provider_type = "nas"

    def __init__(self, dataname, config):
        self.nas_nodes = NAS_NODES
        if "ip" not in self.__dict__:
            self.ip = None
        if "port" not in self.__dict__:
            self.port = None
        if "username" not in self.__dict__:
            self.username = None
        if "password" not in self.__dict__:
            self.password = None
        if "timeout" not in self.__dict__:
            self.timeout = 30
        super().__init__(dataname, config)

    def check_source(self, config):
        if "mirrorlocal" in self.config:
            if not os.path.exists(self.config["mirrorlocal"]):
                raise RuntimeError("mirrorlocal is a non-existent path.")
            if not os.path.exists(self.config["mirrorlocal"] + "/" + self.dataname):
                os.makedirs(
                    self.config["mirrorlocal"] + "/" + self.dataname, exist_ok=True
                )

        console = Console()
        # Check the connection to the NAS node in th list of nas_nodes
        for nas_node in self.nas_nodes:
            self.name = nas_node["NAME"]
            self.ip = nas_node["HOST"]
            self.port = nas_node["PORT"]
            self.username = nas_node["USERNAME"]
            self.password = nas_node["PASSWORD"]
            print(f"Checking NAS connection to {self.name} at {self.ip}:{self.port}...")
            if self.check_nas_connection(console):
                print(f"Connected to NAS {self.name} at {self.ip}:{self.port}")
                return True
            else:
                print(f"Failed to connect to NAS {self.name} at {self.ip}:{self.port}")

        print("Failed to connect to all available NAS nodes.")
        return False

    def check_nas_connection(self, console):
        try:
            console.sh(
                "timeout "
                + str(self.timeout)
                + " bash -c '</dev/tcp/"
                + self.ip
                + "/"
                + self.port
                + "'"
            )
            status = console.sh(
                "ssh -o BatchMode=yes -o ConnectTimeout=5 "
                + self.username
                + "@"
                + self.ip
                + " -p "
                + self.port
                + " echo 'SSH login ok'",
                canFail=True,
            )
            # if either ssh login succeeded or if permission was denied, ssh access to node exists
            if "Permission denied" in status or "SSH login ok" in status:
                return True
            return False
        except Exception as e:
            print("Failed pinging NAS, Error: ", e)

        return False

    def get_mountpath(self):
        if "mirrorlocal" in self.config:
            return {
                "path": self.config["mirrorlocal"] + "/" + self.dataname,
                "home": self.get_env()["MAD_DATAHOME"],
                "readwrite": "true",
            }
        return False

    def prepare_data(self, model_docker):

        datahome = self.get_env()["MAD_DATAHOME"]

        if "mirrorlocal" in self.config:
            # copy data from NAS locally
            cmd = """
                if [ -f \"$(which apt)\" ]; then 
                    apt update && apt install -y sshpass sshfs rsync
                elif [ -f \"$(which yum)\" ]; then 
                    yum install -y sshpass rsync
                else 
                    echo 'Unable to detect Host OS'
                    exit 1
                fi

                echo 'NAS is getting connected to {ip}:{port}.'
                mkdir -p ~/.ssh
                touch ~/.ssh/known_hosts
                ssh-keyscan -p {port} {ip} >> ~/.ssh/known_hosts
                echo '#!/bin/bash' > /tmp/ssh.sh
                echo 'sshpass -p {password} rsync --progress -avz -e \\"ssh -p {port} \\" \\"\\$@\\"' >> /tmp/ssh.sh
                cat /tmp/ssh.sh
                chmod u+x /tmp/ssh.sh
                timeout --preserve-status {timeout} /tmp/ssh.sh {username}@{ip}:{datapath}/* {datahome} && rm -f /tmp/ssh.sh
               """
            cmd = cmd.format(
                ip=self.ip,
                port=self.port,
                username=self.username,
                password=self.password,
                datapath=self.config["path"],
                datahome=datahome,
                timeout=2400,
            )
            # Measure time taken to copy data from NAS to local
            start = time.time()
            print(model_docker.sh(cmd, timeout=2400))  # 40 min timeout
            end = time.time()
            self.duration = end - start
            print("Copy data from NAS to local")
            print("Data Download Duration: {} seconds".format(self.duration))
        else:
            cmd = """
                if [ -f \"$(which apt)\" ]; then 
                    apt update && apt install -y sshpass sshfs
                elif [ -f \"$(which yum)\" ]; then 
                    yum install -y sshpass sshfs
                else 
                    echo 'Unable to detect Host OS'
                    exit 1
                fi

                echo 'NAS is getting connected to {ip}:{port}.'
                mkdir -p ~/.ssh
                mkdir -p {datahome}
                touch ~/.ssh/known_hosts
                ssh-keyscan -p {port} {ip} >> ~/.ssh/known_hosts
                echo '#!/bin/bash' > /tmp/ssh.sh
                echo 'sshpass -p {password} ssh -v \\$*' >> /tmp/ssh.sh
                chmod u+x /tmp/ssh.sh
                timeout --preserve-status {timeout} mount -t fuse sshfs#{username}@{ip}:{datapath} {datahome} -o ssh_command=/tmp/ssh.sh,port={port} && rm -f /tmp/ssh.sh
               """
            cmd = cmd.format(
                ip=self.ip,
                port=self.port,
                username=self.username,
                password=self.password,
                datapath=self.config["path"],
                datahome=datahome,
                timeout=self.timeout,
            )
            # Measure time taken to mount data from NAS to local
            start = time.time()
            print(model_docker.sh(cmd, timeout=120))  # 2 min timeout
            end = time.time()
            self.duration = end - start
            print("Copy data from NAS to local")
            print("Data Download Duration: {} seconds".format(self.duration))
        return True


class AWSDataProvider(DataProvider):
    """AWSDataProvider class"""

    provider_type = "aws"

    def __init__(self, dataname, config):
        if "username" not in self.__dict__:
            self.username = MAD_AWS_S3["USERNAME"]
        if "password" not in self.__dict__:
            self.password = MAD_AWS_S3["PASSWORD"]
        if "timeout" not in self.__dict__:
            self.timeout = 30
        super().__init__(dataname, config)

    def check_source(self, config):
        if "mirrorlocal" in self.config:
            if not os.path.exists(self.config["mirrorlocal"]):
                raise RuntimeError("mirrorlocal is a non-existent path.")
            if not os.path.exists(self.config["mirrorlocal"] + "/" + self.dataname):
                os.makedirs(
                    self.config["mirrorlocal"] + "/" + self.dataname, exist_ok=True
                )

        console = Console()
        console.sh(
            "timeout "
            + str(self.timeout)
            + " bash -c '</dev/tcp/s3.us-east-2.amazonaws.com/443'"
        )
        return True

    def get_mountpath(self):
        if "mirrorlocal" in self.config:
            return {
                "path": self.config["mirrorlocal"] + "/" + self.dataname,
                "home": self.get_env()["MAD_DATAHOME"],
                "readwrite": "true",
            }
        return False

    def prepare_data(self, model_docker):

        datahome = self.get_env()["MAD_DATAHOME"]

        cmd = """
            pip3 --no-cache-dir install --upgrade awscli
            export AWS_ACCESS_KEY_ID={username}
            export AWS_SECRET_ACCESS_KEY={password}
            mkdir -p {datahome}
            if ( aws --region=us-east-2 s3 ls {datapath} | grep \"PRE\" ); then
                aws --region=us-east-2 s3 sync {datapath} {datahome}
            else
                aws --region=us-east-2 s3 sync $( dirname {datapath} ) {datahome} --exclude=\"*\" --include=\"$( basename {datapath} )\"
            fi
           """
        cmd = cmd.format(
            username=self.username,
            password=self.password,
            datapath=self.config["path"],
            datahome=datahome,
            dataname=self.dataname,
        )
        # Measure time taken to copy data from AWS to local
        start = time.time()
        model_docker.sh(cmd, timeout=3600)  # 60 min timeout
        end = time.time()
        self.duration = end - start
        print("Copy data from AWS to local")
        print("Data Download Duration: {} seconds".format(self.duration))
        # Get the size of the data of dataname in the path of datahome and store it in the config
        # cmd = f"du -sh {datahome}/{self.dataname} | cut -f1"
        cmd = f"du -sh {datahome} | cut -f1"
        data_size = model_docker.sh(cmd)
        self.size = data_size
        print("Data Size: ", self.size)
        return True


class MinioDataProvider(DataProvider):
    """MinioDataProvider class"""

    provider_type = "minio"

    def __init__(self, dataname, config):
        if "username" not in self.__dict__:
            self.username = MAD_MINIO["USERNAME"]
        if "password" not in self.__dict__:
            self.password = MAD_MINIO["PASSWORD"]
        if "timeout" not in self.__dict__:
            self.timeout = 30
        if "minio_endpoint" not in self.__dict__:
            self.minio_endpoint = MAD_MINIO["MINIO_ENDPOINT"]
        if "aws_endpoint_url_s3" not in self.__dict__:
            self.aws_endpoint_url_s3 = MAD_MINIO["AWS_ENDPOINT_URL_S3"]
        super().__init__(dataname, config)

    def check_source(self, config):
        if "mirrorlocal" in self.config:
            if not os.path.exists(self.config["mirrorlocal"]):
                raise RuntimeError("mirrorlocal is a non-existent path.")
            if not os.path.exists(self.config["mirrorlocal"] + "/" + self.dataname):
                os.makedirs(
                    self.config["mirrorlocal"] + "/" + self.dataname, exist_ok=True
                )

        console = Console()
        try:
            console.sh(
                f"timeout {self.timeout} curl -s {self.minio_endpoint} -o /dev/null"
            )
        except Exception as e:
            print(f"Failed to connect to Minio endpoint ({self.minio_endpoint}): {e}")
            return False

        return True

    def get_mountpath(self):
        if "mirrorlocal" in self.config:
            return {
                "path": self.config["mirrorlocal"] + "/" + self.dataname,
                "home": self.get_env()["MAD_DATAHOME"],
                "readwrite": "true",
            }
        return False

    def prepare_data(self, model_docker):

        datahome = self.get_env()["MAD_DATAHOME"]

        cmd = """
            pip3 --no-cache-dir install --upgrade awscli
            export AWS_ACCESS_KEY_ID={username}
            export AWS_SECRET_ACCESS_KEY={password}
            export MINIO_ENDPOINT={minio_endpoint}
            export AWS_ENDPOINT_URL_S3={aws_endpoint_url_s3}
            mkdir -p {datahome}
            if ( aws --endpoint-url {minio_endpoint} s3 ls {datapath} | grep PRE ); then
                aws --endpoint-url {minio_endpoint} s3 sync {datapath} {datahome}
            else
                aws --endpoint-url {minio_endpoint} s3 sync $( dirname {datapath} ) {datahome} --exclude=\"*\" --include=\"$( basename {datapath} )\"
            fi
           """
        cmd = cmd.format(
            username=self.username,
            password=self.password,
            minio_endpoint=self.minio_endpoint,
            aws_endpoint_url_s3=self.aws_endpoint_url_s3,
            datapath=self.config["path"],
            datahome=datahome,
            dataname=self.dataname,
        )

        # Measure time taken to copy data from MinIO to local
        start = time.time()
        model_docker.sh(cmd, timeout=3600)  # 60 min timeout
        end = time.time()
        self.duration = end - start
        print("Copy data from MinIO to local")
        print("Data Download Duration: {} seconds".format(self.duration))

        # Get the size of the data of dataname in the path of datahome and store it in the config
        cmd = f"du -sh {datahome} | cut -f1"
        data_size = model_docker.sh(cmd)
        self.size = data_size
        print("Data Size: ", self.size)

        return True


########################### ADD MORE PROVIDERS ABOVE THIS LINE ###########################


def DataProviderFactory(
    dataname: str, data_provider_type: str, data_provider_config: typing.Dict
) -> typing.Optional[DataProvider]:
    """Factory Method for DataProviders

    Args:
        dataname (str): Name of the data
        data_provider_type (str): Type of data provider
        data_provider_config (dict): Configuration of the data provider

    Returns:
        DataProvider: The data provider

    Raises:
        DataSourceException: Raised when data source not valid
    """
    # list of data providers, add more data providers here.
    data_providers = [
        CustomDataProvider,
        LocalDataProvider,
        MinioDataProvider,
        NASDataProvider,
        AWSDataProvider,
    ]
    # iterate through the data providers and return the data provider if the provider type matches the data_provider_type, else return None.
    for data_provider in data_providers:
        if data_provider.provider_type == data_provider_type:
            try:
                return data_provider(dataname, data_provider_config)
            except DataSourceException as e:
                print("DataProviderFactory failed with type = ", data_provider_type)
    return None


class Data:
    """Class to provide data

    Attributes:
        data_provider_config (dict): Configuration of the data provider
        data_provider_list (dict): List of data providers

    Methods:
        find_dataprovider: Find data provider
        get_mountpaths: Get the mount paths
        get_env: Get the environment variables
        prepare_data: Prepare the data
    """

    def __init__(
        self,
        context: typing.Optional[Context] = None,
        filename: str = "data.json",
        force_mirrorlocal: typing.Optional[str] = None,
    ):
        """Constructor of the Data class.

        Args:
            context: The context object
            filename (str): The name of the data configuration file
            force_mirrorlocal: Force mirror local
        """

        # instance variables with default values.
        self.data_provider_config = {}
        self.data_provider_list = {}
        self.selected_data_provider = {}

        # read data config from file
        with open(filename) as f:
            self.data_provider_config.update(json.load(f))

        # 'data' context override self.data_provider_config in file
        if context and "data" in context.ctx:
            self.data_provider_config.update(context.ctx["data"])

        # force mirror local will overwrite any existing, or non-existent "mirrorlocal" setting across ALL data
        if force_mirrorlocal:
            for dataname in self.data_provider_config:
                for data_provider_type in self.data_provider_config[dataname]:
                    # "local" dataprovider does not support mirrorlocal
                    if data_provider_type != "local":
                        self.data_provider_config[dataname][data_provider_type][
                            "mirrorlocal"
                        ] = force_mirrorlocal

    def reorder_data_provider_config(self, dataname: str) -> None:
        """Reorder the data provider config to match the order of the ordered_data_provider_types"""
        ordered_data_provider_types = [
            CustomDataProvider.provider_type,
            LocalDataProvider.provider_type,
            MinioDataProvider.provider_type,
            NASDataProvider.provider_type,
            AWSDataProvider.provider_type,
        ]
        # reorder the data provider types in the config to match the order of the ordered_data_provider_types
        self.data_provider_config[dataname] = {
            k: self.data_provider_config[dataname][k]
            for k in ordered_data_provider_types
            if k in self.data_provider_config[dataname]
        }
        print(
            "MAD_DATA_PROVIDER::"
            + dataname
            + ": reordered list of data provider types to: "
            + str(self.data_provider_config[dataname])
            + " ..."
        )

    def find_dataprovider(self, dataname: str) -> typing.Optional[DataProvider]:
        """Find data provider, determine which data provider to use

        Args:
            dataname (str): Name of the data

        Returns:
            DataProvider: The data provider
        """
        # reuse data already located on node
        if dataname in self.data_provider_list:
            print(
                "MAD_DATA_PROVIDER::"
                + dataname
                + ": searched for previously. Reusing ..."
            )
            return self.data_provider_list[dataname]

        self.reorder_data_provider_config(dataname)

        # iterate through the data provider types and find the data provider.
        for data_provider_type in self.data_provider_config[dataname]:
            print(
                "MAD_DATA_PROVIDER::"
                + dataname
                + ": searching in "
                + data_provider_type
                + ". "
            )
            data_provider = DataProviderFactory(
                dataname,
                data_provider_type,
                self.data_provider_config[dataname][data_provider_type],
            )
            if data_provider:
                print(
                    "MAD_DATA_PROVIDER::"
                    + dataname
                    + ": found in "
                    + data_provider_type
                    + ". "
                )
                self.data_provider_list[dataname] = data_provider
                # save the dataname, the current data provider type, and the current data_provider_config to a json file
                self.selected_data_provider = {
                    "dataname": dataname,
                    "data_provider_type": data_provider_type,
                    "data_provider_config": self.data_provider_config[dataname][
                        data_provider_type
                    ],
                    "duration": data_provider.duration,
                    "size": data_provider.size,
                }
                break

        # set to None if not found
        if dataname not in self.data_provider_list:
            print("MAD_DATA_PROVIDER::" + dataname + ": not found.")
            self.data_provider_list[dataname] = None

        return self.data_provider_list[dataname]

    def get_mountpaths(self, datanames: str) -> typing.List:
        """Get the mount paths

        Args:
            datanames (str): Names of the data

        Returns:
            list: The mount paths
        """
        # iterate through the data names and get the mount paths.
        mountpath = None
        for dataname in datanames.split(","):
            dp = self.find_dataprovider(dataname)
            if dp:
                mp = dp.get_mountpath()
                if not mountpath:
                    mountpath = [mp]
                else:
                    mountpath.append(mp)
        return mountpath

    def get_env(self, datanames):
        """Get the environment variables

        Args:
            datanames: Names of the data

        Returns:
            dict: The environment variables
        """
        env = None
        for dataname in datanames.split(","):
            dp = self.find_dataprovider(dataname)
            if dp:
                ev = dp.get_env()
                if not env:
                    env = ev
                else:
                    env["MAD_DATAHOME"] += "," + ev["MAD_DATAHOME"]
        return env

    def prepare_data(self, datanames: str, model_docker: Docker) -> bool:
        """Prepare the data

        Args:
            datanames: Names of the data
            model_docker: The model docker object

        Returns:
            bool: The status of preparing the data
        """
        flag = True
        # iterate through the data names and prepare the data.
        for dataname in datanames.split(","):
            dp = self.find_dataprovider(dataname)
            if dp:
                flag = flag and dp.prepare_data(model_docker)
                self.selected_data_provider["duration"] = dp.duration
                self.selected_data_provider["size"] = dp.size
                print(f"Selected Data Provider::{self.selected_data_provider}")
        return flag
