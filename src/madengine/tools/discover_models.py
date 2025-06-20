"""Module for dynamically discovering models in the project.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import argparse
import os
import json
import importlib.util
import typing
from dataclasses import dataclass, field, asdict

@dataclass
class CustomModel:
    """Dataclass used to pass custom models to madengine."""

    name: str
    dockerfile: str = ""
    dockercontext: str = ""
    scripts: str = ""
    url: str = ""
    cred: str = ""
    owner: str = ""
    data: str = ""
    n_gpus: str = "-1"
    timeout: int = 7200
    training_precision: str = ""
    tags: typing.List[str] = field(default_factory=list)
    args: str = ""
    multiple_results: str = ""
    skip_gpu_arch: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def update_model(self) -> None:
        """Override this method to update your custom model data after initialization.
        Please note that overriding the name or tags in this function will not work
        and may cause undefined behavior!
        """
        pass


class DiscoverModels:
    """Class to discover models in the project."""

    def __init__(self, args: argparse.Namespace):
        """Initialize the DiscoverModels class.
        
        Args:
            args (argparse.Namespace): Arguments passed to the script.
        """
        self.args = args
        # list of models from models.json and scripts/model_dir/models.json
        self.models: typing.List[dict] = []
        # list of custom models from scripts/model_dir/get_models_json.py
        self.custom_models: typing.List[CustomModel] = []
        # list of model names
        self.model_list: typing.List[str] = []
        # list of selected models parsed using --tags argument
        self.selected_models: typing.List[dict] = []

    def discover_models(self) -> None:
        """Discover models in models.json and models.json in model_dir under scripts directory.
        
        Raises:
            FileNotFoundError: models.json file not found.
        """
        model_dir = os.getcwd()
        model_path = os.path.join(model_dir, "models.json")

        # check the models.json file exists in the path of model_dir
        if os.path.exists(model_path):
            # read the models.json file
            with open(model_path) as f:
                model_dict_list: typing.List[dict] = json.load(f)
                self.models = model_dict_list
                self.model_list = [model_dict["name"] for model_dict in model_dict_list]
        else:
            raise FileNotFoundError("models.json file not found.")
        
        # walk through the subdirs in model_dir/scripts directory to find the models.json file
        for dirname in os.listdir(os.path.join(model_dir, "scripts")):
            root = os.path.join(model_dir, "scripts", dirname)
            if os.path.isdir(root):
                files = os.listdir(root)

                if 'models.json' in files and 'get_models_json.py' in files:
                    raise ValueError(f"Both models.json and get_models_json.py found in {root}.")

                if 'models.json' in files:
                    with open(f"{root}/models.json") as f:
                        model_dict_list: typing.List[dict] = json.load(f)
                        for model_dict in model_dict_list:
                            # Update model name using backslash-separated path
                            model_dict["name"] = dirname + '/' + model_dict["name"]
                            # Update relative path for dockerfile and scripts
                            model_dict["dockerfile"] = os.path.normpath(os.path.join("scripts", dirname, model_dict["dockerfile"]))
                            model_dict["scripts"] = os.path.normpath(os.path.join("scripts", dirname, model_dict["scripts"]))
                            self.models.append(model_dict)
                            self.model_list.append(model_dict["name"])

                if 'get_models_json.py' in files:
                    try:
                        # load the module get_models_json.py
                        spec = importlib.util.spec_from_file_location("get_models_json", f"{root}/get_models_json.py")
                        get_models_json = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(get_models_json)
                        assert hasattr(
                            get_models_json, "list_models"
                        ), "Please define a list_models function in get_models_json.py."
                        custom_model_list = get_models_json.list_models()

                        for custom_model in custom_model_list:
                            assert isinstance(
                                custom_model, CustomModel
                            ), "Please use or subclass madengine.tools.discover_models.CustomModel to define your custom model."
                            # Update model name using backslash-separated path
                            custom_model.name = dirname + '/' + custom_model.name
                            # Defer updating script and dockerfile paths until update_model is called
                            self.custom_models.append(custom_model)
                            self.model_list.append(custom_model.name)
                    except AssertionError:
                        print("See madengine/tests/fixtures/dummy/scripts/dummy3/get_models_json.py for an example.")
                        raise

    def select_models(self) -> None:
        """Get the selected models by parsing the --tags argument and expanding custom models.

        Raises:
            ValueError: No models found corresponding to the given tags.
        """
        if self.args.tags:
            # iterate over tags which is a list.
            for tag in self.args.tags:
                # models corresponding to the given tag
                tag_models = []
                # split the tags by ':', strip the tags and remove empty tags.
                tag_list = [tag_.strip() for tag_ in tag.split(':') if tag_.strip()]

                model_name = tag_list[0]

                # if the length of tag_list is greater than 1, then the rest 
                # of the tags are extra args to be passed into the model script.
                if len(tag_list) > 1:
                    extra_args = [tag_ for tag_ in tag_list[1:]]
                    extra_args = [tag_.strip().replace("=", " ") for tag_ in extra_args]
                    extra_args = " --".join(extra_args)
                    extra_args = " --" + extra_args
                else:
                    extra_args = ""
                
                for model in self.models:
                    if model["name"] == model_name or tag in model["tags"] or tag == "all":
                        model_dict = model.copy()
                        model_dict["args"] = model_dict["args"] + extra_args
                        tag_models.append(model_dict)

                for custom_model in self.custom_models:
                    if custom_model.name == model_name or tag in custom_model.tags or tag == "all":
                        custom_model.update_model()
                        # Update relative path for dockerfile and scripts
                        dirname = custom_model.name.split("/")[0]
                        custom_model.dockerfile = os.path.normpath(os.path.join("scripts", dirname, custom_model.dockerfile))
                        custom_model.scripts = os.path.normpath(os.path.join("scripts", dirname, custom_model.scripts))
                        model_dict = custom_model.to_dict()
                        model_dict["args"] = model_dict["args"] + extra_args
                        tag_models.append(model_dict)

                if not tag_models:
                    raise ValueError(f"No models found corresponding to the given tag: {tag}")
                
                self.selected_models.extend(tag_models)

    def print_models(self) -> None:
        if self.selected_models:
            # print selected models using parsed tags and adding backslash-separated extra args
            print(json.dumps(self.selected_models, indent=4))
        else:
            # print list of all model names
            print(f"Number of models in total: {len(self.model_list)}")
            for model_name in self.model_list:
                print(f"{model_name}")

    def run(self, live_output: bool = True):

        self.discover_models()
        self.select_models()
        if live_output:
            self.print_models()
        
        return self.selected_models
                

