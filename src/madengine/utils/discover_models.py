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
from rich.console import Console as RichConsole


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
    _fs_rel_path: str = field(default="", repr=False)  # Internal: filesystem relative path

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove internal fields from dict representation
        d.pop("_fs_rel_path", None)
        return d

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
        self.rich_console = RichConsole()
        # list of models from models.json and scripts/model_dir/models.json
        self.models: typing.List[dict] = []
        # list of custom models from scripts/model_dir/get_models_json.py
        self.custom_models: typing.List[CustomModel] = []
        # list of model names
        self.model_list: typing.List[str] = []
        # list of selected models parsed using --tags argument
        self.selected_models: typing.List[dict] = []

        # Setup MODEL_DIR if environment variable is set
        self._setup_model_dir_if_needed()

    def _setup_model_dir_if_needed(self) -> None:
        """Setup model directory if MODEL_DIR environment variable is set.

        This copies docker/, scripts/, and config files (models.json, credential.json, data.json)
        from MODEL_DIR to the current working directory to support the model discovery process.
        This operation is safe for build-only (CPU) nodes as it only involves file operations.
        
        MODEL_DIR defaults to "." (current directory) if not set.
        Only copies if MODEL_DIR points to a different directory than current working directory.
        """
        model_dir_env = os.environ.get("MODEL_DIR", ".")
        
        # Get absolute paths to compare
        model_dir_abs = os.path.abspath(model_dir_env)
        cwd_abs = os.path.abspath(".")
        
        # Only copy if MODEL_DIR points to a different directory (not current dir)
        if model_dir_abs != cwd_abs:
            import shlex
            import subprocess
            from pathlib import Path

            self.rich_console.print(f"[bold cyan]📁 MODEL_DIR environment variable detected:[/bold cyan] [yellow]{model_dir_env}[/yellow]")
            print(f"Copying required files to current working directory: {cwd_abs}")

            try:
                # Check if source directory exists
                if not os.path.exists(model_dir_env):
                    self.rich_console.print(f"[yellow]⚠️  Warning: MODEL_DIR path does not exist: {model_dir_env}[/yellow]")
                    return

                # Copy specific directories and files only (not everything with /*)
                # This prevents copying unwanted subdirectories from MODEL_DIR
                items_to_copy = []
                
                # Directories to copy
                for subdir in ["docker", "scripts"]:
                    src_path = Path(model_dir_env) / subdir
                    if src_path.exists():
                        items_to_copy.append((src_path, subdir, "directory"))
                
                # Files to copy
                for file in ["models.json", "credential.json", "data.json"]:
                    src_file = Path(model_dir_env) / file
                    if src_file.exists():
                        items_to_copy.append((src_file, file, "file"))
                
                if not items_to_copy:
                    self.rich_console.print(f"[yellow]⚠️  No required files/directories found in MODEL_DIR[/yellow]")
                    return
                
                # Copy each item
                copied_count = 0
                for src_path, item_name, item_type in items_to_copy:
                    try:
                        cmd = f"cp -vLR --preserve=all {shlex.quote(str(src_path))} {shlex.quote(str(cwd_abs))}/"
                        result = subprocess.run(
                            cmd, shell=True, capture_output=True, text=True, check=True
                        )
                        copied_count += 1
                        
                        if result.stdout:
                            # Show summary for directories, full output for files
                            if item_type == "directory":
                                lines = result.stdout.splitlines()
                                if len(lines) < 10:
                                    print(result.stdout)
                                else:
                                    print(f"  ✓ Copied {item_name}/ ({len(lines)} files)")
                            else:
                                print(f"  ✓ Copied {item_name}")
                    except subprocess.CalledProcessError as e:
                        self.rich_console.print(f"[yellow]⚠️  Warning: Failed to copy {item_name}: {e}[/yellow]")
                        if e.stderr:
                            print(f"    Error details: {e.stderr}")
                        # Continue with other items even if one fails
                
                if copied_count > 0:
                    self.rich_console.print(f"[green]✅ Successfully copied {copied_count} item(s) from MODEL_DIR[/green]")
                
                print(f"Model dir: {model_dir_env} → current dir: {cwd_abs}")
            except Exception as e:
                self.rich_console.print(f"[yellow]⚠️  Warning: Unexpected error copying MODEL_DIR: {e}[/yellow]")
                # Continue execution even if copy fails

    def discover_models(self) -> None:
        """Discover models in models.json and models.json in model_dir under scripts directory.

        Supports both flat structure (scripts/<dir>/models.json) and nested structure
        (scripts/<submodule>/<dir>/models.json) for git submodules like MAD and MAD-private.

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
            self.rich_console.print("[red]❌ models.json file not found.[/red]")
            raise FileNotFoundError("models.json file not found.")

        # Recursively walk through scripts directory to find all models.json files
        # This supports both flat structure and nested submodule structure
        scripts_dir = os.path.join(model_dir, "scripts")
        for root, dirs, files in os.walk(scripts_dir):
            # Calculate relative path from scripts directory
            rel_path = os.path.relpath(root, scripts_dir)

            # Skip the scripts root directory itself (rel_path == ".")
            if rel_path == ".":
                continue

            if "models.json" in files and "get_models_json.py" in files:
                self.rich_console.print(f"[red]❌ Both models.json and get_models_json.py found in {root}.[/red]")
                raise ValueError(
                    f"Both models.json and get_models_json.py found in {root}."
                )

            if "models.json" in files:
                with open(os.path.join(root, "models.json")) as f:
                    model_dict_list: typing.List[dict] = json.load(f)
                    for model_dict in model_dict_list:
                        # Strip "scripts" from path and keep from the directory before last "scripts"
                        # e.g., Model-Repo1/scripts/Model-Repo2/scripts/dummy -> Model-Repo2/dummy
                        # This identifies submodule boundaries: the dir before "scripts" is the submodule name
                        path_parts = rel_path.split(os.sep)

                        # Find the last occurrence of "scripts" directory
                        last_scripts_idx = -1
                        for i, part in enumerate(path_parts):
                            if part.lower() == "scripts":
                                last_scripts_idx = i

                        # If "scripts" found and there's a directory before it, include that directory
                        # plus everything after "scripts"
                        if last_scripts_idx >= 1:
                            # Include the directory before "scripts" + everything after "scripts"
                            clean_parts = [path_parts[last_scripts_idx - 1]] + path_parts[last_scripts_idx + 1:]
                        elif last_scripts_idx == 0:
                            # "scripts" is first element, just take what's after
                            clean_parts = path_parts[last_scripts_idx + 1:]
                        else:
                            # No "scripts" in path, keep all parts
                            clean_parts = path_parts

                        clean_path = "/".join(clean_parts) if clean_parts else rel_path

                        # Update model name using cleaned path
                        model_dict["name"] = clean_path + "/" + model_dict["name"]
                        # Store the original filesystem path for later use (internal field)
                        model_dict["_fs_rel_path"] = rel_path
                        # Update relative path for dockerfile and scripts (keep full path with "scripts")
                        model_dict["dockerfile"] = os.path.normpath(
                            os.path.join(
                                "scripts", rel_path, model_dict["dockerfile"]
                            )
                        )
                        model_dict["scripts"] = os.path.normpath(
                            os.path.join("scripts", rel_path, model_dict["scripts"])
                        )
                        # Keep tags as-is (do not prefix with dirname); tags are logical names for filtering
                        self.models.append(model_dict)
                        self.model_list.append(model_dict["name"])

            if "get_models_json.py" in files:
                try:
                    # load the module get_models_json.py
                    spec = importlib.util.spec_from_file_location(
                        "get_models_json", os.path.join(root, "get_models_json.py")
                    )
                    get_models_json = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(get_models_json)
                    assert hasattr(
                        get_models_json, "list_models"
                    ), "Please define a list_models function in get_models_json.py."
                    custom_model_list = get_models_json.list_models()

                    for custom_model in custom_model_list:
                        if not isinstance(custom_model, CustomModel):
                            raise TypeError(
                                "Please use or subclass "
                                "madengine.utils.discover_models.CustomModel "
                                "to define your custom model."
                            )
                        # Strip "scripts" from path - keep from directory before last "scripts"
                        path_parts = rel_path.split(os.sep)

                        # Find the last occurrence of "scripts" directory
                        last_scripts_idx = -1
                        for i, part in enumerate(path_parts):
                            if part.lower() == "scripts":
                                last_scripts_idx = i

                        # If "scripts" found and there's a directory before it, include that directory
                        if last_scripts_idx >= 1:
                            clean_parts = [path_parts[last_scripts_idx - 1]] + path_parts[last_scripts_idx + 1:]
                        elif last_scripts_idx == 0:
                            clean_parts = path_parts[last_scripts_idx + 1:]
                        else:
                            clean_parts = path_parts

                        clean_path = "/".join(clean_parts) if clean_parts else rel_path

                        # Update model name using cleaned path
                        custom_model.name = clean_path + "/" + custom_model.name
                        # Store the original filesystem path for later use
                        custom_model._fs_rel_path = rel_path
                        # Defer updating script and dockerfile paths until update_model is called
                        self.custom_models.append(custom_model)
                        self.model_list.append(custom_model.name)
                except AssertionError:
                    self.rich_console.print(
                        "[yellow]💡 See madengine/tests/fixtures/dummy/scripts/dummy3/get_models_json.py for an example.[/yellow]"
                    )
                    raise

    @staticmethod
    def _model_entry_has_tag(tags: typing.Any, tag_value: str) -> bool:
        """True if tag_value is present in tags (list or comma-separated string)."""
        if tags is None:
            return False
        if isinstance(tags, list):
            return tag_value in tags
        if isinstance(tags, str):
            parts = [p.strip() for p in tags.split(",") if p.strip()]
            return tag_value in parts
        return False

    def select_models(self) -> None:
        """Get the selected models by parsing the --tags argument and expanding custom models.

        Scoped tags: ``scope/tag`` (exactly one ``/``, no ``:``) limits to models under
        ``scripts/<scope>/`` (names ``scope/...``) and matches by:
        - tag field value
        - ``all`` to select all models in scope
        - exact model name (e.g. ``MAD/inference``)
        - directory path prefix (e.g. ``MAD/dummy_multi`` matches ``MAD/dummy_multi/model1``)

        The directory path matching provides backward compatibility when models are organized
        in nested directories within submodules.

        Raises:
            ValueError: No models found corresponding to the given tags.
        """
        if self.args.tags:
            # iterate over tags which is a list.
            for tag in self.args.tags:
                # models corresponding to the given tag
                tag_models = []
                # split the tags by ':', strip the tags and remove empty tags.
                tag_list = [tag_.strip() for tag_ in tag.split(":") if tag_.strip()]

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

                # scope/tag: restrict to models from scripts/<scope>/ (e.g. MAD-private/inference)
                scoped: typing.Optional[typing.Tuple[str, str]] = None
                if tag.count("/") == 1 and ":" not in tag:
                    scope_part, filter_part = tag.split("/", 1)
                    if scope_part and filter_part:
                        scoped = (scope_part, filter_part)

                if scoped:
                    scope, tag_filter = scoped
                    prefix = scope + "/"
                    full_name_match = prefix + tag_filter
                    # Also match directory paths: MAD/dummy_multi matches MAD/dummy_multi/model1
                    dir_prefix_match = full_name_match + "/"

                    for model in self.models:
                        if not model["name"].startswith(prefix):
                            continue
                        if (
                            tag_filter == "all"
                            or self._model_entry_has_tag(model.get("tags"), tag_filter)
                            or model["name"] == full_name_match
                            or model["name"].startswith(dir_prefix_match)
                        ):
                            model_dict = model.copy()
                            model_dict["args"] = model_dict["args"] + extra_args
                            tag_models.append(model_dict)

                    for custom_model in self.custom_models:
                        if not custom_model.name.startswith(prefix):
                            continue
                        if (
                            tag_filter == "all"
                            or self._model_entry_has_tag(custom_model.tags, tag_filter)
                            or custom_model.name == full_name_match
                            or custom_model.name.startswith(dir_prefix_match)
                        ):
                            custom_model.update_model()
                            # Use the stored filesystem path (includes "scripts" directories)
                            rel_path = custom_model._fs_rel_path
                            custom_model.dockerfile = os.path.normpath(
                                os.path.join("scripts", rel_path, custom_model.dockerfile)
                            )
                            custom_model.scripts = os.path.normpath(
                                os.path.join("scripts", rel_path, custom_model.scripts)
                            )
                            model_dict = custom_model.to_dict()
                            model_dict["args"] = model_dict["args"] + extra_args
                            tag_models.append(model_dict)
                else:
                    for model in self.models:
                        if (
                            model["name"] == model_name
                            or model["name"].split("/")[-1] == model_name
                            or self._model_entry_has_tag(model.get("tags"), model_name)
                            or model_name == "all"
                        ):
                            model_dict = model.copy()
                            model_dict["args"] = model_dict["args"] + extra_args
                            tag_models.append(model_dict)

                    for custom_model in self.custom_models:
                        if (
                            custom_model.name == model_name
                            or custom_model.name.split("/")[-1] == model_name
                            or self._model_entry_has_tag(custom_model.tags, model_name)
                            or model_name == "all"
                        ):
                            custom_model.update_model()
                            # Use the stored filesystem path (includes "scripts" directories)
                            rel_path = custom_model._fs_rel_path
                            custom_model.dockerfile = os.path.normpath(
                                os.path.join("scripts", rel_path, custom_model.dockerfile)
                            )
                            custom_model.scripts = os.path.normpath(
                                os.path.join("scripts", rel_path, custom_model.scripts)
                            )
                            model_dict = custom_model.to_dict()
                            model_dict["args"] = model_dict["args"] + extra_args
                            tag_models.append(model_dict)

                if not tag_models:
                    self.rich_console.print(f"[red]❌ No models found corresponding to the given tag: {tag}[/red]")
                    raise ValueError(
                        f"No models found corresponding to the given tag: {tag}"
                    )

                self.selected_models.extend(tag_models)

    def print_models(self) -> None:
        if self.selected_models:
            # print selected models using parsed tags and adding backslash-separated extra args
            self.rich_console.print(f"[bold green]📋 Selected Models ({len(self.selected_models)} models):[/bold green]")
            print(json.dumps(self.selected_models, indent=4))
        else:
            # print list of all model names
            self.rich_console.print(f"[bold cyan]📊 Available Models ({len(self.model_list)} total):[/bold cyan]")
            for model_name in self.model_list:
                print(f"  {model_name}")

    def run(self, live_output: bool = True):

        self.discover_models()
        self.select_models()
        if live_output:
            self.print_models()

        return self.selected_models
