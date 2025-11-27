"""Template generator for MADEngine distributed execution.

This module provides Jinja2-based template generation for Ansible playbooks
and Kubernetes manifests, supporting environment-specific configurations.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from datetime import datetime


class TemplateGenerator:
    """Template generator for distributed execution configurations."""

    def __init__(
        self, template_dir: Optional[str] = None, values_dir: Optional[str] = None
    ):
        """Initialize the template generator.

        Args:
            template_dir: Path to template directory (defaults to runners/templates)
            values_dir: Path to values directory (defaults to runners/values)
        """
        self.base_dir = Path(__file__).parent
        self.template_dir = (
            Path(template_dir) if template_dir else self.base_dir / "templates"
        )
        self.values_dir = Path(values_dir) if values_dir else self.base_dir / "values"

        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Add custom filters
        self.env.filters["to_yaml"] = self._to_yaml_filter
        self.env.filters["to_json"] = self._to_json_filter
        self.env.filters["basename"] = lambda x: os.path.basename(x)
        self.env.filters["timestamp"] = lambda x: datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

    def _to_yaml_filter(self, value: Any) -> str:
        """Convert value to YAML format."""
        return yaml.dump(value, default_flow_style=False)

    def _to_json_filter(self, value: Any) -> str:
        """Convert value to JSON format."""
        return json.dumps(value, indent=2)

    def load_values(self, environment: str = "default") -> Dict[str, Any]:
        """Load values from environment-specific YAML file.

        Args:
            environment: Environment name (default, dev, prod, test)

        Returns:
            dict: Loaded values
        """
        values_file = self.values_dir / f"{environment}.yaml"
        if not values_file.exists():
            raise FileNotFoundError(f"Values file not found: {values_file}")

        with open(values_file, "r") as f:
            return yaml.safe_load(f) or {}

    def merge_values(
        self, base_values: Dict[str, Any], manifest_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge base values with manifest data.

        Args:
            base_values: Base values from environment file
            manifest_data: Data from build manifest

        Returns:
            dict: Merged values
        """
        merged = base_values.copy()

        # Extract relevant data from manifest
        manifest_values = {
            "manifest": manifest_data,
            "images": manifest_data.get("built_images", {}),
            "models": manifest_data.get("built_models", {}),
            "context": manifest_data.get("context", {}),
            "registry": manifest_data.get("registry", ""),
            "build_timestamp": manifest_data.get("build_timestamp", ""),
            "gpu_vendor": manifest_data.get("context", {}).get("gpu_vendor", ""),
            "docker_build_args": manifest_data.get("context", {}).get(
                "docker_build_arg", {}
            ),
            "docker_env_vars": manifest_data.get("context", {}).get(
                "docker_env_vars", {}
            ),
            "docker_mounts": manifest_data.get("context", {}).get("docker_mounts", {}),
            "docker_gpus": manifest_data.get("context", {}).get("docker_gpus", ""),
        }

        # Deep merge the values
        merged.update(manifest_values)

        # Add generation metadata
        merged["generation"] = {
            "timestamp": datetime.now().isoformat(),
            "generator": "MADEngine Template Generator",
            "version": "1.0.0",
        }

        return merged

    def generate_ansible_playbook(
        self,
        manifest_file: str,
        environment: str = "default",
        output_file: str = "madengine_distributed.yml",
    ) -> str:
        """Generate Ansible playbook from template.

        Args:
            manifest_file: Path to build manifest JSON file
            environment: Environment name for values
            output_file: Output playbook file path

        Returns:
            str: Generated playbook content
        """
        # Load manifest data
        with open(manifest_file, "r") as f:
            manifest_data = json.load(f)

        # Load and merge values
        base_values = self.load_values(environment)
        values = self.merge_values(base_values, manifest_data)

        # Load template
        template = self.env.get_template("ansible/playbook.yml.j2")

        # Generate content
        content = template.render(**values)

        # Write to file
        with open(output_file, "w") as f:
            f.write(content)

        return content

    def generate_kubernetes_manifests(
        self,
        manifest_file: str,
        environment: str = "default",
        output_dir: str = "k8s-manifests",
    ) -> List[str]:
        """Generate Kubernetes manifests from templates.

        Args:
            manifest_file: Path to build manifest JSON file
            environment: Environment name for values
            output_dir: Output directory for manifests

        Returns:
            list: List of generated manifest files
        """
        # Load manifest data
        with open(manifest_file, "r") as f:
            manifest_data = json.load(f)

        # Load and merge values
        base_values = self.load_values(environment)
        values = self.merge_values(base_values, manifest_data)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        generated_files = []

        # Generate each manifest type
        manifest_types = ["namespace", "configmap", "job", "service"]

        for manifest_type in manifest_types:
            template_file = f"k8s/{manifest_type}.yaml.j2"

            try:
                template = self.env.get_template(template_file)
                content = template.render(**values)

                output_file = os.path.join(output_dir, f"{manifest_type}.yaml")
                with open(output_file, "w") as f:
                    f.write(content)

                generated_files.append(output_file)

            except Exception as e:
                print(f"Warning: Could not generate {manifest_type}.yaml: {e}")

        return generated_files

    def generate_slurm_job_array(
        self,
        manifest_file: str,
        environment: str = "default",
        output_file: str = "madengine_job_array.sh",
    ) -> str:
        """Generate SLURM job array script from template.

        Args:
            manifest_file: Path to build manifest JSON file
            environment: Environment name for values
            output_file: Output job script file path

        Returns:
            str: Generated job script content
        """
        # Load manifest data
        with open(manifest_file, "r") as f:
            manifest_data = json.load(f)

        # Load and merge values
        base_values = self.load_values(environment)
        values = self.merge_values(base_values, manifest_data)

        # Extract model tags from manifest for job array
        model_tags = []
        if "models" in manifest_data:
            model_tags = list(manifest_data["models"].keys())
        elif "built_models" in manifest_data:
            model_tags = list(manifest_data["built_models"].keys())
        elif "model_tags" in manifest_data:
            model_tags = manifest_data["model_tags"]

        values["model_tags"] = model_tags

        # Load template
        template = self.env.get_template("slurm/job_array.sh.j2")

        # Generate content
        content = template.render(**values)

        # Write to file
        with open(output_file, "w") as f:
            f.write(content)

        # Make script executable
        os.chmod(output_file, 0o755)

        return content

    def generate_slurm_single_job(
        self,
        manifest_file: str,
        model_tag: str,
        environment: str = "default",
        output_file: str = None,
    ) -> str:
        """Generate SLURM single job script from template.

        Args:
            manifest_file: Path to build manifest JSON file
            model_tag: Specific model tag for this job
            environment: Environment name for values
            output_file: Output job script file path

        Returns:
            str: Generated job script content
        """
        if output_file is None:
            safe_tag = model_tag.replace(":", "-").replace("_", "-")
            output_file = f"madengine_{safe_tag}.sh"

        # Load manifest data
        with open(manifest_file, "r") as f:
            manifest_data = json.load(f)

        # Load and merge values
        base_values = self.load_values(environment)
        values = self.merge_values(base_values, manifest_data)

        # Add specific model tag
        values["model_tag"] = model_tag

        # Load template
        template = self.env.get_template("slurm/single_job.sh.j2")

        # Generate content
        content = template.render(**values)

        # Write to file
        with open(output_file, "w") as f:
            f.write(content)

        # Make script executable
        os.chmod(output_file, 0o755)

        return content

    def generate_slurm_setup_script(
        self,
        manifest_file: str,
        environment: str = "default",
        output_file: str = "setup_environment.sh",
    ) -> str:
        """Generate SLURM environment setup script from template.

        Args:
            manifest_file: Path to build manifest JSON file
            environment: Environment name for values
            output_file: Output setup script file path

        Returns:
            str: Generated setup script content
        """
        # Load manifest data
        with open(manifest_file, "r") as f:
            manifest_data = json.load(f)

        # Load and merge values
        base_values = self.load_values(environment)
        values = self.merge_values(base_values, manifest_data)

        # Add config files that should be copied
        config_files = []
        for file_name in ["credential.json", "data.json", "models.json"]:
            if os.path.exists(file_name):
                config_files.append(file_name)
        values["config_files"] = config_files

        # Load template
        template = self.env.get_template("slurm/setup_environment.sh.j2")

        # Generate content
        content = template.render(**values)

        # Write to file
        with open(output_file, "w") as f:
            f.write(content)

        # Make script executable
        os.chmod(output_file, 0o755)

        return content

    def generate_slurm_inventory(
        self,
        manifest_file: str,
        environment: str = "default",
        output_file: str = "inventory.yml",
    ) -> str:
        """Generate SLURM inventory file from template.

        Args:
            manifest_file: Path to build manifest JSON file
            environment: Environment name for values
            output_file: Output inventory file path

        Returns:
            str: Generated inventory content
        """
        # Load manifest data
        with open(manifest_file, "r") as f:
            manifest_data = json.load(f)

        # Load and merge values
        base_values = self.load_values(environment)
        values = self.merge_values(base_values, manifest_data)

        # Load template
        template = self.env.get_template("slurm/inventory.yml.j2")

        # Generate content
        content = template.render(**values)

        # Write to file
        with open(output_file, "w") as f:
            f.write(content)

        return content

    def list_templates(self) -> Dict[str, List[str]]:
        """List available templates.

        Returns:
            dict: Dictionary of template types and their files
        """
        templates = {}

        for template_type in ["ansible", "k8s", "slurm"]:
            template_path = self.template_dir / template_type
            if template_path.exists():
                templates[template_type] = [
                    f.name
                    for f in template_path.iterdir()
                    if f.is_file() and f.suffix == ".j2"
                ]

        return templates

    def validate_template(self, template_path: str) -> bool:
        """Validate template syntax.

        Args:
            template_path: Path to template file

        Returns:
            bool: True if template is valid
        """
        try:
            template = self.env.get_template(template_path)
            # Try to render with minimal context
            template.render()
            return True
        except Exception as e:
            print(f"Template validation failed: {e}")
            return False


# Convenience functions for backward compatibility
def create_ansible_playbook(
    manifest_file: str = "build_manifest.json",
    environment: str = "default",
    playbook_file: str = "madengine_distributed.yml",
) -> None:
    """Create an Ansible playbook for distributed execution.

    Args:
        manifest_file: Build manifest file
        environment: Environment name for values
        playbook_file: Output Ansible playbook file
    """
    generator = TemplateGenerator()
    generator.generate_ansible_playbook(manifest_file, environment, playbook_file)
    print(f"Ansible playbook created: {playbook_file}")


def create_kubernetes_manifests(
    manifest_file: str = "build_manifest.json",
    environment: str = "default",
    output_dir: str = "k8s-manifests",
) -> None:
    """Create Kubernetes manifests for distributed execution.

    Args:
        manifest_file: Build manifest file
        environment: Environment name for values
        output_dir: Output directory for manifests
    """
    generator = TemplateGenerator()
    generated_files = generator.generate_kubernetes_manifests(
        manifest_file, environment, output_dir
    )
    print(f"Kubernetes manifests created in {output_dir}:")
    for file in generated_files:
        print(f"  - {file}")
