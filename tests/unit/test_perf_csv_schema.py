#!/usr/bin/env python3
"""
Tests for the declared shape of a perf.csv row.

Three writers create perf.csv — the container runner, the SLURM/base deployment path and
the Kubernetes results mixin — and each used to carry its own copy of the header under a
comment asking the reader to keep them in sync. These tests hold the writers to the
schema, and hold the schema to the manifest it borrows field names from.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json

import pytest

from madengine.schemas import (
    load_schema,
    perf_csv_columns,
    perf_csv_header,
    unknown_perf_columns,
)


@pytest.fixture(scope="module")
def perf_schema():
    """The declared result-row shape."""
    return load_schema("perf_csv.schema.json")


@pytest.fixture(scope="module")
def manifest_schema():
    """The declared manifest shape the result row borrows names from."""
    return load_schema("build_manifest.schema.json")


def _resolve_pointer(document, pointer):
    """Resolve an RFC 6901 JSON pointer, returning None when it does not exist."""
    node = document
    for part in pointer.lstrip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


class TestPerfCsvColumns:
    """The column list itself."""

    def test_header_is_the_columns_in_order(self):
        """The header line is the declaration order, not an independently kept string."""
        assert perf_csv_header() == ",".join(perf_csv_columns())

    def test_columns_match_the_published_order(self):
        """Rows are appended positionally into existing files; the order is the contract."""
        assert perf_csv_columns() == (
            "model",
            "n_gpus",
            "nnodes",
            "gpus_per_node",
            "training_precision",
            "pipeline",
            "args",
            "tags",
            "docker_file",
            "base_docker",
            "docker_sha",
            "docker_image",
            "git_commit",
            "machine_name",
            "deployment_type",
            "launcher",
            "gpu_architecture",
            "performance",
            "metric",
            "relative_change",
            "status",
            "build_duration",
            "test_duration",
            "dataname",
            "data_provider_type",
            "data_size",
            "data_download_duration",
            "build_number",
            "additional_docker_run_options",
        )

    def test_every_column_is_documented(self, perf_schema):
        """A column nobody can explain is a column nobody can consume."""
        undocumented = [
            name
            for name, spec in perf_schema["properties"].items()
            if not spec.get("description")
        ]
        assert undocumented == []

    def test_no_duplicate_columns(self):
        """JSON object keys make this hard to get wrong; say so anyway."""
        assert len(perf_csv_columns()) == len(set(perf_csv_columns()))

    def test_schema_is_valid(self, perf_schema):
        """The schema itself has to be a schema."""
        import jsonschema

        jsonschema.Draft202012Validator.check_schema(perf_schema)


class TestManifestLinkage:
    """Columns that restate a manifest field say which one."""

    def test_manifest_pointers_resolve(self, perf_schema, manifest_schema):
        """Renaming a manifest field without updating the result contract fails here."""
        dangling = {
            column: spec["x-manifest-source"]
            for column, spec in perf_schema["properties"].items()
            if "x-manifest-source" in spec
            and _resolve_pointer(manifest_schema, spec["x-manifest-source"]) is None
        }
        assert dangling == {}

    def test_linked_columns_cover_the_obvious_ones(self, perf_schema):
        """The fields an operator reads off a manifest are the ones that must stay linked."""
        linked = {
            column
            for column, spec in perf_schema["properties"].items()
            if "x-manifest-source" in spec
        }
        assert {"model", "docker_image", "nnodes", "launcher", "deployment_type"} <= linked

    def test_pointers_reach_a_declared_property(self, perf_schema, manifest_schema):
        """A pointer must land on a property definition, not on some intermediate node."""
        for column, spec in perf_schema["properties"].items():
            pointer = spec.get("x-manifest-source")
            if pointer is None:
                continue
            target = _resolve_pointer(manifest_schema, pointer)
            assert isinstance(target, dict), column
            assert "type" in target or "$ref" in target, column


class TestWritersUseTheSchema:
    """No writer keeps its own copy of the header."""

    def test_reporting_header_comes_from_the_schema(self):
        """update_perf_csv exports the header other code imports."""
        from madengine.reporting.update_perf_csv import PERF_CSV_HEADER

        assert PERF_CSV_HEADER == perf_csv_header()

    def test_deployment_writes_the_schema_header(self, tmp_path, monkeypatch):
        """The SLURM/base aggregation path creates perf.csv with the declared columns."""
        from madengine.deployment.base import BaseDeployment

        monkeypatch.chdir(tmp_path)
        # The method touches no instance state, and BaseDeployment is abstract.
        BaseDeployment._ensure_perf_csv_exists(None)

        assert (tmp_path / "perf.csv").read_text().strip() == perf_csv_header()

    def test_existing_file_is_left_alone(self, tmp_path, monkeypatch):
        """A perf.csv from an earlier run keeps its own column order."""
        from madengine.deployment.base import BaseDeployment

        monkeypatch.chdir(tmp_path)
        (tmp_path / "perf.csv").write_text("model,performance\n")
        BaseDeployment._ensure_perf_csv_exists(None)

        assert (tmp_path / "perf.csv").read_text() == "model,performance\n"

    def test_no_writer_hardcodes_the_column_list(self):
        """The header string should exist once, in the schema."""
        from pathlib import Path

        import madengine

        src = Path(madengine.__file__).parent
        needle = "data_download_duration,build_number"
        offenders = [
            str(path.relative_to(src))
            for path in src.rglob("*.py")
            if needle in path.read_text(encoding="utf-8", errors="replace")
        ]
        assert offenders == []


class TestUnknownColumns:
    """Rows are written with extrasaction='ignore'; callers can still ask what was dropped."""

    def test_declared_keys_are_not_reported(self):
        """A row made of columns reports nothing."""
        row = {name: "" for name in perf_csv_columns()}

        assert unknown_perf_columns(row) == []

    def test_undeclared_keys_are_reported(self):
        """A typo'd key would otherwise vanish silently into the CSV writer."""
        row = {"model": "llama", "perfromance": 1.0, "nnodes": 2}

        assert unknown_perf_columns(row) == ["perfromance"]

    def test_partial_rows_are_fine(self):
        """Missing columns are normal: not every run has data provider fields."""
        assert unknown_perf_columns({"model": "llama"}) == []
