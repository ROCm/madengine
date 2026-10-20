"""Unit tests for reporting: update_perf_csv and PERF_CSV_HEADER."""

import json
import os
import tempfile

import pandas as pd

from madengine.reporting.update_perf_csv import (
    PERF_CSV_HEADER,
    update_perf_csv,
)


class TestPerfCsvHeader:
    """PERF_CSV_HEADER constant and compatibility."""

    def test_perf_csv_header_contains_required_columns(self):
        """Header must contain model, status, and other required columns for perf table."""
        assert "model" in PERF_CSV_HEADER
        assert "status" in PERF_CSV_HEADER
        assert "performance" in PERF_CSV_HEADER
        assert "gpu_architecture" in PERF_CSV_HEADER

    def test_perf_csv_header_is_comma_separated(self):
        """Header is a single line of comma-separated column names."""
        parts = PERF_CSV_HEADER.split(",")
        assert len(parts) >= 20


class TestUpdatePerfCsvCreatesFileWhenMissing:
    """update_perf_csv creates perf CSV with header when file does not exist."""

    def test_exception_result_creates_perf_csv_if_missing(self):
        """When perf_csv does not exist and exception_result is provided, file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            perf_csv = os.path.join(tmpdir, "perf.csv")
            exception_json = os.path.join(tmpdir, "exception.json")
            # Minimal exception entry (status FAILURE for failed run)
            minimal = {
                "model": "test/model",
                "status": "FAILURE",
                "tags": "tag1",
                "performance": "",
                "metric": "",
            }
            with open(exception_json, "w") as f:
                json.dump(minimal, f)

            assert not os.path.exists(perf_csv)
            update_perf_csv(perf_csv, exception_result=exception_json)

            assert os.path.exists(perf_csv)
            df = pd.read_csv(perf_csv)
            assert "model" in df.columns
            assert "status" in df.columns
            assert len(df) == 1
            assert df.iloc[0]["model"] == "test/model"
            assert df.iloc[0]["status"] == "FAILURE"
