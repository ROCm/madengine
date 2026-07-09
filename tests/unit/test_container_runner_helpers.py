"""Unit tests for container_runner_helpers (including log error scan config)."""

import pytest

from madengine.execution.container_runner_helpers import (
    DEFAULT_LOG_ERROR_PATTERNS,
    log_text_has_error_pattern,
    resolve_log_error_scan_config,
    resolve_run_status,
)


class TestResolveLogErrorScanConfig:
    def test_defaults_enable_scan_and_default_patterns(self):
        enabled, patterns, extra = resolve_log_error_scan_config({}, {})
        assert enabled is True
        assert patterns == list(DEFAULT_LOG_ERROR_PATTERNS)
        assert extra == []

    def test_log_error_pattern_scan_false_string(self):
        enabled, _, _ = resolve_log_error_scan_config(
            {}, {"log_error_pattern_scan": "false"}
        )
        assert enabled is False

    def test_context_overrides_model(self):
        enabled, _, _ = resolve_log_error_scan_config(
            {"log_error_pattern_scan": True},
            {"log_error_pattern_scan": False},
        )
        assert enabled is False

    def test_model_only_false_when_no_context(self):
        enabled, _, _ = resolve_log_error_scan_config(
            {"log_error_pattern_scan": False},
            {},
        )
        assert enabled is False

    def test_extra_benign_merges_model_then_context(self):
        _, _, extra = resolve_log_error_scan_config(
            {"log_error_benign_patterns": ["a", "b"]},
            {"log_error_benign_patterns": ["c"]},
        )
        assert extra == ["a", "b", "c"]

    def test_custom_log_error_patterns(self):
        enabled, patterns, _ = resolve_log_error_scan_config(
            {},
            {"log_error_patterns": ["OOM", "Killed"]},
        )
        assert enabled is True
        assert patterns == ["OOM", "Killed"]

    def test_invalid_custom_patterns_falls_back_to_default(self):
        enabled, patterns, _ = resolve_log_error_scan_config(
            {},
            {"log_error_patterns": []},
        )
        assert enabled is True
        assert patterns == list(DEFAULT_LOG_ERROR_PATTERNS)

    def test_invalid_benign_type_skipped(self):
        _, _, extra = resolve_log_error_scan_config(
            {"log_error_benign_patterns": "not-a-list"},
            {},
        )
        assert extra == []

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (False, False),
            ("no", False),
            ("off", False),
            ("0", False),
            (True, True),
            ("true", True),
            (1, True),
        ],
    )
    def test_scan_toggle_coercion(self, raw, expected):
        enabled, _, _ = resolve_log_error_scan_config(
            {}, {"log_error_pattern_scan": raw}
        )
        assert enabled is expected


class TestLogTextHasErrorPattern:
    def test_finds_literal_pattern(self):
        log = "line1\nRuntimeError: boom\n"
        assert log_text_has_error_pattern(log, "RuntimeError:", [])

    def test_respects_benign_substring(self):
        log = "ignore FutureWarning in this line\n"
        assert not log_text_has_error_pattern(
            log,
            "FutureWarning",
            ["FutureWarning"],
            (),
        )

    def test_quotes_in_pattern_no_shell(self):
        """Patterns with quotes must match literally; must not raise or crash."""
        log = "msg: can't happen\n"
        assert log_text_has_error_pattern(log, "can't happen", [])

    def test_excludes_grep_meta_line(self):
        log = "some grep -q stuff RuntimeError: x\nreal RuntimeError: bad\n"
        # First line matches exclusion grep -q.*RuntimeError
        assert log_text_has_error_pattern(log, "RuntimeError:", [], ())

    def test_regex_benign_excludes_rocprof_style_line(self):
        log = (
            "E12345678  generateRocpd.cpp: noise\n"
            "clean RuntimeError: real issue\n"
        )
        assert log_text_has_error_pattern(
            log,
            "RuntimeError:",
            [],
            (r"^E[0-9]{8}.*generateRocpd\.cpp",),
        )

    def test_user_benign_literal_parentheses(self):
        # User-config benign strings must be literal substrings (not broken by ad-hoc regex escaping).
        log = "info (benign marker) ok\nRuntimeError: real\n"
        assert log_text_has_error_pattern(
            log,
            "RuntimeError:",
            ["(benign marker)"],
            (),
        )


class TestResolveRunStatus:
    """ROCM-27774: valid performance metrics must win over a log error-pattern match,
    since the scan cannot tell framework diagnostics apart from a model's own
    generated stdout (e.g. an LLM benchmark response containing "ValueError:")."""

    def test_performance_and_no_errors_is_success(self):
        status, reason = resolve_run_status(has_performance=True, has_errors=False)
        assert status == "SUCCESS"
        assert "no errors" in reason

    def test_performance_with_errors_is_still_success(self):
        status, reason = resolve_run_status(has_performance=True, has_errors=True)
        assert status == "SUCCESS"
        assert "error pattern also matched" in reason

    def test_errors_without_performance_is_failure(self):
        status, reason = resolve_run_status(has_performance=False, has_errors=True)
        assert status == "FAILURE"
        assert "error patterns detected" in reason

    def test_worker_node_without_performance_is_success(self):
        status, reason = resolve_run_status(
            has_performance=False, has_errors=False, is_worker_node=True
        )
        assert status == "SUCCESS"
        assert "worker node" in reason

    def test_worker_node_with_errors_is_still_failure(self):
        # Worker-node exemption only covers missing performance data, not error matches.
        status, reason = resolve_run_status(
            has_performance=False, has_errors=True, is_worker_node=True
        )
        assert status == "FAILURE"

    def test_skip_perf_collection_without_performance_is_success(self):
        status, reason = resolve_run_status(
            has_performance=False, has_errors=False, skip_perf_collection=True
        )
        assert status == "SUCCESS"
        assert "deferred" in reason

    def test_no_performance_no_errors_no_exemption_is_failure(self):
        status, reason = resolve_run_status(has_performance=False, has_errors=False)
        assert status == "FAILURE"
        assert "no performance metrics" in reason
