"""Tests for madengine.core.additional_context_defaults."""

from madengine.core.additional_context_defaults import (
    DEFAULT_GUEST_OS,
    DEFAULT_GPU_VENDOR,
    apply_build_context_defaults,
)


def test_apply_fills_missing_keys():
    ctx = {}
    apply_build_context_defaults(ctx)
    assert ctx == {"gpu_vendor": DEFAULT_GPU_VENDOR, "guest_os": DEFAULT_GUEST_OS}


def test_apply_preserves_overrides():
    ctx = {"gpu_vendor": "NVIDIA", "guest_os": "CENTOS", "tools": []}
    apply_build_context_defaults(ctx)
    assert ctx["gpu_vendor"] == "NVIDIA"
    assert ctx["guest_os"] == "CENTOS"
    assert ctx["tools"] == []
