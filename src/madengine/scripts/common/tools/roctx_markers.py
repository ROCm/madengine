"""roctx marker helpers for madengine model scripts.

When running under RTL (HSA_TOOLS_LIB=librtl.so + LD_PRELOAD=librtl.so),
markers appear in the trace timeline. When RTL is not loaded, all calls
are silent no-ops with zero overhead.

Usage:
    from roctx_markers import roctx_range, roctx_mark

    with roctx_range("warmup"):
        for i in range(warmup_iters):
            model(input)
        torch.cuda.synchronize()

    with roctx_range("timed"):
        for i in range(timed_iters):
            with roctx_range(f"step_{i}"):
                output = model(input)
            torch.cuda.synchronize()

    roctx_mark("done")

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import ctypes
import contextlib

_push = None
_pop = None
_mark = None

try:
    _lib = ctypes.CDLL(None)
    _push = _lib.roctxRangePushA
    _push.argtypes = [ctypes.c_char_p]
    _push.restype = ctypes.c_int
    _pop = _lib.roctxRangePop
    _pop.restype = ctypes.c_int
    _mark_fn = _lib.roctxMarkA
    _mark_fn.argtypes = [ctypes.c_char_p]
    _mark_fn.restype = None
    _mark = _mark_fn
except (OSError, AttributeError):
    pass


@contextlib.contextmanager
def roctx_range(name: str):
    """Context manager: emits a push/pop roctx range. No-ops without RTL."""
    if _push is not None:
        _push(name.encode())
    try:
        yield
    finally:
        if _pop is not None:
            _pop()


def roctx_mark(name: str):
    """Emit an instant roctx marker. No-ops without RTL."""
    if _mark is not None:
        _mark(name.encode())
