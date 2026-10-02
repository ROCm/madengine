"""A stand-in for the real TraceLens package, for use in tests.

The real package pins ``protobuf`` and ``xprof`` and only produces reports from
recorded GPU traces, so CI can neither install it nor feed it real input. This
stand-in is placed on ``PYTHONPATH`` instead, which lets the tests drive the
whole madengine analysis pipeline — the packaged analyzer, the in-container
post-script, and ``madengine report tracelens`` — with no GPU and no network.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

__version__ = "0.0.0.dummy"
