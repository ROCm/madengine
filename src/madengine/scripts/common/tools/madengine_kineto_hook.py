"""Register Kineto's optimizer-step hook for on-demand dynolog traces.

torch.profiler installs an ``optimizer.step`` post-hook only when this module
is imported and ``KINETO_USE_DAEMON`` is set. The dynolog daemon still sees a
PyTorch process that never imports torch.profiler, but iteration capture then
never finishes and no trace file is written. dynolog_start.sh drops this
module into site-packages for the run so an unmodified workload gets the hook.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import sys

if os.environ.get("KINETO_USE_DAEMON"):
    try:
        import torch.profiler  # noqa: F401

        print(
            f"madengine: Kineto optimizer-step hook registered (pid {os.getpid()})",
            file=sys.stderr,
        )
    except Exception:
        pass
