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

_SAVED_DAEMON = "MADENGINE_KINETO_USE_DAEMON"


def _is_torchrun_launcher() -> bool:
    """True for the torchrun process, not for a script whose name contains it."""
    argv = sys.argv
    if not argv:
        return False
    base = os.path.basename(argv[0])
    if base == "torchrun" or base.startswith("torchrun."):
        return True
    joined = " ".join(argv)
    return (
        "torch.distributed.run" in joined
        or "torch.distributed.elastic" in joined
        or "torch/distributed/run.py" in joined
    )


if os.environ.get("KINETO_USE_DAEMON") or os.environ.get(_SAVED_DAEMON):
    # torchrun imports torch before it spawns the workers. With the daemon
    # enabled, that import registers the launcher too, and the worker then
    # segfaults while flushing its on-demand trace. Drop the daemon for the
    # launcher only; the worker puts it back before importing torch.
    if _is_torchrun_launcher():
        os.environ[_SAVED_DAEMON] = os.environ.get("KINETO_USE_DAEMON", "")
        os.environ.pop("KINETO_USE_DAEMON", None)
    else:
        saved = os.environ.get(_SAVED_DAEMON)
        if saved:
            os.environ["KINETO_USE_DAEMON"] = saved
        if os.environ.get("KINETO_USE_DAEMON"):
            try:
                import torch.profiler  # noqa: F401

                print(
                    "madengine: Kineto optimizer-step hook registered "
                    f"(pid {os.getpid()})",
                    file=sys.stderr,
                )
            except Exception:
                pass
