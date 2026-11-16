"""nan-doctor, host side: deterministic tools that run NaN investigations of PyTorch workloads.

Standard library only, so the package can be copied next to a reproducer.  Everything here
runs on the host (it never imports torch); the code that runs inside the model's container is
the sibling package `nan_doctor_rt` under scripts/common/tools/nan_doctor/.

Entry point: `madengine nan <command>` or `python3 -m madengine.nan_doctor <command>`
(a vendored copy: `python3 -m nan_doctor <command>`).
"""
