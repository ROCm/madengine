"""Container-side runtime of nan-doctor: capture, replay and analysis of NaN capsules.

Imports no torch at package import time, so a .pth hook can import it into every interpreter.
"""
