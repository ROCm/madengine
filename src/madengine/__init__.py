"""
madengine - AI Models automation and dashboarding command-line tool.

An AI Models automation and dashboarding command-line tool to run LLMs and Deep Learning
models locally or remotely with CI. The madengine library supports AI automation with:
- AI Models run reliably on supported platforms and drive software quality
- Simple, minimalistic, out-of-the-box solution that enables confidence on hardware and software stack
- Real-time, audience-relevant AI Models performance metrics tracking, presented in clear, intuitive manner
- Best-practices for handling internal projects and external open-source projects

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("madengine")
except PackageNotFoundError:
    # Package is not installed, use a default version
    __version__ = "dev"

__all__ = ["__version__"]
