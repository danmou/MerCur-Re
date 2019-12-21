# system.py: System utils
#
# (C) 2019, Daniel Mouritzen

import os
import sys
from typing import cast

import psutil


def is_debugging() -> bool:
    """Check if a debugger is attached"""
    try:
        return bool(sys.gettrace())
    except AttributeError:
        return False


def get_memory_usage() -> float:
    """Returns memory usage of current process in gigabytes"""
    return cast(int, psutil.Process(os.getpid()).memory_info().rss) / (1024 ** 3)
