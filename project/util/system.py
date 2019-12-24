# system.py: System utils
#
# (C) 2019, Daniel Mouritzen

import os
import sys
from glob import glob
from typing import Optional, cast

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


def find_shared_library(name: str) -> Optional[str]:
    if '.' not in name:
        name += '.so'
    prefixes = ['/usr/lib*', '/usr/lib*/*-linux-gnu']
    if 'CONDA_PREFIX' in os.environ:
        prefixes.insert(0, os.environ['CONDA_PREFIX'] + '/lib')
    for prefix in prefixes:
        results = glob(f'{prefix}/{name}*')
        if results:
            results.sort(key=len)
            return results[0]
    return None
