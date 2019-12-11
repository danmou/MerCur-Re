# system.py: System utils
#
# (C) 2019, Daniel Mouritzen

import sys


def is_debugging() -> bool:
    """Check if a debugger is attached"""
    try:
        return bool(sys.gettrace())
    except AttributeError:
        return False
