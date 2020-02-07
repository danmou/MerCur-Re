#!/usr/bin/env python
# run.py: Script to allow running without installing first
#
# (C) 2019, Daniel Mouritzen

import multiprocessing
import os
import sys

from project.util.system import find_shared_library


def run() -> None:
    from project.cli import cli
    cli()


if __name__ == '__main__':
    libtcmalloc = find_shared_library('libtcmalloc')
    if libtcmalloc:
        os.environ['LD_PRELOAD'] = libtcmalloc
    else:
        print('Could not find libtcmalloc.so (part of gperftools); memory leakage may occur.', file=sys.stderr)
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(target=run)  # type: ignore[attr-defined]
    p.start()
    p.join()
