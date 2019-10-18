#!/usr/bin/env python
# run.py: Script to allow running without installing first
#
# (C) 2019, Daniel Mouritzen

import sys

sys.path.append('planet')

from project.cli import cli

if __name__ == '__main__':
    cli()
