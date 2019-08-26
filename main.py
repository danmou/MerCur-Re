#!/usr/bin/env python
# main.py
#
# (C) 2019, Daniel Mouritzen

from typing import Optional, List

import tensorflow as tf
from tensorflow.python.util import deprecation

from planet.scripts.train import main as planet_main
from planet.tools import AttrDict


def main(argv: Optional[List[str]] = None) -> None:
    args = AttrDict({
        'logdir': '/tmp/planet-logs',
        'num_runs': 1000,
        'config': 'debug',
        'params': AttrDict({
            'tasks': ['cheetah_run'],
            # 'tasks': ['gym_racecar'],
        }),
        'ping_every': 0,
        'resume_runs': False,
    })
    planet_main(args)


if __name__ == '__main__':
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.app.run()
