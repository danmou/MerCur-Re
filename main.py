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
    args = AttrDict()
    with args.unlocked:
        args.logdir = '/tmp/planet-logs'
        args.num_runs = 1000
        args.ping_every = 0
        args.resume_runs = False
        args.config = 'default'
        params = AttrDict()
        with params.unlocked:
            params.tasks = ['cheetah_run']
            # params.tasks = ['gym_racecar']
            params.action_repeat = 50
            params.num_seed_episodes = 1
            params.train_steps = 10
            params.test_steps = 10
            params.max_steps = 500
            params.train_collects = [dict(after=10, every=10)]
            params.test_collects = [dict(after=10, every=10)]
            params.model_size = 10
            params.state_size = 5
            params.num_layers = 1
            params.num_units = 10
            params.batch_shape = [5, 10]
            params.loader_every = 5
            params.loader_window = 2
            params.planner_amount = 5
            params.planner_topk = 2
            params.planner_iterations = 2
        args.params = params

    planet_main(args)


if __name__ == '__main__':
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.app.run()
