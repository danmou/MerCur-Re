#!/usr/bin/env python
# main.py
#
# (C) 2019, Daniel Mouritzen

from typing import Optional, List

import habitat
import tensorflow as tf
from planet.scripts.train import main as planet_main
from planet.tools import AttrDict
from tensorflow.python.util import deprecation


def test_planet() -> None:
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


def test_habitat() -> None:
    env = habitat.Env(config=habitat.get_config("configs/habitat/task_pointnav.yaml"))

    print("Environment creation successful")
    observations = env.reset()

    print("Agent stepping around inside environment.")
    count_steps = 0
    while not env.episode_over:
        observations = env.step(env.action_space.sample())
        count_steps += 1
    print("Episode finished after {} steps.".format(count_steps))


def main(argv: Optional[List[str]] = None) -> None:
    # test_planet()
    test_habitat()


if __name__ == '__main__':
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.app.run()
