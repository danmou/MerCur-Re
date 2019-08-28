#!/usr/bin/env python
# main.py
#
# (C) 2019, Daniel Mouritzen

from typing import Optional, List

import habitat
import numpy as np
import tensorflow as tf
from habitat import SimulatorActions
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


class RandomAgent(habitat.Agent):
    def __init__(self, success_distance, goal_sensor_uuid):
        self.dist_threshold_to_stop = success_distance
        self.goal_sensor_uuid = goal_sensor_uuid

    def reset(self):
        print('\n*** reset ***\n')

    def is_goal_reached(self, observations):
        dist = observations[self.goal_sensor_uuid][0]
        print(f'dist: {dist}')
        return dist <= self.dist_threshold_to_stop

    def act(self, observations):
        print(f'got observations {list(observations.keys())}')
        if self.is_goal_reached(observations):
            action = SimulatorActions.STOP
        else:
            action = np.random.choice(
                [
                    SimulatorActions.MOVE_FORWARD,
                    SimulatorActions.TURN_LEFT,
                    SimulatorActions.TURN_RIGHT,
                ]
            )
        print(f'action: {action}')
        return action

def test_habitat() -> None:
    config = habitat.get_config('configs/habitat/task_pointnav.yaml')
    agent = RandomAgent(
        success_distance=config.TASK.SUCCESS_DISTANCE,
        goal_sensor_uuid=config.TASK.GOAL_SENSOR_UUID,
    )
    benchmark = habitat.Benchmark(config_paths='configs/habitat/task_pointnav.yaml')
    metrics = benchmark.evaluate(agent, num_episodes=10)

    for k, v in metrics.items():
        habitat.logger.info('{}: {:.3f}'.format(k, v))


def main(argv: Optional[List[str]] = None) -> None:
    # test_planet()
    test_habitat()


if __name__ == '__main__':
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.app.run()
