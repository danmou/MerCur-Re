# planet.py: Functionality for interfacing with PlaNet
#
# (C) 2019, Daniel Mouritzen

from typing import Any, Dict, List, Mapping, Optional

import gin
import gym
import habitat
import planet.control.wrappers as planet_wrappers
import planet.tools
import planet.training.running
import planet.training.utility
import tensorflow as tf
from loguru import logger
from planet.scripts.configs import tasks_lib
from planet.scripts.tasks import Task as PlanetTask
from planet.scripts.train import main as planet_main

from .environments import Habitat, wrappers


@gin.configurable('planet')
def run(logdir: str,
        num_runs: int = 1000,
        ping_every: int = 0,
        resume_runs: bool = False,
        config: str = 'default',
        params: Optional[Dict[str, Any]] = None,
        ) -> None:
    if params is None:
        params = {'tasks': ['habitat']}
    args = planet.tools.AttrDict()
    with args.unlocked:
        args.logdir = logdir
        args.num_runs = num_runs
        args.ping_every = ping_every
        args.resume_runs = resume_runs
        args.config = config
        args.params = planet.tools.AttrDict(params)

    planet_main(args)


def habitat_env_ctor(action_repeat: int, min_length: int, max_length: int) -> gym.Env:
    assert min_length <= max_length, f'{min_length}>{max_length}!'
    logger.debug(f'Collecting episodes between {min_length} and {max_length} steps in length.')
    env = Habitat(max_steps=max_length*action_repeat)
    env = planet_wrappers.ActionRepeat(env, action_repeat)
    env = wrappers.DiscreteWrapper(env)
    env = wrappers.MinimumDuration(env, min_length, stop_index=habitat.SimulatorActions.STOP)
    return env


def planet_habitat_task(config: planet.tools.AttrDict, params: planet.tools.AttrDict) -> PlanetTask:
    action_repeat = params.get('action_repeat', 1)
    max_length = params['max_task_length']
    state_components = ['reward', 'goal']
    env_ctor = planet.tools.bind(
        habitat_env_ctor, action_repeat, config.batch_shape[1], max_length)
    return PlanetTask('habitat', env_ctor, max_length, state_components)


@gin.configurable('planet.tf')
def create_tf_session(options: Optional[Dict] = None, gpu_options: Optional[Dict] = None) -> tf.Session:
    if options is None:
        options = {}
    if gpu_options is not None:
        options['gpu_options'] = tf.GPUOptions(**gpu_options)
    config = tf.ConfigProto(**options)
    try:
        return tf.Session('local', config=config)
    except tf.errors.NotFoundError:
        return tf.Session(config=config)


class PlanetTrainer(planet.training.Trainer):
    def iterate(self, max_step=None, sess=None):
        sess = create_tf_session()
        for score in super().iterate(max_step, sess):
            yield score


# Monkey patch PlaNet to add `habitat` task and use loguru instead of print for logging
tasks_lib.habitat = planet_habitat_task  # type: ignore
planet.control.wrappers.print = logger.info  # type: ignore
planet.training.utility.print = logger.info  # type: ignore
planet.training.running.print = logger.info  # type: ignore
planet.training.trainer.Trainer = PlanetTrainer  # type: ignore
