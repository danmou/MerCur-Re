# habitat.py: Habitat task
#
# (C) 2019, Daniel Mouritzen

from typing import Any, Callable, Dict, List, Tuple, Type

import gin
import gym
from loguru import logger

import project.models.planet.training
import project.util.planet
from project.environments import habitat, wrappers
from project.util import AttrDict

from .planet_tasks import Task


def habitat_env_ctor(*params_tuple: Tuple[str, Any]) -> gym.Env:
    params = dict(params_tuple)
    wrappers = params.pop('wrappers')
    min_duration = params['min_duration']
    max_duration = params['max_duration']
    assert min_duration <= max_duration, f'{min_duration}>{max_duration}!'
    logger.trace(f'Collecting episodes between {min_duration} and {max_duration} steps in length.')
    env = habitat.Habitat(**params)
    for Wrapper, params in wrappers:
        env = Wrapper(env, **params)
    return env


@gin.configurable(whitelist=['wrappers'])
def habitat_task(config: AttrDict,
                 params: AttrDict,
                 wrappers: List[Tuple[Type[wrappers.Wrapper],
                                      Callable[[Dict[str, Any]], Dict[str, Any]]]] = gin.REQUIRED,
                 ) -> Task:
    action_repeat = params.get('action_repeat', 1)
    max_length = params.max_task_length
    state_components = ['reward']
    observation_components = ['image', 'goal']
    metrics = ['success', 'spl', 'path_length', 'optimal_path_length', 'remaining_distance', 'collisions']
    env_params = {'action_repeat': action_repeat,
                  'min_duration': config.batch_shape[1],
                  'max_duration': max_length,
                  'capture_video': False}
    env_params['wrappers'] = [(Wrapper, kwarg_fn(env_params)) for Wrapper, kwarg_fn in wrappers]
    env_params.update(habitat.get_config(max_steps=max_length*action_repeat*3))  # times 3 because TURN_ANGLE is really 3 actions
    env_ctor = project.util.planet.bind(habitat.VectorHabitat, habitat_env_ctor, env_params)
    return Task('habitat', env_ctor, max_length, state_components, observation_components, metrics)
