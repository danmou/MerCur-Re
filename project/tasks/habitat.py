# habitat.py: Habitat task
#
# (C) 2019, Daniel Mouritzen

import functools
from typing import Any, Callable, Dict, List, Tuple, Type

import gin
import gym

from project.environments import habitat, wrappers

from .planet_tasks import Task


def habitat_env_ctor(*params_tuple: Tuple[str, Any]) -> gym.Env:
    params = dict(params_tuple)
    wrappers = params.pop('wrappers')
    env = habitat.Habitat(**params)
    for Wrapper, params in wrappers:
        env = Wrapper(env, **params)
    return env


@gin.configurable(whitelist=['action_repeat', 'max_length', 'wrappers'])
def habitat_task(action_repeat: int = 1,
                 max_length: int = 150,
                 wrappers: List[Tuple[Type[wrappers.Wrapper],
                                      Callable[[Dict[str, Any]], Dict[str, Any]]]] = gin.REQUIRED,
                 ) -> Task:
    state_components = ['reward']
    observation_components = ['image', 'goal']
    metrics = ['success', 'spl', 'path_length', 'optimal_path_length', 'remaining_distance', 'collisions']
    env_params = {'action_repeat': action_repeat,
                  'max_duration': max_length,
                  'capture_video': False}
    env_params['wrappers'] = [(Wrapper, kwarg_fn(env_params)) for Wrapper, kwarg_fn in wrappers]
    env_params.update(habitat.get_config(max_steps=max_length*action_repeat*3))  # times 3 because TURN_ANGLE is really 3 actions

    def env_ctor(**kwargs: Any) -> habitat.VectorHabitat:
        params = env_params.copy()
        params.update(kwargs)
        return habitat.VectorHabitat(habitat_env_ctor, params)
    return Task('habitat', env_ctor, max_length, state_components, observation_components, metrics)
