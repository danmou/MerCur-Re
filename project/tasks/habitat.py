# habitat.py: Habitat task
#
# (C) 2019, Daniel Mouritzen

from typing import Any, Callable, Dict, List, Tuple, Type, cast

import gin
import gym

from project.environments import habitat
from project.environments.wrappers import Wrapper

from .planet_tasks import Task


def habitat_env_ctor(*params_tuple: Tuple[str, Any]) -> gym.Env:
    params = dict(params_tuple)
    wrappers = params.pop('wrappers')
    env = habitat.Habitat(**params)
    for wrapper_cls, params in wrappers:
        env = wrapper_cls(env, **params)
    return env


@gin.configurable(whitelist=['max_length', 'wrappers'])
def habitat_train_task(max_length: int = 150,
                       wrappers: List[Tuple[Type[Wrapper], Callable[[Dict[str, Any]], Dict[str, Any]]]] = gin.REQUIRED,
                       ) -> Task:
    return cast(Task, habitat_task(training=True, max_length=max_length, wrappers=wrappers))


@gin.configurable(whitelist=['max_length', 'wrappers'])
def habitat_eval_task(max_length: int = 150,
                      wrappers: List[Tuple[Type[Wrapper], Callable[[Dict[str, Any]], Dict[str, Any]]]] = gin.REQUIRED,
                      ) -> Task:
    return cast(Task, habitat_task(training=False, max_length=max_length, wrappers=wrappers))


@gin.configurable(whitelist=['action_repeat'])
def habitat_task(training: bool,
                 max_length: int,
                 wrappers: List[Tuple[Type[Wrapper], Callable[[Dict[str, Any]], Dict[str, Any]]]],
                 action_repeat: int = 1,
                 ) -> Task:
    state_components = ['reward']
    observation_components = ['image', 'goal']
    metrics = ['success', 'spl', 'path_length', 'optimal_path_length', 'remaining_distance', 'collisions']

    def env_ctor(**kwargs: Any) -> habitat.VectorHabitat:
        kwargs.setdefault('action_repeat', action_repeat)
        kwargs.setdefault('max_duration', max_length)
        kwargs.setdefault('wrappers', [(wrapper, kwarg_fn(kwargs)) for wrapper, kwarg_fn in wrappers])
        kwargs = dict(habitat.get_config(training=training,
                                         top_down_map=kwargs.get('capture_video', False),
                                         max_steps=max_length * action_repeat * 3),  # times 3 because TURN_ANGLE is really 3 actions
                      **kwargs)
        return habitat.VectorHabitat(habitat_env_ctor, kwargs)
    return Task('habitat', env_ctor, max_length, state_components, observation_components, metrics)
