# wrappers.py: Additional wrappers to supplement planet.control.wrappers
#
# (C) 2019, Daniel Mouritzen

from typing import Any, Dict, Tuple, Optional, cast

import gin
import gym.spaces
import numpy as np
from loguru import logger
from planet.control.wrappers import Wrapper as PlanetWrapper

Observations = Dict[str, np.ndarray]
ObsTuple = Tuple[Observations, float, bool, Dict[str, Any]]  # obs, reward, done, info


class Wrapper(PlanetWrapper):
    """Simply a type-annotated version of planet_wrappers.Wrapper"""
    def step(self, action: int) -> ObsTuple:
        return cast(ObsTuple, super().step(action))

    def reset(self, **kwargs: Any) -> Observations:
        return cast(Observations, super().reset(**kwargs))


@gin.configurable(whitelist=['sample'])
class DiscreteWrapper(Wrapper):
    """
    Wraps a discrete action-space environment into a continuous control task.
    Inspired by https://github.com/piojanu/planet/blob/master/planet/control/wrappers.py#L731-L747
    """

    def __init__(self, env: gym.Env, sample: bool = False) -> None:
        super().__init__(env)
        self._sample = sample
        self.action_space = gym.spaces.Box(low=-1, high=1,  # PlaNet returns numbers in this range
                                           shape=(self.env.action_space.n,),
                                           dtype=np.float32)

    def step(self, action: np.ndarray) -> ObsTuple:
        if self._sample:
            action = action + 1  # shift to make values positive
            act = np.random.choice(len(action), p=action/np.sum(action))
        else:
            act = np.argmax(action)
        return cast(ObsTuple, self.env.step(act))


class MinimumDuration(Wrapper):
    """Extends the episode to a given lower number of decision points by preventing stop actions."""
    def __init__(self, env: gym.Env, duration: int) -> None:
        super().__init__(env)
        self._duration = duration
        self._step = 0

    def step(self, action: np.ndarray) -> ObsTuple:
        self._step += 1
        if self._step < self._duration:
            action[self.env.stop_action] = self.action_space.low[self.env.stop_action]  # set stop probability to zero
        obs, reward, done, info = cast(ObsTuple, self.env.step(action))
        if done:
            if self._step < self._duration:
                logger.warning(f'Episode finished at step {self._step}, but requirement is {self._duration}.')
            else:
                logger.debug(f'Episode finished at step {self._step}.')
        return obs, reward, done, info

    def reset(self, **kwargs: Any) -> Observations:
        self._step = 0
        return cast(Observations, self.env.reset(**kwargs))


@gin.configurable(whitelist=['enable'])
class AutomaticStop(Wrapper):
    """Removes the stop action from the action space and triggers it automatically when the goal is reached."""
    def __init__(self, env: gym.Env, enable: bool = False, minimum_duration: int = 0) -> None:
        super().__init__(env)
        self._enable = enable
        self._duration = minimum_duration
        self._step = 0
        if self._enable:
            self.action_space = gym.spaces.Discrete(self.env.action_space.n - 1)

    def step(self, action: int) -> ObsTuple:
        self._step += 1
        if self._enable:
            if self._step >= self._duration and self.env.distance_to_target() < self.env.success_distance:
                action = self.env.stop_action
            elif action >= self.env.stop_action:
                action += 1
        return super().step(action)

    def reset(self, **kwargs: Any) -> Observations:
        self._step = 0
        return cast(Observations, self.env.reset(**kwargs))
