# wrappers.py: Additional wrappers to supplement planet.control.wrappers
#
# (C) 2019, Daniel Mouritzen

from typing import Any, Dict, Optional, Tuple, cast

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


class DiscreteWrapper(Wrapper):
    """
    Wraps a discrete action-space environment into a continuous control task.
    Inspired by https://github.com/piojanu/planet/blob/master/planet/control/wrappers.py#L731-L747
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(self.env.action_space.n,),
                                           dtype=np.float32)

    def step(self, action: np.ndarray) -> ObsTuple:
        return cast(ObsTuple, self.env.step(np.argmax(action)))
        # TODO: Maybe sample?


class MinimumDurationRepeat(Wrapper):
    """Extends the episode to a given lower number of decision points by repeating the last observation."""
    def __init__(self, env: gym.Env, duration: int) -> None:
        super().__init__(env)
        self._duration = duration
        self._step = 0
        self._last_episode: Optional[ObsTuple] = None

    def step(self, action: int) -> ObsTuple:
        self._step += 1
        if self._last_episode is not None:
            obs, reward, _, info = self._last_episode
        else:
            obs, reward, done, info = cast(ObsTuple, self.env.step(action))
            if done:
                logger.debug(f'Finished at step {self._step}, repeating for remaining {self._duration - self._step} steps.')
                self._last_episode = obs, reward, False, info
        done = self._step >= self._duration
        return obs, reward, done, info

    def reset(self, **kwargs: Any) -> Observations:
        self._step = 0
        self._last_episode = None
        return cast(Observations, self.env.reset(**kwargs))
