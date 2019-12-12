# wrappers.py: Additional wrappers to supplement planet_wrappers.
# Note: Classes with @gin.configurable decorator are unpicklable and hence can't be used with Habitat's VectorEnv, so
# instead each class has an associated function that adds the gin-config arguments to a kwargs dict.
#
# (C) 2019, Daniel Mouritzen

from typing import Any, Callable, Dict, Tuple, Type, cast

import gin
import gym.spaces
import numpy as np
from loguru import logger

from project.util.typing import Action, Observations, ObsTuple

from .base import Wrapper


class DiscreteWrapper(Wrapper):
    """
    Wraps a discrete action-space environment into a continuous control task.
    Inspired by https://github.com/piojanu/planet/blob/master/planet/control/wrappers.py#L731-L747
    """

    def __init__(self, env: gym.Env, sample: bool) -> None:
        super().__init__(env)
        self._sample = sample
        self.action_space = gym.spaces.Box(low=-1, high=1,  # PlaNet returns numbers in this range
                                           shape=(self.env.action_space.n,),
                                           dtype=np.float32)

    def step(self, action: Action) -> ObsTuple:
        assert isinstance(action, np.ndarray)
        if self._sample:
            action = action + 1  # shift to make values positive
            if np.sum(action) < 0.01:
                action += 0.01
            assert np.sum(action) > 0, action
            act = np.random.choice(len(action), p=action / np.sum(action))
        else:
            act = np.argmax(action)
        return super().step(act)


@gin.configurable(whitelist=['sample'])
def discrete_wrapper(sample: bool = False) -> Tuple[Type[DiscreteWrapper], Callable[[Dict[str, Any]], Dict[str, Any]]]:
    return DiscreteWrapper, lambda kwargs: {'sample': sample}


class HabitatMinimumDuration(Wrapper):
    """Extends the episode to a given lower number of decision points by preventing stop actions."""
    def __init__(self, env: gym.Env, duration: int) -> None:
        super().__init__(env)
        self._duration = duration
        self._step = 0

    def step(self, action: Action) -> ObsTuple:
        assert isinstance(action, np.ndarray)
        self._step += 1
        if self._step < self._duration:
            action[self.env.stop_action] = self.action_space.low[self.env.stop_action]  # set stop probability to zero
        obs, reward, done, info = super().step(action)
        if done:
            if self._step < self._duration:
                logger.warning(f'Episode finished at step {self._step}, but requirement is {self._duration}.')
            else:
                logger.debug(f'Episode finished at step {self._step}.')
        return obs, reward, done, info

    def reset(self) -> Observations:
        self._step = 0
        return super().reset()


@gin.configurable(whitelist=[])
def habitat_minimum_duration() -> Tuple[Type[HabitatMinimumDuration], Callable[[Dict[str, Any]], Dict[str, Any]]]:
    return HabitatMinimumDuration, lambda kwargs: {'duration': kwargs['min_duration']}


class AutomaticStop(Wrapper):
    """Removes the stop action from the action space and triggers it automatically when the goal is reached."""
    def __init__(self, env: gym.Env, enable: bool, min_duration: int = 0) -> None:
        super().__init__(env)
        self._enable = enable
        self._duration = min_duration
        self._step = 0
        if self._enable:
            self.action_space = gym.spaces.Discrete(self.env.action_space.n - 1)

    def step(self, action: Action) -> ObsTuple:
        assert isinstance(action, int)
        self._step += 1
        if self._enable:
            if self._step >= self._duration and self.env.distance_to_target() < self.env.success_distance:
                action = self.env.stop_action
            elif action >= self.env.stop_action:
                action += 1
        return super().step(action)

    def reset(self) -> Observations:
        self._step = 0
        return super().reset()


@gin.configurable(whitelist=['enable'])
def automatic_stop(enable: bool = False) -> Tuple[Type[AutomaticStop], Callable[[Dict[str, Any]], Dict[str, Any]]]:
    return AutomaticStop, lambda kwargs: {'enable': enable, 'min_duration': kwargs['min_duration']}


class Curriculum(Wrapper):
    """Rejects episodes longer than a gradually increasing threshold."""

    def __init__(self,
                 env: gym.Env,
                 enabled: bool,
                 start_threshold: float,
                 initial_delay: int,
                 increase_rate: float,
                 ) -> None:
        """
        Args:
            env: Environment.
            start_threshold: Initial threshold in meters.
            initial_delay: Number of episodes to wait before increasing threshold.
            increase_rate: Rate of increase in meters per episode.
        """
        super().__init__(env)
        self._enabled = enabled
        self._episodes = -initial_delay
        self._start_threshold = start_threshold
        self._increase_rate = increase_rate

    @property
    def threshold(self) -> float:
        return self._start_threshold + max(0.0, self._episodes * self._increase_rate)

    @property
    def episode_length(self) -> float:
        return cast(float, self.env.habitat_env.current_episode.info["geodesic_distance"])

    def enable_curriculum(self, enable: bool = True) -> None:
        self._enabled = enable

    def reset(self) -> Observations:
        obs = super().reset()
        if self._enabled:
            count = 0
            while self.episode_length > self.threshold:
                logger.trace(f'Curriculum: Rejected episode with length {self.episode_length:.2f}m due to threshold '
                             f'{self.threshold:.2f}m')
                if count >= 500:
                    logger.warning(f'Curriculum: Failed to find a suitable episode with threshold {self.threshold:.2f}m '
                                   f'in 500 steps; increasing start_threshold to '
                                   f'{self._start_threshold + self._increase_rate:.2f}m.')
                    self._start_threshold += self._increase_rate
                    count = 0
                else:
                    count += 1
                obs = self.env.reset()
            self._episodes += 1
        return obs


@gin.configurable(whitelist=['enabled', 'start_threshold', 'initial_delay', 'increase_rate'])
def curriculum(enabled: bool = True,
               start_threshold: float = 1.0,
               initial_delay: int = 0,
               increase_rate: float = 0.1,
               ) -> Tuple[Type[Curriculum], Callable[[Dict[str, Any]], Dict[str, Any]]]:
    return Curriculum, lambda kwargs: {'enabled': enabled,
                                       'start_threshold': start_threshold,
                                       'initial_delay': initial_delay,
                                       'increase_rate': increase_rate}
