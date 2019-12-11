# base.py: Base classes for agents
#
# (C) 2019, Daniel Mouritzen

import abc
from typing import Dict, Optional, Union

import gym
import tensorflow as tf

Observations = Union[tf.Tensor, Dict[str, tf.Tensor]]


class Agent(abc.ABC):
    def __init__(self, action_space: gym.Space) -> None:
        self._action_space = action_space

    def reset(self) -> None:
        """Reset agent's state"""
        pass

    @abc.abstractmethod
    def observe(self, observations: Observations, action: Optional[tf.Tensor]) -> None:
        """Update agent's state based on observations"""
        raise NotImplementedError

    @abc.abstractmethod
    def act(self) -> tf.Tensor:
        """Decide the next action"""
        raise NotImplementedError


class BlindAgent(Agent):
    def __init__(self, action_space: gym.Space) -> None:
        super().__init__(action_space)

    def observe(self, observations: Observations, action: Optional[tf.Tensor]) -> None:
        """Ignore observations; child classes should not override this method"""
        pass

    @abc.abstractmethod
    def act(self) -> tf.Tensor:
        """Decide the next action"""
        raise NotImplementedError
