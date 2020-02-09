# simple.py: Simple agents for baselines
#
# (C) 2019, Daniel Mouritzen

from typing import Optional

import gym
import tensorflow as tf

from .base import BlindAgent


class RandomAgent(BlindAgent):
    def act(self) -> tf.Tensor:
        return tf.convert_to_tensor(self.action_space.sample())


class ConstantAgent(BlindAgent):
    def __init__(self, action_space: gym.Space, value: Optional[tf.Tensor] = None) -> None:
        super().__init__(action_space)
        if value is None:
            value = tf.zeros_like(self.action_space.sample())
        assert self.action_space.contains(value.numpy())
        self._value = value

    def act(self) -> tf.Tensor:
        return self._value
