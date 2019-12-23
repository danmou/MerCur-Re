# base.py: Planner base class
#
# (C) 2019, Daniel Mouritzen

import abc
from typing import Callable, Tuple, Union

import gym.spaces
import tensorflow as tf

from project.networks.predictors import Predictor


class Planner(abc.ABC):
    def __init__(self,
                 predictor: Predictor,
                 objective_fn: Callable[[Tuple[tf.Tensor, ...]], tf.Tensor],
                 action_space: gym.spaces.box,
                 ) -> None:
        self._predictor = predictor
        self._objective_fn = objective_fn
        self._action_space = action_space

    @abc.abstractmethod
    def __call__(self, initial_state: Tuple[Union[tf.Tensor, tf.Variable], ...]) -> tf.Tensor:
        raise NotImplementedError
