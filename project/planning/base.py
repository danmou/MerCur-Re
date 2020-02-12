# base.py: Planner base class
#
# (C) 2019, Daniel Mouritzen

from __future__ import annotations

import abc
from typing import Optional, Tuple, Union
from typing_extensions import Protocol

import gym.spaces
import tensorflow as tf

from project.model import Model
from project.networks.predictors import OpenLoopPredictor


class DecoderFunction(Protocol):
    def __call__(self, __state: tf.Tensor, training: bool) -> tf.Tensor:
        ...


class Planner(abc.ABC):
    def __init__(self,
                 predictor: OpenLoopPredictor,
                 objective_decoder: DecoderFunction,
                 action_space: gym.spaces.box,
                 ) -> None:
        self._predictor = predictor
        self._objective_decoder = objective_decoder
        self.action_space = action_space

    def _objective_fn(self, state: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        obj = self._objective_decoder(self._predictor.state_to_features(state), training=False)
        return tf.reduce_sum(obj, axis=1)

    @classmethod
    @abc.abstractmethod
    def from_model(cls, model: Model, action_space: gym.spaces.box) -> Planner:
        raise NotImplementedError

    @abc.abstractmethod
    def get_action(self,
                   initial_state: Tuple[Union[tf.Tensor, tf.Variable], ...],
                   visualization_goal: Optional[tf.Tensor] = None,
                   ) -> tf.Tensor:
        raise NotImplementedError
