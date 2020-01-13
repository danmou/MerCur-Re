# base.py: Planner base class
#
# (C) 2019, Daniel Mouritzen

from __future__ import annotations

import abc
from typing import Any, Callable, Optional, Tuple, Union

import gym.spaces
import tensorflow as tf

from project.networks.predictors import OpenLoopPredictor
from project.networks.rnns import RNN


class Planner(abc.ABC):
    def __init__(self,
                 predictor: OpenLoopPredictor,
                 objective_fn: Callable[[Tuple[tf.Tensor, ...]], tf.Tensor],
                 action_space: gym.spaces.box,
                 ) -> None:
        self._predictor = predictor
        self._objective_fn = objective_fn
        self.action_space = action_space

    @classmethod
    def from_rnn(cls, rnn: RNN, *args: Any, **kwargs: Any) -> Planner:
        return cls(rnn.predictor.open_loop_predictor, *args, **kwargs)

    @abc.abstractmethod
    def __call__(self,
                 initial_state: Tuple[Union[tf.Tensor, tf.Variable], ...],
                 initial_mean: Optional[tf.Tensor] = None,
                 initial_std_dev: Optional[tf.Tensor] = None,
                 ) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError
