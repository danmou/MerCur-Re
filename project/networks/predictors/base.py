# base.py: Provides base class for predictors
#
# (C) 2019, Daniel Mouritzen

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import tensorflow as tf

from project.util.tf import auto_shape
from project.util.tf.nested import FlatDataClass


@dataclass(init=False)
class State(FlatDataClass[tf.Tensor]):
    def to_features(self) -> tf.Tensor:
        """Convert state to features."""
        raise NotImplementedError


class Predictor(auto_shape.AbstractRNNCell):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dynamic=False, **kwargs)

    def prior(self, prev_state_unpacked: Tuple[tf.Tensor, ...], prev_action: tf.Tensor) -> Tuple[tf.Tensor, ...]:
        raise NotImplementedError

    def posterior(self,
                  prev_state_unpacked: Tuple[tf.Tensor, ...],
                  prev_action: tf.Tensor,
                  latent_obs: tf.Tensor,
                  ) -> Tuple[tf.Tensor, ...]:
        raise NotImplementedError

    @staticmethod
    def state_to_features(state: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        raise NotImplementedError

    @staticmethod
    def state_divergence(state1: Tuple[tf.Tensor, ...],
                         state2: Tuple[tf.Tensor, ...],
                         mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        raise NotImplementedError

    @property
    def output_size(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return self.state_size, self.state_size

    @property
    def state_size(self) -> Tuple[int, ...]:
        raise NotImplementedError

    def zero_state(self, batch_size: int = 1, dtype: tf.DType = tf.float32) -> Tuple[tf.Tensor, ...]:
        return tuple(tf.zeros([batch_size, size], dtype) for size in self.state_size)

    @tf.function
    def call(self,
             inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
             prev_state_unpacked: Tuple[tf.Tensor, ...],
             ) -> Tuple[Tuple[Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]], Tuple[tf.Tensor, ...]]:
        obs, action, use_obs = inputs
        prior = self.prior(prev_state_unpacked, action)
        posterior = tf.cond(tf.reduce_any(use_obs),
                            lambda: self.posterior(prev_state_unpacked, action, obs),
                            lambda: prior)
        return (prior, posterior), posterior
