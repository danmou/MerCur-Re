# base.py: Provides base class for predictors
#
# (C) 2019, Daniel Mouritzen

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, cast

import tensorflow as tf

from project.util.tf import auto_shape
from project.util.tf.nested import FlatDataClass


@dataclass(init=False)
class State(FlatDataClass[tf.Tensor]):
    def to_features(self) -> tf.Tensor:
        """Convert state to features."""
        raise NotImplementedError


class OpenLoopPredictor(auto_shape.AbstractRNNCell):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dynamic=False, **kwargs)

    def prior(self, prev_action: tf.Tensor, prev_state_unpacked: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        raise NotImplementedError

    @classmethod
    def state_to_features(cls, state: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        raise NotImplementedError

    @classmethod
    def state_divergence(cls,
                         state1: Tuple[tf.Tensor, ...],
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
             action: tf.Tensor,
             prev_state_unpacked: Tuple[tf.Tensor, ...],
             ) -> Tuple[Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]:
        prior = self.prior(action, prev_state_unpacked)
        return prior, prior


class Predictor(auto_shape.AbstractRNNCell):
    open_loop_predictor_class = OpenLoopPredictor
    open_loop_predictor: OpenLoopPredictor

    def __init__(self, name: str = 'predictor', **kwargs: Any) -> None:
        super().__init__(dynamic=False, name=name, **kwargs)

    def prior(self, prev_action: tf.Tensor, prev_state_unpacked: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        return cast(Tuple[tf.Tensor, ...], self.open_loop_predictor(prev_action, prev_state_unpacked)[0])

    def posterior(self,
                  prev_action: tf.Tensor,
                  latent_obs: tf.Tensor,
                  prev_state_unpacked: Tuple[tf.Tensor, ...],
                  ) -> Tuple[tf.Tensor, ...]:
        raise NotImplementedError

    @classmethod
    def state_to_features(cls, state: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        return cls.open_loop_predictor_class.state_to_features(state)

    @classmethod
    def state_divergence(cls,
                         state1: Tuple[tf.Tensor, ...],
                         state2: Tuple[tf.Tensor, ...],
                         mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        return cls.open_loop_predictor_class.state_divergence(state1, state2, mask)

    @property
    def output_size(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return self.open_loop_predictor.output_size

    @property
    def state_size(self) -> Tuple[int, ...]:
        return self.open_loop_predictor.state_size

    def zero_state(self, batch_size: int = 1, dtype: tf.DType = tf.float32) -> Tuple[tf.Tensor, ...]:
        return self.open_loop_predictor.zero_state(batch_size, dtype)

    @tf.function
    def call(self,
             inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
             prev_state_unpacked: Tuple[tf.Tensor, ...],
             ) -> Tuple[Tuple[Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]], Tuple[tf.Tensor, ...]]:
        obs, action, use_obs = inputs
        prior = self.prior(action, prev_state_unpacked)
        posterior = tf.cond(tf.reduce_any(use_obs),
                            lambda: self.posterior(action, obs, prev_state_unpacked),
                            lambda: prior)
        return (prior, posterior), posterior
