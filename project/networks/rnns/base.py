# base.py: Basic RNN class for use with Predictor cells
#
# (C) 2019, Daniel Mouritzen

import abc
from typing import Any, Optional, Tuple, Type

import gin
import tensorflow as tf

from project.networks.predictors import Predictor
from project.util.tf import auto_shape


class RNN(abc.ABC, auto_shape.Layer):
    def __init__(self, predictor_class: Type[Predictor], *, name: str = 'rnn', **kwargs: Any) -> None:
        self.predictor_class = predictor_class
        kwargs['min_batch_shape'] = kwargs.get('min_batch_shape', [1, 2])
        super().__init__(batch_dims=2, name=name, **kwargs)

    @property
    @abc.abstractmethod
    def predictor(self) -> Predictor:
        raise NotImplementedError

    @abc.abstractmethod
    def state_to_features(self, state: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def state_divergence(self,
                         state1: Tuple[tf.Tensor, ...],
                         state2: Tuple[tf.Tensor, ...],
                         mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def closed_loop(self,
                    observations: tf.Tensor,
                    actions: tf.Tensor,
                    initial_state: Optional[tf.Tensor] = None,
                    mask: Optional[tf.Tensor] = None,
                    ) -> Tuple[Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]:
        raise NotImplementedError

    @abc.abstractmethod
    def open_loop(self,
                  actions: tf.Tensor,
                  initial_state: Optional[tf.Tensor] = None,
                  mask: Optional[tf.Tensor] = None,
                  ) -> Tuple[tf.Tensor, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def call(self,
             inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
             initial_state: Optional[tf.Tensor] = None,
             mask: Optional[tf.Tensor] = None,
             ) -> Tuple[Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]:
        raise NotImplementedError


@gin.configurable(module='rnns', whitelist=[])
class SimpleRNN(RNN):
    def __init__(self, predictor_class: Type[Predictor], *, name: str = 'simple_rnn') -> None:
        self._predictor = predictor_class(name=f'{name}_predictor')
        self.rnn = auto_shape.RNN(self._predictor, return_sequences=True, name=f'{name}_inner')
        super().__init__(predictor_class=predictor_class, name=name)

    @property
    def predictor(self) -> Predictor:
        return self._predictor

    def state_to_features(self, state: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        return self._predictor.state_to_features(state)

    def state_divergence(self,
                         state1: Tuple[tf.Tensor, ...],
                         state2: Tuple[tf.Tensor, ...],
                         mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        return self._predictor.state_divergence(state1, state2, mask)

    def closed_loop(self,
                    observations: tf.Tensor,
                    actions: tf.Tensor,
                    initial_state: Optional[tf.Tensor] = None,
                    mask: Optional[tf.Tensor] = None,
                    ) -> Tuple[Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]:
        use_obs = tf.ones(observations.shape[:2] + [1], tf.bool)
        prior, posterior = self((observations, actions, use_obs), initial_state=initial_state, mask=mask)
        return prior, posterior

    def open_loop(self,
                  actions: tf.Tensor,
                  initial_state: Optional[tf.Tensor] = None,
                  mask: Optional[tf.Tensor] = None,
                  ) -> Tuple[tf.Tensor, ...]:
        obs_spec: tf.keras.layers.InputSpec = self._predictor.input_spec[0]  # type: ignore[index]
        obs = tf.zeros(actions.shape[:2] + obs_spec.shape[1:], obs_spec.dtype)
        use_obs = tf.zeros(actions.shape[:2] + [1], tf.bool)
        prior: Tuple[tf.Tensor, ...]
        prior, _ = self((obs, actions, use_obs), initial_state=initial_state, mask=mask)
        return prior

    def call(self,
             inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
             initial_state: Optional[tf.Tensor] = None,
             mask: Optional[tf.Tensor] = None,
             ) -> Tuple[Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]:
        prior: Tuple[tf.Tensor, ...]
        posterior: Tuple[tf.Tensor, ...]
        prior, posterior = self.rnn(inputs, initial_state=initial_state, mask=mask)
        return prior, posterior
