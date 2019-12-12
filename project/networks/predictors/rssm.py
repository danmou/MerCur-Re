# Copyright 2019 The PlaNet Authors. All rights reserved.
# Modifications copyright 2019 Daniel Mouritzen.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import gin
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from project.util.planet.mask import apply_mask
from project.util.tf import auto_shape

from ..basic import SequentialBlock
from .base import Predictor, State


@dataclass(init=False)
class StateDist(State):
    mean: tf.Tensor = None
    stddev: tf.Tensor = None
    sample: tf.Tensor = None

    def to_features(self) -> tf.Tensor:
        return self.sample

    def to_dist(self, mask: Optional[tf.Tensor] = None) -> tfd.MultivariateNormalDiag:
        """Convert to distribution."""
        stddev = self.stddev
        if mask is not None:
            stddev = apply_mask(stddev, mask, value=1)
        dist = tfd.MultivariateNormalDiag(self.mean, stddev)
        return dist

    def divergence(self, other: StateDist, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Compute the divergence measure with other state."""
        divergence = tfd.kl_divergence(self.to_dist(mask), other.to_dist(mask))
        if mask is not None:
            divergence = apply_mask(divergence, mask)
        return divergence


@dataclass(init=False)
class FullState(State):
    state: StateDist = field(default_factory=StateDist)
    belief: tf.Tensor = None

    def to_features(self) -> tf.Tensor:
        return tf.concat([self.state.to_features(), self.belief], -1)


class SequentialNormalBlock(auto_shape.Layer):
    """A number of sequential layers producing a normal distribution"""
    def __init__(self,
                 num_layers: int,
                 num_units: int,
                 activation: str,
                 output_size: int,
                 min_stddev: float = 0.1,
                 mean_only: bool = False,
                 **kwargs: Any,
                 ) -> None:
        super().__init__(**kwargs)
        self._mean_only = mean_only
        self._min_stddev = min_stddev
        self._hidden_layers = SequentialBlock(num_units=num_units, num_layers=num_layers, activation=activation)
        self._mean_layer = auto_shape.Dense(output_size, activation=None)
        self._stddev_layer = auto_shape.Dense(output_size, activation='softplus')

    def call(self, inputs: tf.Tensor) -> List[tf.Tensor]:
        hidden = self._hidden_layers(inputs)
        mean = self._mean_layer(hidden)
        stddev = self._stddev_layer(hidden)
        stddev += self._min_stddev
        if self._mean_only:
            sample = mean
        else:
            sample = tfd.MultivariateNormalDiag(mean, stddev).sample()
        return list(StateDist(mean, stddev, sample))


@gin.configurable(module='predictors')
class RSSM(Predictor):
    r"""
    Deterministic and stochastic state model.

    The stochastic latent is computed from the hidden state at the same time
    step. If an observation is present, the posterior latent is compute from both
    the hidden state and the observation.

    Prior:    Posterior:

    (a)       (a)
       \         \
        v         v
    [h]->[h]  [h]->[h]
        ^ |       ^ :
       /  v      /  v
    (s)  (s)  (s)  (s)
                    ^
                    :
                   (o)
    """

    def __init__(self,
                 state_size: int = gin.REQUIRED,
                 belief_size: int = gin.REQUIRED,
                 embed_size: int = gin.REQUIRED,
                 mean_only: bool = False,
                 min_stddev: float = 0.1,
                 activation: str = 'relu',
                 num_layers: int = 1,
                 name: str = 'rssm',
                 ) -> None:
        self._state_size = state_size
        self._belief_size = belief_size
        self._input_layers = SequentialBlock(num_units=embed_size,
                                             num_layers=num_layers,
                                             activation=activation,
                                             name=f'{name}_input_block')
        self._cell = auto_shape.GRUCell(self._belief_size, name=f'{name}_gru_cell')
        kwargs = dict(num_layers=num_layers,
                      num_units=embed_size,
                      activation=activation,
                      output_size=state_size,
                      min_stddev=min_stddev,
                      mean_only=mean_only)
        self._prior_dist = SequentialNormalBlock(**kwargs, name=f'{name}_prior_block')  # type: ignore[arg-type]
        self._posterior_dist = SequentialNormalBlock(**kwargs, name=f'{name}_posterior_block')  # type: ignore[arg-type]
        super().__init__(name=name)

    def state_to_features(self, state: List[tf.Tensor]) -> tf.Tensor:
        return FullState(*state).to_features()

    def state_divergence(self,
                         state1: List[tf.Tensor],
                         state2: List[tf.Tensor],
                         mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        return FullState(*state1).state.divergence(FullState(*state2).state, mask)

    @property
    def state_size(self) -> List[int]:
        return [self._state_size, self._state_size, self._state_size, self._belief_size]

    def _transition(self, prev_state_unpacked: List[tf.Tensor], prev_action: tf.Tensor) -> tf.Tensor:
        """Compute next (deterministic) belief."""
        prev_state = FullState(*prev_state_unpacked)
        hidden = tf.concat([prev_state.state.sample, prev_action], -1)
        hidden = self._input_layers(hidden)
        belief, _ = self._cell(hidden, [prev_state.belief])
        return belief

    def _prior(self, prev_state_unpacked: List[tf.Tensor], prev_action: tf.Tensor) -> List[tf.Tensor]:
        """Compute prior next state by applying the transition dynamics."""
        belief = self._transition(prev_state_unpacked, prev_action)
        prior_state = self._prior_dist(belief)
        prior = FullState(state=StateDist(*prior_state), belief=belief)
        return list(prior)

    def _posterior(self, prev_state_unpacked: List[tf.Tensor], prev_action: tf.Tensor, latent_obs: tf.Tensor) -> List[tf.Tensor]:
        """Compute posterior state from previous state and current observation."""
        belief = self._transition(prev_state_unpacked, prev_action)
        hidden = tf.concat([belief, latent_obs], -1)
        posterior_state_unpacked = self._posterior_dist(hidden)
        posterior = FullState(state=StateDist(*posterior_state_unpacked), belief=belief)
        return list(posterior)
