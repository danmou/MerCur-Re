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

from typing import Callable, Tuple, Union

import gin
import gym.spaces
import tensorflow as tf
from tensorflow.keras.layers import RNN

from project.networks.predictors import Predictor

from .base import Planner


@gin.configurable(whitelist=['horizon', 'amount', 'top_k', 'iterations'])
class CrossEntropyMethod(Planner):
    def __init__(self,
                 predictor: Predictor,
                 objective_fn: Callable[[Tuple[tf.Tensor, ...]], tf.Tensor],
                 action_space: gym.spaces.box,
                 horizon: int = 12,
                 amount: int = 1000,
                 top_k: int = 100,
                 iterations: int = 10,
                 ) -> None:
        super().__init__(predictor, objective_fn, action_space)
        self._horizon = horizon
        self._amount = amount
        self._top_k = top_k
        self._iterations = iterations
        self._embedded_shape = getattr(predictor, 'input_shape', None)
        self._rnn = RNN(predictor, return_sequences=True, name='planner_rnn')

    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ASSERT_STATEMENTS)
    def __call__(self, initial_state: Tuple[Union[tf.Tensor, tf.Variable], ...]) -> tf.Tensor:
        """
        Calculates an action sequence of length `horizon` using the following method:
        ```
        initialize mean and std_dev with shape [horizon] + action_shape
        for i in range(iterations):
            sample `amount` action sequences from mean and stddev
            predict objective for all action sequences
            update mean and stddev based on best `top_k` action sequences
        return mean
        ```
        """
        action_shape = self._action_space.low.shape
        assert initial_state[0].shape[0] == 1, f'Initial state can only have a single batch element, ' \
                                               f'not {initial_state[0].shape[0]}'
        initial_state = tf.nest.map_structure(lambda x: tf.concat([x] * self._amount, 0), initial_state)
        use_obs = tf.zeros([self._amount, self._horizon, 1], tf.bool)
        obs = tf.zeros([self._amount, self._horizon] + self._predictor.input_shape[0][1:])

        mean = tf.stack([(self._action_space.high + self._action_space.low) / 2] * self._horizon, 0)
        std_dev = tf.stack([(self._action_space.high - self._action_space.low) / 2] * self._horizon, 0)
        for i in range(self._iterations):
            # Sample action proposals from belief.
            normal = tf.random.normal((self._amount, self._horizon) + action_shape)
            actions = normal * std_dev[tf.newaxis, :, :] + mean[tf.newaxis, :, :]
            actions = tf.clip_by_value(actions, self._action_space.low, self._action_space.high)

            # Evaluate proposal actions.
            states, _ = self._rnn((obs, actions, use_obs), initial_state=initial_state)
            objective = self._objective_fn(states)

            # Re-fit belief to the best ones.
            _, indices = tf.nn.top_k(objective, self._top_k, sorted=False)
            best_actions = tf.gather(actions, indices)
            mean, variance = tf.nn.moments(best_actions, 0)
            std_dev = tf.sqrt(variance + 1e-6)

        return mean
