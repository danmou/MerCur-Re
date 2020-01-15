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

from typing import Callable, Optional, Tuple, Union

import gin
import gym.spaces
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RNN, AbstractRNNCell

from project.networks.predictors import OpenLoopPredictor

from .base import Planner


class SimulationRNNCell(AbstractRNNCell):
    """Simple idealized simulation cell for visualization"""
    def __init__(self) -> None:
        super().__init__()

    @property
    def state_size(self) -> int:
        return 3

    def call(self, action: tf.Tensor, states: Tuple[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        state = states[0]
        angle_change = action * np.pi / 2
        mid_angle = state[:, 2, tf.newaxis] + angle_change / 2
        new_state = state + tf.concat([tf.cos(mid_angle) * 0.25, tf.sin(mid_angle) * 0.25, angle_change], axis=1)
        return new_state, new_state


@tf.function
def simulate_plan(actions: tf.Tensor) -> tf.Tensor:
    """Convert sequences of actions into sequences of positions (for visualization purposes)"""
    assert len(actions.shape) == 3
    assert actions.shape[2] == 1, 'This simulation assumes single-dimensional action space'
    rnn = RNN(SimulationRNNCell(), return_sequences=True)
    positions = tf.concat([tf.zeros([actions.shape[0], 1, 2]), rnn(actions)[:, :, :2]], axis=1)
    return positions


def plot_positions(positions: np.ndarray, values: np.ndarray, goal: np.ndarray, step: int) -> None:
    """Plot lines from `positions` with colors according to `values` (which must be between 0 and 1)"""
    goal *= [1.0, -1.0]  # Invert y axis
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list('BlueRed', ['b', 'r'])
    matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=color_map(values))
    plt.plot(positions[:, :, 0].T, positions[:, :, 1].T, alpha=0.3)
    plt.plot(0.0, 0.0, '.k', markersize=10.0)
    plt.plot(goal[0], goal[1], '.g', markersize=10.0)
    plt.axis('equal')
    margin = np.linalg.norm(goal, 2) * 0.2
    plt.xlim(min(0.0, goal[0]) - margin, max(0.0, goal[0]) + margin)
    plt.ylim(min(0.0, goal[1]) - margin, max(0.0, goal[1]) + margin)
    plt.grid(True)
    plt.savefig(f'cem_step_{step:03d}.png')
    plt.close()


@tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ASSERT_STATEMENTS)
def cross_entropy_method(initial_state: Tuple[Union[tf.Tensor, tf.Variable], ...],
                         rnn: RNN,
                         objective_fn: Callable[[Tuple[tf.Tensor, ...]], tf.Tensor],
                         action_space: gym.spaces.box,
                         horizon: int = 12,
                         amount: int = 1000,
                         top_k: int = 100,
                         iterations: int = 10,
                         mean: Optional[np.ndarray] = None,
                         std_dev: Optional[np.ndarray] = None,
                         visualization_goal: Optional[tf.Tensor] = None,
                         ) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Calculates an action sequence of length `horizon` using the following method:
    ```
    initialize mean and std_dev with shape [horizon] + action_shape
    for i in range(iterations):
        sample `amount` action sequences from mean and stddev
        predict objective for all action sequences
        update mean and stddev based on best `top_k` action sequences
    return mean, std_dev
    ```
    """
    action_shape = action_space.low.shape
    assert initial_state[0].shape[0] == 1, 'Initial state can only have a single batch element.'
    initial_state = tf.nest.map_structure(lambda x: tf.concat([x] * amount, 0), initial_state)

    if mean is None:
        mean = tf.stack([(action_space.high + action_space.low) / 2] * horizon, 0)
    else:
        mean = mean[:horizon]
    if std_dev is None:
        std_dev = tf.stack([(action_space.high - action_space.low) / 2] * horizon, 0)
    else:
        std_dev = std_dev[:horizon]

    for i in range(iterations):
        # Sample action proposals from belief.
        normal = tf.random.normal((amount, horizon) + action_shape)
        actions = normal * std_dev[tf.newaxis, :, :] + mean[tf.newaxis, :, :]
        actions = tf.clip_by_value(actions, action_space.low, action_space.high)

        # Evaluate proposal actions.
        states = rnn(actions, initial_state=initial_state)
        objective = objective_fn(states)

        # Re-fit belief to the best ones.
        best_scores, indices = tf.nn.top_k(objective, top_k, sorted=False)
        best_actions = tf.gather(actions, indices)
        mean, variance = tf.nn.moments(best_actions, 0)
        std_dev = tf.sqrt(variance + 1e-6)

        if visualization_goal is not None:
            positions = simulate_plan(best_actions)
            scores_normalized = best_scores - tf.reduce_min(best_scores)
            scores_normalized = scores_normalized / tf.reduce_max(scores_normalized)
            tf.numpy_function(plot_positions, [positions, scores_normalized, visualization_goal, tf.constant(i)], [])

    return mean, std_dev


@gin.configurable(whitelist=['horizon', 'amount', 'top_k', 'iterations'])
class CrossEntropyMethod(Planner):
    def __init__(self,
                 predictor: OpenLoopPredictor,
                 objective_fn: Callable[[Tuple[tf.Tensor, ...]], tf.Tensor],
                 action_space: gym.spaces.box,
                 horizon: int = 12,
                 amount: int = 1000,
                 top_k: int = 100,
                 iterations: int = 10,
                 ) -> None:
        super().__init__(predictor, objective_fn, action_space)
        self.horizon = horizon
        self.amount = amount
        self.top_k = top_k
        self.iterations = iterations
        self._rnn = RNN(predictor, return_sequences=True, name='planner_rnn')

    @tf.function
    def __call__(self,
                 initial_state: Tuple[Union[tf.Tensor, tf.Variable], ...],
                 initial_mean: Optional[tf.Tensor] = None,
                 initial_std_dev: Optional[tf.Tensor] = None,
                 visualization_goal: Optional[tf.Tensor] = None,
                 ) -> Tuple[tf.Tensor, tf.Tensor]:
        mean: tf.Tensor
        std_dev: tf.Tensor
        mean, std_dev = cross_entropy_method(initial_state,
                                             self._rnn,
                                             self._objective_fn,
                                             self.action_space,
                                             self.horizon,
                                             self.amount,
                                             self.top_k,
                                             self.iterations,
                                             initial_mean,
                                             initial_std_dev,
                                             visualization_goal)
        return mean, std_dev
