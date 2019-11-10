# Copyright 2019 The PlaNet Authors. All rights reserved.
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

"""Batch of environments inside the TensorFlow graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf


class InGraphBatchEnv(object):
    """Batch of environments inside the TensorFlow graph.

    The batch of environments will be stepped and reset inside of the graph using
    a tf.py_func(). The current batch of observations, actions, rewards, and done
    flags are held in according variables.
    """

    def __init__(self, batch_env):
        """Batch of environments inside the TensorFlow graph.

        Args:
          batch_env: Batch environment.
        """
        self._batch_env = batch_env
        batch_dims = (len(self._batch_env),)
        observ_shape = self._parse_shape(self._batch_env.observation_space)
        observ_dtype = self._parse_dtype(self._batch_env.observation_space)
        action_shape = self._parse_shape(self._batch_env.action_space)
        action_dtype = self._parse_dtype(self._batch_env.action_space)
        with tf.compat.v1.variable_scope('env_temporary'):
            self._observ = {k: tf.compat.v1.get_variable(
                'observ_' + k, batch_dims + observ_shape[k], observ_dtype[k],
                tf.constant_initializer(0), trainable=False) for k in observ_shape.keys()}
            self._action = tf.compat.v1.get_variable(
                'action', batch_dims + action_shape, action_dtype,
                tf.constant_initializer(0), trainable=False)
            self._reward = tf.compat.v1.get_variable(
                'reward', batch_dims, tf.float32,
                tf.constant_initializer(0), trainable=False)
            # This variable should be boolean, but tf.compat.v1.scatter_update() does not
            # support boolean resource variables yet.
            self._done = tf.compat.v1.get_variable(
                'done', batch_dims, tf.int32,
                tf.constant_initializer(False), trainable=False)
            self._metrics = {}
            if hasattr(batch_env, 'metric_names'):
                self._metrics = {k: tf.compat.v1.get_variable(
                    'metric_' + k, batch_dims, tf.float32,
                    tf.constant_initializer(0), trainable=False) for k in batch_env.metric_names}

    def __getattr__(self, name):
        """Forward unimplemented attributes to one of the original environments.

        Args:
          name: Attribute that was accessed.

        Returns:
          Value behind the attribute name in one of the original environments.
        """
        return getattr(self._batch_env, name)

    def __len__(self):
        """Number of combined environments."""
        return len(self._batch_env)

    def __getitem__(self, index):
        """Access an underlying environment by index."""
        return self._batch_env[index]

    def step(self, action):
        """Step the batch of environments.

        The results of the step can be accessed from the variables defined below.

        Args:
          action: Tensor holding the batch of actions to apply.

        Returns:
          Operation.
        """
        with tf.name_scope('environment/simulate'):
            observ_dtype = self._parse_dtype(self._batch_env.observation_space)

            def step_func(action):
                observ, reward, done, info = self._batch_env.step(action)
                return ([observ[k] for k in observ_dtype.keys()] +
                        [reward, done] +
                        [info[k].astype(np.float32) for k in self._metrics.keys()])

            step_vals = tf.py_func(
                step_func,
                [action],
                list(observ_dtype.values()) + [tf.float32, tf.bool] + [tf.float32] * len(self._metrics),
                name='step')
            observ, step_vals = step_vals[:len(observ_dtype)], step_vals[len(observ_dtype):]
            observ = dict(zip(observ_dtype.keys(), observ))
            reward, step_vals = step_vals[0], step_vals[1:]
            done, step_vals = step_vals[0], step_vals[1:]
            metrics = step_vals
            assert len(metrics) == len(self._metrics)
            metrics = dict(zip(self._metrics.keys(), metrics))
            return tf.group(
                *(self._observ[k].assign(v) for k, v in observ.items()),
                self._action.assign(action),
                self._reward.assign(reward),
                self._done.assign(tf.to_int32(done)),
                *(self._metrics[k].assign(tf.cast(v, tf.float32)) for k, v in metrics.items()))

    def reset(self, indices=None):
        """Reset the batch of environments.

        Args:
          indices: The batch indices of the environments to reset; defaults to all.

        Returns:
          Batch tensor of the new observations.
        """
        if indices is None:
            indices = tf.range(len(self._batch_env))
        observ_dtype = self._parse_dtype(self._batch_env.observation_space)

        def reset_func(indices):
            observ = self._batch_env.reset(indices)
            return [observ[k] for k in observ_dtype.keys()]

        observ = dict(zip(observ_dtype.keys(), tf.py_func(
            reset_func, [indices], list(observ_dtype.values()), name='reset')))
        reward = tf.zeros_like(indices, tf.float32)
        done = tf.zeros_like(indices, tf.int32)
        metrics = {k: tf.zeros_like(indices, tf.float32) for k in self._metrics.keys()}
        return tf.group(
            *(tf.compat.v1.scatter_update(self._observ[k], indices, v) for k, v in observ.items()),
            tf.compat.v1.scatter_update(self._reward, indices, reward),
            tf.compat.v1.scatter_update(self._done, indices, tf.to_int32(done)),
            *(tf.compat.v1.scatter_update(self._metrics[k], indices, v) for k, v in metrics.items()))

    @property
    def observ(self):
        """Access the variable holding the current observation."""
        return {k: v + 0 for k, v in self._observ.items()}

    @property
    def action(self):
        """Access the variable holding the last received action."""
        return self._action + 0

    @property
    def reward(self):
        """Access the variable holding the current reward."""
        return self._reward + 0

    @property
    def done(self):
        """Access the variable indicating whether the episode is done."""
        return tf.cast(self._done, tf.bool)

    @property
    def metrics(self):
        """Access the variable holding the latest metrics."""
        return {k: v + 0 for k, v in self._metrics.items()}

    def close(self):
        """Send close messages to the external process and join them."""
        self._batch_env.close()

    def _parse_shape(self, space):
        """Get a tensor shape from a OpenAI Gym space.

        Args:
          space: Gym space.

        Raises:
          NotImplementedError: For spaces other than Box and Discrete.

        Returns:
          Shape tuple.
        """
        if isinstance(space, gym.spaces.Discrete):
            return ()
        if isinstance(space, gym.spaces.Box):
            return space.shape
        if isinstance(space, gym.spaces.Dict):
            return {k: self._parse_shape(v) for k, v in space.spaces.items()}
        raise NotImplementedError("Unsupported space '{}.'".format(space))

    def _parse_dtype(self, space):
        """Get a tensor dtype from a OpenAI Gym space.

        Args:
          space: Gym space.

        Raises:
          NotImplementedError: For spaces other than Box and Discrete.

        Returns:
          TensorFlow data type.
        """
        if isinstance(space, gym.spaces.Discrete):
            return tf.int32
        if isinstance(space, gym.spaces.Box):
            if space.low.dtype == np.uint8:
                return tf.uint8
            else:
                return tf.float32
        if isinstance(space, gym.spaces.Dict):
            return {k: self._parse_dtype(v) for k, v in space.spaces.items()}
        raise NotImplementedError()
