# Copyright 2019 The Dreamer Authors. All rights reserved.
# Modifications copyright 2020 Daniel Mouritzen.
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

from typing import Any, Sequence

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from project.util.tf import auto_shape

from .basic import ShapedDense
from .wrappers import ExtraBatchDim


class TanhNormalDistribution(tfd.Distribution):
    """Normal distribution transformed with a tanh function. Mean, stddev and mode are approximated by sampling."""
    def __init__(self, mean: tf.Tensor, std: tf.Tensor, feature_dims: int = 1, num_samples: int = 100) -> None:
        dist = tfd.Normal(mean, std)
        dist = tfd.TransformedDistribution(dist, TanhBijector())
        dist = tfd.Independent(dist, feature_dims)
        self._dist = dist
        self._num_samples = num_samples
        super().__init__(dtype=self._dist.dtype,
                         reparameterization_type=self._dist.reparameterization_type,
                         validate_args=False,
                         allow_nan_stats=self._dist.allow_nan_stats,
                         name='TanhNormalDistribution')

    def _batch_shape(self) -> tf.TensorShape:
        return self._dist.batch_shape

    def _event_shape(self) -> tf.TensorShape:
        return self._dist.event_shape

    def sample(self, *args: Any, **kwargs: Any) -> tf.Tensor:
        return self._dist.sample(*args, **kwargs)

    def _mean(self) -> tf.Tensor:
        samples = self._dist.sample(self._num_samples)
        return tf.reduce_mean(samples, 0)

    def _stddev(self) -> tf.Tensor:
        samples = self._dist.sample(self._num_samples)
        mean = tf.reduce_mean(samples, 0, keep_dims=True)
        return tf.sqrt(tf.reduce_mean(tf.pow(samples - mean, 2), 0))

    def _mode(self) -> tf.Tensor:
        samples = self._dist.sample(self._num_samples)
        log_probs = self._dist.log_prob(samples)
        mask = tf.one_hot(tf.argmax(log_probs, axis=0), self._num_samples, axis=0)
        return tf.reduce_sum(samples * mask[..., None], 0)

    def _log_prob(self, value: tf.Tensor) -> tf.Tensor:
        return self._dist.log_prob(value)

    def _entropy(self) -> tf.Tensor:
        sample = self._dist.sample(self._num_samples)
        log_prob = self._dist.log_prob(sample)
        return -tf.reduce_mean(log_prob, 0)


class TanhBijector(tfp.bijectors.Bijector):
    def __init__(self, validate_args: bool = False, name: str = 'tanh') -> None:
        super(TanhBijector, self).__init__(forward_min_event_ndims=0,
                                           validate_args=validate_args,
                                           name=name)

    def _forward(self, x: tf.Tensor) -> tf.Tensor:
        return tf.nn.tanh(x)

    def _inverse(self, y: tf.Tensor) -> tf.Tensor:
        precision = 0.99999997
        clipped = tf.where(tf.less_equal(tf.abs(y), 1.),
                           tf.clip_by_value(y, -precision, precision),
                           y)
        return tf.atanh(clipped)

    def _forward_log_det_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        log2 = tf.math.log(tf.constant(2.0, dtype=x.dtype))
        return 2.0 * (log2 - x - tf.nn.softplus(-2.0 * x))


class TanhNormalTanh(auto_shape.Layer):
    def __init__(self,
                 output_shape: Sequence[int],
                 extra_batch_dim: bool = False,
                 init_std: float = 5.0,
                 min_std: float = 1e-4,
                 mean_scaling: float = 5.0,
                 name: str = 'tanh_normal_tanh',
                 ) -> None:
        self._mean_layer = ShapedDense(output_shape, activation=None, name=f'{name}_mean')
        self._stddev_layer = ShapedDense(output_shape, activation=None, name=f'{name}_stddev')
        if extra_batch_dim:
            self._mean_layer = ExtraBatchDim(self._mean_layer, name=f'{name}_mean_ebd')
            self._stddev_layer = ExtraBatchDim(self._stddev_layer, name=f'{name}_stddev_ebd')
        self._feature_dims = len(output_shape)
        self._init_std = np.log(np.exp(init_std) - 1)
        self._min_std = min_std
        self._mean_scaling = mean_scaling
        super().__init__(batch_dims=self._mean_layer.batch_dims, name=name)

    def call(self, inputs: tf.Tensor) -> TanhNormalDistribution:
        mean = self._mean_layer(inputs)
        # TODO: Does the next line really make any difference?
        mean = self._mean_scaling * tf.tanh(mean / self._mean_scaling)
        stddev = self._stddev_layer(inputs)
        stddev = tf.math.softplus(stddev + self._init_std) + self._min_std
        dist = TanhNormalDistribution(mean, stddev, feature_dims=self._feature_dims)
        return dist
