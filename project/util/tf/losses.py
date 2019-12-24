# losses.py: Loss functions
#
# (C) 2019, Daniel Mouritzen

import numpy as np
import tensorflow as tf


def mse(pred: tf.Tensor, true: tf.Tensor, batch_dims: int = 1) -> tf.Tensor:
    return tf.reduce_mean(tf.math.squared_difference(pred, true), axis=list(range(batch_dims, pred.shape.ndims)))


def log_prob(pred: tf.Tensor, true: tf.Tensor, batch_dims: int = 1) -> tf.Tensor:
    """Calculate log probability from a distribution with `pred` as mean and unit variance"""
    return tf.reduce_sum(-0.5 * tf.math.squared_difference(pred, true) - 0.5 * np.log(2. * np.pi),
                         axis=list(range(batch_dims, pred.shape.ndims)))
