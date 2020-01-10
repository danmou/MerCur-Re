# losses.py: Loss functions
#
# (C) 2019, Daniel Mouritzen

from typing import Optional

import tensorflow as tf


def apply_mask(loss: tf.Tensor, mask: Optional[tf.Tensor]) -> tf.Tensor:
    if mask is not None:
        loss = tf.boolean_mask(loss, mask)
    if loss.shape[0] == 0:
        return tf.constant(0.0)
    return tf.reduce_mean(loss)


def mse(pred: tf.Tensor, true: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
    return apply_mask(tf.math.squared_difference(pred, true), mask)
