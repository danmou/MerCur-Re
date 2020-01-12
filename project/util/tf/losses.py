# losses.py: Loss functions
#
# (C) 2019, Daniel Mouritzen

from typing import Optional

import tensorflow as tf


def apply_mask(loss: tf.Tensor, mask: Optional[tf.Tensor], name: Optional[str] = None) -> tf.Tensor:
    if mask is not None:
        loss = tf.boolean_mask(loss, mask)
    if loss.shape[0] == 0:
        return tf.constant(0.0, name=name)
    return tf.reduce_mean(loss, name=name)


def mse(pred: tf.Tensor, true: tf.Tensor, mask: Optional[tf.Tensor] = None, name: Optional[str] = None) -> tf.Tensor:
    return apply_mask(tf.math.squared_difference(pred, true), mask, name=name)
