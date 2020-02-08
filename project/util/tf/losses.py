# losses.py: Loss functions
#
# (C) 2019, Daniel Mouritzen

from typing import Optional

import tensorflow as tf


def reduce_loss(loss: tf.Tensor,
                mask: Optional[tf.Tensor],
                reduce: bool = True,
                name: Optional[str] = None,
                ) -> tf.Tensor:
    if mask is not None:
        loss = tf.boolean_mask(loss, mask)
    if loss.shape[0] == 0:
        return tf.constant(0.0, name=name)
    if reduce:
        return tf.reduce_mean(loss, name=name)
    return loss


def mse(pred: tf.Tensor,
        true: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        reduce: bool = True,
        name: Optional[str] = None,
        ) -> tf.Tensor:
    return reduce_loss(tf.math.squared_difference(pred, true), mask, reduce, name=name)


def binary_crossentropy(pred: tf.Tensor,
                        true: tf.Tensor,
                        mask: Optional[tf.Tensor] = None,
                        reduce: bool = True,
                        from_logits: bool = False,
                        name: Optional[str] = None,
                        ) -> tf.Tensor:
    return reduce_loss(tf.keras.backend.binary_crossentropy(true, pred, from_logits=from_logits),
                       mask,
                       reduce,
                       name=name)


def kl_divergence(mean: tf.Tensor,
                  log_var: tf.Tensor,
                  mask: Optional[tf.Tensor] = None,
                  reduce: bool = True,
                  name: Optional[str] = None,
                  ) -> tf.Tensor:
    return reduce_loss(-0.5 * tf.reduce_sum(1 + log_var - mean**2 - tf.exp(log_var), 1), mask, reduce, name=name)
