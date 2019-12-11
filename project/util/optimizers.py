# optimizers.py: Custom optimizer utils
#
# (C) 2019, Daniel Mouritzen

from types import MethodType
from typing import Any, Callable, Iterable, List, NoReturn, Optional, Tuple

import gin
import tensorflow as tf
from tensorflow.keras import optimizers


@gin.configurable(module='optimizers', whitelist=['optimizer', 'clip_norm'])
def with_global_norm_clipping(optimizer: optimizers.Optimizer = gin.REQUIRED,
                              clip_norm: Optional[float] = gin.REQUIRED) -> optimizers.Optimizer:
    original_compute_gradients = optimizer._compute_gradients

    def _compute_gradients(self,
                           loss: Callable[[], tf.Tensor],
                           var_list: Iterable[tf.Variable],
                           grad_loss: Optional[tf.Tensor] = None
                           ) -> List[Tuple[Optional[tf.Tensor], tf.Variable]]:
        grads_and_vars = original_compute_gradients(loss, var_list, grad_loss)
        print('clipping')
        if clip_norm is None:
            return grads_and_vars
        grads, var_list = zip(*grads_and_vars)
        grads = tf.clip_by_global_norm(grads, clip_norm)
        return list(zip(grads, var_list))

    def get_config(self) -> NoReturn:
        raise NotImplementedError

    optimizer._compute_gradients = MethodType(_compute_gradients, optimizer)
    optimizer.get_config = MethodType(get_config, optimizer)
    return optimizer
