# basic.py: Basic feed-forward neural networks
#
# (C) 2019, Daniel Mouritzen

from typing import Any, Optional, Sequence

import numpy as np
import tensorflow as tf

from project.util.tf import auto_shape


class SequentialBlock(auto_shape.Sequential):
    """A number of sequential dense layers of the same size"""
    def __init__(self,
                 num_units: int,
                 num_layers: int,
                 activation: Optional[Type[tf.keras.layers.Layer]],
                 batch_norm: bool = True,
                 initial_layers: Optional[Sequence[tf.keras.layers.Layer]] = None,
                 name: str = 'sequential_block',
                 ) -> None:
        layers = [] if initial_layers is None else list(initial_layers)
        for i in range(num_layers):
            layers.append(auto_shape.Dense(num_units, activation=activation, name=f'{name}_dense_{i}'))
            if batch_norm:
                layers.append(auto_shape.BatchNormalization())
        super().__init__(layers, name=name)


class ShapedDense(auto_shape.Sequential):
    """A dense layer with given output shape"""
    def __init__(self,
                 shape: Sequence[int],
                 name: str = 'shaped_dense',
                 **kwargs: Any,
                 ) -> None:
        units = np.prod(shape)
        super().__init__([auto_shape.Dense(units, **kwargs, name=f'{name}_dense'),
                          auto_shape.Reshape(shape, name=f'{name}_reshape')], name=name)
