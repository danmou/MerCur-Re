# basic.py: Basic feed-forward neural networks
#
# (C) 2019, Daniel Mouritzen

from typing import Any, Optional, Sequence, Type

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
            layers.append(auto_shape.Dense(num_units, name=f'{name}_dense_{i}'))
            if activation is not None:
                layers.append(activation(name=f'{name}_activation_{i}'))
            if batch_norm:
                layers.append(auto_shape.BatchNormalization())
        super().__init__(layers, name=name)


class ShapedDense(auto_shape.Sequential):
    """A dense layer with given output shape"""
    def __init__(self,
                 shape: Sequence[int],
                 activation: Optional[Type[tf.keras.layers.Layer]] = None,
                 name: str = 'shaped_dense',
                 **kwargs: Any,
                 ) -> None:
        units = np.prod(shape)
        layers = [auto_shape.Dense(units, **kwargs, name=f'{name}_dense')]
        if activation is not None:
            layers.append(activation(name=f'{name}_activation'))
        layers.append(auto_shape.Reshape(shape, name=f'{name}_reshape'))
        super().__init__(layers, name=name)
