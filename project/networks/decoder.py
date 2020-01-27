# decoder.py: Decoder network
#
# (C) 2019, Daniel Mouritzen

from typing import Optional, Type

import gin
import tensorflow as tf

from project.util.tf import auto_shape


@gin.configurable(whitelist=['activation', 'batch_norm'])
class Decoder(auto_shape.Layer):
    """Decoder with architecture from World Models (D. Ha and J. Schmidhuber)"""
    def __init__(self,
                 name: str = 'image_decoder',
                 activation: Optional[Type[tf.keras.layers.Layer]] = auto_shape.ReLU,
                 batch_norm: bool = False,
                 ) -> None:
        super().__init__()
        layers = [auto_shape.Dense(1024, activation=None, name=f'{name}_dense'),
                  auto_shape.Reshape([1, 1, 1024], name=f'{name}_reshape')]
        filter_counts = [128, 64, 32, 3]
        kernel_sizes = [5, 5, 6, 6]
        for i, (filters, kernel_size) in enumerate(zip(filter_counts, kernel_sizes)):
            layers.append(auto_shape.Conv2DTranspose(filters=filters,
                                                     kernel_size=kernel_size,
                                                     strides=2,
                                                     name=f'{name}_conv_t_{i}'))
            if i < 3:
                if activation is not None:
                    layers.append(activation(name=f'{name}_activation_{i}'))
                if batch_norm:
                    layers.append(auto_shape.BatchNormalization())
        self._decoder = auto_shape.Sequential(layers, name=name)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        assert input.shape.ndims == 2
        return self._decoder(input)
