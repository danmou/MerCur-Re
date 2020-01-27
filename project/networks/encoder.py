# encoder.py: Encoder network
#
# (C) 2019, Daniel Mouritzen

from typing import Iterable, Mapping, Optional

import gin
import tensorflow as tf

from project.util.tf import auto_shape


@gin.configurable(whitelist=['batch_norm'])
class Encoder(auto_shape.Layer):
    """Encoder with architecture from World Models (D. Ha and J. Schmidhuber)"""
    def __init__(self,
                 image_input: str = 'image',
                 vector_inputs: Optional[Iterable[str]] = None,
                 batch_norm: bool = False,
                 name: str = 'image_encoder') -> None:
        super().__init__(name=name)
        self._image_input = image_input
        self._vector_inputs = [] if vector_inputs is None else list(vector_inputs)
        kwargs = dict(kernel_size=4, strides=2, activation='relu')
        filter_counts = [32, 64, 128, 256]
        layers = []
        for i, filters in enumerate(filter_counts):
            layers.append(auto_shape.Conv2D(filters=filters, **kwargs, name=f'{name}_conv_{i}'))
            if batch_norm:
                layers.append(auto_shape.BatchNormalization())
        layers.append(auto_shape.Flatten(name=f'{name}_flatten'))
        self._image_enc = auto_shape.Sequential(layers, name=f'{name}_sequential')
        self._concat = auto_shape.Concatenate(axis=-1)

    def call(self, inputs: Mapping[str, tf.Tensor]) -> tf.Tensor:
        vectors = [inputs[key] for key in self._vector_inputs]
        return self._concat([self._image_enc(inputs[self._image_input])] + vectors)
