# encoder.py: Encoder network
#
# (C) 2019, Daniel Mouritzen

from typing import Dict, Iterable, Optional

import tensorflow as tf

from project.util.tf import auto_shape


class Encoder(auto_shape.Layer):
    """Encoder with architecture from World Models (D. Ha and J. Schmidhuber)"""
    def __init__(self,
                 image_input: str = 'image',
                 vector_inputs: Optional[Iterable[str]] = None,
                 name: str = 'image_encoder') -> None:
        super().__init__(name=name)
        self._image_input = image_input
        self._vector_inputs = [] if vector_inputs is None else list(vector_inputs)
        kwargs = dict(strides=2, activation='relu')
        self._image_enc = auto_shape.Sequential([
            auto_shape.Conv2D(filters=32, kernel_size=4, **kwargs, name=f'{name}_conv_0'),
            auto_shape.Conv2D(filters=64, kernel_size=4, **kwargs, name=f'{name}_conv_1'),
            auto_shape.Conv2D(filters=128, kernel_size=4, **kwargs, name=f'{name}_conv_2'),
            auto_shape.Conv2D(filters=256, kernel_size=4, **kwargs, name=f'{name}_conv_3'),
            auto_shape.Flatten(name=f'{name}_flatten'),
        ], name=f'{name}_sequential')
        self._concat = auto_shape.Concatenate(axis=-1)

    def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:  # type: ignore[override]
        vectors = [inputs[key] for key in self._vector_inputs]
        return self._concat([self._image_enc(inputs[self._image_input])] + vectors)
