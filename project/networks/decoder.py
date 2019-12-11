# decoder.py: Decoder network
#
# (C) 2019, Daniel Mouritzen

import tensorflow as tf

from project.util.tf import auto_shape


class Decoder(auto_shape.Layer):
    """Decoder with architecture from World Models (D. Ha and J. Schmidhuber)"""
    def __init__(self, name: str = 'image_decoder') -> None:
        super().__init__()
        kwargs = dict(strides=2, activation='relu')
        self._decoder = auto_shape.Sequential([
            auto_shape.Dense(1024, activation=None, name=f'{name}_dense'),
            auto_shape.Reshape([1, 1, 1024], name=f'{name}_reshape'),
            auto_shape.Conv2DTranspose(filters=128, kernel_size=5, **kwargs, name=f'{name}_conv_t_0'),
            auto_shape.Conv2DTranspose(filters=64, kernel_size=5, **kwargs, name=f'{name}_conv_t_1'),
            auto_shape.Conv2DTranspose(filters=32, kernel_size=6, **kwargs, name=f'{name}_conv_t_2'),
            auto_shape.Conv2DTranspose(filters=3, kernel_size=6, strides=2, activation=None, name=f'{name}_conv_t_3'),
        ], name=name)

    def call(self, input: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        assert input.shape.ndims == 2
        return self._decoder(input)
