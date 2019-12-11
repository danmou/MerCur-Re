# auto_shape.py: Wrappers for tf.keras layers and models with automatic shape definition
#
# (C) 2019, Daniel Mouritzen

from typing import Any, Optional, Union

import tensorflow as tf
from tensorflow.keras import layers

from project.util.typing import Nested


class AutoShapeMixin:
    """
    Mixin for `tf.keras.layers.Layer`s and subclasses to automatically define input and output specs when model is
    built using `build_with_input`. Must be listed before `tf.keras.layers.Layer` when subclassing. Only works for
    models and layers with static input and output shapes. First dimension is assumed to be batch dimension.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.batch_dims: int = kwargs.pop('batch_dims', 1)
        super().__init__(*args, **kwargs)
        assert not getattr(self, 'dynamic'), 'AutoShapeMixin should not be used with dynamic layers!'
        self._input_spec: Optional[layers.InputSpec] = None
        self._output_spec: Optional[layers.InputSpec] = None
        self.built_with_input = False

    def build_with_input(self, input: Union[Nested[tf.TensorSpec], Nested[tf.Tensor]], *args: Any, **kwargs: Any) -> None:
        bd = self.batch_dims
        self._input_spec = tf.nest.map_structure(
            lambda x: layers.InputSpec(shape=[None]*bd + x.shape[bd:], dtype=x.dtype), input)
        dummy_input = tf.nest.map_structure(lambda t: tf.zeros([2]*bd + t.shape[bd:], t.dtype), input)
        dummy_output = super().__call__(dummy_input, *args, **kwargs)
        self._output_spec = tf.nest.map_structure(lambda x: layers.InputSpec(shape=[None]*bd + x.shape[bd:],
                                                                             dtype=x.dtype), dummy_output)
        self.built_with_input = True

    def __call__(self, inputs: Nested[tf.Tensor], *args: Any, **kwargs: Any) -> Any:
        if not self.built_with_input:
            self.build_with_input(inputs, *args, **kwargs)
        return super().__call__(inputs, *args, **kwargs)

    @property
    def input_spec(self) -> Optional[Nested[layers.InputSpec]]:
        return self._input_spec

    @input_spec.setter
    def input_spec(self, value: Optional[layers.InputSpec]) -> None:
        self._input_spec = value

    @property
    def output_spec(self) -> Optional[Nested[layers.InputSpec]]:
        return self._output_spec

    @output_spec.setter
    def output_spec(self, value: Optional[layers.InputSpec]) -> None:
        self._output_spec = value

    @property
    def input_shape(self) -> Nested[tf.TensorShape]:
        assert self.input_spec is not None, 'build_with_input has not been called; input shape is not defined'
        return tf.nest.map_structure(lambda x: x.shape, self.input_spec)

    @property
    def output_shape(self) -> Nested[tf.TensorShape]:
        assert self.output_spec is not None, 'build_with_input has not been called; output shape is not defined'
        return tf.nest.map_structure(lambda x: x.shape, self.output_spec)

    @property
    def input_dtype(self) -> Nested[tf.TensorShape]:
        assert self.input_spec is not None, 'build_with_input has not been called; input dtype is not defined'
        return tf.nest.map_structure(lambda x: x.dtype, self.input_spec)

    @property
    def output_dtype(self) -> Nested[tf.TensorShape]:
        assert self.output_spec is not None, 'build_with_input has not been called; output dtype is not defined'
        return tf.nest.map_structure(lambda x: x.dtype, self.output_spec)

    def compute_output_shape(self, input_shape: Nested[tf.TensorShape]) -> Nested[tf.TensorShape]:
        if self.output_spec is None:
            return super().compute_output_shape(input_shape)
        batch_shape = tf.nest.flatten(input_shape)[0][:self.batch_dims]
        return tf.nest.map_structure(lambda x: batch_shape + x[self.batch_dims:], self.output_shape)


class Layer(AutoShapeMixin, layers.Layer):
    pass


class Wrapper(AutoShapeMixin, layers.Wrapper):
    def __init__(self, layer: layers.Layer, **kwargs: Any) -> None:
        kwargs['name'] = kwargs.get('name', f'{layer.name}_{self.__class__.__name__}')
        if hasattr(layer, 'batch_dims'):
            kwargs['batch_dims'] = kwargs.get('batch_dims', layer.batch_dims)
        super().__init__(layer, **kwargs)


class Dense(AutoShapeMixin, layers.Dense):
    pass


class Conv2D(AutoShapeMixin, layers.Conv2D):
    pass


class Conv2DTranspose(AutoShapeMixin, layers.Conv2DTranspose):
    pass


class Flatten(AutoShapeMixin, layers.Flatten):
    pass


class Reshape(AutoShapeMixin, layers.Reshape):
    pass


class Concatenate(AutoShapeMixin, layers.Concatenate):
    pass


class RNN(AutoShapeMixin, layers.RNN):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs['batch_dims'] = kwargs.get('batch_dims', 2)
        super().__init__(*args, **kwargs)

    def build_with_input(self, input: Union[Nested[tf.TensorSpec], Nested[tf.Tensor]], *args: Any, **kwargs: Any) -> None:
        super().build_with_input(input, *args, **kwargs)
        if not self.cell.built_with_input:
            def zero_state(size: int):
                return tf.zeros([2, size])
            self.cell.build_with_input(tf.nest.map_structure(lambda x: x[:, 0], input),
                                       tf.nest.map_structure(zero_state, self.cell.state_size))
        self.built_with_input = True

    def build(self, input_shape: tf.TensorShape) -> None:
        if not self.cell.built:
            def get_step_input_shape(shape: tf.TensorShape) -> tf.TensorShape:
                return shape[1:] if self.time_major else shape[0] + shape[2:]
            step_input_shape = tf.nest.map_structure(get_step_input_shape, input_shape)
            self.cell.build(step_input_shape)


class AbstractRNNCell(AutoShapeMixin, layers.AbstractRNNCell):
    pass


class GRUCell(AutoShapeMixin, layers.GRUCell):
    pass


class Model(AutoShapeMixin, tf.keras.Model):
    pass


class Sequential(AutoShapeMixin, tf.keras.Sequential):
    pass
