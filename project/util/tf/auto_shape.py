# auto_shape.py: Wrappers for tf.keras layers and models with automatic shape definition
#
# (C) 2019, Daniel Mouritzen

from typing import Any, Dict, List, Optional, Union

import gin
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.training.tracking.layer_utils import filter_empty_layer_containers
from tensorflow_probability import distributions as tfd

from project.util.typing import Nested


class AutoShapeMixin:
    """
    Mixin for `tf.keras.layers.Layer`s and subclasses to automatically define input and output specs when model is
    built using `build_with_input`. Must be listed before `tf.keras.layers.Layer` when subclassing. Only works for
    models and layers with static input and output shapes.
    Args:
        batch_dims: Number of dimensions to treat as batch dimensions (default 1)
        min_batch_shape: List of positive integers giving the number of elements for each batch dimension to use
            when building the graph (default [1]*batch_dims).
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.batch_dims: int = kwargs.pop('batch_dims', 1)
        self.min_batch_shape: List[int] = kwargs.pop('min_batch_shape', [1] * self.batch_dims)
        assert len(self.min_batch_shape) == self.batch_dims, 'min_batch_shape incompatible with batch_dims'
        super().__init__(*args, **kwargs)  # type: ignore[call-arg]  # mypy/issues/5887
        assert not getattr(self, 'dynamic'), 'AutoShapeMixin should not be used with dynamic layers!'
        self._input_spec: Optional[layers.InputSpec] = None
        self._output_spec: Optional[layers.InputSpec] = None
        self.built_with_input = False
        self._named_losses: Dict[str, tf.Tensor] = {}

    def build_with_input(self, input: Union[Nested[tf.TensorSpec], Nested[tf.Tensor]], *args: Any, **kwargs: Any) -> None:
        bd = self.batch_dims
        self._input_spec = tf.nest.map_structure(
            lambda x: layers.InputSpec(shape=([None] * bd + x.shape[bd:])[:x.shape.ndims], dtype=x.dtype),
            input)
        dummy_input = tf.nest.map_structure(
            lambda x: tf.zeros((list(self.min_batch_shape) + x.shape[bd:])[:x.shape.ndims], x.dtype),
            input)
        if 'mask' in kwargs:
            kwargs['mask'] = tf.ones(self.min_batch_shape, tf.bool)
        kwargs['training'] = False
        dummy_output = super().__call__(dummy_input, *args, **kwargs)  # type: ignore[misc]  # mypy/issues/5887
        # if isinstance(tf.nest.flatten(dummy_output)[0], tf.Tensor):
        if isinstance(dummy_output, tfd.Distribution):
            self._output_spec = layers.InputSpec(shape=[None] * len(dummy_output.batch_shape) + dummy_output.event_shape,
                                                 dtype=dummy_output.dtype)
        else:
            self._output_spec = tf.nest.map_structure(lambda x: layers.InputSpec(shape=[None] * bd + x.shape[bd:],
                                                                                 dtype=x.dtype), dummy_output)
        self.built_with_input = True

    def __call__(self, inputs: Nested[tf.Tensor], *args: Any, **kwargs: Any) -> Any:
        if not self.built_with_input:
            self.build_with_input(inputs, *args, **kwargs)
        return super().__call__(inputs, *args, **kwargs)  # type: ignore[misc]  # mypy/issues/5887

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
        assert self.input_spec is not None, (f'build_with_input has not been called for layer {self.name}; '  # type: ignore[attr-defined]
                                             'input shape is not defined')
        return tf.nest.map_structure(lambda x: x.shape, self.input_spec)

    @property
    def output_shape(self) -> Nested[tf.TensorShape]:
        assert self.output_spec is not None, (f'build_with_input has not been called for layer {self.name}; '  # type: ignore[attr-defined]
                                              'output shape is not defined')
        return tf.nest.map_structure(lambda x: x.shape, self.output_spec)

    @property
    def input_dtype(self) -> Nested[tf.TensorShape]:
        assert self.input_spec is not None, (f'build_with_input has not been called for layer {self.name}; '  # type: ignore[attr-defined]
                                             'input dtype is not defined')
        return tf.nest.map_structure(lambda x: x.dtype, self.input_spec)

    @property
    def output_dtype(self) -> Nested[tf.TensorShape]:
        assert self.output_spec is not None, (f'build_with_input has not been called for layer {self.name}; '  # type: ignore[attr-defined]
                                              'output dtype is not defined')
        return tf.nest.map_structure(lambda x: x.dtype, self.output_spec)

    def compute_output_shape(self, input_shape: Nested[tf.TensorShape]) -> Nested[tf.TensorShape]:
        if self.output_spec is None:
            return super().compute_output_shape(input_shape)  # type: ignore[misc]  # mypy/issues/5887
        batch_shape = tf.nest.flatten(input_shape)[0][:self.batch_dims]
        return tf.nest.map_structure(lambda x: batch_shape + x[self.batch_dims:], self.output_shape)

    @property
    def named_losses(self) -> Dict[str, tf.Tensor]:
        losses = self._named_losses.copy()
        if hasattr(self, '_layers'):
            for layer in filter_empty_layer_containers(self._layers):  # type: ignore[attr-defined]
                layer_losses = getattr(layer, 'named_losses', None)
                if layer_losses:
                    for name, loss in layer_losses.items():
                        assert name not in losses, f'Loss names must be unique, but there are two losses called {name}!'
                        losses[name] = loss
        return losses

    @property
    def total_loss(self) -> tf.Tensor:
        return sum(loss * loss._scaling for loss in self.named_losses.values())

    @property
    def per_layer_losses(self) -> Dict[layers.Layer, tf.Tensor]:
        losses: Dict[layers.Layer, tf.Tensor] = {}
        for loss in self.named_losses.values():
            layer = loss._layer if loss._layer is not None else self
            losses[layer] = losses.get(layer, 0.0) + loss * loss._scaling
        return losses

    def add_named_loss(self,
                       loss: tf.Tensor,
                       name: str,
                       scaling: float = 1.0,
                       layer: Optional[layers.Layer] = None,
                       input_dependent: bool = True,
                       ) -> None:
        loss._unconditional_loss = not input_dependent
        loss._scaling = scaling
        loss._layer = layer
        self._named_losses[name] = loss
        self.add_metric(loss, aggregation='mean', name=name)  # type: ignore[attr-defined]


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


class BatchNormalization(AutoShapeMixin, layers.BatchNormalization):
    _USE_V2_BEHAVIOR = False  # https://github.com/tensorflow/tensorflow/issues/32477


class Activation(AutoShapeMixin, layers.Activation):
    pass


@gin.configurable
class ReLU(AutoShapeMixin, layers.ReLU):
    pass


@gin.configurable
class LeakyReLU(AutoShapeMixin, layers.LeakyReLU):
    pass


@gin.configurable
class PReLU(AutoShapeMixin, layers.PReLU):
    pass


class RNN(AutoShapeMixin, layers.RNN):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs['batch_dims'] = kwargs.get('batch_dims', 2)
        super().__init__(*args, **kwargs)

    def build_with_input(self, input: Union[Nested[tf.TensorSpec], Nested[tf.Tensor]], *args: Any, **kwargs: Any) -> None:
        super().build_with_input(input, *args, **kwargs)
        if not self.cell.built_with_input:
            self.cell.build_with_input(tf.nest.map_structure(lambda x: x[:, 0], input))
        self.built_with_input = True

    def build(self, input_shape: tf.TensorShape) -> None:
        if not self.cell.built:
            def get_step_input_shape(shape: tf.TensorShape) -> tf.TensorShape:
                return shape[1:] if self.time_major else shape[0] + shape[2:]
            step_input_shape = tf.nest.map_structure(get_step_input_shape, input_shape)
            self.cell.build(step_input_shape)


class AbstractRNNCell(AutoShapeMixin, layers.AbstractRNNCell):
    def build_with_input(self, input: Union[Nested[tf.TensorSpec], Nested[tf.Tensor]], *args: Any, **kwargs: Any) -> None:
        def zero_state(size: int) -> tf.Tensor:
            return tf.zeros([self.min_batch_shape[0], size])
        super().build_with_input(input, tf.nest.map_structure(zero_state, self.state_size))
        self.built_with_input = True


class GRUCell(AutoShapeMixin, layers.GRUCell):
    pass


class Model(AutoShapeMixin, tf.keras.Model):
    pass


class Sequential(AutoShapeMixin, tf.keras.Sequential):
    pass
