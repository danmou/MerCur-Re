# wrappers.py: Wrappers modifying the behavior of network layers
#
# (C) 2019, Daniel Mouritzen

from typing import Any, Dict, Mapping, Optional, Sequence, Union

import tensorflow as tf

from project.util.tf import auto_shape, combine_dims, split_dim


class ExtraBatchDim(auto_shape.Wrapper):
    """Turns a layer expecting a single batch dimension into one expecting two."""
    def __init__(self, layer: auto_shape.Layer, **kwargs: Any) -> None:
        super().__init__(layer, batch_dims=2, **kwargs)

    def _combine_shape(self, shape: Optional[tf.TensorShape]) -> Optional[tf.TensorShape]:
        if not shape or shape.ndims < 2:
            return shape
        return shape[0] * shape[1] + shape[2:]

    def build(self, input_shape: Union[None, tf.TensorShape, Dict[str, tf.TensorShape]] = None) -> None:
        super().build(tf.nest.map_structure(self._combine_shape, input_shape))

    @tf.function
    def call(self, input: Union[tf.Tensor, Dict[str, tf.Tensor]], **kwargs: Any) -> tf.Tensor:
        input_flattened = tf.nest.flatten(input)
        batch_shape = input_flattened[0].shape[:2]
        assert all(i.shape[:2].as_list() == batch_shape.as_list() for i in input_flattened), (
            f'Mismatched batch dimensions in input: {input_flattened}')
        input_combined = tf.nest.map_structure(lambda t: combine_dims(t, [0, 1]), input)
        output_combined = self.layer(input_combined, **kwargs)
        return tf.nest.map_structure(lambda t: split_dim(t, 0, batch_shape), output_combined)


class SelectItems(auto_shape.Wrapper):
    """Ensure a layer expecting dict inputs only gets the relevant items and gets them in the correct order."""
    def __init__(self, layer: auto_shape.Layer, keys: Sequence[str], **kwargs: Any) -> None:
        super().__init__(layer, **kwargs)
        self._select_keys = keys

    def __call__(self, input: Mapping[str, tf.Tensor], *args: Any, **kwargs: Any) -> tf.Tensor:  # type: ignore[override]
        """We select keys here instead of in `call` so input shape gets correctly defined"""
        return super().__call__({k: input[k] for k in self._select_keys})

    def call(self, input: Mapping[str, tf.Tensor], *args: Any, **kwargs: Any) -> tf.Tensor:
        return self.layer(input)
