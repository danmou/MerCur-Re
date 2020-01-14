# general.py: General TensorFlow-related utilities
#
# (C) 2019, Daniel Mouritzen

from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional, Sequence

import gin
import tensorflow as tf
from loguru import logger

from project.util.typing import Nested


def tf_nested_py_func(func: Callable,
                      inp: Sequence[Nested[tf.Tensor]],
                      out_types: Nested[tf.DType],
                      ) -> Nested[tf.Tensor]:
    """tf.numpy_function replacement that allows nested structures"""
    def flat_func(*flattened: Any) -> Any:
        nested = tf.nest.pack_sequence_as(inp, flattened)
        nested_output = func(*nested)
        tf.nest.assert_same_structure(nested_output, out_types)
        return tf.nest.flatten(nested_output)

    assert isinstance(inp, (list, tuple)), 'Inputs to tf_nested_py_func must be a sequence'
    flat_output = tf.numpy_function(flat_func, tf.nest.flatten(inp), tf.nest.flatten(out_types))
    return tf.nest.pack_sequence_as(out_types, tf.nest.flatten(flat_output))


def _gpu_id_from_name(name: str) -> int:
    return int(name.split(':')[-1])


@gin.configurable('tf.gpus', whitelist=['gpu_ids', 'memory_growth'])
def get_distribution_strategy(gpu_ids: Optional[Sequence[int]] = None, memory_growth: bool = True) -> tf.distribute.Strategy:
    available_gpus = tf.config.experimental.list_physical_devices("GPU")
    if not available_gpus:
        logger.warning('No GPUs found; running on CPU')
        return tf.distribute.OneDeviceStrategy('/cpu:0')

    if gpu_ids is not None:
        available_gpu_ids = list(_gpu_id_from_name(gpu.name) for gpu in available_gpus)
        assert set(available_gpu_ids).issuperset(set(gpu_ids)), (f'Config specifies gpus {gpu_ids}, but only these '
                                                                 f'devices are available to CUDA: {available_gpu_ids}')
        available_gpus = [gpu for gpu in available_gpus if _gpu_id_from_name(gpu.name) in gpu_ids]
        assert available_gpus

    for gpu in available_gpus:
        tf.config.experimental.set_memory_growth(gpu, memory_growth)

    if len(available_gpus) == 1:
        logger.info('Running on single GPU')
        return tf.distribute.OneDeviceStrategy(f'/gpu:{_gpu_id_from_name(available_gpus[0].name)}')

    logger.info(f'Running on {len(available_gpus)} GPUs')
    return tf.distribute.MirroredStrategy()


@contextmanager
def trace_graph(writer: tf.summary.SummaryWriter) -> Generator[None, None, None]:
    """Context manager that traces the graph for a model constructed within it"""
    tf.summary.trace_on(graph=True, profiler=False)
    yield
    with writer.as_default():
        tf.summary.trace_export(name="graph", step=0)


@contextmanager
def trace_profile(writer: tf.summary.SummaryWriter) -> Generator[None, None, None]:
    """Context manager that profiles a model called within it"""
    logger.debug('Running profiler...')
    tf.summary.trace_on(graph=False, profiler=True)
    yield
    with writer.as_default():
        tf.summary.trace_export(name="profile", step=0)


def map_fn(fn: Callable,
           *args: Nested[tf.Tensor],
           axis: int = 0,
           vectorized: bool = False,
           **kwargs: Any,
           ) -> Nested[tf.Tensor]:
    """
    Improved version of tf.map_fn and tf.vectorized_map that allows using a function that take multiple args
    and vectorizing over any axis
    """
    assert axis >= 0, 'Axis to map_fn must be non-negative'
    perm_in = [axis] + list(range(axis))  # e.g. [2, 0, 1]
    perm_out = list(range(1, axis + 1)) + [0]  # e.g. [1, 2, 0]
    args_transposed = tf.nest.map_structure(lambda x: tf.transpose(x, perm_in + list(range(axis + 1, x.shape.ndims))),
                                            args)
    output_transposed = (tf.vectorized_map if vectorized else tf.map_fn)(lambda a: fn(*a), args_transposed, **kwargs)
    return tf.nest.map_structure(lambda x: tf.transpose(x, perm_out + list(range(axis + 1, x.shape.ndims))),
                                 output_transposed)


def sliding_window(tensor: tf.Tensor, size: int, axis: int = 0) -> tf.Tensor:
    """
    Expand the specified axis of size N into two axes of size (N - size + 1, size), with element (..., i, j, ...) corresponding to
    element (..., i+j, ...) in the original tensor.

    Examples:
        sliding_window(tf.constant([1, 2, 3, 4, 5]), 3) == tf.constant([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        sliding_window(tf.ones(shape=[13, 14, 15, 16]), 4, axis=2).shape == [13, 14, 12, 4, 16]
    """
    def window_fn(i: tf.Tensor) -> tf.Tensor:
        return tf.gather(tensor, tf.range(i, i + size), axis=axis)
    res = tf.map_fn(window_fn, tf.range(tensor.shape[axis] - size + 1), dtype=tensor.dtype)
    reshaped = tf.transpose(res, perm=list(range(1, axis + 1)) + [0] + list(range(axis + 1, res.shape.ndims)))
    correct_shape = tensor.shape[:axis] + [tensor.shape[axis] - size + 1, size] + tensor.shape[axis + 1:]
    return tf.ensure_shape(reshaped, correct_shape)
