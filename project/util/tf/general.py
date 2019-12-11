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
    def flat_func(*flattened: Any) -> None:
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
