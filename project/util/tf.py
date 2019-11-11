# tf.py: TensorFlow-related utilities
#
# (C) 2019, Daniel Mouritzen

import os.path
from typing import Any, List, cast

import gin
import tensorflow as tf
from loguru import logger
from tensorflow.python import debug as tf_debug

from project.util.logging import capture_output
from project.util.timing import Timer

from .planet import AttrDict


def tf_print(*args: Any) -> tf.Tensor:
    """tf.print replacement that uses Python logging"""
    try:
        nest = tf.nest
    except AttributeError:
        nest = tf.contrib.framework.nest

    def print_fn(*flattened: List[Any]) -> None:
        unflattened = nest.pack_sequence_as(args, flattened)
        logger.info(' '.join(str(arg.decode() if isinstance(arg, bytes) else arg) for arg in unflattened))

    return tf.py_func(print_fn, nest.flatten(args), [])


@gin.configurable('tf.options')
class TFOptions(AttrDict):
    pass


@gin.configurable('tf.gpu_options')
class TFGPUOptions(AttrDict):
    pass


@gin.configurable('tf')
def create_tf_session(debugger: bool = False) -> tf.compat.v1.Session:
    with Timer() as t:
        options = TFOptions()
        gpu_options = TFGPUOptions()
        if gpu_options:
            devices = [int(d) for d in gpu_options.get('visible_device_list', '').split(',') if d]
            if devices:
                num_visible_devices = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
                max_d = max(devices)
                min_d = min(devices)
                assert max_d - min_d < num_visible_devices, (f'Config specifies devices {devices} for planet, but only '
                                                             f'{num_visible_devices} devices are visible to CUDA.')
                if max_d >= num_visible_devices:
                    shift = max_d - num_visible_devices + 1
                    logger.warning(f'Config specifies devices {devices} for planet, but only {num_visible_devices} devices '
                                   f'are visible to CUDA. Shifting device list down by {shift} to compensate.')
                    devices = [d - shift for d in devices]
                    with gpu_options.unlocked():
                        gpu_options.visible_device_list = ','.join([str(d) for d in devices])
            with options.unlocked():
                options.gpu_options = tf.compat.v1.GPUOptions(**gpu_options)
        config = tf.compat.v1.ConfigProto(**options)
        with capture_output('tensorflow'):
            try:
                sess = tf.compat.v1.Session('local', config=config)
            except tf.errors.NotFoundError:
                sess = tf.compat.v1.Session(config=config)
            if debugger:
                sess = cast(tf.compat.v1.Session,
                            tf_debug.TensorBoardDebugWrapperSession(sess,
                                                                    'localhost:6064',
                                                                    send_traceback_and_source_code=False))
    logger.debug(f'Initialized TF in {t.interval:.3g}s')
    logger.trace(f'Config:\n{gin.operative_config_str()}')
    return sess
