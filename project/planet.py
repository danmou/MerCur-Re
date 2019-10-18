# planet.py: Functionality for interfacing with PlaNet
#
# (C) 2019, Daniel Mouritzen

import os.path
from typing import Any, Dict, Generator, List, cast

import gin
import gym
import planet.control.wrappers as planet_wrappers
import planet.tools
import planet.training
import planet.training.running
import tensorflow as tf
from loguru import logger
from planet.scripts.configs import tasks_lib
from planet.scripts.tasks import Task as PlanetTask
from planet.scripts.train import process as planet_train
from tensorflow.python import debug as tf_debug

from .environments import Habitat, wrappers
from .util import capture_output


@gin.configurable('planet')
class PlanetParams(planet.tools.AttrDict):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not self.get('tasks'):
            self.tasks = ['habitat']


def train(logdir: str) -> None:
    params = PlanetParams()
    args = planet.tools.AttrDict()
    with args.unlocked:
        args.config = 'default'
        args.params = params

    for score in planet_train(logdir, args):
        pass
    logger.info('Run completed.')


def habitat_env_ctor(action_repeat: int, min_length: int, max_length: int) -> gym.Env:
    assert min_length <= max_length, f'{min_length}>{max_length}!'
    logger.debug(f'Collecting episodes between {min_length} and {max_length} steps in length.')
    env = Habitat(max_steps=max_length*action_repeat)
    env = wrappers.AutomaticStop(env, minimum_duration=min_length)
    env = planet_wrappers.ActionRepeat(env, action_repeat)
    env = wrappers.DiscreteWrapper(env)
    env = wrappers.MinimumDuration(env, min_length)
    return env


def planet_habitat_task(config: planet.tools.AttrDict, params: planet.tools.AttrDict) -> PlanetTask:
    action_repeat = params.get('action_repeat', 1)
    max_length = params.max_task_length
    state_components = ['reward']
    observation_components = ['image', 'goal']
    metrics = ['success', 'spl', 'path_length', 'optimal_path_length', 'remaining_distance', 'collisions']
    env_ctor = planet.tools.bind(
        habitat_env_ctor, action_repeat, config.batch_shape[1], max_length)
    return PlanetTask('habitat', env_ctor, max_length, state_components, observation_components, metrics)


@gin.configurable('planet.tf.options')
class PlanetTFOptions(planet.tools.AttrDict):
    pass


@gin.configurable('planet.tf.gpu_options')
class PlanetTFGPUOptions(planet.tools.AttrDict):
    pass


@gin.configurable('planet.tf')
def create_tf_session(debugger: bool = False) -> tf.Session:
    options = PlanetTFOptions()
    gpu_options = PlanetTFGPUOptions()
    if gpu_options:
        devices = [int(d) for d in gpu_options.get('visible_device_list', '').split(',')]
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
                with gpu_options.unlocked:
                    gpu_options.visible_device_list = ','.join([str(d) for d in devices])
        with options.unlocked:
            options.gpu_options = tf.GPUOptions(**gpu_options)
    config = tf.ConfigProto(**options)
    with capture_output('tensorflow'):
        try:
            sess = tf.Session('local', config=config)
        except tf.errors.NotFoundError:
            sess = tf.Session(config=config)
        if debugger:
            sess = cast(tf.Session, tf_debug.TensorBoardDebugWrapperSession(sess,
                                                                            'localhost:6064',
                                                                            send_traceback_and_source_code=False))
    logger.debug('Initialized TF')
    logger.trace(f'Config:\n{gin.operative_config_str()}')
    return sess


class PlanetTrainer(planet.training.Trainer):
    def iterate(self, max_step: Any = None, sess: Any = None) -> Generator[Any, None, None]:
        """Simple patch to replace tf session"""
        sess = create_tf_session()
        for score in super().iterate(max_step, sess):
            yield score


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


# Monkey patch PlaNet to add `habitat` task and use loguru instead of print for logging
tasks_lib.habitat = planet_habitat_task  # type: ignore
planet.control.wrappers.print = logger.info  # type: ignore
planet.training.utility.print = logger.info  # type: ignore
planet.training.trainer.Trainer = PlanetTrainer  # type: ignore
tf.print = tf_print
