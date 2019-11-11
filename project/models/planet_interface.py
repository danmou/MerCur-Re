# planet_interface.py: Functionality for interfacing with PlaNet
#
# (C) 2019, Daniel Mouritzen

import os.path
from typing import Any, Callable, Dict, Generator, List, Tuple, Type, cast

import gin
import gym
import numpy as np
import tensorflow as tf
from loguru import logger
from tensorflow.python import debug as tf_debug

import project.models.planet.control.wrappers
import project.models.planet.tools
import project.models.planet.training
from project.environments import habitat, wrappers
from project.models.planet.scripts.configs import tasks_lib
from project.models.planet.scripts.tasks import Task as PlanetTask
from project.models.planet.tools import AttrDict
from project.util.logging import capture_output
from project.util.timing import Timer


@gin.configurable('planet')
class PlanetParams(AttrDict):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not self.get('tasks'):
            self.tasks = ['habitat']


def habitat_env_ctor(*params_tuple: Tuple[str, Any]) -> gym.Env:
    params = dict(params_tuple)
    wrappers = params.pop('wrappers')
    min_duration = params['min_duration']
    max_duration = params['max_duration']
    assert min_duration <= max_duration, f'{min_duration}>{max_duration}!'
    logger.trace(f'Collecting episodes between {min_duration} and {max_duration} steps in length.')
    env = habitat.Habitat(**params)
    for Wrapper, params in wrappers:
        env = Wrapper(env, **params)
    return env


@gin.configurable(whitelist=['wrappers'])
def planet_habitat_task(config: AttrDict,
                        params: AttrDict,
                        wrappers: List[Tuple[Type[wrappers.Wrapper],
                                       Callable[[Dict[str, Any]], Dict[str, Any]]]] = gin.REQUIRED,
                        ) -> PlanetTask:
    action_repeat = params.get('action_repeat', 1)
    max_length = params.max_task_length
    state_components = ['reward']
    observation_components = ['image', 'goal']
    metrics = ['success', 'spl', 'path_length', 'optimal_path_length', 'remaining_distance', 'collisions']
    env_params = {'action_repeat': action_repeat,
                  'min_duration': config.batch_shape[1],
                  'max_duration': max_length,
                  'capture_video': False}
    env_params['wrappers'] = [(Wrapper, kwarg_fn(env_params)) for Wrapper, kwarg_fn in wrappers]
    env_params.update(habitat.get_config(max_steps=max_length*action_repeat*3))  # times 3 because TURN_ANGLE is really 3 actions
    env_ctor = project.models.planet.tools.bind(habitat.VectorHabitat, habitat_env_ctor, env_params)
    return PlanetTask('habitat', env_ctor, max_length, state_components, observation_components, metrics)


@gin.configurable('planet.tf.options')
class PlanetTFOptions(AttrDict):
    pass


@gin.configurable('planet.tf.gpu_options')
class PlanetTFGPUOptions(AttrDict):
    pass


@gin.configurable('planet.tf')
def create_tf_session(debugger: bool = False) -> tf.compat.v1.Session:
    with Timer() as t:
        options = PlanetTFOptions()
        gpu_options = PlanetTFGPUOptions()
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
                    with gpu_options.unlocked:
                        gpu_options.visible_device_list = ','.join([str(d) for d in devices])
            with options.unlocked:
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


class PlanetTrainer(project.models.planet.training.Trainer):
    def iterate(self, max_step: Any = None, sess: Any = None) -> Generator[np.float32, None, None]:
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
tasks_lib.habitat = planet_habitat_task  # type: ignore[attr-defined]
project.models.planet.control.wrappers.print = logger.info  # type: ignore[attr-defined]
project.models.planet.training.utility.print = logger.info  # type: ignore[attr-defined]
project.models.planet.training.trainer.Trainer = PlanetTrainer  # type: ignore[misc]
tf.print = tf_print
