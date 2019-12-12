# train.py: Train model
#
# (C) 2019, Daniel Mouritzen

from pathlib import Path
from typing import Optional, Sequence, Tuple

import gin
import tensorflow as tf
from loguru import logger

from project.agents import MPCAgent, RandomAgent
from project.model import get_model, restore_model
from project.tasks import Task
from project.util.callbacks import (CheckpointCallback,
                                    DataCollectionCallback,
                                    EvaluateCallback,
                                    LoggingCallback,
                                    PredictionSummariesCallback)
from project.util.files import link_directory_contents
from project.util.planet.numpy_episodes import numpy_episodes
from project.util.tf import get_distribution_strategy, trace_graph
from project.util.timing import measure_time

from .simulator import Simulator


@measure_time
@gin.configurable('training', whitelist=['tasks', 'num_seed_episodes', 'num_epochs', 'train_steps', 'test_steps',
                                         'batch_shape'])
def train(logdir: Path,
          initial_data: Optional[str],
          checkpoint: Optional[Path] = None,
          tasks: Sequence[Task] = gin.REQUIRED,
          num_seed_episodes: int = 10,
          num_epochs: int = 200,
          train_steps: int = gin.REQUIRED,
          test_steps: int = gin.REQUIRED,
          batch_shape: Tuple[int, int] = (64, 64),
          ) -> None:
    logger.info('Creating environments.')
    envs = {}
    sims = {}
    for task in tasks:
        envs[task.name] = task.env_ctor()
        sims[task.name] = {phase: Simulator(envs[task.name],
                                            metrics=task.metrics,
                                            save_dir=logdir / f'{phase}_episodes',
                                            save_data=True)
                           for phase in ['train', 'test']}

    distribution_strategy = get_distribution_strategy()
    batch_shape = (distribution_strategy.num_replicas_in_sync * batch_shape[0], batch_shape[1])

    dataset_dirs = {name: logdir / f'{name}_episodes' for name in ['train', 'test']}
    if initial_data:
        logger.info('Linking initial dataset.')
        for dataset in dataset_dirs.values():
            link_directory_contents(Path(initial_data).absolute() / dataset.name, dataset)
    else:
        for task, task_sims in sims.items():
            for phase, sim in task_sims.items():
                logger.info(f'Collecting {num_seed_episodes} initial episodes ({task} {phase}).')
                sim.run(RandomAgent(sim.action_space), num_seed_episodes)

    with distribution_strategy.scope():
        train_data, test_data = numpy_episodes(dataset_dirs['train'], dataset_dirs['test'], batch_shape)
        observation_components = {name for task in tasks for name in task.observation_components}

        writer = tf.summary.create_file_writer(str(logdir / 'tb_logs' / 'train'))
        if checkpoint is None:
            with trace_graph(writer):
                model = get_model(observation_components, train_data.output_shapes, train_data.output_types)
            start_epoch = 0
        else:
            model, start_epoch = restore_model(checkpoint, logdir)
        model.summary(line_length=100, print_fn=logger.debug)
        # model.encoder.layer.layer._image_enc.summary(line_length=100, print_fn=logger.debug)
        # model.decoders['image'].layer._decoder.summary(line_length=100, print_fn=logger.debug)

        agents = {task_name: MPCAgent(env.action_space, model, objective='reward') for task_name, env in envs.items()}

        logger.info('Training...')
        callbacks = [
            LoggingCallback(),
            tf.keras.callbacks.TensorBoard(log_dir=str(logdir / 'tb_logs'),
                                           write_graph=True,
                                           profile_batch=2,
                                           write_grads=True,  # currently no effect, see https://github.com/tensorflow/tensorflow/issues/31173
                                           histogram_freq=0,
                                           update_freq='epoch'),
            CheckpointCallback(filepath=str(logdir / 'checkpoint_epoch_{epoch:03d}_loss_{val_loss:.2f}.h5'),
                               verbose=1),
            DataCollectionCallback(sims, agents),
            PredictionSummariesCallback(model, dataset_dirs),
            EvaluateCallback(logdir, model, envs),
        ]
        measure_time(model.fit)(train_data,
                                validation_data=test_data,
                                initial_epoch=start_epoch,
                                epochs=int(num_epochs),
                                steps_per_epoch=train_steps,
                                validation_steps=test_steps,
                                callbacks=callbacks,
                                verbose=0)

    logger.success('Run completed.')
