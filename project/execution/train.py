# train.py: Train model
#
# (C) 2019, Daniel Mouritzen

from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Type

import gin
import tensorflow as tf
from loguru import logger
from tensorflow.python.keras.callbacks import configure_callbacks
from tensorflow.python.keras.engine.training_v2 import TrainingContext
from tensorflow.python.keras.utils.mode_keys import ModeKeys

from project.agents import ModelBasedAgent, RandomAgent
from project.model import get_model, restore_model
from project.tasks import Task
from project.util.files import link_directory_contents
from project.util.planet.numpy_episodes import numpy_episodes
from project.util.tf import get_distribution_strategy, reshape_known_dims, trace_graph
from project.util.tf.callbacks import (CheckpointCallback,
                                       DataCollectionCallback,
                                       EvaluateCallback,
                                       LoggingCallback,
                                       PredictionSummariesCallback,
                                       WandbCommitCallback)
from project.util.timing import measure_time

from .evaluator import Evaluator, get_evaluation_agent
from .simulator import Simulator


@measure_time
@gin.configurable('training', whitelist=['tasks', 'num_seed_episodes', 'num_epochs', 'train_steps', 'test_steps',
                                         'batch_shape', 'agent_cls'])
def train(logdir: Path,
          initial_data: Optional[str],
          checkpoint: Optional[Path] = None,
          tasks: Sequence[Task] = gin.REQUIRED,
          num_seed_episodes: int = 10,
          num_epochs: int = 200,
          train_steps: int = gin.REQUIRED,
          test_steps: int = gin.REQUIRED,
          batch_shape: Tuple[int, int] = (64, 64),
          agent_cls: Type[ModelBasedAgent] = gin.REQUIRED,
          ) -> None:
    logger.info('Creating training environments.')
    sims = {task.name: Simulator(task) for task in tasks}
    evaluator = Evaluator(logdir=logdir, video=True)

    distribution_strategy = get_distribution_strategy()
    batch_shape = (distribution_strategy.num_replicas_in_sync * batch_shape[0], batch_shape[1])

    dataset_dirs = {name: logdir / f'{name}_episodes' for name in ['train', 'test']}
    if initial_data:
        logger.info('Linking initial dataset.')
        for dataset in dataset_dirs.values():
            link_directory_contents(Path(initial_data).absolute() / dataset.name, dataset)
    else:
        for task, sim in sims.items():
            for phase, save_dir in dataset_dirs.items():
                logger.info(f'Collecting {num_seed_episodes} initial episodes ({task} {phase}).')
                sim.run(RandomAgent(sim.action_space), num_seed_episodes, save_dir=save_dir, save_data=True)

    train_data, test_data = numpy_episodes(dataset_dirs['train'], dataset_dirs['test'], batch_shape)
    observation_components = {name for task in tasks for name in task.observation_components}

    writer = tf.summary.create_file_writer(str(logdir / 'tb_logs' / 'train'))
    if checkpoint is None:
        with trace_graph(writer):
            model = get_model(observation_components, train_data.element_spec)
        start_epoch = 0
    else:
        model, start_epoch = restore_model(checkpoint, logdir)
    model.summary(line_length=100, print_fn=logger.debug)
    # model.encoder.layer.layer._image_enc.summary(line_length=100, print_fn=logger.debug)
    # model.decoders['image'].layer._decoder.summary(line_length=100, print_fn=logger.debug)

    train_agents = {task_name: agent_cls(sim.action_space, model) for task_name, sim in sims.items()}
    eval_agents = {task_name: get_evaluation_agent(agent.action_space, model, train_agent=agent)
                   for task_name, agent in train_agents.items()}

    logger.info('Training...')
    callbacks = [
        LoggingCallback(),
        tf.keras.callbacks.TensorBoard(log_dir=str(logdir / 'tb_logs'),
                                       write_graph=True,
                                       profile_batch=2,
                                       histogram_freq=0,
                                       update_freq='epoch'),
        CheckpointCallback(filepath=str(logdir / 'checkpoint_epoch_{epoch:03d}_loss_{val_loss:.2f}.h5'),
                           verbose=1),
        DataCollectionCallback(sims, train_agents, dataset_dirs),
        EvaluateCallback(evaluator, eval_agents),
        PredictionSummariesCallback(model, dataset_dirs),
        WandbCommitCallback(),
    ]
    fit_model(model=model,
              train_data=train_data,
              val_data=test_data,
              initial_epoch=start_epoch,
              epochs=int(num_epochs),
              steps_per_epoch=train_steps,
              validation_steps=test_steps,
              validation_freq=1,
              callbacks=callbacks)

    logger.success('Run completed.')


@measure_time
def fit_model(model: tf.keras.Model,
              train_data: tf.data.Dataset,
              val_data: tf.data.Dataset,
              initial_epoch: int,
              epochs: int,
              steps_per_epoch: int,
              validation_steps: int,
              validation_freq: int,
              callbacks: Sequence[tf.keras.callbacks.Callback],
              ) -> None:
    train_context = TrainingContext()
    train_data_iter = iter(train_data)
    val_data_iter = iter(val_data)
    train_callbacks = configure_callbacks(callbacks,
                                          model,
                                          epochs=epochs,
                                          steps_per_epoch=steps_per_epoch,
                                          samples=steps_per_epoch,
                                          verbose=0,
                                          mode=ModeKeys.TRAIN)

    with train_context.on_start(model, train_callbacks, use_samples=False, mode=ModeKeys.TRAIN):
        for epoch in range(initial_epoch, epochs):
            if train_callbacks.model.stop_training:
                break
            with train_context.on_epoch(epoch, ModeKeys.TRAIN) as epoch_logs:
                train_result = run_one_epoch(model,
                                             train_data_iter,
                                             steps_per_epoch=steps_per_epoch,
                                             mode=ModeKeys.TRAIN,
                                             context=train_context)
                if train_result is not None:
                    epoch_logs.update(train_result)
                if train_callbacks.model.stop_training:
                    break
                if (epoch + 1) % validation_freq == 0:
                    validation_callbacks = configure_callbacks(train_callbacks,
                                                               model,
                                                               epochs=1,
                                                               steps_per_epoch=validation_steps,
                                                               samples=validation_steps,
                                                               verbose=0,
                                                               mode=ModeKeys.TEST)
                    val_context = TrainingContext()
                    with val_context.on_start(model, validation_callbacks, use_samples=False, mode=ModeKeys.TEST):
                        with val_context.on_epoch(epoch, ModeKeys.TEST):
                            val_result = run_one_epoch(model,
                                                       val_data_iter,
                                                       steps_per_epoch=validation_steps,
                                                       mode=ModeKeys.TEST,
                                                       context=val_context)
                            if val_result is not None:
                                epoch_logs.update({f'val_{k}': v for k, v in val_result.items()})


def run_one_epoch(model: tf.keras.Model,
                  iterator: Iterator[Mapping[str, tf.Tensor]],
                  steps_per_epoch: int,
                  mode: str,
                  context: TrainingContext,
                  ) -> Optional[Dict[str, float]]:
    model.reset_metrics()
    metrics = None
    for step in range(steps_per_epoch):
        with context.on_batch(step=step, mode=mode) as batch_logs:
            inputs = next(iterator)
            metrics = run_on_batch(model, inputs, training=mode == ModeKeys.TRAIN)
            batch_logs.update(metrics)
        if context.callbacks.model.stop_training:
            break
    return metrics


@gin.configurable(whitelist=['gradient_clip_norm'])
@tf.function
def run_on_batch(model: tf.keras.Model,
                 inputs: Mapping[str, tf.Tensor],
                 training: bool = False,
                 gradient_clip_norm: Optional[float] = None,
                 ) -> Dict[str, tf.Tensor]:
    """Runs a single training or validation step on a single batch of data."""
    inputs = {key: reshape_known_dims(tf.cast(inputs[key], spec.dtype), spec.shape)
              for key, spec in model.input_spec.items()}
    with tf.GradientTape(persistent=True) as tape:
        model(inputs, training=training)
        losses = model.per_layer_losses
        total_loss = model.total_loss
    metrics = {'loss': total_loss}
    metrics.update({m.name: m.result() for m in model.metrics})
    if training:
        optimizers = dict(zip(model.optimizer_targets, model.optimizer))
        gradients_per_optimizer: Dict[str, Dict[Any, tf.Tensor]] = {optimizer_name: {} for optimizer_name in optimizers.keys()}
        for layer, loss in losses.items():
            variables = layer.trainable_variables
            optimizer_name = layer.name if layer.name in gradients_per_optimizer.keys() else 'model'
            gradients = tape.gradient(loss, variables)
            for var, grad in zip(variables, gradients):
                if grad is not None:
                    ref = var.experimental_ref()
                    gradients_per_optimizer[optimizer_name][ref] = gradients_per_optimizer.get(ref, 0.0) + grad
        for optimizer_name, gradients in gradients_per_optimizer.items():
            if not gradients:
                logger.warning(f'No gradients for optimizer {optimizer_name}')
                continue
            vars_ = list(var.deref() for var in gradients.keys())
            grads = list(gradients.values())
            if gradient_clip_norm:
                grads, norm = tf.clip_by_global_norm(grads, gradient_clip_norm)
                metrics[f'grad_norm_{optimizer_name}'] = norm
            optimizers[optimizer_name].apply_gradients(zip(grads, vars_))
    return metrics
