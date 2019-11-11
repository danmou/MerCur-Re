# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import functools
import os

import tensorflow as tf
from loguru import logger

from project.environments import wrappers
from project.models.planet import control
from project.models.planet.training import trainer as trainer_
from project.util import planet as tools

Objective = collections.namedtuple(
    'Objective', 'name, value, goal, include, exclude')


def get_batch(datasets, phase, reset):
    """Read batches from multiple datasets based on the training phase.

    The test dataset is reset at the beginning of every test phase. The training
    dataset is repeated infinitely and doesn't need a reset.

    Args:
      datasets: Dictionary of datasets with training phases as keys.
      phase: Tensor of the training phase name.
      reset: Whether to reset the datasets.

    Returns:
      data: a batch of data from either the train or test set.
    """
    with datasets.unlocked():
        datasets.train = datasets.train.make_one_shot_iterator()
        datasets.test = datasets.test.make_one_shot_iterator()
    data = tf.cond(
        tf.equal(phase, 'train'),
        datasets.train.get_next,
        datasets.test.get_next)
    if not isinstance(data, dict):
        data = {'data': data}
    if 'length' not in data:
        example = data[list(data.keys())[0]]
        data['length'] = (
                tf.zeros((tf.shape(example)[0],), tf.int32) + tf.shape(example)[1])
    return data


def train(model_fn, datasets, logdir, config, envs):
    """Train a model on a datasets.

    The model function receives the following arguments: data batch, trainer
    phase, whether it should log, and the config. The configuration object should
    contain the attributes `batch_shape`, `train_steps`, `test_steps`,
    `max_epochs`, in addition to the attributes expected by the model function.

    Args:
      model_fn: Function greating the model graph.
      datasets: Dictionary with keys `train` and `test` and datasets as values.
      logdir: Optional logging directory for summaries and checkpoints.
      config: Configuration object.
      envs: Dictionary of initialized environments.

    Yields:
      Test score of every epoch.

    Raises:
      KeyError: if config is falsey.
    """
    if not config:
        raise KeyError('You must specify a configuration.')
    logdir = logdir and os.path.expanduser(logdir)
    if logdir:
        with config.unlocked():
            config.logdir = logdir
    trainer = trainer_.Trainer(logdir, config=config)
    cleanups = []
    try:
        with tf.compat.v1.variable_scope('graph', use_resource=True):
            data = get_batch(datasets, trainer.phase, trainer.reset)
            summary, collect_scores, collect_summaries, cleanups = model_fn(
                data, trainer, config, envs)
            message = 'Graph contains {} trainable variables.'
            tf.compat.v1.logging.info(message.format(tools.count_weights()))
            if config.train_steps:
                trainer.add_phase(
                    'train', config.train_steps, tf.constant(0.0), summary,
                    yield_every=None,
                    log_every=config.train_log_every)
            if config.test_steps:
                trainer.add_phase(
                    'test', config.test_steps, tf.constant(0.0), summary,
                    yield_every=None,
                    log_every=config.test_log_every)
            if config.checkpoint_every:
                trainer.add_checkpoint_phase(config.checkpoint_every)
            for name, params in config.collects.items():
                score, summary = collect_scores[name], collect_summaries[name]
                trainer.add_phase(
                    name, params.num_episodes, score, summary,
                    yield_every=params.eval_every, collect_every=params.every
                )
        for saver in config.savers:
            trainer.add_saver(**saver)
        for score in trainer.iterate(config.max_epochs):
            yield score, envs
    finally:
        for cleanup in cleanups:
            cleanup()


def compute_objectives(posterior, prior, target, graph, config):
    raw_features = graph.cell.features_from_state(posterior)
    heads = graph.heads
    objectives = []
    for name, scale in config.loss_scales.items():
        if config.loss_scales[name] == 0.0:
            continue
        if name in config.heads and name not in config.gradient_heads:
            features = tf.stop_gradient(raw_features)
            include = r'.*/head_{}/.*'.format(name)
            exclude = None
        else:
            features = raw_features
            include = r'.*'
            exclude = None

        if name == 'divergence':
            loss = graph.cell.divergence_from_states(posterior, prior)
            if config.free_nats is not None:
                loss = tf.maximum(0.0, loss - float(config.free_nats))
            objectives.append(Objective('divergence', loss, min, include, exclude))

        elif name == 'overshooting':
            shape = tools.shape(graph.data['action'])
            length = tf.tile(tf.constant(shape[1])[None], [shape[0]])
            _, priors, posteriors, mask = tools.overshooting(
                graph.cell, {}, graph.embedded, graph.data['action'], length,
                config.overshooting_distance, posterior)
            posteriors, priors, mask = tools.nested.map(
                lambda x: x[:, :, 1:-1], (posteriors, priors, mask))
            if config.os_stop_posterior_grad:
                posteriors = tools.nested.map(tf.stop_gradient, posteriors)
            loss = graph.cell.divergence_from_states(posteriors, priors)
            if config.free_nats is not None:
                loss = tf.maximum(0.0, loss - float(config.free_nats))
            objectives.append(Objective('overshooting', loss, min, include, exclude))

        else:
            logprob = heads[name](features).log_prob(target[name])
            objectives.append(Objective(name, logprob, max, include, exclude))

    objectives = [o._replace(value=tf.reduce_mean(o.value)) for o in objectives]
    return objectives


def apply_optimizers(objectives, trainer, config):
    # Make sure all losses are computed and apply loss scales.
    processed = []
    values = [ob.value for ob in objectives]
    for ob in objectives:
        loss = {min: ob.value, max: -ob.value}[ob.goal]
        loss *= config.loss_scales[ob.name]
        with tf.control_dependencies(values):
            loss = tf.identity(loss)
        processed.append(ob._replace(value=loss, goal=min))
    # Merge objectives that operate on the whole model to compute only one
    # backward pass and to share optimizer statistics.
    objectives = []
    losses = []
    for ob in processed:
        if ob.include == r'.*' and ob.exclude is None:
            assert ob.goal == min
            losses.append(ob.value)
        else:
            objectives.append(ob)
    objectives.append(Objective('main', tf.reduce_sum(losses), min, r'.*', None))
    # Apply optimizers and collect loss summaries.
    summaries = []
    grad_norms = {}
    with tf.compat.v1.variable_scope('optimizer'):
        for ob in objectives:
            assert ob.name in list(config.loss_scales.keys()) + ['main'], ob
            assert ob.goal == min, ob
            assert ob.name in config.optimizers, ob
            optimizer = config.optimizers[ob.name](
                include=ob.include,
                exclude=ob.exclude,
                step=trainer.step,
                log=trainer.log,
                debug=config.debug,
                name=ob.name)
            condition = tf.equal(trainer.phase, 'train')
            summary, grad_norm = optimizer.maybe_minimize(condition, ob.value)
            summaries.append(summary)
            grad_norms[ob.name] = grad_norm
    return summaries, grad_norms


def simulate_episodes(
        config, params, graph, cleanups, env,
        expensive_summaries, gif_summary, metrics_summaries, name):
    if params.save_episode_dir:
        env = wrappers.CollectGymDataset(env, params.save_episode_dir)
    env = wrappers.SelectObservations(env, params.task.observation_components)
    env = wrappers.SelectMetrics(env, params.task.metrics)
    cell = graph.cell
    agent_config = tools.AttrDict(
        cell=cell,
        encoder=graph.encoder,
        planner=functools.partial(params.planner, graph=graph),
        objective=params.objective and functools.partial(params.objective, graph=graph),
        exploration=params.exploration,
        preprocess_fn=config.preprocess_fn,
        postprocess_fn=config.postprocess_fn)
    params = params.copy()
    with params.unlocked():
        params.update(agent_config)
    with agent_config.unlocked():
        agent_config.update(params)
    summary, return_, metrics = control.simulate(
        graph.step, env, params.task.max_length, agent_config,
        expensive_summaries, gif_summary, metrics_summaries, name=name)
    return summary, return_, metrics


def print_metrics(metrics, step, every, name='metrics'):
    def print_fn(*values):
        tf.compat.v1.logging.info(f'{name}:')
        names = [f'{m:9s}' for m in ['step'] + list(metrics.keys())]
        lengths = [len(k) for k in names]
        tf.compat.v1.logging.info('  '.join(names))
        values = [f'{v:<{l}.0f}'
                  if v % 1 == 0 and abs(v) < 1e7
                  else f'{v:<{l}.3g}'
                  for l, v in zip(lengths, values)]
        tf.compat.v1.logging.info('  '.join(values))

    means, updates = [], []
    for key, value in metrics.items():
        key = 'metrics_{}_{}'.format(name, key)
        mean = tools.StreamingMean((), tf.float32, key)
        means.append(mean)
        updates.append(mean.submit(tf.cast(value, tf.float32)))
    with tf.control_dependencies(updates):
        print_metrics = tf.cond(
            tf.equal((step + 1) % every, 0),
            lambda: tf.py_func(
                print_fn, [step] + [mean.clear() for mean in means], []),
            tf.no_op)
    return print_metrics


def collect_initial_episodes(config, envs):
    items = config.random_collects.items()
    items = sorted(items, key=lambda x: x[0])
    existing = {}
    for name, params in items:
        outdir = params.save_episode_dir
        tf.compat.v1.gfile.MakeDirs(outdir)
        if outdir not in existing:
            existing[outdir] = len(tf.compat.v1.gfile.Glob(os.path.join(outdir, '*.npz')))
        if params.num_episodes <= existing[outdir]:
            existing[outdir] -= params.num_episodes
        else:
            remaining = params.num_episodes - existing[outdir]
            existing[outdir] = 0
            env = envs[params.task.name]
            logger.info('Collecting {} initial episodes ({}).'.format(remaining, name))
            control.random_episodes(env, remaining, outdir)
