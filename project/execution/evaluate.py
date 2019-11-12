# evaluate.py: Evaluate model
#
# (C) 2019, Daniel Mouritzen

import functools
import random
from collections import namedtuple
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import gin
import gym
import habitat
import numpy as np
import tensorflow as tf
import wandb
from loguru import logger

import project.models.planet
from project.environments import wrappers
from project.environments.habitat import VectorHabitat
from project.models.planet.scripts.configs import default as planet_config
from project.models.planet.training.define_model import build_network
from project.util import AttrDict, PrettyPrinter, Statistics
from project.util.planet_interface import PlanetParams
from project.util.tf import create_tf_session
from project.util.timing import Timer, measure_time


def evaluate(logdir: Path,
             checkpoint: Path,
             num_episodes: int = 10,
             video: bool = True,
             seed: Optional[int] = None,
             sync_wandb: bool = True,
             existing_env: VectorHabitat = None) -> None:
    if seed is not None:
        logger.info('Running evaluation without parallel loops (this is deterministic but ~50% slower).')
        random.seed(seed)
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
    params = PlanetParams()
    with params.unlocked():
        params.logdir = str(logdir)
        params.batch_shape = [1, 1]
    config = AttrDict()
    with config.unlocked():
        config = planet_config(config, params)
    collect_params = next(iter(config.collects.values()))  # We assume all collects have same task and planner
    if existing_env is None:
        env = create_env(collect_params.task, video, seed)
        original_config = None
        original_params = None
    else:
        env = existing_env
        original_config, original_params = reconfigure_env(env, video, seed)
    env = wrap_env(env, collect_params.task)
    data = create_dummy_data(env, config.preprocess_fn)
    graph = create_graph(data, config)
    agent = create_agent(graph, env, collect_params, config, deterministic=seed is not None)
    steps_op, score_op, metrics_op = define_episode(env, agent)
    logger.debug(f'Graph contains {project.util.planet.count_weights()} trainable variables')
    sess = create_tf_session()
    with sess:
        restore_checkpoint(sess, checkpoint, logdir, config.savers[0])
        sess.graph.finalize()
        statistics = Statistics(['steps', 'score', 'step_time'] + list(metrics_op.keys()),
                                save_file=logdir / 'eval.csv')
        pp = PrettyPrinter(['episode', 'steps', 'score', 'step_time'] + list(metrics_op.keys()))
        pp.print_header()
        for episode in range(num_episodes):
            num_steps = -1
            while num_steps < 1:
                if num_steps == 0:
                    logger.warning('Episode lasted 0 steps; retrying...')
                with Timer() as t:
                    num_steps, score, metrics = sess.run([steps_op, score_op, metrics_op])
            statistics.update(dict(steps=num_steps, score=score, step_time=t.interval/num_steps, **metrics))
            pp.print_row(dict(episode=episode, steps=num_steps, score=score, step_time=t.interval/num_steps, **metrics))
            if video:
                env.save_video(logdir / f'episode_{episode}_spl_{metrics["spl"]:.2f}')
        logger.info('')
        logger.info('Finished evaluation.')
        logger.info('Results:')
        statistics.print()
        if sync_wandb:
            # First delete existing summary items
            for k in list(wandb.run.summary._json_dict.keys()):
                wandb.run.summary._root_del((k,))

            wandb.run.summary.update(statistics.mean)
            wandb.run.summary['seed'] = seed
            if video:
                for vid in logdir.glob('episode*.mp4'):
                    wandb.run.summary[vid.stem] = wandb.Video(str(vid), fps=20, format="mp4")
    if existing_env is not None:
        assert original_config is not None
        assert original_params is not None
        existing_env.reconfigure(config=original_config, **original_params)
        existing_env.call_at(0, 'enable_curriculum', {'enable': gin.query_parameter('curriculum.enabled')})


@measure_time()
def create_env(task: AttrDict, capture_video: bool, seed: Optional[int]) -> VectorHabitat:
    params = task.env_ctor._args[1]  # TODO: do this in a less hacky way
    params['capture_video'] = capture_video
    params['seed'] = seed
    params['min_duration'] = 0
    config = params['config']
    config.defrost()
    if capture_video and 'TOP_DOWN_MAP' not in config.TASK.MEASUREMENTS:
        # Top-down map is expensive to compute, so we only enable it for evaluation.
        config.TASK.MEASUREMENTS.append('TOP_DOWN_MAP')
    config.freeze()
    env: VectorHabitat = task.env_ctor()
    return env


@measure_time()
def reconfigure_env(env: VectorHabitat,
                    capture_video: bool = False,
                    seed: Optional[int] = None,
                    ) -> Tuple[habitat.Config, Dict[str, Any]]:
    original_config: habitat.Config = env._config
    original_params: Dict[str, Any] = {'capture_video': env._capture_video, 'min_duration': env._min_duration}
    config = original_config.clone()
    config.defrost()
    if capture_video and 'TOP_DOWN_MAP' not in config.TASK.MEASUREMENTS:
        # Top-down map is expensive to compute, so we only enable it for evaluation.
        config.TASK.MEASUREMENTS.append('TOP_DOWN_MAP')
    config.freeze()
    env.reconfigure(config=config, capture_video=capture_video, seed=seed, min_duration=0)
    return original_config, original_params


@measure_time()
def wrap_env(env: VectorHabitat, task: AttrDict) -> project.models.planet.control.InGraphBatchEnv:
    env.call_at(0, 'enable_curriculum', {'enable': False})
    wrapped_env: gym.Env = wrappers.SelectObservations(env, task.observation_components)
    wrapped_env = wrappers.SelectMetrics(wrapped_env, task.metrics)
    with tf.compat.v1.variable_scope('environment', use_resource=True):
        batch_env = project.models.planet.control.InGraphBatchEnv(project.models.planet.control.BatchEnv([wrapped_env]))
    return batch_env


@measure_time()
def create_dummy_data(env: project.models.planet.control.InGraphBatchEnv,
                      preprocess_fn: Callable[[tf.Tensor], tf.Tensor],
                      ) -> Dict[str, tf.Tensor]:
    tensors = env.observ.copy()
    tensors['action'] = env.action
    tensors['reward'] = env.reward
    data = {k: tf.compat.v1.placeholder(shape=v[0].shape, dtype=v[0].dtype) for k, v in tensors.items()}
    data['image'] = preprocess_fn(data['image'])
    sequence = {k: tf.expand_dims(v, 0) for k, v in data.items()}
    sequence['length'] = tf.constant(1, dtype=tf.int32)
    batch = {k: tf.expand_dims(v, 0) for k, v in sequence.items()}
    return batch


@measure_time()
def create_graph(data: Dict[str, tf.Tensor],
                 config: AttrDict,
                 ) -> AttrDict:
    with tf.compat.v1.variable_scope('graph', use_resource=True):
        graph = AttrDict(_unlocked=True, step=tf.constant(0, dtype=tf.int32, name='step'))
        graph.update(build_network(data, config))
        graph.embedded = graph.encoder(data)
        graph.prior, graph.posterior = project.util.planet.unroll.closed_loop(
            graph.cell, graph.embedded, data['action'], debug=config.debug)
    return graph


@measure_time()
def create_agent(graph: AttrDict,
                 env: project.models.planet.control.InGraphBatchEnv,
                 params: AttrDict,
                 config: AttrDict,
                 deterministic: bool,
                 ) -> project.models.planet.control.MPCAgent:
    agent_config = AttrDict(
        cell=graph.cell,
        encoder=graph.encoder,
        planner=functools.partial(params.planner, graph=graph),
        objective=functools.partial(params.objective, graph=graph),
        exploration=params.exploration,
        preprocess_fn=config.preprocess_fn,
        postprocess_fn=config.postprocess_fn)
    return project.models.planet.control.MPCAgent(env, graph.step, False, False, agent_config, deterministic)


@measure_time()
def define_episode(env: project.models.planet.control.InGraphBatchEnv,
                   agent: project.models.planet.control.MPCAgent,
                   ) -> Tuple[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
    # tf.while_loop supports namedtuples but not dicts, so we define a namedtuple to store the metrics
    Metrics = namedtuple('Metrics', list(sorted(env.metrics.keys())))  # type: ignore[misc]

    assert len(env) == 1
    agent_indices = tf.range(1)

    with tf.control_dependencies([env.reset(agent_indices)]):
        begin = agent.begin_episode(agent_indices)

    def not_done(steps: tf.Tensor, score: tf.Tensor, metrics: Metrics) -> tf.Tensor:
        with tf.control_dependencies([begin, steps, score]):
            return tf.math.logical_not(env.done[0])

    def update_score(steps: tf.Tensor, score: tf.Tensor, metrics: Metrics) -> Tuple[tf.Tensor, tf.Tensor, Metrics]:
        with tf.control_dependencies([begin]):
            action = agent.perform(agent_indices, env.observ)
        action.set_shape(env.action.shape)
        do_step = env.step(action)
        with tf.control_dependencies([do_step]):
            new_steps = tf.add(steps, 1)
            new_score = tf.add(score, env.reward[0])
            new_metrics = Metrics(**{k: tf.identity(v[0]) for k, v in env.metrics.items()})  # type: ignore[call-arg]
        return new_steps, new_score, new_metrics

    initializer = (tf.constant(0),  # steps
                   tf.zeros_like(env.reward[0]),  # score
                   Metrics(**{k: v[0] for k, v in env.metrics.items()}))  # type: ignore[call-arg]  # metrics
    num_steps, final_score, final_metrics = tf.while_loop(not_done,
                                                          update_score,
                                                          initializer,
                                                          parallel_iterations=1)
    final_metrics = dict(final_metrics._asdict())  # convert namedtuple to dict
    return num_steps, final_score, final_metrics


def restore_checkpoint(sess: tf.compat.v1.Session, checkpoint: Path, logdir: Path, params: Dict[str, str]) -> None:
    to_initialize = set(sess.graph.get_collection(tf.compat.v1.GraphKeys.LOCAL_VARIABLES) +
                        project.util.planet.filter_variables(include=params.get('exclude')))
    to_restore = set(project.util.planet.filter_variables(**params))
    both = to_initialize.intersection(to_restore)
    assert not both, f'These variables are being initialized and restored: {both}'
    all_vars = set(sess.graph.get_collection(tf.compat.v1.GraphKeys.LOCAL_VARIABLES) +
                   sess.graph.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES))
    remainder = all_vars - to_initialize.union(to_restore)
    assert not remainder, f"These variables aren't being initialized: {remainder}"
    sess.run([tf.compat.v1.variables_initializer(list(to_initialize))])

    saver = tf.compat.v1.train.Saver(to_restore)
    original_checkpoint = checkpoint
    logdir = logdir.expanduser()
    checkpoint = checkpoint.expanduser()
    if not checkpoint.is_absolute():
        checkpoint = logdir / checkpoint
    checkpoint = checkpoint.resolve()
    if checkpoint.is_dir():
        checkpoint = tf.train.latest_checkpoint(str(checkpoint))
    if not checkpoint:
        raise ValueError(f'Could not find checkpoint in {original_checkpoint}.')

    restore_shapes = {var.name.rsplit(':')[0]: var.shape.as_list() for var in to_restore}
    checkpoint_shapes = dict(tf.train.list_variables(str(checkpoint)))
    num_vars = int(sum(np.prod(shape) for shape in checkpoint_shapes.values()))
    logger.debug(f'Checkpoint contains {num_vars} variables')
    for name, shape in restore_shapes.items():
        if name not in checkpoint_shapes.keys():
            logger.warning(f'Variable {name} with shape {shape} not found in checkpoint')
        elif checkpoint_shapes[name] != shape:
            logger.warning(f'Variable {name} has different shape in checkpoint: {checkpoint_shapes[name]}!={shape}')
    for name, shape in checkpoint_shapes.items():
        if name not in restore_shapes.keys():
            logger.debug(f'Checkpoint variable {name} with shape {shape} not being restored')

    saver.restore(sess, str(checkpoint))
