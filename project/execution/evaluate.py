# evaluate.py: 
#
# (C) 2019, Daniel Mouritzen

import functools
import os.path
from collections import namedtuple
from typing import Callable, Dict, Optional, Tuple

import planet
import planet.control.wrappers
import tensorflow as tf
from loguru import logger
from planet.scripts.configs import default as planet_config
from planet.training.define_model import build_network

from project.models.planet import AttrDict, PlanetParams, create_tf_session
from project.util import PrettyPrinter, Statistics, Timer, measure_time


def evaluate(logdir: str, checkpoint: Optional[str], num_episodes: int) -> None:
    params = PlanetParams()
    with params.unlocked:
        params.logdir = logdir
        params.batch_shape = [1, 1]
    config = AttrDict()
    with config.unlocked:
        config = planet_config(config, params)
    checkpoint = checkpoint or config.checkpoint_to_load
    if not checkpoint:
        raise ValueError('No checkpoint specified!')
    collect_params = next(iter(config.collects.values()))  # We assume all collects have same task and planner
    env = create_env(collect_params.task)
    data = create_dummy_data(env, config.preprocess_fn)
    graph = create_graph(data, config)
    agent = create_agent(graph, env, collect_params, config)
    steps_op, score_op, metrics_op = define_episode(env, agent)
    logger.debug(f'Graph contains {planet.tools.count_weights()} trainable variables')
    sess = create_tf_session()
    with sess:
        sess.run(tf.group(tf.compat.v1.local_variables_initializer(),
                          tf.compat.v1.global_variables_initializer()))
        restore_checkpoint(sess, checkpoint, logdir)
        sess.graph.finalize()
        statistics = Statistics(['steps', 'score', 'step_time'] + list(metrics_op.keys()),
                                save_file=f'{logdir}/eval.csv')
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
            env.save_video(f'{logdir}/episode_{episode}_spl_{metrics["spl"]}')
        logger.info('')
        logger.info('Finished evaluation.')
        logger.info('Results:')
        statistics.print()


@measure_time()
def create_env(task: AttrDict):
    params = task.env_ctor._args[1]  # TODO: do this in a less hacky way
    params['capture_video'] = True
    config = params['config']
    config.defrost()
    if 'TOP_DOWN_MAP' not in config.TASK.MEASUREMENTS:
        # Top-down map is expensive to compute, so we only enable it for evaluation.
        config.TASK.MEASUREMENTS.append('TOP_DOWN_MAP')
    config.freeze()
    env = task.env_ctor()
    env = planet.control.wrappers.SelectObservations(env, task.observation_components)
    env = planet.control.wrappers.SelectMetrics(env, task.metrics)
    env = planet.control.InGraphBatchEnv(planet.control.BatchEnv([env], blocking=True))
    return env


@measure_time()
def create_dummy_data(env: planet.control.InGraphBatchEnv,
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
        graph.prior, graph.posterior = planet.tools.unroll.closed_loop(
            graph.cell, graph.embedded, data['action'], config.debug)
    return graph


@measure_time()
def create_agent(graph: AttrDict,
                 env: planet.control.InGraphBatchEnv,
                 params: AttrDict,
                 config: AttrDict,
                 ) -> planet.control.MPCAgent:
    agent_config = AttrDict(
        cell=graph.cell,
        encoder=graph.encoder,
        planner=functools.partial(params.planner, graph=graph),
        objective=functools.partial(params.objective, graph=graph),
        exploration=params.exploration,
        preprocess_fn=config.preprocess_fn,
        postprocess_fn=config.postprocess_fn)
    return planet.control.MPCAgent(env, graph.step, False, False, agent_config)


@measure_time()
def define_episode(env: planet.control.InGraphBatchEnv,
                   agent: planet.control.MPCAgent,
                   ) -> Tuple[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
    # tf.while_loop supports namedtuples but not dicts, so we define a namedtuple to store the metrics
    Metrics = namedtuple('Metrics', sorted(env.metrics.keys()))

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
            new_metrics = Metrics(**{k: tf.identity(v[0]) for k, v in env.metrics.items()})
        return new_steps, new_score, new_metrics

    initializer = (tf.constant(0),                                        # steps
                   tf.zeros_like(env.reward[0]),                          # score
                   Metrics(**{k: v[0] for k, v in env.metrics.items()}))  # metrics
    num_steps, final_score, final_metrics = tf.while_loop(not_done,
                                                          update_score,
                                                          initializer,
                                                          parallel_iterations=1)
    final_metrics = dict(final_metrics._asdict())  # convert namedtuple to dict
    return num_steps, final_score, final_metrics


def restore_checkpoint(sess: tf.compat.v1.Session, checkpoint: str, logdir: str) -> None:
    # variables = planet.tools.filter_variables(exclude=[r'.*_temporary.*',
    #                                                    r'graph/collection.*',
    #                                                    r'graph/optimizer.*',
    #                                                    r'graph/phase_evaluate.*',
    #                                                    'global_step'])
    variables = planet.tools.filter_variables(include=[r'graph/head_.*', r'graph/encoder/.*', r'graph/rnn/.*'])
    saver = tf.compat.v1.train.Saver(variables)
    original_checkpoint = checkpoint
    logdir = os.path.expanduser(logdir)
    checkpoint = os.path.expanduser(checkpoint)
    if not os.path.isabs(checkpoint):
        checkpoint = os.path.join(logdir, checkpoint)
    checkpoint = os.path.abspath(checkpoint)
    if os.path.isdir(checkpoint):
        checkpoint = tf.train.latest_checkpoint(checkpoint)
    if checkpoint:
        saver.restore(sess, checkpoint)
    else:
        raise ValueError(f'Could not find checkpoint in {original_checkpoint}.')
