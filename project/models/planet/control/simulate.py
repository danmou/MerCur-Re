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

"""In-graph simulation step of a vectorized algorithm with environments."""

import tensorflow as tf

from project.models.planet import tools
from project.models.planet.control import batch_env, in_graph_batch_env, mpc_agent
from project.models.planet.tools import streaming_mean


def simulate(
        step, env, duration, agent_config, expensive_summaries=False,
        gif_summary=True, metrics_summaries=True, name='simulate'):
    summaries = []
    with tf.compat.v1.variable_scope(name):
        return_, obs, action, reward, metrics = collect_rollouts(
            step=step,
            env=env,
            duration=duration,
            agent_config=agent_config)
        return_mean = tf.reduce_mean(return_)
        summaries.append(tf.compat.v1.summary.scalar('return', return_mean))
        metric_means = {k: tf.reduce_mean(v) for k, v in metrics.items()}
        metric_means['batched_episodes'] = tf.shape(return_)[0]
        if metrics_summaries:
            with tf.name_scope('metrics'):
                for name, value in metric_means.items():
                    summaries.append(tf.compat.v1.summary.scalar(name, value))
        if expensive_summaries:
            summaries.append(tf.compat.v1.summary.histogram('return_hist', return_))
            summaries.append(tf.compat.v1.summary.histogram('reward_hist', reward))
            summaries.append(tf.compat.v1.summary.histogram('action_hist', action))
            summaries.append(tools.image_strip_summary(
                'image', obs['image'], max_length=duration))
        if gif_summary:
            summaries.append(tools.gif_summary(
                'animation', obs['image'], max_outputs=1, fps=20))
    summary = tf.compat.v1.summary.merge(summaries)
    return summary, return_mean, metric_means


def collect_rollouts(step, env, duration, agent_config):
    batch_env = define_batch_env(env)
    agent = mpc_agent.MPCAgent(batch_env, step, False, False, agent_config)

    def simulate_fn(unused_last, step):
        done, score, unused_summary = simulate_step(
            batch_env, agent,
            log=False,
            reset=tf.equal(step, 0))
        with tf.control_dependencies([done, score]):
            obs = batch_env.observ
            batch_action = batch_env.action
            batch_reward = batch_env.reward
            batch_metrics = batch_env.metrics
        return done, score, obs, batch_action, batch_reward, batch_metrics

    initializer = (
        tf.zeros([1], tf.bool),
        tf.zeros([1], tf.float32),
        {k: 0 * v for k, v in batch_env.observ.items()},
        0 * batch_env.action,
        tf.zeros([1], tf.float32),
        {k: tf.zeros([1], tf.float32) for k in batch_env.metrics.keys()})
    done, score, observ, action, reward, metrics = tf.scan(
        simulate_fn, tf.range(duration),
        initializer, parallel_iterations=1)
    score = tf.boolean_mask(score, done)
    metrics = {k: tf.boolean_mask(v, done) for k, v in metrics.items()}
    done_indices = tf.pad(tf.where(done)[:, 0], tf.constant([[1, 0]]))
    lengths = done_indices[1:] - done_indices[:-1]
    metrics['steps_taken'] = lengths
    for key in observ.keys():
        observ[key] = tf.transpose(observ[key], [1, 0] + list(range(2, len(observ[key].shape))))
    action = tf.transpose(action, [1, 0, 2])
    reward = tf.transpose(reward)
    return score, observ, action, reward, metrics


def define_batch_env(env):
    with tf.compat.v1.variable_scope('environments'):
        env = batch_env.BatchEnv([env], blocking=True)
        env = in_graph_batch_env.InGraphBatchEnv(env)
    return env


def simulate_step(batch_env, algo, log=True, reset=False):
    """Simulation step of a vectorized algorithm with in-graph environments.

    Integrates the operations implemented by the algorithm and the environments
    into a combined operation.

    Args:
      batch_env: In-graph batch environment.
      algo: Algorithm instance implementing required operations.
      log: Tensor indicating whether to compute and return summaries.
      reset: Tensor causing all environments to reset.

    Returns:
      Tuple of tensors containing done flags for the current episodes, possibly
      intermediate scores for the episodes, and a summary tensor.
    """

    def _define_begin_episode(agent_indices):
        """Reset environments, intermediate scores and durations for new episodes.

        Args:
          agent_indices: Tensor containing batch indices starting an episode.

        Returns:
          Summary tensor, new score tensor, and new length tensor.
        """
        assert agent_indices.shape.ndims == 1
        zero_scores = tf.zeros_like(agent_indices, tf.float32)
        zero_durations = tf.zeros_like(agent_indices)
        update_score = tf.compat.v1.scatter_update(score_var, agent_indices, zero_scores)
        update_length = tf.compat.v1.scatter_update(
            length_var, agent_indices, zero_durations)
        reset_ops = [
            batch_env.reset(agent_indices), update_score, update_length]
        with tf.control_dependencies(reset_ops):
            return algo.begin_episode(agent_indices), update_score, update_length

    def _define_step():
        """Request actions from the algorithm and apply them to the environments.

        Increments the lengths of all episodes and increases their scores by the
        current reward. After stepping the environments, provides the full
        transition tuple to the algorithm.

        Returns:
          Summary tensor, new score tensor, and new length tensor.
        """
        prevob = batch_env.observ
        agent_indices = tf.range(len(batch_env))
        action = algo.perform(agent_indices, prevob)
        action.set_shape(batch_env.action.shape)
        with tf.control_dependencies([batch_env.step(action)]):
            add_score = score_var.assign_add(batch_env.reward)
            inc_length = length_var.assign_add(tf.ones(len(batch_env), tf.int32))
        with tf.control_dependencies([add_score, inc_length]):
            agent_indices = tf.range(len(batch_env))
            experience_summary = algo.experience(
                agent_indices, prevob,
                batch_env.action,
                batch_env.reward,
                batch_env.done,
                batch_env.observ)
        return experience_summary, add_score, inc_length

    def _define_end_episode(agent_indices):
        """Notify the algorithm of ending episodes.

        Also updates the mean score and length counters used for summaries.

        Args:
          agent_indices: Tensor holding batch indices that end their episodes.

        Returns:
          Summary tensor.
        """
        assert agent_indices.shape.ndims == 1
        submit_score = mean_score.submit(tf.gather(score, agent_indices))
        submit_length = mean_length.submit(
            tf.cast(tf.gather(length, agent_indices), tf.float32))
        with tf.control_dependencies([submit_score, submit_length]):
            return algo.end_episode(agent_indices)

    def _define_summaries():
        """Reset the average score and duration, and return them as summary.

        Returns:
          Summary string.
        """
        score_summary = tf.cond(
            tf.logical_and(log, tf.cast(mean_score.count, tf.bool)),
            lambda: tf.compat.v1.summary.scalar('mean_score', mean_score.clear()), str)
        length_summary = tf.cond(
            tf.logical_and(log, tf.cast(mean_length.count, tf.bool)),
            lambda: tf.compat.v1.summary.scalar('mean_length', mean_length.clear()), str)
        return tf.compat.v1.summary.merge([score_summary, length_summary])

    with tf.name_scope('simulate'):
        log = tf.convert_to_tensor(log)
        reset = tf.convert_to_tensor(reset)
        with tf.compat.v1.variable_scope('simulate_temporary'):
            score_var = tf.compat.v1.get_variable(
                'score', (len(batch_env),), tf.float32,
                tf.constant_initializer(0),
                trainable=False, collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES])
            length_var = tf.compat.v1.get_variable(
                'length', (len(batch_env),), tf.int32,
                tf.constant_initializer(0),
                trainable=False, collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES])
        mean_score = streaming_mean.StreamingMean((), tf.float32, 'mean_score')
        mean_length = streaming_mean.StreamingMean((), tf.float32, 'mean_length')
        agent_indices = tf.cond(
            reset,
            lambda: tf.range(len(batch_env)),
            lambda: tf.cast(tf.where(batch_env.done)[:, 0], tf.int32))
        begin_episode, score, length = tf.cond(
            tf.cast(tf.shape(agent_indices)[0], tf.bool),
            lambda: _define_begin_episode(agent_indices),
            lambda: (tf.no_op(), score_var, length_var))
        with tf.control_dependencies([begin_episode]):
            step, score, length = _define_step()
        with tf.control_dependencies([step]):
            agent_indices = tf.cast(tf.where(batch_env.done)[:, 0], tf.int32)
            end_episode = tf.cond(
                tf.cast(tf.shape(agent_indices)[0], tf.bool),
                lambda: _define_end_episode(agent_indices), tf.no_op)
        with tf.control_dependencies([end_episode]):
            summary = tf.compat.v1.summary.merge([_define_summaries(), step])
        with tf.control_dependencies([summary]):
            score = 0.0 + score
            done = batch_env.done
        return done, score, summary
