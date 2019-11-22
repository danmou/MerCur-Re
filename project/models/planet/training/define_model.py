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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from project.models.planet import tools
from project.models.planet.training import define_summaries
from project.models.planet.training import utility


def define_model(data, trainer, config, envs):
  tf.compat.v1.logging.info('Build TensorFlow compute graph.')
  cleanups = []
  graph = tools.AttrDict(
      _unlocked=True, step=trainer.step,
      global_step=trainer.global_step, data=data)

  graph.update(build_network(data, config))
  summaries, graph_ = apply_network(graph, data, trainer, config)
  graph.update(graph_)

  collect_summaries, collect_scores, collect_metrics = define_collection(
      graph, config, cleanups, envs)

  # Compute summaries.
  summary = tf.cond(
      trainer.log,
      lambda: define_summaries.define_summaries(graph, config, cleanups),
      lambda: tf.constant(''),
      name='summaries')
  summaries = tf.compat.v1.summary.merge([summaries, summary])
  prints = []
  prints.append(utility.print_metrics(
      {ob.name: ob.value for ob in graph.objectives},
      graph.step, config.print_objectives_every, 'objectives'))
  prints.append(utility.print_metrics(
      graph.grad_norms, graph.step, config.print_objectives_every, 'grad_norms'))
  with tf.control_dependencies(prints):
    summaries = tf.identity(summaries)
  return summaries, collect_scores, collect_summaries, cleanups


def build_network(data, config):
  cell = config.cell()
  kwargs = dict(create_scope_now_=True)
  encoder = tf.compat.v1.make_template('encoder', config.encoder, **kwargs)
  heads = tools.AttrDict(_unlocked=True)
  dummy_features = cell.features_from_state(cell.zero_state(1, tf.float32))
  for key, head in config.heads.items():
    name = 'head_{}'.format(key)
    kwargs = dict(create_scope_now_=True)
    if key in data:
      kwargs['data_shape'] = data[key].shape[2:].as_list()
    elif key == 'action_target':
      kwargs['data_shape'] = data['action'].shape[2:].as_list()
    heads[key] = tf.compat.v1.make_template(name, head, **kwargs)
    heads[key](dummy_features)  # Initialize weights.
  return tools.AttrDict(cell=cell, encoder=encoder, heads=heads)


def apply_network(graph, data, trainer, config):
  embedded = graph.encoder(data)
  prior, posterior = tools.unroll.closed_loop(
      graph.cell, embedded, data['action'], config.debug)
  objectives = utility.compute_objectives(
      posterior, prior, data, graph, config)
  summaries, grad_norms = utility.apply_optimizers(
      objectives, trainer, config)
  return summaries, tools.AttrDict(
      embedded=embedded, objectives=objectives, grad_norms=grad_norms)


def define_collection(graph, config, cleanups, envs):
  summaries = {}
  scores = {}
  metrics = {}
  with tf.compat.v1.variable_scope('collection'):
    with tf.control_dependencies(summaries):  # Make sure to train first.
      for name, params in config.collects.items():
        env = envs[params.task.name]
        summary, score, metrics_ = utility.simulate_episodes(
            config, params, graph, cleanups, env, expensive_summaries=False,
            gif_summary=bool(params.eval_every), metrics_summaries=bool(params.eval_every), name=name)
        print_every = utility.print_metrics(metrics_, graph.step, 1, name)
        with tf.control_dependencies([print_every]):
          print_mean = utility.print_metrics(
              metrics_, graph.step, params.num_episodes, f'{name}_mean')
        with tf.control_dependencies([print_mean]):
          summary = tf.identity(summary)
        summaries[name] = summary
        scores[name] = score
        metrics[name] = metrics_
  return summaries, scores, metrics
