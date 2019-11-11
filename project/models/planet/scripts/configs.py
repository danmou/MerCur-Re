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

import os

import tensorflow as tf

import project.tasks as tasks_lib
from project.environments import wrappers
from project.models.planet import control, models, networks
from project.models.planet.scripts import objectives as objectives_lib
from project.util import planet as tools

ACTIVATIONS = {
    'relu': tf.nn.relu,
    'elu': tf.nn.elu,
    'tanh': tf.tanh,
    'swish': lambda x: x * tf.sigmoid(x),
    'softplus': tf.nn.softplus,
    'none': None,
}


def default(config, params):
    config.debug = False
    config.loss_scales = tools.AttrDict(_unlocked=True)
    config = _data_processing(config, params)
    config = _model_components(config, params)
    config = _tasks(config, params)
    config = _loss_functions(config, params)
    config = _training_schedule(config, params)
    return config


def debug(config, params):
    defaults = tools.AttrDict(_unlocked=True)
    defaults.action_repeat = 50
    defaults.num_seed_episodes = 1
    defaults.train_steps = 2
    defaults.test_steps = 2
    defaults.max_epochs = 2
    defaults.train_collect_every = 1
    defaults.num_train_episodes = 1
    defaults.test_collect_every = 1
    defaults.num_test_episodes = 1
    defaults.model_size = 10
    defaults.state_size = 5
    defaults.num_layers = 1
    defaults.num_units = 10
    defaults.batch_shape = [5, 10]
    defaults.loader_every = 5
    defaults.loader_window = 2
    defaults.planner_amount = 5
    defaults.planner_topk = 2
    defaults.planner_iterations = 2
    with params.unlocked():
        for key, value in defaults.items():
            if key not in params:
                params[key] = value
    config = default(config, params)
    config.debug = True
    return config


def _data_processing(config, params):
    config.batch_shape = params.get('batch_shape', (50, 50))
    config.num_chunks = params.get('num_chunks', 1)
    image_bits = params.get('image_bits', 5)
    config.preprocess_fn = tools.bind(
        tools.preprocess.preprocess, bits=image_bits)
    config.postprocess_fn = tools.bind(
        tools.preprocess.postprocess, bits=image_bits)
    config.open_loop_context = 5
    config.data_reader = tools.numpy_episodes.episode_reader
    config.data_loader = {
        'cache': tools.bind(
            tools.numpy_episodes.cache_loader,
            every=params.get('loader_every', 1000)),
        'recent': tools.bind(
            tools.numpy_episodes.recent_loader,
            every=params.get('loader_every', 1000)),
        'reload': tools.numpy_episodes.reload_loader,
        'dummy': tools.numpy_episodes.dummy_loader,
    }[params.get('loader', 'recent')]
    return config


def _model_components(config, params):
    config.gradient_heads = params.get('gradient_heads', ['image', 'reward'])
    network = getattr(networks, params.get('network', 'conv_ha'))
    config.activation = ACTIVATIONS[params.get('activation', 'relu')]
    config.num_layers = params.get('num_layers', 3)
    config.num_units = params.get('num_units', 300)
    config.head_network = tools.bind(
        networks.feed_forward,
        num_layers=config.num_layers,
        units=config.num_units,
        activation=config.activation)
    config.encoder = network.encoder
    config.decoder = network.decoder
    config.heads = tools.AttrDict(_unlocked=True)
    config.heads.image = config.decoder
    size = params.get('model_size', 200)
    state_size = params.get('state_size', 30)
    model = params.get('model', 'rssm')
    if model == 'rssm':
        config.cell = tools.bind(
            models.RSSM, state_size, size, size,
            params.get('mean_only', False),
            params.get('min_stddev', 1e-1),
            config.activation,
            params.get('model_layers', 1))
    else:
        raise NotImplementedError("Unknown model '{}.".format(params.model))
    return config


def _tasks(config, params):
    tasks = params.get('tasks', ['cheetah_run'])
    tasks = [getattr(tasks_lib, name)(config, params) for name in tasks]
    tasks = {task.name: task for task in tasks}
    config.isolate_envs = params.get('isolate_envs', 'thread')

    def common_spaces_ctor(task, action_spaces):
        env = task.env_ctor()
        env = wrappers.SelectObservations(env, task.observation_components)
        env = wrappers.SelectMetrics(env, task.metrics)
        env = wrappers.PadActions(env, action_spaces)
        return env

    if len(tasks) > 1:
        action_spaces = [task.env_ctor().action_space for task in tasks]
        for name, task in tasks.items():
            env_ctor = tools.bind(common_spaces_ctor, task, action_spaces)
            tasks[name] = tasks_lib.Task(
                task.name, env_ctor, task.max_length, ['reward'], task.observation_components, task.metrics)
    additional_components = {name
                             for task in tasks.values()
                             for name in task.observation_components
                             if name not in config.heads}
    for name in additional_components:
        config.heads[name] = config.head_network
    encoder = config.encoder
    config.encoder = lambda obs: tf.concat(
        [encoder(obs)] + [obs[name] for name in additional_components], -1)
    config.heads['reward'] = tools.bind(
        config.head_network,
        stop_gradient='reward' not in config.gradient_heads)
    config.loss_scales['reward'] = 1.0
    config.tasks = tasks
    return config


def _loss_functions(config, params):
    for head in config.gradient_heads:
        assert head in config.heads, head
    config.loss_scales.divergence = params.get('divergence_scale', 1.0)
    config.loss_scales.global_divergence = params.get('global_div_scale', 0.0)
    config.loss_scales.overshooting = params.get('overshooting_scale', 0.0)
    for head in config.heads:
        defaults = {'reward': 10.0}
        scale = defaults[head] if head in defaults else 1.0
        config.loss_scales[head] = params.get(head + '_loss_scale', scale)
    config.free_nats = params.get('free_nats', 3.0)
    config.overshooting_distance = params.get('overshooting_distance', 0)
    config.os_stop_posterior_grad = params.get('os_stop_posterior_grad', True)
    config.optimizers = tools.AttrDict(_unlocked=True)
    config.optimizers.main = tools.bind(
        tools.CustomOptimizer,
        optimizer_cls=tools.bind(tf.compat.v1.train.AdamOptimizer, epsilon=1e-4),
        # schedule=tools.bind(tools.schedule.linear, ramp=0),
        learning_rate=params.get('main_learning_rate', 1e-3),
        clipping=params.get('main_gradient_clipping', 1000.0))
    return config


def _training_schedule(config, params):
    config.train_steps = int(params.get('train_steps', 1000))
    config.test_steps = int(params.get('test_steps', 1))
    config.max_epochs = int(params.get('max_epochs', 1000))
    config.train_log_every = config.train_steps * int(params.get('log_summaries_every', 1))
    config.test_log_every = config.test_steps * int(params.get('log_summaries_every', 1))
    config.checkpoint_every = params.get('checkpoint_every', 10)
    config.checkpoint_to_load = params.get('checkpoint_to_load')
    config.savers = [dict(exclude=(r'.*_temporary.*',))]
    load_exclude = params.get('checkpoint_load_exclude')
    if config.checkpoint_to_load and load_exclude:
        config.savers[0]['load'] = False
        config.savers.append(dict(
            exclude=(r'.*_temporary.*',) + tuple(load_exclude),
            save=False))
    config.print_objectives_every = max(1, config.train_steps // 10)
    config.train_dir = os.path.join(params.logdir, 'train_episodes')
    config.test_dir = os.path.join(params.logdir, 'test_episodes')
    config.random_collects = _initial_collection(config, params)
    config.collects = tools.AttrDict()
    with config.collects.unlocked():
        config.collects.update(_active_collection(
            params.get('train_collects', [{}]), dict(
                every=params.get('train_collect_every', 100),
                num_episodes=params.get('num_train_episodes', 1),
                eval_every=None,
                prefix='collect_train',
                save_episode_dir=config.train_dir,
                action_noise=params.get('train_action_noise', 0.3),
            ), config, params))
        config.collects.update(_active_collection(
            params.get('test_collects', [{}]), dict(
                every=params.get('test_collect_every', 100),
                num_episodes=params.get('num_test_episodes', 1),
                eval_every=params.get('eval_every', 10),
                prefix='collect_test',
                save_episode_dir=config.test_dir,
                action_noise=0.0,
            ), config, params))
    return config


def _initial_collection(config, params):
    num_seed_episodes = params.get('num_seed_episodes', 5)
    sims = tools.AttrDict(_unlocked=True)
    for name, task in config.tasks.items():
        sims['train-' + name] = tools.AttrDict(
            task=task,
            save_episode_dir=config.train_dir,
            num_episodes=num_seed_episodes)
        sims['test-' + name] = tools.AttrDict(
            task=task,
            save_episode_dir=config.test_dir,
            num_episodes=num_seed_episodes)
    return sims


def _active_collection(collects, defaults, config, params):
    defs = dict(
        name='main',
        horizon=params.get('planner_horizon', 12),
        objective=params.get('collect_objective', 'reward'),
        action_noise=0.0,
        action_noise_ramp=params.get('action_noise_ramp', 0),
        action_noise_min=params.get('action_noise_min', 0.0),
    )
    defs.update(defaults)
    sims = tools.AttrDict(_unlocked=True)
    for name, task in config.tasks.items():
        for collect in collects:
            collect = tools.AttrDict(collect, _defaults=defs)
            sim = _define_simulation(task, params, collect.horizon,
                                     collect.objective)
            sim.unlock()
            sim.save_episode_dir = collect.save_episode_dir
            sim.every = int(collect.every)
            sim.num_episodes = int(collect.num_episodes)
            sim.eval_every = collect.eval_every
            sim.exploration = tools.AttrDict(
                scale=collect.action_noise,
                schedule=tools.bind(
                    tools.schedule.linear,
                    ramp=collect.action_noise_ramp,
                    min=collect.action_noise_min,
                ))
            collect_name = '{}_{}_{}'.format(collect.prefix, collect.name, name)
            assert collect_name not in sims, (set(sims.keys()), collect_name)
            sims[collect_name] = sim
            assert not collect.untouched, collect.untouched
    return sims


def _define_simulation(task, params, horizon, objective='reward'):
    planner = params.get('planner', 'cem')
    if planner == 'cem':
        planner_fn = tools.bind(
            control.planning.cross_entropy_method,
            amount=params.get('planner_amount', 1000),
            iterations=params.get('planner_iterations', 10),
            topk=params.get('planner_topk', 100),
            horizon=horizon)
    else:
        raise NotImplementedError(planner)
    return tools.AttrDict(
        task=task,
        planner=planner_fn,
        objective=tools.bind(getattr(objectives_lib, objective), params=params))
