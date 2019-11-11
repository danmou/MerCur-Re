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

import gym
import numpy as np

from project.environments import wrappers
from project.util import planet as tools

Task = collections.namedtuple(
    'Task', 'name, env_ctor, max_length, state_components, observation_components, metrics')


def cartpole_balance(config: tools.AttrDict, params: tools.AttrDict) -> Task:
    action_repeat = params.get('action_repeat', 8)
    max_length = 1000 // action_repeat
    state_components = ['reward', 'position', 'velocity']
    observation_components = ['image']
    env_ctor = tools.bind(
        _dm_control_env, action_repeat, max_length, 'cartpole', 'balance',
        params)
    return Task('cartpole_balance', env_ctor, max_length, state_components, observation_components, [])


def cartpole_swingup(config: tools.AttrDict, params: tools.AttrDict) -> Task:
    action_repeat = params.get('action_repeat', 8)
    max_length = 1000 // action_repeat
    state_components = ['reward', 'position', 'velocity']
    observation_components = ['image']
    env_ctor = tools.bind(
        _dm_control_env, action_repeat, max_length, 'cartpole', 'swingup',
        params)
    return Task('cartpole_swingup', env_ctor, max_length, state_components, observation_components, [])


def finger_spin(config: tools.AttrDict, params: tools.AttrDict) -> Task:
    action_repeat = params.get('action_repeat', 2)
    max_length = 1000 // action_repeat
    state_components = ['reward', 'position', 'velocity', 'touch']
    observation_components = ['image']
    env_ctor = tools.bind(
        _dm_control_env, action_repeat, max_length, 'finger', 'spin', params)
    return Task('finger_spin', env_ctor, max_length, state_components, observation_components, [])


def cheetah_run(config: tools.AttrDict, params: tools.AttrDict) -> Task:
    action_repeat = params.get('action_repeat', 4)
    max_length = 1000 // action_repeat
    state_components = ['reward', 'position', 'velocity']
    observation_components = ['image']
    env_ctor = tools.bind(
        _dm_control_env, action_repeat, max_length, 'cheetah', 'run', params)
    return Task('cheetah_run', env_ctor, max_length, state_components, observation_components, [])


def cup_catch(config: tools.AttrDict, params: tools.AttrDict) -> Task:
    action_repeat = params.get('action_repeat', 4)
    max_length = 1000 // action_repeat
    state_components = ['reward', 'position', 'velocity']
    observation_components = ['image']
    env_ctor = tools.bind(
        _dm_control_env, action_repeat, max_length, 'ball_in_cup', 'catch',
        params)
    return Task('cup_catch', env_ctor, max_length, state_components, observation_components, [])


def walker_walk(config: tools.AttrDict, params: tools.AttrDict) -> Task:
    action_repeat = params.get('action_repeat', 2)
    max_length = 1000 // action_repeat
    state_components = ['reward', 'height', 'orientations', 'velocity']
    observation_components = ['image']
    env_ctor = tools.bind(
        _dm_control_env, action_repeat, max_length, 'walker', 'walk', params)
    return Task('walker_walk', env_ctor, max_length, state_components, observation_components, [])


def reacher_easy(config: tools.AttrDict, params: tools.AttrDict) -> Task:
    action_repeat = params.get('action_repeat', 4)
    max_length = 1000 // action_repeat
    state_components = ['reward', 'position', 'velocity', 'to_target']
    observation_components = ['image']
    env_ctor = tools.bind(
        _dm_control_env, action_repeat, max_length, 'reacher', 'easy', params)
    return Task('reacher_easy', env_ctor, max_length, state_components, observation_components, [])


def gym_cheetah(config: tools.AttrDict, params: tools.AttrDict) -> Task:
    # Works with `isolate_envs: process`.
    action_repeat = params.get('action_repeat', 1)
    max_length = 1000 // action_repeat
    state_components = ['reward', 'state']
    observation_components = ['image']
    env_ctor = tools.bind(
        _gym_env, action_repeat, config.batch_shape[1], max_length,
        'HalfCheetah-v3')
    return Task('gym_cheetah', env_ctor, max_length, state_components, observation_components, [])


def gym_racecar(config: tools.AttrDict, params: tools.AttrDict) -> Task:
    # Works with `isolate_envs: thread`.
    action_repeat = params.get('action_repeat', 1)
    max_length = 1000 // action_repeat
    state_components = ['reward']
    observation_components = ['image']
    env_ctor = tools.bind(
        _gym_env, action_repeat, config.batch_shape[1], max_length,
        'CarRacing-v0', obs_is_image=True)
    return Task('gym_racing', env_ctor, max_length, state_components, observation_components, [])


def _dm_control_env(action_repeat: int,
                    max_length: int,
                    domain: str,
                    task: str,
                    params: tools.AttrDict,
                    normalize: bool = False) -> gym.Env:
    from dm_control import suite
    env = suite.load(domain, task)
    camera_id = int(params.get('camera_id', 0))
    env = wrappers.DeepMindWrapper(env, (64, 64), camera_id=camera_id)
    env = wrappers.ActionRepeat(env, action_repeat)
    if normalize:
        env = wrappers.NormalizeActions(env)
    env = wrappers.MaximumDuration(env, max_length)
    env = wrappers.PixelObservations(env, (64, 64), np.uint8, 'image')
    env = wrappers.ConvertTo32Bit(env)
    return env


def _gym_env(action_repeat: int, min_length: int, max_length: int, name: str, obs_is_image: bool = False) -> gym.Env:
    env = gym.make(name)
    env = wrappers.ActionRepeat(env, action_repeat)
    env = wrappers.NormalizeActions(env)
    env = wrappers.MinimumDuration(env, min_length)
    env = wrappers.MaximumDuration(env, max_length)
    if obs_is_image:
        env = wrappers.ObservationDict(env, 'image')
        env = wrappers.ObservationToRender(env)
    else:
        env = wrappers.ObservationDict(env, 'state')
    env = wrappers.PixelObservations(env, (64, 64), np.uint8, 'image')
    env = wrappers.ConvertTo32Bit(env)
    return env
