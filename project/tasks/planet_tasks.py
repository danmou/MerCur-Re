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

import gin
import gym
import numpy as np

from project.environments import wrappers

Task = collections.namedtuple(
    'Task', 'name, env_ctor, max_length, state_components, observation_components, metrics')


@gin.configurable(whitelist=['action_repeat'])
def cartpole_balance(action_repeat: int = 8) -> Task:
    max_length = 1000 // action_repeat
    state_components = ['reward', 'position', 'velocity']
    observation_components = ['image']

    def env_ctor() -> gym.Env:
        return _dm_control_env(action_repeat, max_length, 'cartpole', 'balance')
    return Task('cartpole_balance', env_ctor, max_length, state_components, observation_components, [])


@gin.configurable(whitelist=['action_repeat'])
def cartpole_swingup(action_repeat: int = 8) -> Task:
    max_length = 1000 // action_repeat
    state_components = ['reward', 'position', 'velocity']
    observation_components = ['image']

    def env_ctor() -> gym.Env:
        return _dm_control_env(action_repeat, max_length, 'cartpole', 'swingup')
    return Task('cartpole_swingup', env_ctor, max_length, state_components, observation_components, [])


@gin.configurable(whitelist=['action_repeat'])
def finger_spin(action_repeat: int = 2) -> Task:
    max_length = 1000 // action_repeat
    state_components = ['reward', 'position', 'velocity', 'touch']
    observation_components = ['image']

    def env_ctor() -> gym.Env:
        return _dm_control_env(action_repeat, max_length, 'finger', 'spin')
    return Task('finger_spin', env_ctor, max_length, state_components, observation_components, [])


@gin.configurable(whitelist=['action_repeat'])
def cheetah_run(action_repeat: int = 4) -> Task:
    max_length = 1000 // action_repeat
    state_components = ['reward', 'position', 'velocity']
    observation_components = ['image']

    def env_ctor() -> gym.Env:
        return _dm_control_env(action_repeat, max_length, 'cheetah', 'run')
    return Task('cheetah_run', env_ctor, max_length, state_components, observation_components, [])


@gin.configurable(whitelist=['action_repeat'])
def cup_catch(action_repeat: int = 4) -> Task:
    max_length = 1000 // action_repeat
    state_components = ['reward', 'position', 'velocity']
    observation_components = ['image']

    def env_ctor() -> gym.Env:
        return _dm_control_env(action_repeat, max_length, 'ball_in_cup', 'catch')
    return Task('cup_catch', env_ctor, max_length, state_components, observation_components, [])


@gin.configurable(whitelist=['action_repeat'])
def walker_walk(action_repeat: int = 2) -> Task:
    max_length = 1000 // action_repeat
    state_components = ['reward', 'height', 'orientations', 'velocity']
    observation_components = ['image']

    def env_ctor() -> gym.Env:
        return _dm_control_env(action_repeat, max_length, 'walker', 'walk')
    return Task('walker_walk', env_ctor, max_length, state_components, observation_components, [])


@gin.configurable(whitelist=['action_repeat'])
def reacher_easy(action_repeat: int = 4) -> Task:
    max_length = 1000 // action_repeat
    state_components = ['reward', 'position', 'velocity', 'to_target']
    observation_components = ['image']

    def env_ctor() -> gym.Env:
        return _dm_control_env(action_repeat, max_length, 'reacher', 'easy')
    return Task('reacher_easy', env_ctor, max_length, state_components, observation_components, [])


@gin.configurable(whitelist=['action_repeat'])
def gym_cheetah(action_repeat: int = 1) -> Task:
    # Works with `isolate_envs: process`.
    max_length = 1000 // action_repeat
    state_components = ['reward', 'state']
    observation_components = ['image']

    def env_ctor() -> gym.Env:
        return _gym_env(action_repeat, max_length, 'HalfCheetah-v3')
    return Task('gym_cheetah', env_ctor, max_length, state_components, observation_components, [])


@gin.configurable(whitelist=['action_repeat'])
def gym_racecar(action_repeat: int = 1) -> Task:
    # Works with `isolate_envs: thread`.
    max_length = 1000 // action_repeat
    state_components = ['reward']
    observation_components = ['image']

    def env_ctor() -> gym.Env:
        return _gym_env(action_repeat, max_length, 'CarRacing-v0', obs_is_image=True)
    return Task('gym_racing', env_ctor, max_length, state_components, observation_components, [])


def _dm_control_env(action_repeat: int,
                    max_length: int,
                    domain: str,
                    task: str,
                    normalize: bool = False) -> gym.Env:
    from dm_control import suite
    env = suite.load(domain, task)
    camera_id = 0
    env = wrappers.DeepMindWrapper(env, (64, 64), camera_id=camera_id)
    env = wrappers.ActionRepeat(env, action_repeat)
    if normalize:
        env = wrappers.NormalizeActions(env)
    env = wrappers.MaximumDuration(env, max_length)
    env = wrappers.PixelObservations(env, (64, 64), np.uint8, 'image')
    env = wrappers.ConvertTo32Bit(env)
    return env


def _gym_env(action_repeat: int, max_length: int, name: str, obs_is_image: bool = False) -> gym.Env:
    env = gym.make(name)
    env = wrappers.ActionRepeat(env, action_repeat)
    env = wrappers.NormalizeActions(env)
    env = wrappers.MaximumDuration(env, max_length)
    if obs_is_image:
        env = wrappers.ObservationDict(env, 'image')
        env = wrappers.ObservationToRender(env)
    else:
        env = wrappers.ObservationDict(env, 'state')
    env = wrappers.PixelObservations(env, (64, 64), np.uint8, 'image')
    env = wrappers.ConvertTo32Bit(env)
    return env
