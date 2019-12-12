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

"""Environment wrappers."""
import datetime
import io
import os
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import gin
import gym
import gym.spaces
import numpy as np
import skimage.transform
import tensorflow as tf
from loguru import logger

from project.util.typing import Action, Observations, ObsTuple, Reward

from .base import Wrapper


class ObservationDict(Wrapper):

    def __init__(self, env: gym.Env, key: str = 'observ') -> None:
        super().__init__(env)
        self._key = key
        self.observation_space = gym.spaces.Dict({self._key: self.env.observation_space})

    def step(self, action: Action) -> ObsTuple:
        obs, reward, done, info = super().step(action)
        obs = {self._key: np.array(obs)}
        return obs, reward, done, info

    def reset(self) -> Observations:
        obs = super().reset()
        obs = {self._key: np.array(obs)}
        return obs


class SelectObservations(Wrapper):

    def __init__(self, env: gym.Env, keys: Iterable[str]) -> None:
        super().__init__(env)
        self._keys = keys
        self.observation_space = gym.spaces.Dict(
            {key: self.env.observation_space.spaces[key] for key in self._keys})

    def step(self, action: Action) -> ObsTuple:
        obs, reward, done, info = super().step(action)
        obs = {key: obs[key] for key in self._keys}
        return obs, reward, done, info

    def reset(self) -> Observations:
        obs = super().reset()
        obs = {key: obs[key] for key in self._keys}
        return obs


class SelectMetrics(Wrapper):

    def __init__(self, env: gym.Env, keys: Iterable[str]) -> None:
        super().__init__(env)
        self.metric_names = keys

    def step(self, action: Action) -> ObsTuple:
        obs, reward, done, info = super().step(action)
        info = {key: info[key] for key in self.metric_names}
        return obs, reward, done, info


class PixelObservations(Wrapper):

    def __init__(self,
                 env: gym.Env,
                 size: Tuple[int, int] = (64, 64),
                 dtype: Union[Type[np.uint8], Type[np.float]] = np.uint8,
                 key: str = 'image') -> None:
        assert isinstance(env.observation_space, gym.spaces.Dict)
        super().__init__(env)
        self._size = size
        self._dtype = dtype
        self._key = key
        self.observation_space = self._get_observation_space()

    def _get_observation_space(self) -> gym.spaces.Dict:
        high = {np.uint8: 255, np.float: 1.0}[self._dtype]
        image = gym.spaces.Box(0, high, self._size + (3,), dtype=self._dtype)
        spaces = self.env.observation_space.spaces.copy()
        assert self._key not in spaces
        spaces[self._key] = image
        return gym.spaces.Dict(spaces)

    def step(self, action: Action) -> ObsTuple:
        obs, reward, done, info = super().step(action)
        obs[self._key] = self._render_image()
        return obs, reward, done, info

    def reset(self) -> Observations:
        obs = super().reset()
        obs[self._key] = self._render_image()
        return obs

    def _render_image(self) -> np.ndarray:
        image: np.ndarray = self.env.render('rgb_array')
        if image.shape[:2] != self._size:
            kwargs = dict(
                output_shape=self._size, mode='edge', order=1, preserve_range=True)
            image = skimage.transform.resize(image, **kwargs).astype(image.dtype)
        if self._dtype and image.dtype != self._dtype:
            if image.dtype in (np.float32, np.float64) and self._dtype == np.uint8:
                image = (image * 255).astype(self._dtype)
            elif image.dtype == np.uint8 and self._dtype in (np.float32, np.float64):
                image = image.astype(self._dtype) / 255
            else:
                message = 'Cannot convert observations from {} to {}.'
                raise NotImplementedError(message.format(image.dtype, self._dtype))
        return image


class ObservationToRender(Wrapper):

    def __init__(self, env: gym.Env, key: str = 'image') -> None:
        super().__init__(env)
        self._key = key
        self._image = None
        self.observation_space = gym.spaces.Dict({})

    def step(self, action: Action) -> ObsTuple:
        obs, reward, done, info = super().step(action)
        self._image = obs.pop(self._key)
        return obs, reward, done, info

    def reset(self) -> Observations:
        obs = super().reset()
        self._image = obs.pop(self._key)
        return obs

    def render(self, *args: Any, **kwargs: Any) -> Optional[np.ndarray]:
        return self._image


class ActionRepeat(Wrapper):
    """Repeat the agent action multiple steps."""

    def __init__(self, env: gym.Env, amount: int) -> None:
        super().__init__(env)
        assert amount > 0
        self._amount = amount

    def step(self, action: Action) -> ObsTuple:
        done = False
        total_reward = 0.0
        current_step = 0
        while current_step < self._amount and not done:
            observ, reward, done, info = super().step(action)
            total_reward += reward
            current_step += 1
        return observ, total_reward, done, info


@gin.configurable(whitelist=[])
def action_repeat() -> Tuple[Type[ActionRepeat], Callable[[Dict[str, Any]], Dict[str, Any]]]:
    return ActionRepeat, lambda kwargs: {'amount': kwargs['action_repeat']}


class NormalizeActions(Wrapper):

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        low, high = env.action_space.low, env.action_space.high
        self._enabled = np.logical_and(np.isfinite(low), np.isfinite(high))
        self._low = np.where(self._enabled, low, -np.ones_like(low))
        self._high = np.where(self._enabled, high, np.ones_like(low))
        self.action_space = self._get_action_space()

    def _get_action_space(self) -> gym.spaces.Box:
        space = self.env.action_space
        low = np.where(self._enabled, -np.ones_like(space.low), space.low)
        high = np.where(self._enabled, np.ones_like(space.high), space.high)
        return gym.spaces.Box(low, high, dtype=space.dtype)

    def step(self, action: Action) -> ObsTuple:
        action = (action + 1) / 2 * (self._high - self._low) + self._low
        return super().step(action)


class DeepMindWrapper(gym.Env):
    """Wraps a DM Control environment into a Gym interface."""

    metadata = {'render.modes': ['rgb_array']}
    reward_range = (-np.inf, np.inf)

    def __init__(self, env: gym.Env, render_size: Tuple[int, int] = (64, 64), camera_id: int = 0) -> None:
        self.env = env
        self._render_size = render_size
        self._camera_id = camera_id
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

    def _get_observation_space(self) -> gym.spaces.Dict:
        components = {}
        for key, value in self.env.observation_spec().items():
            components[key] = gym.spaces.Box(
                -np.inf, np.inf, value.shape, dtype=np.float32)
        return gym.spaces.Dict(components)

    def _get_action_space(self) -> gym.spaces.Box:
        action_spec = self.env.action_spec()
        return gym.spaces.Box(
            action_spec.minimum, action_spec.maximum, dtype=np.float32)

    def step(self, action: Action) -> ObsTuple:
        time_step = super().step(action)
        obs = dict(time_step.observation)
        reward = time_step.reward or 0
        done = time_step.last()
        info = {'discount': time_step.discount}
        return obs, reward, done, info

    def reset(self) -> Observations:
        time_step = super().reset()
        return dict(time_step.observation)

    def render(self, *args: Any, **kwargs: Any) -> np.ndarray:
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        del args  # Unused
        del kwargs  # Unused
        return self.env.physics.render(
            *self._render_size, camera_id=self._camera_id)


class MaximumDuration(Wrapper):
    """Limits the episode to a given upper number of decision points."""

    def __init__(self, env: gym.Env, duration: int) -> None:
        super().__init__(env)
        self._duration = duration
        self._step: Optional[int] = None

    def step(self, action: Action) -> ObsTuple:
        if self._step is None:
            raise RuntimeError('Must reset environment.')
        observ, reward, done, info = super().step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            self._step = None
        return observ, reward, done, info

    def reset(self) -> Observations:
        self._step = 0
        return super().reset()


class MinimumDuration(Wrapper):
    """Extends the episode to a given lower number of decision points."""

    def __init__(self, env: gym.Env, duration: int) -> None:
        super().__init__(env)
        self._duration = duration
        self._step = 0

    def step(self, action: Action) -> ObsTuple:
        observ, reward, done, info = super().step(action)
        self._step += 1
        if self._step < self._duration:
            done = False
        return observ, reward, done, info

    def reset(self) -> Observations:
        self._step = 0
        return super().reset()


class PadActions(Wrapper):
    """Pad action space to the largest action space."""

    def __init__(self, env: gym.Env, spaces: Iterable[gym.spaces.Box]) -> None:
        super().__init__(env)
        self.action_space = self._pad_box_space(spaces)

    def step(self, action: Action) -> ObsTuple:
        assert isinstance(action, np.ndarray)
        action = action[:len(self.env.action_space.low)]
        return super().step(action)

    def reset(self) -> Observations:
        return super().reset()

    def _pad_box_space(self, spaces: Iterable[gym.spaces.Box]) -> gym.spaces.Box:
        assert all(len(space.low.shape) == 1 for space in spaces)
        length = max(len(space.low) for space in spaces)
        low, high = np.inf * np.ones(length), -np.inf * np.ones(length)
        for space in spaces:
            low[:len(space.low)] = np.minimum(space.low, low[:len(space.low)])
            high[:len(space.high)] = np.maximum(space.high, high[:len(space.high)])
        return gym.spaces.Box(low, high, dtype=np.float32)


class CollectGymDataset(Wrapper):
    """Collect transition tuples and store episodes as Numpy files.

    The time indices of the collected episode use the convention that at each
    time step, the agent first decides on an action, and the environment then
    returns the reward and observation.

    This means the action causes the environment state and thus observation and
    rewards at the same time step. A dynamics model can thus predict the sequence
    of observations and rewards from the sequence of actions.

    The first transition tuple contains the observation returned from resetting
    the environment, together with zeros for the action and reward. Thus, the
    episode length is one more than the number of decision points.
    """

    def __init__(self, env: gym.Env, outdir: Optional[str]) -> None:
        super().__init__(env)
        self._outdir = outdir and os.path.expanduser(outdir)
        self._episode: List[Dict[str, Any]] = []

    def step(self, action: Action) -> ObsTuple:
        observ, reward, done, info = super().step(action)
        transition = self._process_observ(observ).copy()
        transition['action'] = action
        transition['reward'] = reward
        for key in ['taken_action', 'success']:
            # Optional items
            if key in info:
                transition[key] = info[key]
        self._episode.append(transition)
        if done:
            episode = self._get_episode()
            if self._outdir:
                filename = self._get_filename()
                self._write(episode, filename)
        return observ, reward, done, info

    def reset(self) -> Observations:
        # Resetting the environment provides the observation for time step zero.
        # The action and reward are not known for this time step, so we zero them.
        observ = super().reset()
        transition: Dict[str, Any] = self._process_observ(observ).copy()
        self._episode = [transition]
        return observ

    def _process_observ(self, observ: Observations) -> Observations:
        if not isinstance(observ, dict):
            observ = {'observ': observ}
        return observ

    def _get_filename(self) -> str:
        assert self._outdir is not None
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        identifier = str(uuid.uuid4()).replace('-', '')
        filename = '{}-{}.npz'.format(timestamp, identifier)
        filename = os.path.join(self._outdir, filename)
        return filename

    def _get_episode(self) -> Dict[str, np.ndarray]:
        for k in self._episode[-1]:
            if k not in self._episode[0]:
                # First transition only has observation, so we put zeros for the other items
                self._episode[0][k] = np.zeros_like(self._episode[-1][k])
        np_episode = {k: np.array([t[k] for t in self._episode]) for k in self._episode[0]}
        for key, sequence in np_episode.items():
            if sequence.dtype == 'object':
                raise RuntimeError(f"Sequence '{key}' is not numeric:\n{sequence}")
        return np_episode

    def _write(self, episode: Dict[str, np.ndarray], filename: str) -> None:
        assert self._outdir is not None
        if not tf.io.gfile.exists(self._outdir):
            tf.io.gfile.makedirs(self._outdir)
        with io.BytesIO() as file_:
            np.savez_compressed(file_, **episode)
            file_.seek(0)
            with tf.io.gfile.GFile(filename, 'w') as ff:
                ff.write(file_.read())
        folder = os.path.basename(self._outdir)
        name = os.path.splitext(os.path.basename(filename))[0]
        logger.debug('Recorded episode {} to {}.'.format(name, folder))


class ConvertTo32Bit(Wrapper):
    """Convert data types of an OpenAI Gym environment to 32 bit."""

    def step(self, action: Action) -> ObsTuple:
        observ, reward, done, info = super().step(action)
        observ = tf.nest.map_structure(self._convert_observ, observ)
        reward = self._convert_reward(reward)
        return observ, reward, done, info

    def reset(self) -> Observations:
        observ = super().reset()
        observ = tf.nest.map_structure(self._convert_observ, observ)
        return observ

    def _convert_observ(self, observ: np.ndarray) -> np.ndarray:
        if not np.isfinite(observ).all():
            raise ValueError('Infinite observation encountered.')
        if observ.dtype == np.float64:
            return observ.astype(np.float32)
        if observ.dtype == np.int64:
            return observ.astype(np.int32)
        return observ

    def _convert_reward(self, reward: Reward) -> np.ndarray:
        if not np.isfinite(reward).all():
            raise ValueError('Infinite reward encountered.')
        return np.array(reward, dtype=np.float32)
