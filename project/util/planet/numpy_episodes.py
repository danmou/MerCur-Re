# Copyright 2019 The PlaNet Authors. All rights reserved.
# Modifications copyright 2019 Daniel Mouritzen.
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

"""Load tensors from a directory of numpy files."""

import functools
import os
import random
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, TypeVar

import gin
import numpy as np
import tensorflow as tf
from scipy.ndimage import interpolation

from .chunk_sequence import chunk_sequence
from .preprocess import preprocess

Episode = Dict[str, np.ndarray]


@gin.configurable(whitelist=['num_chunks', 'loader_update_every', 'train_action_noise'])
def numpy_episodes(train_dir: str,
                   test_dir: str,
                   shape: Tuple[int, int],
                   num_chunks: Optional[int] = 1,
                   loader_update_every: int = 1000,
                   train_action_noise: float = 0.3) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Read sequences stored as compressed Numpy files as a TensorFlow dataset.

    Args:
      train_dir: Directory containing NPZ files of the training dataset.
      test_dir: Directory containing NPZ files of the testing dataset.
      shape: Tuple of batch size and chunk length for the datasets.
      num_chunks: Number of chunks to extract from each sequence.
      loader_update_every: Number of episodes between loader cache updates.
      train_action_noise: Amount of noise to add to actions of training episodes

    Returns:
      Structured data from numpy episodes as Tensors.
    """
    loader = recent_loader
    dtypes, shapes, _ = _read_spec(train_dir)
    train = tf.data.Dataset.from_generator(
        functools.partial(loader, train_dir, loader_update_every, train_action_noise),
        dtypes, shapes)
    test = tf.data.Dataset.from_generator(
        functools.partial(loader, test_dir, loader_update_every),
        dtypes, shapes)

    def chunking(x: Dict[str, tf.Tensor]) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices(
            chunk_sequence(x, shape[1], True, num_chunks))

    def sequence_preprocess_fn(sequence: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        sequence['image'] = preprocess(sequence['image'])
        return sequence

    train, test = (dataset
                   .flat_map(chunking)
                   .batch(shape[0], drop_remainder=True)
                   .map(sequence_preprocess_fn, tf.data.experimental.AUTOTUNE)
                   .prefetch(tf.data.experimental.AUTOTUNE)
                   for dataset in (train, test))
    return train, test


def cache_loader(directory: str,
                 update_every: int,
                 action_noise: Optional[float] = None,
                 ) -> Generator[Episode, None, None]:
    """Loads all files into a cache, which is updated after `update_every` episodes"""
    cache: Dict[str, Episode] = {}
    while True:
        episodes = _sample(cache.values(), update_every)
        for episode in _permuted(episodes, update_every):
            yield episode
        filenames = tf.io.gfile.glob(os.path.join(directory, '*.npz'))
        filenames = [filename for filename in filenames if filename not in cache]
        for filename in filenames:
            cache[filename] = episode_reader(filename, action_noise=action_noise)


def recent_loader(directory: str,
                  update_every: int,
                  action_noise: Optional[float] = None,
                  ) -> Generator[Episode, None, None]:
    """Same as cache_loader, but 50% of the episodes come from the latest added set of files"""
    recent: Dict[str, Episode] = {}
    cache: Dict[str, Episode] = {}
    while True:
        episodes: List[Episode] = []
        episodes += _sample(recent.values(), update_every // 2)
        episodes += _sample(cache.values(), update_every // 2)
        for episode in _permuted(episodes, update_every):
            yield episode
        cache.update(recent)
        recent = {}
        filenames = tf.io.gfile.glob(os.path.join(directory, '*.npz'))
        filenames = [filename for filename in filenames if filename not in cache]
        for filename in filenames:
            recent[filename] = episode_reader(filename, action_noise=action_noise)


def reload_loader(directory: str,
                  update_every: Optional[int] = None,
                  action_noise: Optional[float] = None,
                  ) -> Generator[Episode, None, None]:
    """Simple loader without cache"""
    directory = os.path.expanduser(directory)
    while True:
        filenames = tf.io.gfile.glob(os.path.join(directory, '*.npz'))
        random.shuffle(filenames)
        for filename in filenames:
            yield episode_reader(filename, action_noise=action_noise)


def dummy_loader(directory: str,
                 update_every: Optional[int] = None,
                 action_noise: Optional[float] = None,
                 ) -> Generator[Episode, None, None]:
    random = np.random.RandomState(seed=0)
    dtypes, shapes, length = _read_spec(directory, numpy_types=True)
    while True:
        episode = {}
        for key in dtypes:
            dtype, shape = dtypes[key], (length,) + shapes[key][1:]
            if dtype in (np.float32, np.float64):
                episode[key] = random.uniform(0, 1, shape).astype(dtype)
            elif dtype in (np.int32, np.int64, np.uint8):
                episode[key] = random.uniform(0, 255, shape).astype(dtype)
            else:
                raise NotImplementedError('Unsupported dtype {}.'.format(dtype))
        yield episode


def episode_reader(filename: str,
                   resize: Optional[float] = None,
                   max_length: Optional[int] = None,
                   action_noise: Optional[float] = None,
                   ) -> Episode:
    episode: Episode = np.load(filename)
    episode = {key: _convert_type(episode[key]) for key in episode.keys()}
    episode['return'] = np.cumsum(episode['reward'])
    if max_length:
        episode = {key: value[:max_length] for key, value in episode.items()}
    if resize and resize != 1.0:
        factors = (1, resize, resize, 1)
        episode['image'] = interpolation.zoom(episode['image'], factors)
    if action_noise:
        seed = np.fromstring(filename, dtype=np.uint8)
        episode['action'] += np.random.RandomState(seed).normal(
            0, action_noise, episode['action'].shape)
    return episode


def _read_spec(directory: str,
               numpy_types: bool = False,
               ) -> Tuple[Dict[str, Any], Dict[str, Tuple[Optional[int]]], int]:
    episodes = reload_loader(directory)
    episode = next(episodes)
    episodes.close()
    dtypes = {key: value.dtype for key, value in episode.items()}
    if not numpy_types:
        dtypes = {key: tf.as_dtype(value) for key, value in dtypes.items()}
    shapes = {key: value.shape for key, value in episode.items()}
    shapes = {key: (None,) + shape[1:] for key, shape in shapes.items()}
    length = len(episode[list(shapes.keys())[0]])
    return dtypes, shapes, length


def _convert_type(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.float64:
        return array.astype(np.float32)
    if array.dtype == np.int64:
        return array.astype(np.int32)
    return array


T = TypeVar('T')


def _sample(sequence: Iterable[T], amount: int) -> Iterable[T]:
    sequence = list(sequence)
    amount = min(amount, len(sequence))
    return random.sample(sequence, amount)


def _permuted(sequence: Iterable[T], amount: int) -> Generator[T, None, None]:
    sequence = list(sequence)
    if not sequence:
        return
    index = 0
    while True:
        for element in np.random.permutation(sequence):
            if index >= amount:
                return
            yield element
            index += 1
