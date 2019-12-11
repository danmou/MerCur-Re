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

from typing import Dict, Optional

import tensorflow as tf


def chunk_sequence(sequence: Dict[str, tf.Tensor],
                   chunk_length: int,
                   randomize: bool = True,
                   num_chunks: Optional[int] = None,
                   ) -> Dict[str, tf.Tensor]:
    """Split a nested dict of sequence tensors into a batch of chunks.

    This function does not expect a batch of sequences, but a single sequence. A
    `length` key is added if it did not exist already. When `randomize` is set,
    up to `chunk_length - 1` initial frames will be discarded. Final frames that
    do not fit into a chunk are always discarded. If the sequence is shorter than
    chunk_length, it will be padded with zeros (but the length key will still be
    the true length)

    Args:
      sequence: Nested dict of tensors with time dimension.
      chunk_length: Size of chunks the sequence will be split into.
      randomize: Start chunking from a random offset in the sequence,
          enforcing that at least one chunk is generated.
      num_chunks: Optionally specify the exact number of chunks to be extracted
          from the sequence. Requires input to be long enough.

    Returns:
      Nested dict of sequence tensors with chunk dimension.
    """
    with tf.device('/cpu:0'):
        if 'length' in sequence:
            length = sequence.pop('length')
        else:
            length = tf.shape(tf.nest.flatten(sequence)[0])[0]

        def pad_sequence(tensor):
            pad_amount = tf.maximum(0, chunk_length - length)
            paddings = [[0, pad_amount]] + [[0, 0]] * (len(tensor.shape) - 1)
            return tf.pad(tensor, paddings)
        sequence = tf.nest.map_structure(pad_sequence, sequence)

        if randomize:
            if num_chunks is None:
                num_chunks = tf.maximum(1, length // chunk_length - 1)
            else:
                num_chunks = num_chunks + 0 * length
            used_length = num_chunks * chunk_length
            max_offset = tf.maximum(0, length - used_length)
            offset = tf.random.uniform((), 0, max_offset + 1, dtype=tf.int32)
        else:
            if num_chunks is None:
                num_chunks = length // chunk_length
            else:
                num_chunks = num_chunks + 0 * length
            used_length = num_chunks * chunk_length
            offset = 0
        clipped = tf.nest.map_structure(lambda tensor: tensor[offset: offset + used_length], sequence)
        chunks = tf.nest.map_structure(
            lambda tensor: tf.reshape(tensor, [num_chunks, chunk_length] + tensor.shape[1:].as_list()),
            clipped)
        chunks['length'] = tf.minimum(chunk_length, length) * tf.ones((num_chunks,), dtype=tf.int32)
        return chunks
