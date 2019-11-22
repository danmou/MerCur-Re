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

import collections
import os

import tensorflow as tf

from project.models.planet import tools


_Phase = collections.namedtuple(
    'Phase',
    'name, writer, op, steps, feed, yield_every, log_every,'
    'collect_every')


_StoreCheckpoint = collections.namedtuple('StoreCheckpoint', 'steps, every')


class Trainer(object):
  """Execute operations in a trainer and coordinate logging and checkpoints.

  Supports multiple phases, that define their own operations to run, and
  intervals for reporting scores, logging summaries, and storing checkpoints.
  All class state is stored in-graph to properly recover from checkpoints.
  """

  def __init__(self, logdir, config=None):
    """Execute operations in a trainer and coordinate logging and checkpoints.

    The `reset` property is used to indicate switching to a new phase, so that
    the model can start a new computation in case its computation is split over
    multiple training steps.

    Args:
      logdir: Will contain checkpoints and summaries for each phase.
      config: configuration AttrDict.
    """
    self._logdir = logdir
    self._global_step = tf.compat.v1.train.get_or_create_global_step()
    self._next_step = self._global_step.assign_add(1)
    self._step = tf.compat.v1.placeholder(tf.int32, name='step')
    self._phase = tf.compat.v1.placeholder(tf.string, name='phase')
    self._log = tf.compat.v1.placeholder(tf.bool, name='log')
    self._reset = tf.compat.v1.placeholder(tf.bool, name='reset')
    self._do_run = tf.compat.v1.placeholder(tf.bool, name='do_run')
    self._phases = []
    # Checkpointing.
    self._loaders = []
    self._savers = []
    self._logdirs = []
    self._checkpoints = []
    self._config = config or tools.AttrDict()

  @property
  def global_step(self):
    """Global number of steps performed over all phases."""
    return self._global_step

  @property
  def step(self):
    """Number of steps performed in the current phase."""
    return self._step

  @property
  def phase(self):
    """Name of the current training phase."""
    return self._phase

  @property
  def log(self):
    """Whether the model should compute summaries."""
    return self._log

  @property
  def reset(self):
    """Whether the model should reset its state."""
    return self._reset

  @property
  def do_run(self):
    """Whether to do anything in this step."""
    return self._do_run

  def add_saver(
      self, include=r'.*', exclude=r'.^', load=True, save=True,
      checkpoint=None):
    """Add a saver to save or load variables.

    Args:
      include: One or more regexes to match variable names to include.
      exclude: One or more regexes to match variable names to exclude.
      load: Whether to use the saver to restore variables.
      save: Whether to use the saver to save variables.
      checkpoint: Checkpoint name to load; None for newest.
    """
    assert save or load
    variables = tools.filter_variables(include, exclude)
    saver = tf.compat.v1.train.Saver(variables, keep_checkpoint_every_n_hours=2, save_relative_paths=True)
    if load:
      self._loaders.append(saver)
      if checkpoint is None and self._config.checkpoint_to_load:
        self._checkpoints.append(self._config.checkpoint_to_load)
      else:
        self._checkpoints.append(checkpoint)
    if save:
      self._savers.append(saver)

  def add_phase(
      self, name, steps, score, summary, yield_every=None, log_every=None,
      feed=None, collect_every=None):
    """Add a phase to the trainer protocol.

    The score tensor can either be a scalar or vector, to support single and
    batched computations.

    Args:
      name: Name for the phase, used for the summary writer.
      steps: Duration of the phase in steps. For data collection, number of
             episodes to collect.
      score: Tensor holding the current scores.
      summary: Tensor holding summary string to write if not an empty string.
      yield_every: Yield every this number of epochs to allow independent evaluation.
      log_every: Request summaries via `log` tensor every this number of steps.
      feed: Additional feed dictionary for the session run call.
      collect_every: For data collection, only run every this number of epochs.
    """
    score = tf.convert_to_tensor(score, tf.float32)
    summary = tf.convert_to_tensor(summary, tf.string)
    feed = feed or {}
    if not score.shape.ndims:
      score = score[None]
    writer = self._logdir and tf.compat.v1.summary.FileWriter(
        os.path.join(self._logdir, name),
        tf.compat.v1.get_default_graph(), flush_secs=30)
    op = self._define_step(name, score, summary)
    self._phases.append(_Phase(
        name, writer, op, int(steps), feed, yield_every,
        log_every, collect_every))

  def add_checkpoint_phase(self, every):
    self._phases.append(_StoreCheckpoint(steps=1, every=every))

  def run(self, max_epochs=None, sess=None, unused_saver=None):
    """Run the schedule for a specified number of steps and log scores.

    Args:
      max_epochs: Run the operations until the epoch reaches this limit.
      sess: Session to use to run the phase operation.
    """
    for _ in self.iterate(max_epochs, sess):
      pass

  def iterate(self, max_epochs=None, sess=None):
    """Run the schedule for a specified number of steps and yield scores.

    Call the operation of the current phase until the global step reaches the
    specified maximum step. Phases are repeated over and over in the order they
    were added.

    Args:
      max_epochs: Run the operations until the epoch reaches this limit.
      sess: Session to use to run the phase operation.

    Yields:
      Reported mean scores.
    """
    sess = sess or self._create_session()
    with sess:
      self._initialize_variables(sess, self._loaders, self._checkpoints)
      sess.graph.finalize()
      while True:
        global_step = sess.run(self._global_step)
        phase, epoch, steps_in = self._find_current_phase(global_step)
        if max_epochs and epoch >= max_epochs:
          break
        if isinstance(phase, _StoreCheckpoint):
          if self._is_every_steps(epoch, phase.every):
            tf.compat.v1.logging.info('Saving checkpoint.')
            for saver in self._savers:
              self._store_checkpoint(sess, saver, epoch)
          sess.run(self._next_step)
          continue
        phase_step = epoch * phase.steps + steps_in
        if steps_in % phase.steps == 0:
          message = '\n' + ('-' * 50) + '\n'
          message += 'Epoch {} phase {} (phase step {}, global step {}).'
          tf.compat.v1.logging.info(message.format(
              epoch + 1, phase.name, phase_step, global_step))
        # Populate book keeping tensors.
        phase.feed[self._step] = phase_step
        phase.feed[self._phase] = phase.name
        phase.feed[self._reset] = (steps_in == 0)
        phase.feed[self._log] = phase.writer and self._is_every_steps(
            phase_step, phase.log_every)
        phase.feed[self._do_run] = not phase.collect_every or self._is_every_steps(epoch, phase.collect_every)
        summary, score, global_step = sess.run(phase.op, phase.feed)
        if self._is_every_steps(epoch, phase.yield_every):
          yield score
        if summary and phase.writer:
          # We want smaller phases to catch up at the beginnig of each epoch so
          # that their graphs are aligned.
          longest_phase = max(phase_.steps for phase_ in self._phases)
          summary_step = epoch * longest_phase + steps_in
          phase.writer.add_summary(summary, summary_step)

  def _is_every_steps(self, phase_step, every):
    """Determine whether a periodic event should happen at this step.

    Args:
      phase_step: The incrementing step.
      every: The interval of the period.

    Returns:
      Boolean of whether the event should happen.
    """
    if not every:
      return False
    return (phase_step + 1) % every == 0

  def _find_current_phase(self, global_step):
    """Determine the current phase based on the global step.

    This ensures continuing the correct phase after restoring checkoints.

    Args:
      global_step: The global number of steps performed across all phases.

    Returns:
      Tuple of phase object, epoch number, and phase steps within the epoch.
    """
    epoch_size = sum(phase.steps for phase in self._phases)
    epoch = int(global_step // epoch_size)
    steps_in = global_step % epoch_size
    for phase in self._phases:
      if steps_in < phase.steps:
        return phase, epoch, steps_in
      steps_in -= phase.steps

  def _define_step(self, name, score, summary):
    """Combine operations of a phase.

    Args:
      name: Name of the phase used for the score summary.
      score: Tensor holding the current scores.
      summary: Tensor holding summary string to write if not an empty string.

    Returns:
      Tuple of summary tensor, score, and new global step.
    """
    with tf.compat.v1.variable_scope('phase_{}'.format(name)):
      summary, score = tf.cond(self._do_run, lambda: (summary, score), lambda: (tf.constant(''), tf.constant(0.0)))
      with tf.control_dependencies([summary, score, self._next_step]):
        return (
            tf.identity(summary),
            tf.identity(score),
            tf.identity(self._next_step))

  def _create_session(self):
    """Create a TensorFlow session with sensible default parameters.

    Returns:
      Session.
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    try:
      return tf.compat.v1.Session('local', config=config)
    except tf.errors.NotFoundError:
      return tf.compat.v1.Session(config=config)

  def _initialize_variables(self, sess, savers, checkpoints):
    """Initialize or restore variables from a checkpoint if available.

    Args:
      sess: Session to initialize variables in.
      savers: List of savers to restore variables.
      checkpoints: List of checkpoint names for each saver; None for newest.
    """
    sess.run(tf.group(
        tf.compat.v1.local_variables_initializer(),
        tf.compat.v1.global_variables_initializer()))
    assert len(savers) == len(checkpoints)
    for saver, checkpoint in zip(savers, checkpoints):
      if not checkpoint:
        continue
      original_checkpoint = checkpoint
      logdir = os.path.expanduser(self._logdir)
      checkpoint = os.path.expanduser(checkpoint)
      if not os.path.isabs(checkpoint):
        checkpoint = os.path.join(logdir, checkpoint)
      checkpoint = os.path.abspath(checkpoint)
      if os.path.isdir(checkpoint):
        checkpoint = tf.train.latest_checkpoint(checkpoint)
      if checkpoint:
        saver.restore(sess, checkpoint)
      else:
        tf.compat.v1.logging.info(f'Could not find checkpoint in {original_checkpoint}.')

    # Make sure global step is set to start of the epoch
    global_step = sess.run(self._global_step)
    epoch_size = sum(phase.steps for phase in self._phases)
    epoch = int(global_step // epoch_size)
    steps_in = global_step % epoch_size
    if steps_in > 0:
      sess.run(self._global_step.assign((epoch + 1) * epoch_size))

  def _store_checkpoint(self, sess, saver, step):
    """Store a checkpoint if a log directory was provided to the constructor.

    The directory will be created if needed.

    Args:
      sess: Session containing variables to store.
      saver: Saver used for checkpointing.
      step: Step number of the checkpoint name.
    """
    if not self._logdir or not saver:
      return
    tf.compat.v1.gfile.MakeDirs(self._logdir)
    filename = os.path.join(self._logdir, 'model.ckpt')
    saver.save(sess, filename, step)
