# hierarchical_cross_entropy_method.py: Version of CrossEntropyMethod planner working on multiple time scales
#
# (C) 2019, Daniel Mouritzen

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, Union

import gin
import gym.spaces
import numpy as np
import tensorflow as tf

from project.networks import DenseVAE
from project.networks.predictors import OpenLoopPredictor
from project.networks.rnns import RNN

from .base import Planner
from .cross_entropy_method import CrossEntropyMethod


@gin.configurable(whitelist=['horizon', 'amount', 'top_k', 'iterations'])
class HierarchicalCrossEntropyMethod(Planner):
    def __init__(self,
                 predictor: OpenLoopPredictor,
                 objective_fn: Callable[[Tuple[tf.Tensor, ...]], tf.Tensor],
                 action_space: gym.spaces.box,
                 horizon: int = 12,
                 amount: int = 1000,
                 top_k: int = 100,
                 iterations: int = 10,
                 ) -> None:
        super().__init__(predictor, objective_fn, action_space)
        self.horizon = horizon
        self.amount = amount
        self.top_k = top_k
        self.iterations = iterations
        self._planners: List[CrossEntropyMethod] = []
        self._action_vaes: List[Optional[DenseVAE]] = []

    @classmethod
    def from_rnn(cls, rnn: RNN, *args: Any, **kwargs: Any) -> HierarchicalCrossEntropyMethod:
        self = cls(rnn.predictor.open_loop_predictor, *args, **kwargs)
        self._planners = [CrossEntropyMethod(self._predictor,
                                             self._objective_fn,
                                             self.action_space,
                                             self.horizon,
                                             self.amount,
                                             self.top_k,
                                             self.iterations)]
        self._planners += [CrossEntropyMethod(predictor,
                                              self._objective_fn,
                                              gym.spaces.Box(-2.0, 2.0, shape=(action_embedding,), dtype=np.float32),
                                              self.horizon,
                                              self.amount,
                                              self.top_k,
                                              self.iterations)
                           for predictor, action_embedding in zip(rnn.predictors, rnn.action_embedding_sizes[1:])]
        self._action_vaes = [None] + rnn.action_vaes
        return self

    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ASSERT_STATEMENTS)
    def __call__(self,
                 initial_state: Tuple[Union[tf.Tensor, tf.Variable], ...],
                 initial_mean: Optional[tf.Tensor] = None,
                 initial_std_dev: Optional[tf.Tensor] = None,
                 visualization_goal: Optional[tf.Tensor] = None,
                 ) -> Tuple[tf.Tensor, tf.Tensor]:
        assert len(self._planners) > 0, 'HierarchicalCrossEntropyMethod must be initialized using the `from_rnn` method.'
        mean = initial_mean
        std_dev = initial_std_dev
        for planner, vae in zip(reversed(self._planners), reversed(self._action_vaes)):
            if mean is not None:
                mean = mean[:planner.horizon]
            if std_dev is not None:
                std_dev = std_dev[:planner.horizon]
            vis_goal = visualization_goal if vae is None else None
            mean, std_dev = planner(initial_state, mean, std_dev, vis_goal)  # shape: [horizon, action_space]
            if vae is not None:
                mean = vae.decoder(mean[tf.newaxis, :], training=False)  # shape: [1, horizon, factor, new_action_space]
                factor = mean.shape[2]
                mean = tf.reshape(mean, [-1] + mean.shape[3:].as_list())  # shape: [horizon * factor, new_action_space]
                std_dev = tf.reduce_mean(std_dev, axis=-1)  # shape: [horizon]
                std_dev = tf.reshape(tf.tile(std_dev[:, tf.newaxis, tf.newaxis],
                                             [1, factor] + mean.shape[1:]),
                                     [mean.shape[0], -1])  # shape: [horizon * factor, new_action_space]
                std_dev += 0.1  # Add some variance to account for the decoder uncertainty
                # TODO: Think about a smarter way to approximate the standard deviation
        assert mean is not None and std_dev is not None, (mean, std_dev)
        return mean, std_dev
