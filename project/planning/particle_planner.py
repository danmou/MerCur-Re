# particle_planner.py: Monte Carlo particle-filter-style planner
#
# (C) 2020, Daniel Mouritzen

from __future__ import annotations

from typing import Optional, Tuple, Union, cast

import gin
import gym.spaces
import tensorflow as tf

from project.model import Model
from project.networks.predictors import OpenLoopPredictor
from project.networks.predictors.rssm import FullState
from project.util.tf import combine_dims, split_dim
from project.util.tf.discounting import lambda_return

from .base import DecoderFunction, Planner


@tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ASSERT_STATEMENTS)
def particle_planner(initial_state: Tuple[Union[tf.Tensor, tf.Variable], ...],
                     predictor: OpenLoopPredictor,
                     reward_decoder: DecoderFunction,
                     done_decoder: DecoderFunction,
                     value_network: DecoderFunction,
                     action_network: DecoderFunction,
                     action_space: gym.spaces.box,
                     state_samples: int = 10,
                     action_samples: int = 10,
                     horizon: int = 10,
                     lambda_: float = 0.95,
                     ) -> tf.Tensor:
    assert initial_state[0].shape[0] == 1, 'Initial state can only have a single batch element.'

    # Draw `state_samples` samples from the initial state distribution
    initial_state = tf.nest.map_structure(lambda x: tf.tile(x, [state_samples, 1]), initial_state)
    initial_state_object = FullState(*initial_state)
    initial_state_object.state.sample = initial_state_object.state.to_dist().sample()
    initial_state_features = initial_state_object.to_features()
    initial_state = tuple(initial_state_object)

    # Draw `action_samples` samples for each state sample
    initial_actions = action_network(initial_state_features[tf.newaxis, :], training=False).sample(action_samples)
    initial_actions = combine_dims(initial_actions, [0, 1, 2])  # shape: [action_samples * state_samples, action_shape]

    # Try all sampled actions on all sampled states
    initial_state_repeated = tf.nest.map_structure(lambda x: combine_dims(tf.tile(x[tf.newaxis, :],
                                                                                  [state_samples * action_samples, 1, 1]),
                                                                          [0, 1]),
                                                   initial_state)
    initial_actions_repeated = combine_dims(tf.tile(initial_actions[:, tf.newaxis],
                                                    [1, state_samples, 1]),
                                            [0, 1])
    # shape: [state_samples**2 * action_samples, ...]
    _, next_states = predictor(initial_actions_repeated, initial_state_repeated)

    def step_fn(prev: Tuple[tf.Tensor, ...], index: tf.Tensor) -> Tuple[tf.Tensor, ...]:
        features = predictor.state_to_features(prev)
        action = action_network(features[tf.newaxis, :], training=False).sample()[0, :]
        _, state = predictor(action, prev)
        return cast(Tuple[tf.Tensor, ...], state)

    features = predictor.state_to_features(next_states)[tf.newaxis, :]
    if horizon:
        states = tf.scan(step_fn, tf.range(horizon), initializer=next_states, back_prop=False)
        # shape: [horizon, state_samples**2 * action_samples, ...]
        features = tf.concat([features, predictor.state_to_features(states)], 0)

    values = value_network(features, training=False)
    rewards = reward_decoder(features, training=False)
    done_probs = done_decoder(features, training=False)

    if values.shape[0] > 1:
        rewards = rewards[:-1, :]
        final_value = values[-1]
        values = values[:-1, :]
        discounts = 1 - done_probs[:-1, :]
        returns = lambda_return(rewards, values, discounts, lambda_, final_value, axis=0)[0]
    else:
        returns = values[0]
    action_returns = tf.reduce_mean(split_dim(returns, 0, [state_samples * action_samples, state_samples]), axis=1)

    return initial_actions[tf.argmax(action_returns)]


@gin.configurable(whitelist=['state_samples', 'action_samples', 'horizon', 'lambda_'])
class ParticlePlanner(Planner):
    def __init__(self,
                 predictor: OpenLoopPredictor,
                 reward_decoder: DecoderFunction,
                 done_decoder: DecoderFunction,
                 value_network: Optional[DecoderFunction],
                 action_network: Optional[DecoderFunction],
                 action_space: gym.spaces.box,
                 state_samples: int = 10,
                 action_samples: int = 10,
                 horizon: int = 10,
                 lambda_: float = 0.95,
                 ) -> None:
        super().__init__(predictor, reward_decoder, action_space)
        self.done_decoder = done_decoder
        assert value_network is not None
        assert action_network is not None
        self.value_network = value_network
        self.action_network = action_network
        self.state_samples = state_samples
        self.action_samples = action_samples
        self.horizon = horizon
        self.lambda_ = lambda_

    @classmethod
    def from_model(cls, model: Model, action_space: gym.spaces.box) -> ParticlePlanner:
        return cls(predictor=model.rnn.predictor.open_loop_predictor,
                   reward_decoder=model.decoders['reward'],
                   done_decoder=model.decoders['done'],
                   value_network=model.value_network,
                   action_network=model.action_network,
                   action_space=action_space)

    @tf.function
    def get_action(self,
                   initial_state: Tuple[Union[tf.Tensor, tf.Variable], ...],
                   visualization_goal: Optional[tf.Tensor] = None,
                   ) -> tf.Tensor:
        return particle_planner(initial_state,
                                self._predictor,
                                self._objective_decoder,
                                self.done_decoder,
                                self.value_network,
                                self.action_network,
                                self.action_space,
                                self.state_samples,
                                self.action_samples,
                                self.horizon,
                                self.lambda_)
