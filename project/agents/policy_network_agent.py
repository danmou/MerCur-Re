# policy_network_agent.py: Provides PolicyNetworkAgent class
#
# (C) 2020, Daniel Mouritzen

import gin
import gym.spaces
import tensorflow as tf

from project.model import Model

from .base import ModelBasedAgent


@gin.configurable(whitelist=['sample'])
class PolicyNetworkAgent(ModelBasedAgent):
    """At each time step, uses a policy network to choose the best action and executes it."""
    def __init__(self, action_space: gym.Space, model: Model, sample: bool = True) -> None:
        super().__init__(action_space, model)
        assert model.action_network is not None
        self._policy = model.action_network
        self._sample = sample

    @tf.function
    def act(self) -> tf.Tensor:
        action_dist = self._policy(self._predictor.state_to_features(self.state)[tf.newaxis, :], training=False)
        if self._sample:
            action = action_dist.sample()[0, 0, :]
        else:
            action = action_dist.mode()[0, 0, :]
        return action
