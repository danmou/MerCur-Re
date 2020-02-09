# base.py: Base classes for agents
#
# (C) 2019, Daniel Mouritzen

import abc
from typing import Dict, Optional, Tuple, Union

import gym
import tensorflow as tf

from project.model import Model

Observations = Union[tf.Tensor, Dict[str, tf.Tensor]]


class Agent(abc.ABC):
    def __init__(self, action_space: gym.Space) -> None:
        self.action_space = action_space

    def reset(self) -> None:
        """Reset agent's state"""
        pass

    @abc.abstractmethod
    def observe(self, observations: Observations, action: Optional[tf.Tensor]) -> None:
        """Update agent's state based on observations"""
        raise NotImplementedError

    @abc.abstractmethod
    def act(self) -> tf.Tensor:
        """Decide the next action"""
        raise NotImplementedError


class BlindAgent(Agent):
    def __init__(self, action_space: gym.Space) -> None:
        super().__init__(action_space)

    def observe(self, observations: Observations, action: Optional[tf.Tensor]) -> None:
        """Ignore observations; child classes should not override this method"""
        pass

    @abc.abstractmethod
    def act(self) -> tf.Tensor:
        """Decide the next action"""
        raise NotImplementedError


class ModelBasedAgent(Agent):
    """Base class for agents that use a model and need to keep its state up to date."""
    def __init__(self, action_space: gym.Space, model: Model) -> None:
        super().__init__(action_space)
        self._predictor = model.rnn.predictor
        self._encoder = model.encoder
        self._state = tuple(tf.Variable(x) for x in self._predictor.zero_state(1, tf.float32))

    @property
    def state(self) -> Tuple[tf.Variable, ...]:
        return self._state

    @state.setter  # type: ignore[misc]  # mypy/issues/1362
    @tf.function
    def state(self, value: Tuple[tf.Tensor, ...]) -> None:
        tf.nest.assert_same_structure(value, self._state)
        assert all(a.shape == b.shape for a, b in zip(value, self._state))
        for s, v in zip(self._state, value):
            s.assign(v)

    @tf.function
    def reset(self) -> None:
        self.state = tuple(tf.zeros_like(s) for s in self.state)  # type: ignore[misc]  # mypy/issues/1362

    @tf.function
    def observe(self, observations: Observations, action: Optional[tf.Tensor]) -> None:
        """Update model state based on observations."""
        if action is None:
            action = tf.zeros_like(self.action_space.low)
        observations = tf.nest.map_structure(lambda t: t[tf.newaxis, tf.newaxis, :], observations)
        embedded = self._encoder(observations, training=False)[0]
        action = action[tf.newaxis, :]
        use_obs = tf.constant([[True]])
        state = tuple(v.value() for v in self._state)  # This is needed to prevent weird errors with dead weakrefs
        _, self.state = self._predictor((embedded, action, use_obs), state, training=False)  # type: ignore[misc]  # mypy/issues/1362

    @abc.abstractmethod
    def act(self) -> tf.Tensor:
        """Decide the next action"""
        raise NotImplementedError
