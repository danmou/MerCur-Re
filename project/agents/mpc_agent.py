# mpc_agent.py: Provides MPCAgent class
#
# (C) 2019, Daniel Mouritzen

from typing import Optional, Tuple, Type, Union

import gin
import gym.spaces
import tensorflow as tf

from project.model import Model
from project.planning import Planner

from .base import Agent, Observations


@gin.configurable(whitelist=['objective', 'planner'])
class MPCAgent(Agent):
    """
    At each time step, uses a predictive model together with a planning algorithm to choose the best sequence of
    actions and executes the first one.
    """
    def __init__(self,
                 action_space: gym.Space,
                 model: Model,
                 objective: str = 'reward',
                 planner: Type[Planner] = gin.REQUIRED,
                 exploration_noise: float = 0.0,
                 visualize: bool = False,
                 ) -> None:
        assert isinstance(action_space, gym.spaces.Box), f'Unsupported action space {action_space}'
        super().__init__(action_space)
        self._predictor = model.rnn.predictor
        self._encoder = model.encoder
        self._objective_decoder = model.decoders[objective]
        self._state = tuple(tf.Variable(x) for x in self._predictor.zero_state(1, tf.float32))
        self._planner = planner.from_rnn(model.rnn, self._objective_fn, self._action_space)
        self._exploration_noise = exploration_noise
        self._goal = tf.Variable([0.0, 0.0])
        self._visualize = tf.Variable(visualize)

    def _objective_fn(self, state: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        obj = self._objective_decoder(self._predictor.state_to_features(state))
        return tf.reduce_sum(obj, axis=1)

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

    @property
    def visualize(self) -> tf.Variable:
        return self._visualize

    @visualize.setter  # type: ignore[misc]  # mypy/issues/1362
    @tf.function
    def visualize(self, value: Union[tf.Tensor, bool]) -> None:
        self._visualize.assign(value)

    @tf.function
    def reset(self) -> None:
        self.state = tuple(tf.zeros_like(s) for s in self.state)  # type: ignore[misc]  # mypy/issues/1362

    @tf.function
    def observe(self, observations: Observations, action: Optional[tf.Tensor]) -> None:
        if action is None:
            action = tf.zeros_like(self._action_space.low)
        self._goal.assign(observations['goal'])
        observations = tf.nest.map_structure(lambda t: t[tf.newaxis, tf.newaxis, :], observations)
        embedded = self._encoder(observations)[0]
        action = action[tf.newaxis, :]
        use_obs = tf.constant([[True]])
        state = tuple(v.value() for v in self._state)  # This is needed to prevent weird errors with dead weakrefs
        _, self.state = self._predictor((embedded, action, use_obs), state)  # type: ignore[misc]  # mypy/issues/1362

    @tf.function
    def act(self) -> tf.Tensor:
        if self.visualize:
            plan, _ = self._planner(self.state, visualization_goal=self._goal)
        else:
            plan, _ = self._planner(self.state)
        self.visualize = False  # type: ignore[misc]  # mypy/issues/1362
        action = plan[0, :]
        if self._exploration_noise:
            action += tf.random.normal(action.shape, stddev=self._exploration_noise)
        return action
