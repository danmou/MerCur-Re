# mpc_agent.py: Provides MPCAgent class
#
# (C) 2019, Daniel Mouritzen

from typing import Optional, Tuple, Type, Union

import gin
import gym.spaces
import tensorflow as tf

from project.model import Model
from project.planning import Planner

from .base import ModelBasedAgent, Observations


@gin.configurable(whitelist=['objective', 'planner', 'exploration_noise', 'visualize'])
class MPCAgent(ModelBasedAgent):
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
        super().__init__(action_space, model)
        self._objective_decoder = model.decoders[objective]
        self._planner = planner.from_rnn(model.rnn, self._objective_fn, self._action_space)
        self._exploration_noise = exploration_noise
        self._goal = tf.Variable([0.0, 0.0])
        self._visualize = tf.Variable(visualize)

    def _objective_fn(self, state: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        obj = self._objective_decoder(self._predictor.state_to_features(state), training=False)
        return tf.reduce_sum(obj, axis=1)

    @property
    def visualize(self) -> tf.Variable:
        return self._visualize

    @visualize.setter  # type: ignore[misc]  # mypy/issues/1362
    @tf.function
    def visualize(self, value: Union[tf.Tensor, bool]) -> None:
        self._visualize.assign(value)

    @tf.function
    def observe(self, observations: Observations, action: Optional[tf.Tensor]) -> None:
        super().observe(observations, action)
        self._goal.assign(observations['goal'])

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
