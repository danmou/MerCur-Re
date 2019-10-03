# rewards.py: Reward functions
#
# (C) 2019, Daniel Mouritzen

from math import inf
from typing import Dict, Iterable, Tuple, Type

import gin
import gym
import numpy as np

Observations = Dict[str, np.ndarray]


class RewardFunction:
    def __init__(self, env: gym.Env) -> None:
        self._env = env

    def get_reward_range(self) -> Tuple[float, float]:
        raise NotImplementedError

    def get_reward(self, observations: Observations) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        pass


@gin.configurable(whitelist=['rewards'])
def combine_rewards(rewards: Iterable[Type[RewardFunction]]) -> Type[RewardFunction]:
    class CombinedRewards(RewardFunction):
        def __init__(self, env: gym.Env) -> None:
            super().__init__(env)
            self.rewards = [reward(env) for reward in rewards]

        def get_reward_range(self) -> Tuple[float, float]:
            lower, upper = 0.0, 0.0
            for a, b in [reward.get_reward_range() for reward in self.rewards]:
                lower += a
                upper += b
            return lower, upper

        def get_reward(self, observations: Observations) -> float:
            return sum(reward.get_reward(observations) for reward in self.rewards)

        def reset(self) -> None:
            for reward in self.rewards:
                reward.reset()

    return CombinedRewards


@gin.configurable(whitelist=['slack_reward', 'success_reward', 'distance_scaling'])
class DenseReward(RewardFunction):
    """
    Implements a dense reward function based on
    github.com/facebookresearch/habitat-api/blob/master/habitat_baselines/common/environments.py
    """
    def __init__(self,
                 env: gym.Env,
                 slack_reward: float = -0.01,
                 success_reward: float = 10.0,
                 distance_scaling: float = 1.0) -> None:
        # assert all(hasattr(env, name) for name in ('distance_to_target', 'episode_success', 'habitat_env')), \
        #        'Env must be a Habitat environment!'
        super().__init__(env)
        self._slack_reward = slack_reward
        self._success_reward = success_reward
        self._distance_scaling = distance_scaling
        self._previous_target_distance = None

    def get_reward_range(self) -> Tuple[float, float]:
        step_size = self._env.sim.config.FORWARD_STEP_SIZE
        return self._slack_reward - step_size, self._success_reward + step_size

    def get_reward(self, observations: Observations) -> float:
        if self._previous_target_distance is None:
            # New episode
            self._previous_target_distance = self._env.habitat_env.current_episode.info["geodesic_distance"]

        reward = self._slack_reward

        current_target_distance = self._env.distance_to_target()
        reward += self._distance_scaling * (self._previous_target_distance - current_target_distance)
        self._previous_target_distance = current_target_distance

        if self._env.episode_success():
            reward += self._success_reward

        return reward

    def reset(self) -> None:
        self._previous_target_distance = None


@gin.configurable(whitelist=['scaling'])
class OptimalPathLengthReward(RewardFunction):
    """Implements a sparse reward proportional to the optimal path length, as in arxiv.org/abs/1804.00168"""
    def __init__(self, env: gym.Env, scaling: float = 1.0) -> None:
        # assert all(hasattr(env, name) for name in ('episode_success', 'habitat_env')), \
        #        'Env must be a Habitat environment!'
        super().__init__(env)
        self._scaling = scaling

    def get_reward_range(self) -> Tuple[float, float]:
        return 0.0, inf

    def get_reward(self, observations: Observations) -> float:
        if self._env.episode_success():
            optimal_path_length: float = self._env.habitat_env.current_episode.info["geodesic_distance"]
            return self._scaling * optimal_path_length
        return 0.0

    def reset(self) -> None:
        return super().reset()


@gin.configurable(whitelist=['scaling'])
class CollisionPenalty(RewardFunction):
    """Adds a penalty for colliding with an obstacle."""
    def __init__(self, env: gym.Env, scaling: float = 1.0) -> None:
        # assert hasattr(env, 'sim'), 'Env must be a Habitat environment!'
        super().__init__(env)
        self._scaling = scaling

    def get_reward_range(self) -> Tuple[float, float]:
        return -self._scaling, 0.0

    def get_reward(self, observations: Observations) -> float:
        if self._env.sim.previous_step_collided:
            return -self._scaling
        return 0.0


@gin.configurable(whitelist=['threshold', 'scaling'])
class ObstacleDistancePenalty(RewardFunction):
    """
    Adds a penalty for getting closer than `threshold` meters from an obstacle, taking the agent's radius into account.
    The penaly increases linearly up to `scaling`.
    """
    def __init__(self, env: gym.Env, threshold: float = 1.0, scaling: float = 1.0) -> None:
        # assert hasattr(env, 'sim'), 'Env must be a Habitat environment!'
        super().__init__(env)
        self._threshold = threshold
        self._scaling = scaling

    def get_reward_range(self) -> Tuple[float, float]:
        return -self._scaling, 0.0

    def get_reward(self, observations: Observations) -> float:
        distance: float = self._env.sim.distance_to_closest_obstacle(
            self._env.sim.get_agent_state().position, self._threshold) - self._env.sim.config.AGENT_0.RADIUS
        if distance < self._threshold:
            return self._scaling * (distance / self._threshold - 1.0)
        return 0.0
