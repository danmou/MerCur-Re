# habitat.py: provides Habitat class, which interfaces with the habitat environment
#
# (C) 2019, Daniel Mouritzen

from typing import Any, Dict, List, Optional, Tuple, TypeVar, cast

import gym.spaces
import habitat
from habitat.core.simulator import Observations

ObsTuple = Tuple[Observations, Any, bool, dict]


class Habitat(habitat.RLEnv):
    observation_space: gym.spaces.Dict
    action_space: gym.Space

    def __init__(self, config_path: str, dataset: Optional[habitat.Dataset] = None) -> None:
        config = habitat.get_config(config_path)
        super().__init__(config, dataset)
        self._reward_measure = 'spl'
        self._image_key = 'rgb'
        # TODO: get these from config
        self.observation_space = gym.spaces.Dict(self._update_key(self.observation_space.spaces))

    def get_reward_range(self) -> List[float]:
        return [0.0, 1.0]

    def get_reward(self, observations: Observations) -> float:
        return cast(float, self.habitat_env.get_metrics()[self._reward_measure])

    def get_done(self, observations: Observations) -> bool:
        return self.habitat_env.episode_over

    def get_info(self, observations: Observations) -> Dict[Any, Any]:
        return self.habitat_env.get_metrics()

    def step(self, action: int) -> ObsTuple:
        obs, reward, done, info = super().step(action)
        obs = self._update_key(obs)
        return obs, reward, done, info

    def reset(self) -> Observations:
        return self._update_key(super().reset())

    _ObsOrDict = TypeVar('_ObsOrDict', Observations, Dict['str', Any])

    def _update_key(self, obs: '_ObsOrDict') -> '_ObsOrDict':
        obs_dict = obs.copy()  # copy converts to dict
        obs_dict['image'] = obs_dict.pop(self._image_key)
        # Convert back to original type (Observations.__init__ tries to process the input dict)
        obs = type(obs)({})
        obs.update(obs_dict)
        return obs