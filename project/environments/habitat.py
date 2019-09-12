# habitat.py: provides Habitat class, which interfaces with the habitat environment
#
# (C) 2019, Daniel Mouritzen

from typing import Any, Dict, List, Optional, Tuple, TypeVar, cast

import gin
import gym.spaces
import habitat
from habitat.core.simulator import Observations

ObsTuple = Tuple[Observations, Any, bool, dict]


@gin.configurable(whitelist=['task_config', 'dataset_config', 'reward_measure', 'image_key'])
class Habitat(habitat.RLEnv):
    observation_space: gym.spaces.Dict
    action_space: gym.Space

    def __init__(self,
                 max_steps: Optional[int] = None,
                 task_config: str = 'habitat_test',
                 dataset_config: str = 'pointnav',
                 reward_measure: str = 'spl',
                 image_key: str = 'rgb') -> None:
        opts = []
        if max_steps:
            opts = ['ENVIRONMENT.MAX_EPISODE_STEPS', max_steps]
        if not task_config.endswith('.yaml'):
            task_config = f'configs/habitat/tasks/{task_config}.yaml'
        if not dataset_config.endswith('.yaml'):
            dataset_config = f'configs/habitat/datasets/{dataset_config}.yaml'
        config = habitat.get_config([task_config, dataset_config], opts)
        super().__init__(config)
        self._reward_measure = reward_measure
        self._image_key = image_key
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
