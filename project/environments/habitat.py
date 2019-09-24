# habitat.py: provides Habitat class, which interfaces with the habitat environment
#
# (C) 2019, Daniel Mouritzen

from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, cast

import gin
import gym.spaces
import habitat
import wandb
from habitat.core.simulator import Observations
from loguru import logger

from project.util import get_config_dir

ObsTuple = Tuple[Observations, Any, bool, dict]


@gin.configurable(whitelist=['task', 'dataset', 'gpu_id', 'reward_measure', 'image_key'])
class Habitat:
    class __Habitat(habitat.RLEnv):
        observation_space: gym.spaces.Dict
        action_space: gym.Space

        def __init__(self, config: habitat.Config, reward_measure: str = 'spl', image_key: str = 'rgb') -> None:
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

    __instance = None

    def __init__(self,
                 max_steps: Optional[int] = None,
                 task: str = 'habitat_test',
                 dataset: str = 'pointnav',
                 gpu_id: int = 0,
                 reward_measure: str = 'spl',
                 image_key: str = 'rgb') -> None:
        self.config = get_config(max_steps, task, dataset, gpu_id)
        self.reward_measure = reward_measure
        self.image_key = image_key
        self.__create_instance()

    def __create_instance(self) -> None:
        assert self.config is not None
        assert self.reward_measure is not None
        assert self.image_key is not None
        if not Habitat.__instance:
            wandb.config.update({'habitat_config': self.config})
        else:
            logger.debug("Deleting current instance of Habitat before reinit.")
            self.close()
        logger.debug("Creating Habitat instance.")
        Habitat.__instance = Habitat.__Habitat(self.config, self.reward_measure, self.image_key)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__instance, name)

    def reset(self) -> Observations:
        self.__create_instance()
        assert self.__instance is not None
        obs = self.__instance.reset()
        return obs

    def close(self) -> None:
        assert self.__instance is not None
        self.__instance.close()


def get_config(max_steps: Optional[Union[int, float]],
               task: str,
               dataset: str,
               gpu_id: int) -> habitat.Config:
    opts = ['SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID', gpu_id]
    if max_steps:
        opts += ['ENVIRONMENT.MAX_EPISODE_STEPS', int(max_steps)]
    if not task.endswith('.yaml'):
        task = f'{get_config_dir()}/habitat/tasks/{task}.yaml'
    if not dataset.endswith('.yaml'):
        dataset = f'{get_config_dir()}/habitat/datasets/{dataset}.yaml'
    return habitat.get_config([task, dataset], opts)
