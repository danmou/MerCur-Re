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

from project.util import capture_output, get_config_dir

ObsTuple = Tuple[Observations, Any, bool, dict]


@gin.configurable(whitelist=['task', 'dataset', 'gpu_id', 'image_key', 'slack_reward', 'success_reward'])
class Habitat:
    """
    Singleton wrapper for Habitat.

    Reward function is based on
    github.com/facebookresearch/habitat-api/blob/master/habitat_baselines/common/environments.py
    """
    class __Habitat(habitat.RLEnv):
        observation_space: gym.spaces.Dict
        action_space: gym.Space

        def __init__(self, config: habitat.Config, image_key: str, slack_reward: float, success_reward: float) -> None:
            self._image_key = image_key
            self._slack_reward = slack_reward
            self._success_reward = success_reward
            self._success_distance = config.TASK.SUCCESS_DISTANCE
            self._previous_target_distance = None
            self._previous_action = None

            super().__init__(config)
            self.observation_space = gym.spaces.Dict(self._update_key(self.observation_space.spaces))

        def get_reward_range(self) -> List[float]:
            return [
                self._slack_reward - 1.0,
                self._success_reward + 1.0,
            ]

        def get_reward(self, observations: Observations) -> float:
            reward = self._slack_reward

            current_target_distance = self._distance_target()
            reward += self._previous_target_distance - current_target_distance
            self._previous_target_distance = current_target_distance

            if self._episode_success():
                reward += self._success_reward

            return reward

        def _distance_target(self) -> float:
            current_position = self._env.sim.get_agent_state().position.tolist()
            target_position = self._env.current_episode.goals[0].position
            distance = self._env.sim.geodesic_distance(
                current_position, target_position
            )
            return distance

        def get_done(self, observations: Observations) -> bool:
            return self._env.episode_over or self._episode_success()

        def _episode_success(self) -> bool:
            return self._previous_action == habitat.SimulatorActions.STOP and \
                   self._distance_target() < self._success_distance

        def get_info(self, observations: Observations) -> Dict[Any, Any]:
            return self.habitat_env.get_metrics()

        def step(self, action: int) -> ObsTuple:
            self._previous_action = action
            obs, reward, done, info = super().step(action)
            obs = self._update_key(obs)
            return obs, reward, done, info

        def reset(self) -> Observations:
            self._previous_action = None
            observations = super().reset()
            self._previous_target_distance = self.habitat_env.current_episode.info["geodesic_distance"]
            return self._update_key(observations)

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
                 image_key: str = 'rgb',
                 slack_reward: float = -0.01,
                 success_reward: float = 10.0) -> None:
        self.config = get_config(max_steps, task, dataset, gpu_id)
        self.image_key = image_key
        self.slack_reward = slack_reward
        self.success_reward = success_reward
        self.__create_instance()

    def __create_instance(self) -> None:
        if not Habitat.__instance:
            wandb.config.update({'habitat_config': self.config})
        else:
            logger.debug("Deleting current instance of Habitat before reinit.")
            self.close()
        logger.debug("Creating Habitat instance.")
        with capture_output('habitat_sim'):
            Habitat.__instance = Habitat.__Habitat(self.config, self.image_key, self.slack_reward, self.success_reward)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__instance, name)

    def reset(self) -> Observations:
        self.__create_instance()
        assert self.__instance is not None
        obs = self.__instance.reset()
        return obs

    def close(self) -> None:
        assert self.__instance is not None
        with capture_output('habitat_sim'):
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
