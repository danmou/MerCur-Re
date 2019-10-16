# habitat.py: provides Habitat class, which interfaces with the habitat environment
#
# (C) 2019, Daniel Mouritzen

from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

import gin
import gym.spaces
import habitat
import wandb
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.nav import NavigationEpisode
from loguru import logger

from project.util import capture_output, get_config_dir

from .rewards import RewardFunction

ObsTuple = Tuple[Observations, Any, bool, dict]


@habitat.registry.register_measure
class Success(habitat.Measure):
    def __init__(self, *args: Any, sim: habitat.Simulator, config: habitat.Config, **kwargs: Any) -> None:
        self._sim = sim
        self._config = config
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "success"

    def reset_metric(self, *args: Any, episode, **kwargs: Any) -> None:
        self._metric = None

    def update_metric(self, *args: Any, episode: NavigationEpisode, task: habitat.EmbodiedTask, **kwargs: Any) -> None:
        current_position = self._sim.get_agent_state().position.tolist()
        distance_to_target = self._sim.geodesic_distance(current_position, episode.goals[0].position)

        self._metric = int(getattr(task, "is_stop_called", False) and
                           distance_to_target < self._config.SUCCESS_DISTANCE)


@gin.configurable(whitelist=['task', 'dataset', 'gpu_id', 'image_key', 'goal_key', 'reward_function'])
class Habitat:
    """Singleton wrapper for Habitat."""
    class __Habitat(habitat.RLEnv):
        def __init__(self,
                     config: habitat.Config,
                     image_key: str,
                     goal_key: str,
                     reward_function: Type[RewardFunction]) -> None:
            self._image_key = image_key
            self._goal_key = goal_key
            self._reward_function = reward_function(self)
            self._previous_action: Optional[int] = None
            self.success_distance = config.TASK.SUCCESS_DISTANCE
            self.stop_action: int = HabitatSimActions.STOP

            super().__init__(config)
            self.observation_space: gym.spaces.Dict = gym.spaces.Dict(self._update_keys(self._env.observation_space.spaces))
            self.action_space = self._env.action_space

        def get_reward_range(self) -> Tuple[float, float]:
            return self._reward_function.get_reward_range()

        def get_reward(self, observations: Observations) -> float:
            return self._reward_function.get_reward(observations)

        def get_done(self, observations: Observations) -> bool:
            return self._env.episode_over or self.episode_success()

        def episode_success(self) -> bool:
            return self._previous_action == self.stop_action and self.distance_to_target() < self.success_distance

        def distance_to_target(self) -> float:
            current_position = self._env.sim.get_agent_state().position.tolist()
            target_position = self._env.current_episode.goals[0].position  # type: ignore
            distance = self._env.sim.geodesic_distance(
                current_position, target_position
            )
            return distance

        def get_info(self, observations: Observations) -> Dict[Any, Any]:
            metrics = self.habitat_env.get_metrics()
            if 'distance_to_goal' in metrics:
                dist_dict = metrics.pop('distance_to_goal')
                metrics['path_length'] = dist_dict['agent_path_length']
                metrics['optimal_path_length'] = dist_dict['start_distance_to_target']
                metrics['remaining_distance'] = dist_dict['distance_to_target']
            if 'collisions' in metrics:
                metrics['collisions'] = metrics['collisions']['count']
            return metrics

        def step(self, action: int) -> ObsTuple:
            self._previous_action = action
            obs, reward, done, info = super().step(action)
            obs = self._update_keys(obs)
            info['taken_action'] = action
            return obs, reward, done, info

        def reset(self) -> Observations:
            self._previous_action = None
            self._reward_function.reset()
            return self._update_keys(super().reset())

        _ObsOrDict = TypeVar('_ObsOrDict', Observations, Dict['str', Any])

        def _update_keys(self, obs: '_ObsOrDict') -> '_ObsOrDict':
            obs_dict = obs.copy()  # copy converts to dict
            obs_dict['image'] = obs_dict.pop(self._image_key)
            obs_dict['goal'] = obs_dict.pop(self._goal_key)
            # Convert back to original type (Observations.__init__ tries to process the input dict)
            obs = type(obs)({})
            obs.update(obs_dict)
            return obs

        def close(self) -> None:
            with capture_output('habitat_sim'):
                self._env.close()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._env, name)

    __instance = None

    def __init__(self,
                 max_steps: Optional[int] = None,
                 task: str = 'habitat_test',
                 dataset: str = 'pointnav',
                 gpu_id: int = 0,
                 image_key: str = 'rgb',
                 goal_key: str = 'pointgoal_with_gps_compass',
                 reward_function: Type[RewardFunction] = RewardFunction) -> None:
        self.config = get_config(max_steps, task, dataset, gpu_id)
        self.image_key = image_key
        self.goal_key = goal_key
        self.reward_function = reward_function
        self.__create_instance()

    def __create_instance(self) -> None:
        if not Habitat.__instance:
            wandb.config.update({'habitat_config': self.config})
        else:
            logger.debug("Deleting current instance of Habitat before reinit.")
            self.close()
        logger.debug("Creating Habitat instance.")
        with capture_output('habitat_sim'):
            Habitat.__instance = Habitat.__Habitat(self.config, self.image_key, self.goal_key, self.reward_function)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__instance, name)

    def reset(self) -> Observations:
        self.__create_instance()
        assert self.__instance is not None
        obs = self.__instance.reset()
        return obs


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
    config = habitat.get_config([task, dataset], opts)
    config.defrost()
    config.TASK.SUCCESS = habitat.Config()
    config.TASK.SUCCESS.TYPE = "Success"
    config.TASK.SUCCESS.SUCCESS_DISTANCE = config.TASK.SUCCESS_DISTANCE
    config.TASK.MEASUREMENTS.append("SUCCESS")
    config.freeze()
    return config
