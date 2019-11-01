# habitat.py: provides Habitat class, which interfaces with the habitat environment
#
# (C) 2019, Daniel Mouritzen

import random
from mock import MagicMock
from pathlib import Path
from shutil import copyfile
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import gin
import gym.spaces
import habitat
import numpy as np
import skimage.transform
import wandb
from habitat.core.simulator import Observations
from habitat.core.vector_env import VectorEnv
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from loguru import logger

from project.util import capture_output, get_config_dir, measure_time

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

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        self._metric = None

    def update_metric(self,  # type: ignore
                      *args: Any,
                      episode: NavigationEpisode,
                      task: habitat.EmbodiedTask,
                      **kwargs: Any) -> None:
        current_position = self._sim.get_agent_state().position.tolist()
        distance_to_target = self._sim.geodesic_distance(current_position, episode.goals[0].position)

        self._metric = int(getattr(task, "is_stop_called", False) and
                           distance_to_target < self._config.SUCCESS_DISTANCE)


class Habitat(habitat.RLEnv):
    def __init__(self,
                 config: habitat.Config,
                 image_key: str,
                 goal_key: str,
                 reward_function: RewardFunction,
                 capture_video: bool = False,
                 seed: Optional[int] = None,
                 **_: Any) -> None:
        self._image_key = image_key
        self._goal_key = goal_key
        self._reward_function = reward_function
        self._reward_function.set_env(self)
        self._previous_action: Optional[int] = None
        self.success_distance = config.TASK.SUCCESS_DISTANCE
        self.stop_action: int = HabitatSimActions.STOP
        self._capture_video = capture_video
        self._rgb_frames: List[np.ndarray] = []
        if seed is not None:
            # This is needed for reproducible episode shuffling
            random.seed(seed)
            np.random.seed(seed)
        with capture_output('habitat_sim'):
            super().__init__(config)
        if seed is not None:
            self.seed(seed)
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

    def step(self, action: int) -> ObsTuple:  # type: ignore
        self._previous_action = action
        obs, reward, done, info = super().step(action)
        obs = self._update_keys(obs)
        if self._capture_video:
            # upscale image to make the resulting video more viewable
            new_obs = {'rgb': np.repeat(np.repeat(obs['image'], 4, axis=0), 4, axis=1)}
            self._rgb_frames.append(observations_to_image(new_obs, self.habitat_env.get_metrics()))
        info['taken_action'] = action
        return obs, reward, done, info

    def reset(self) -> Observations:
        self._previous_action = None
        self._reward_function.reset()
        self._rgb_frames = []
        with capture_output('habitat_sim'):
            obs = super().reset()
        return self._update_keys(obs)

    _ObsOrDict = TypeVar('_ObsOrDict', Observations, Dict['str', Any])

    def _update_keys(self, obs: '_ObsOrDict') -> '_ObsOrDict':
        obs_dict = {'image': obs[self._image_key],
                    'goal': obs[self._goal_key]}
        # Convert back to original type (Observations.__init__ tries to process the input dict)
        obs = type(obs)({})
        obs.update(obs_dict)
        return obs

    def close(self) -> None:
        with capture_output('habitat_sim'):
            self._env.close()

    @measure_time()
    def save_video(self, file: Union[str, Path], fps: int = 20) -> None:
        assert self._capture_video, 'Not capturing video; nothing to save.'
        if len(self._rgb_frames) == 0:
            return
        file = Path(file)
        with capture_output('save_video'):
            images_to_video(self._rgb_frames, str(file.parent), file.name, fps=fps, quality=5)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class DummyHabitat(gym.Env):
    def __init__(self,
                 config: habitat.Config,
                 image_key: str,
                 goal_key: str,
                 reward_function: RewardFunction,
                 **_: Any) -> None:
        self._image_key = image_key
        self._goal_key = goal_key
        self._reward_function = reward_function
        self._reward_function.set_env(self)  # type: ignore
        self.success_distance = config.TASK.SUCCESS_DISTANCE
        self.stop_action: int = HabitatSimActions.STOP

        self.sim = MagicMock()
        self.sim.config = config.SIMULATOR
        self.sim.previous_step_collided = False
        self.sim.distance_to_closest_obstacle = MagicMock(return_value=0.5)
        self.habitat_env = MagicMock()
        self.habitat_env.current_episode.info = {"geodesic_distance": 5.0}

        self.observation_space = gym.spaces.Dict({'image': gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                                                  'goal': gym.spaces.Box(low=-3e38, high=3e38, shape=(2,), dtype=np.float32)})
        self.action_space = habitat.core.spaces.ActionSpace({'MOVE_FORWARD': habitat.core.spaces.EmptySpace(),
                                                             'STOP': habitat.core.spaces.EmptySpace(),
                                                             'TURN_LEFT': habitat.core.spaces.EmptySpace(),
                                                             'TURN_RIGHT': habitat.core.spaces.EmptySpace()})
        self.reward_range = self._reward_function.get_reward_range()

    def distance_to_target(self) -> float:
        return random.uniform(0, 5)

    def episode_success(self) -> bool:
        return random.random() < 0.05

    def step(self, action: int) -> ObsTuple:  # type: ignore
        obs = self.observation_space.sample()
        reward = self._reward_function.get_reward(obs)
        done = random.random() < 0.05
        info = {'success': 0,
                'spl': 0,
                'path_length': 0,
                'optimal_path_length': 0,
                'remaining_distance': 0,
                'collisions': 0,
                'taken_action': action}
        return obs, reward, done, info

    def reset(self) -> Observations:
        self._reward_function.reset()
        return self.observation_space.sample()


class VectorHabitat(VectorEnv):
    def __init__(self,
                 env_ctor: Callable[..., gym.Env],
                 params: Dict[str, Any],
                 auto_reset_done: bool = False) -> None:
        super().__init__(env_ctor, [tuple(params.items())], auto_reset_done=auto_reset_done, multiprocessing_start_method='fork')
        self.observation_space = self.observation_spaces[0]
        self.action_space = self.action_spaces[0]

    def __getattr__(self, name: str) -> Any:
        return self.call_at(0, '__getattr__', {'name': name})

    def save_video(self, file: Union[str, Path], **kwargs) -> None:
        kwargs['file'] = file
        self.call_at(0, 'save_video', kwargs)

    def seed(self, seed: int) -> None:
        self.call_at(0, 'seed', {'seed': seed})

    def step(self, action: Any) -> ObsTuple:  # type: ignore
        obs, reward, done, info = super().step(data=[{'action': action}])[0]
        return obs, reward, done, info

    def reset(self) -> Observations:  # type: ignore
        return super().reset()[0]


@gin.configurable('Habitat', whitelist=['task', 'dataset', 'gpu_id', 'image_key', 'goal_key', 'reward_function'])
def get_config(max_steps: Optional[Union[int, float]] = None,
               task: str = 'pointnav',
               dataset: str = 'habitat_test',
               gpu_id: int = 0,
               image_key: str = 'rgb',
               goal_key: str = 'pointgoal_with_gps_compass',
               reward_function: RewardFunction = gin.REQUIRED,
               ) -> Dict[str, Any]:
    opts = ['SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID', gpu_id]
    if max_steps:
        opts += ['ENVIRONMENT.MAX_EPISODE_STEPS', int(max_steps)]
    if not task.endswith('.yaml'):
        task = f'{get_config_dir()}/habitat/tasks/{task}.yaml'
    if not dataset.endswith('.yaml'):
        dataset = f'{get_config_dir()}/habitat/datasets/{dataset}.yaml'
    copyfile(task, Path(wandb.run.dir) / 'task.yaml')
    copyfile(dataset, Path(wandb.run.dir) / 'dataset.yaml')
    config = habitat.get_config([task, dataset], opts)
    config.defrost()
    config.TASK.SUCCESS = habitat.Config()
    config.TASK.SUCCESS.TYPE = "Success"
    config.TASK.SUCCESS.SUCCESS_DISTANCE = config.TASK.SUCCESS_DISTANCE
    config.TASK.MEASUREMENTS.append("SUCCESS")
    config.freeze()
    return {'config': config, 'image_key': image_key, 'goal_key': goal_key, 'reward_function': reward_function}
