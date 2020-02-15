# habitat.py: provides Habitat class, which interfaces with the habitat environment
#
# (C) 2019, Daniel Mouritzen

from __future__ import annotations

import random
from pathlib import Path
from shutil import copyfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, TypeVar, Union, cast
from unittest.mock import MagicMock

import gin
import gym.spaces
import habitat
import habitat_sim
import numpy as np
import wandb
from habitat.core.simulator import Observations
from habitat.core.vector_env import VectorEnv
from habitat.sims.habitat_simulator.actions import HabitatSimActions, HabitatSimV1ActionSpaceConfiguration
from habitat.tasks import make_task
from habitat.tasks.nav.nav import NavigationEpisode, SimulatorTaskAction
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from habitat_sim.agent.controls.controls import ActuationSpec
from habitat_sim.agent.controls.default_controls import LookLeft
from loguru import logger

from project.util.config import get_config_dir
from project.util.logging import capture_output
from project.util.timing import measure_time

from .rewards import RewardFunction

ObsTuple = Tuple[Observations, Any, bool, dict]


@habitat.registry.register_measure
class Success(habitat.Measure):
    def __init__(self, *args: Any, sim: habitat.Simulator, config: habitat.Config, **kwargs: Any) -> None:
        self._sim = sim
        self._config = config
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return 'success'

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        self._metric = None

    def update_metric(self,  # type: ignore[override]
                      *args: Any,
                      episode: NavigationEpisode,
                      task: habitat.EmbodiedTask,
                      **kwargs: Any) -> None:
        current_position = self._sim.get_agent_state().position.tolist()
        distance_to_target = self._sim.geodesic_distance(current_position, episode.goals[0].position)

        self._metric = int(getattr(task, 'is_stop_called', False) and distance_to_target < self._config.SUCCESS_DISTANCE)


@habitat_sim.registry.register_move_fn(body_action=True)
class TurnAngle(LookLeft):
    """
    This class defines a simulator action that turns counter-clockwise some number of degrees specified by
    the class variable `angle`
    """

    angle = 0.0

    def __call__(self, scene_node: habitat_sim.SceneNode, actuation_spec: ActuationSpec) -> None:
        actuation_spec.amount = TurnAngle.angle
        super().__call__(scene_node, actuation_spec)


@habitat.registry.register_task_action
class TurnAngleAction(SimulatorTaskAction):
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return 'turn_angle'

    def step(self, *args: Any, **kwargs: Any) -> Observations:
        return self._sim.step(HabitatSimActions.TURN_ANGLE)


@habitat.registry.register_action_space_configuration
class TurnAngleActionSpace(HabitatSimV1ActionSpaceConfiguration):
    def get(self) -> Dict[int, habitat_sim.ActionSpec]:
        config: Dict[int, habitat_sim.ActionSpec] = super().get()
        config[HabitatSimActions.TURN_ANGLE] = habitat_sim.ActionSpec('turn_angle', ActuationSpec(0.0))
        return config


class Habitat(habitat.RLEnv):
    def __init__(self,
                 config: habitat.Config,
                 image_key: str,
                 goal_key: str,
                 reward_function: RewardFunction,
                 capture_video: bool = False,
                 seed: Optional[int] = None,
                 min_duration: int = 0,
                 max_duration: int = 500,
                 **_: Any) -> None:
        self._image_key = image_key
        self._goal_key = goal_key
        self._reward_function = reward_function
        self._reward_function.set_env(self)
        self._called_stop: bool = False
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
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self._min_duration = min_duration
        self._max_duration = max_duration
        self._step_count = 0

    def reconfigure(self,
                    config: habitat.Config,
                    capture_video: Optional[bool] = None,
                    seed: Optional[int] = None,
                    min_duration: Optional[int] = None,
                    max_duration: Optional[int] = None,
                    ) -> None:
        if capture_video is not None:
            self._capture_video = capture_video
        if seed is not None:
            # This is needed for reproducible episode shuffling
            random.seed(seed)
            np.random.seed(seed)
        with capture_output('habitat_sim'):
            self.habitat_env.reconfigure(config)
            # Habitat's reconfigure doesn't update the task config, so we do that manually:
            self.habitat_env._task = make_task(
                config.TASK.TYPE,
                config=config.TASK,
                sim=self.habitat_env._sim,
                dataset=self.habitat_env._dataset,
            )
        if seed is not None:
            self.seed(seed)
        if min_duration is not None:
            self._min_duration = min_duration
        if max_duration is not None:
            self._max_duration = max_duration

    def get_reward_range(self) -> Tuple[float, float]:
        return self._reward_function.get_reward_range()

    def get_reward(self, observations: Observations) -> float:
        return self._reward_function.get_reward(observations)

    def get_done(self, observations: Observations) -> bool:
        return self._env.episode_over or self.episode_success()

    def episode_success(self) -> bool:
        return self._called_stop and self.distance_to_target() < self.success_distance

    def distance_to_target(self) -> float:
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position  # type: ignore[attr-defined]
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
        metrics['scene'] = self.habitat_env.current_episode.scene_id
        return metrics

    def step(self, action: Union[np.ndarray, float]) -> ObsTuple:  # type: ignore[override]
        self._called_stop = False
        action = float(action)
        self._step_count += 1
        if self._step_count >= self._max_duration or \
                self._step_count >= self._min_duration and self.distance_to_target() < self.success_distance:
            self._called_stop = True
            obs, reward, done, info = super().step('STOP')
            info['taken_action'] = 0.0
            info['timeout'] = not self.episode_success()
        else:
            TurnAngle.angle = action * 90.0 / 2
            sum_reward = 0.0
            for sub_action in ['TURN_ANGLE',
                               'MOVE_FORWARD',
                               'TURN_ANGLE']:
                obs, reward, done, info = super().step(sub_action)
                sum_reward += reward
                if done:
                    break
            info['taken_action'] = action
            reward = sum_reward
        obs = self._update_keys(obs)
        if done:
            logger.debug(f'Episode finished at step {self._step_count}.')
        if self._capture_video:
            # upscale image to make the resulting video more viewable
            new_obs = {'rgb': np.repeat(np.repeat(obs['image'], 4, axis=0), 4, axis=1)}
            act = action * 0.9
            if act:
                img_size = new_obs['rgb'].shape[0]
                start = img_size / 2
                end = img_size * (1 - act) / 2
                left = round(min(start, end))
                right = round(max(start, end))
                new_obs['rgb'][round(img_size * 0.9):round(img_size * 0.95),
                               round(left):round(right)] = np.array([0, 0, 255])

            self._rgb_frames.append(observations_to_image(new_obs, self.habitat_env.get_metrics()))
        return obs, reward, done, info

    def reset(self) -> Observations:
        self._called_stop = False
        self._step_count = 0
        self._reward_function.reset()
        self._rgb_frames = []
        with capture_output('habitat_sim'):
            obs = super().reset()
        return self._update_keys(obs)

    _ObsOrDict = TypeVar('_ObsOrDict', Observations, Dict['str', Any])

    def _update_keys(self, obs: _ObsOrDict) -> _ObsOrDict:
        obs_dict = {'image': obs[self._image_key],
                    'goal': obs[self._goal_key]}
        # Convert back to original type (Observations.__init__ tries to process the input dict)
        obs = type(obs)({})
        obs.update(obs_dict)
        return obs

    def close(self) -> None:
        with capture_output('habitat_sim'):
            self._env.close()

    @measure_time
    def save_video(self, file: Union[str, Path], fps: int = 10) -> None:
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
        self._reward_function.set_env(self)
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

    def step(self, action: int) -> ObsTuple:
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
        return cast(Observations, self.observation_space.sample())


class VectorHabitat(VectorEnv):
    def __init__(self,
                 env_ctor: Callable[..., gym.Env],
                 params: Mapping[str, Any],
                 auto_reset_done: bool = False) -> None:
        super().__init__(env_ctor, [tuple(params.items())], auto_reset_done=auto_reset_done, multiprocessing_start_method='fork')
        self.observation_space = self.observation_spaces[0]
        self.action_space = self.action_spaces[0]
        self._valid_attrs = {'observation_space', 'action_space', 'reward_range', 'metadata', 'reward_range', 'spec',
                             '_config', '_capture_video', '_min_duration'}
        self._valid_attrs |= set(Habitat.__dict__.keys())

    def __getattr__(self, name: str) -> Any:
        if name not in self._valid_attrs:
            raise AttributeError(f'Invalid attribute {name}')
        return self.call_at(0, '__getattr__', {'name': name})

    def save_video(self, file: Union[str, Path], **kwargs: Any) -> None:
        kwargs['file'] = file
        self.call_at(0, 'save_video', kwargs)

    def reconfigure(self, **kwargs: Any) -> None:
        self.call_at(0, 'reconfigure', kwargs)

    def seed(self, seed: int) -> None:
        self.call_at(0, 'seed', {'seed': seed})

    def step(self, action: Any) -> ObsTuple:  # type: ignore[override]
        obs, reward, done, info = super().step(data=[{'action': action}])[0]
        return obs, reward, done, info

    def reset(self) -> Observations:
        return cast(Observations, super().reset()[0])

    def close(self) -> None:
        try:
            super().close()
        except (BrokenPipeError, EOFError):
            pass


@gin.configurable('Habitat', whitelist=['task', 'train_dataset', 'train_split', 'eval_dataset', 'eval_split', 'gpu_id',
                                        'image_key', 'goal_key', 'reward_function', 'eval_episodes_per_scene'])
def get_config(training: bool = False,
               top_down_map: bool = False,
               max_steps: Optional[Union[int, float]] = None,
               task: str = 'pointnav',
               train_dataset: str = 'habitat_test',
               train_split: str = 'train',
               eval_dataset: str = 'habitat_test',
               eval_split: str = 'val',
               gpu_id: int = 0,
               image_key: str = 'rgb',
               goal_key: str = 'pointgoal_with_gps_compass',
               reward_function: RewardFunction = gin.REQUIRED,
               eval_episodes_per_scene: int = 3
               ) -> Dict[str, Any]:
    mode = 'train' if training else 'eval'
    dataset = train_dataset if training else eval_dataset
    split = train_split if training else eval_split
    opts = ['SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID', gpu_id]
    if max_steps:
        opts += ['ENVIRONMENT.MAX_EPISODE_STEPS', int(max_steps)]
    opts += ['DATASET.SPLIT', split]
    if not training:
        opts += ['ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE', False]
        opts += ['ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES', eval_episodes_per_scene]
    if not task.endswith('.yaml'):
        task = f'{get_config_dir()}/habitat/tasks/{task}.yaml'
    if not dataset.endswith('.yaml'):
        dataset = f'{get_config_dir()}/habitat/datasets/{dataset}.yaml'
    task_file = Path(wandb.run.dir) / 'task.yaml'
    dataset_file = Path(wandb.run.dir) / f'{mode}_dataset.yaml'
    copyfile(task, task_file)
    copyfile(dataset, dataset_file)
    wandb.save(str(task_file))
    wandb.save(str(dataset_file))
    if not HabitatSimActions.has_action('TURN_ANGLE'):
        HabitatSimActions.extend_action_space('TURN_ANGLE')
    config = habitat.get_config([task, dataset], opts)
    config.defrost()
    config.TASK.SUCCESS = habitat.Config()
    config.TASK.SUCCESS.TYPE = 'Success'
    config.TASK.SUCCESS.SUCCESS_DISTANCE = config.TASK.SUCCESS_DISTANCE
    config.TASK.MEASUREMENTS.append('SUCCESS')
    config.TASK.ACTIONS.TURN_ANGLE = habitat.Config()
    config.TASK.ACTIONS.TURN_ANGLE.TYPE = 'TurnAngleAction'
    config.TASK.POSSIBLE_ACTIONS = ['STOP', 'MOVE_FORWARD', 'TURN_ANGLE']
    if top_down_map and 'TOP_DOWN_MAP' not in config.TASK.MEASUREMENTS:
        # Top-down map is expensive to compute, so we only enable it when needed.
        config.TASK.MEASUREMENTS.append('TOP_DOWN_MAP')
    config.freeze()
    return {'config': config, 'image_key': image_key, 'goal_key': goal_key, 'reward_function': reward_function}
