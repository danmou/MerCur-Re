# slam_agent.py: Wrapper for ORB-SLAM2 agent from habitat_baselines
#
# (C) 2020, Daniel Mouritzen

from typing import Optional

import gym
import tensorflow as tf

from project.habitat_baselines.config.default import get_config as get_default_config
from project.util.config import get_config_dir

from .base import Agent, Observations


class SLAMAgent(Agent):
    def __init__(self, action_space: gym.Space) -> None:
        super().__init__(action_space)
        import torch
        from project.habitat_baselines.agents.slam_agents import ORBSLAM2Agent
        config = get_default_config(f'{get_config_dir()}/baselines/slam_pointnav.yaml')
        config.defrost()
        config.ORBSLAM2.CAMERA_HEIGHT = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.POSITION[1]
        config.ORBSLAM2.H_OBSTACLE_MIN = 0.3 * config.ORBSLAM2.CAMERA_HEIGHT
        config.ORBSLAM2.H_OBSTACLE_MAX = 1.0 * config.ORBSLAM2.CAMERA_HEIGHT
        config.ORBSLAM2.MIN_PTS_IN_OBSTACLE = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH / 2.0
        config.freeze()
        self._agent = ORBSLAM2Agent(config.ORBSLAM2, device=torch.device('cpu'))

    def reset(self) -> None:
        """Reset agent's state"""
        self._agent.reset()

    def observe(self, observations: Observations, action: Optional[tf.Tensor]) -> None:
        """Update agent's state based on observations"""
        self._agent.update_internal_state({k: v.numpy() for k, v in observations.items()})

    def act(self) -> tf.Tensor:
        """Decide the next action"""
        return tf.convert_to_tensor(self._agent.act(None, random_prob=0.0)['action'])
