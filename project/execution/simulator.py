# simulator.py: Provides Simulator class
#
# (C) 2019, Daniel Mouritzen

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union, cast

import gym
import gym.spaces
import numpy as np
import tensorflow as tf
from loguru import logger

from project.agents import Agent
from project.environments.wrappers import CollectGymDataset
from project.util import PrettyPrinter, Statistics
from project.util.planet.preprocess import preprocess
from project.util.tf import tf_nested_py_func
from project.util.timing import Timer
from project.util.typing import Observations, ObsTuple

TensorObs = Union[tf.Tensor, Dict[str, tf.Tensor]]
TensorObsTuple = Tuple[TensorObs, tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]


class Simulator:
    """Allows running an Agent closed-loop in an Env using `Simulator(env).run(agent)`."""
    def __init__(self,
                 env: gym.Env,
                 metrics: Optional[Sequence[str]] = None,
                 save_dir: Union[None, str, Path] = None,
                 save_data: bool = False,
                 save_video: bool = False,
                 ) -> None:
        assert not ((save_data or save_video) and save_dir is None), 'Can\'t save data or videos without save_dir.'
        self.save_dir = None if save_dir is None else Path(save_dir)
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True)
        if save_data:
            self._env = CollectGymDataset(env, str(self.save_dir))
        else:
            self._env = env
        self._observation_dtypes = self._parse_dtype(env.observation_space)
        self._metrics = list(metrics) if metrics else []
        self._seen_scenes = set()
        self._steps_seen = 0
        self._save_video = save_video

    @property
    def action_space(self) -> gym.Space:
        return self._env.action_space

    @property
    def observation_space(self) -> gym.Space:
        return self._env.observation_space

    @property
    def steps_seen(self) -> int:
        return self._steps_seen

    @property
    def scenes_seen(self) -> int:
        return len(self._seen_scenes)

    def run(self, agent: Agent, num_episodes: int = 1, log: bool = False) -> Dict[str, float]:
        """
        Run `num_episodes` episodes
        Returns:
            Dict of mean metrics, including number of steps, score (total reward) and planning time
        """
        log_fn = logger.info if log else logger.trace
        log_fn(f'Simulating {num_episodes} episodes closed-loop.')
        statistics_file = self.save_dir / 'eval.csv' if log and self.save_dir else None
        statistics = Statistics(['steps', 'score', 'plan_time'] + self._metrics, save_file=statistics_file)
        pp = PrettyPrinter(['episode', 'steps', 'score', 'plan_time'] + self._metrics, log_fn=log_fn)
        pp.print_header()
        for episode in range(num_episodes):
            steps, score, metrics = self.run_episode(agent)
            statistics.update(dict(steps=steps, score=score, **metrics))
            pp.print_row(dict(episode=episode, steps=steps, score=score, **metrics))
            if self._save_video:
                assert self.save_dir is not None
                self._env.save_video(self.save_dir / f'episode_{episode}_spl_{metrics["spl"]:.2f}')
        log_fn('Results:')
        statistics.print(log_fn=log_fn)
        return statistics.mean

    def run_episode(self, agent: Agent) -> Tuple[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Run one episode
        Returns:
            The episode duration in steps
            The total reward
            The metrics received in the last step (plus the mean planning time per step as `metrics['plan_time']`)
        """
        done = tf.constant(False)
        score = tf.constant(0.0, tf.float32)
        steps = tf.constant(0, tf.int16)
        plan_time = tf.constant(0.0, tf.float32)
        metrics: Dict[str, tf.Tensor] = {}

        agent.reset()
        obs = self._tf_reset_env()
        agent.observe(obs, action=None)
        while not done:
            with Timer() as t:
                action = agent.act()
            plan_time += t.interval
            obs, reward, done, metrics = self._tf_step_env(action)
            score += reward
            steps += 1
            agent.observe(obs, action)

        metrics['plan_time'] = plan_time / tf.cast(steps, tf.float32)
        return steps, score, metrics

    def _select_obs(self, obs: Observations) -> Observations:
        return {k: obs[k] for k in self._observation_dtypes.keys()}

    def _process_step(self, output: ObsTuple) -> ObsTuple:
        obs, reward, done, info = output
        obs = self._select_obs(obs)
        reward = np.float32(reward)
        metrics = {k: info[k] for k in self._metrics}
        self._steps_seen += 1
        if 'scene' in info:
            self._seen_scenes.add(info['scene'])
        obs, metrics = tf.nest.map_structure(lambda x: np.float32(x), (obs, metrics))
        return obs, reward, done, metrics

    @staticmethod
    def _tf_process_obs(obs: TensorObs) -> TensorObs:
        obs['image'] = preprocess(obs['image'])
        return obs

    def _tf_reset_env(self) -> TensorObs:
        obs = cast(TensorObs, tf_nested_py_func(lambda: self._select_obs(self._env.reset()),
                                                [],
                                                self._observation_dtypes))
        return self._tf_process_obs(obs)

    def _tf_step_env(self, action: tf.Tensor) -> TensorObsTuple:
        return_types = (self._observation_dtypes, tf.float32, tf.bool, {k: tf.float32 for k in self._metrics})
        obs, reward, done, metrics = cast(TensorObsTuple,
                                          tf_nested_py_func(lambda a: self._process_step(self._env.step(a)),
                                                            [action],
                                                            return_types))
        done.set_shape(())
        return self._tf_process_obs(obs), reward, done, metrics

    @staticmethod
    def _parse_dtype(space: gym.Space) -> Union[tf.DType, Dict[str, Union[tf.DType, Dict]]]:
        """Get tensor dtypes from a gym space."""
        if isinstance(space, gym.spaces.Discrete):
            return tf.int32
        if isinstance(space, gym.spaces.Box):
            if space.low.dtype == np.uint8:
                return tf.uint8
            else:
                return tf.float32
        if isinstance(space, gym.spaces.Dict):
            return {k: Simulator._parse_dtype(v) for k, v in space.spaces.items()}
        raise NotImplementedError(f"Unsupported space '{space}.'")
