# evaluate.py: Evaluate model
#
# (C) 2019, Daniel Mouritzen

import contextlib
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gin
import gym
import habitat
import numpy as np
import tensorflow as tf
import wandb
from loguru import logger

from project.agents import MPCAgent
from project.environments import wrappers
from project.environments.habitat import VectorHabitat
from project.model import Model, restore_model
from project.tasks import Task
from project.tasks.habitat import habitat_task
from project.util.tf import get_distribution_strategy
from project.util.timing import measure_time

from .simulator import Simulator


@measure_time
def evaluate(logdir: Path,
             checkpoint: Optional[Path] = None,
             model: Optional[Model] = None,
             num_episodes: int = 10,
             video: bool = True,
             seed: Optional[int] = None,
             sync_wandb: bool = True,
             existing_env: Optional[VectorHabitat] = None,
             ) -> Tuple[Dict[str, float], Optional[List[wandb.Video]]]:
    """Evaluate trained model in a Habitat environment."""
    assert [checkpoint, model].count(None) == 1, 'Exactly one of checkpoint and model must be provided'
    if seed is not None:
        # TODO: Make deterministic mode work again
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    task = habitat_task()
    if existing_env is None:
        logger.info('Creating environments.')
        env = create_env(task, video, seed)
        original_config = None
        original_params = None
    else:
        env = existing_env
        original_config, original_params = reconfigure_env(env, video, seed)
    env = wrap_env(env, task)

    sim = Simulator(env,
                    metrics=task.metrics,
                    save_dir=logdir / 'eval' / f'{datetime.now():%Y%m%d-%H%M%S}',
                    save_video=video)

    if not tf.distribute.has_strategy():
        distribute_scope = get_distribution_strategy().scope()
    else:
        distribute_scope = contextlib.nullcontext()
    with distribute_scope:
        if model is None:
            model, _ = restore_model(checkpoint, logdir)
        agent = MPCAgent(env.action_space, model, objective='reward')
        mean_metrics = sim.run(agent, num_episodes, log=True)

    videos = [wandb.Video(str(vid), fps=10, format="mp4") for vid in sim.save_dir.glob('*.mp4')] if video else None

    if sync_wandb:
        # First delete existing summary items
        for k in list(wandb.run.summary._json_dict.keys()):
            wandb.run.summary._root_del((k,))
        wandb.run.summary.update(mean_metrics)
        wandb.run.summary['seed'] = seed
        if video:
            for i, vid in enumerate(videos):
                wandb.run.summary[f'video_{i}'] = vid

    if existing_env is not None:
        assert original_config is not None
        assert original_params is not None
        existing_env.reconfigure(config=original_config, **original_params)
        existing_env.call_at(0, 'enable_curriculum', {'enable': gin.query_parameter('curriculum.enabled')})

    return mean_metrics, videos


def create_env(task: Task, capture_video: bool, seed: Optional[int]) -> VectorHabitat:
    env: VectorHabitat = task.env_ctor()
    env.reset()  # Required before reconfigure
    reconfigure_env(env, capture_video, seed)
    return env


def reconfigure_env(env: VectorHabitat,
                    capture_video: bool = False,
                    seed: Optional[int] = None,
                    ) -> Tuple[habitat.Config, Dict[str, Any]]:
    original_config: habitat.Config = env._config
    original_params: Dict[str, Any] = {'capture_video': env._capture_video, 'min_duration': env._min_duration}
    config = original_config.clone()
    config.defrost()
    if capture_video and 'TOP_DOWN_MAP' not in config.TASK.MEASUREMENTS:
        # Top-down map is expensive to compute, so we only enable it for evaluation.
        config.TASK.MEASUREMENTS.append('TOP_DOWN_MAP')
    config.freeze()
    env.reconfigure(config=config, capture_video=capture_video, seed=seed, min_duration=0)
    return original_config, original_params


def wrap_env(env: VectorHabitat, task: Task) -> gym.Env:
    env.call_at(0, 'enable_curriculum', {'enable': False})
    wrapped_env: gym.Env = wrappers.SelectObservations(env, task.observation_components)
    wrapped_env = wrappers.SelectMetrics(wrapped_env, task.metrics)
    return wrapped_env
