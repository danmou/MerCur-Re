# evaluator.py: Evaluate model
#
# (C) 2019, Daniel Mouritzen

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import gin
import gym
import tensorflow as tf
import wandb
from loguru import logger

from project.agents import Agent, ConstantAgent, MPCAgent, RandomAgent
from project.model import restore_model
from project.tasks import Task
from project.util.tf import get_distribution_strategy
from project.util.timing import measure_time

from .simulator import Simulator


@gin.configurable('evaluation', whitelist=['tasks'])
class Evaluator:
    def __init__(self, logdir: Path, video: bool = True, tasks: Sequence[Task] = gin.REQUIRED) -> None:
        self.logdir = logdir
        self.video = video
        logger.info('Creating evaluation environments.')
        self.sims = {task.name: Simulator(task, capture_video=video) if task.name == 'habitat' else Simulator(task)
                     for task in tasks}

    @measure_time
    def evaluate(self,
                 checkpoint: Optional[Path] = None,
                 baseline: Optional[str] = None,
                 agents: Optional[Mapping[str, Agent]] = None,
                 num_episodes: int = 10,
                 visualize_planner: bool = False,
                 seed: Optional[int] = None,
                 sync_wandb: bool = True,
                 ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[wandb.Video]]]:
        """Evaluate trained model in all environments."""
        assert [checkpoint, baseline, agents].count(None) == 2, 'Exactly one of checkpoint, baseline and agents must be provided'

        mean_metrics = {}
        videos = {}
        for task, sim in self.sims.items():
            logger.info(f'Evaluating on {task} task.')

            if seed is not None:
                # TODO: Make deterministic mode work again
                sim.seed(seed)

            save_dir = self.logdir / 'eval' / task / f'{datetime.now():%Y%m%d-%H%M%S}'

            get_distribution_strategy()
            if agents is not None:
                agent = agents[task]
            else:
                agent = self.get_agent(sim.action_space, checkpoint, baseline)
            if isinstance(agent, MPCAgent):
                agent.visualize = visualize_planner  # type: ignore[misc]  # mypy/issues/1362
            mean_metrics[task] = sim.run(agent, num_episodes, log=True, save_dir=save_dir, save_video=self.video)

            if self.video:
                videos[task] = [wandb.Video(str(vid), fps=10, format="mp4") for vid in save_dir.glob('*.mp4')]

        if sync_wandb:
            # First delete existing summary items
            for k in list(wandb.run.summary._json_dict.keys()):
                wandb.run.summary._root_del((k,))
            wandb.run.summary.update(mean_metrics)
            wandb.run.summary['seed'] = seed
            if self.video:
                for task, task_videos in videos.items():
                    for i, vid in enumerate(task_videos):
                        wandb.run.summary[f'{task}/video_{i}'] = vid

        return mean_metrics, videos

    def get_agent(self,
                  action_space: gym.Space,
                  checkpoint: Optional[Path] = None,
                  baseline: Optional[str] = None,
                  ) -> Agent:
        if baseline is None:
            assert checkpoint is not None
            model, _ = restore_model(checkpoint, self.logdir)
            return MPCAgent(action_space, model, objective='reward')
        else:
            if baseline == 'random':
                return RandomAgent(action_space)
            elif baseline == 'straight':
                return ConstantAgent(action_space, value=tf.constant([0.0]))
            else:
                raise RuntimeError(f'Unknown baseline {baseline}')
