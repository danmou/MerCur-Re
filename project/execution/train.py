# train.py: 
#
# (C) 2019, Daniel Mouritzen

from pathlib import Path
from typing import Optional

import gin
import tensorflow as tf
from loguru import logger
from planet.scripts.train import process as planet_train

from project.models.planet import PlanetParams, AttrDict
from .evaluate import evaluate


@gin.configurable(whitelist=['num_eval_episodes'])
def train(logdir: str, initial_data: Optional[str], num_eval_episodes: int = 10) -> None:
    params = PlanetParams()
    if initial_data:
        with params.unlocked:
            params.num_seed_episodes = 0
        logger.info('Linking initial dataset.')
        for dataset in ['test_episodes', 'train_episodes']:
            dest = Path(logdir) / dataset
            dest.mkdir()
            for src_file in (Path(initial_data).absolute() / dataset).iterdir():
                dest_file = dest / src_file.name
                dest_file.symlink_to(src_file)

    args = AttrDict()
    with args.unlocked:
        args.config = 'default'
        args.params = params

    def eval_(envs):
        with tf.Graph().as_default():
            evaluate(logdir=Path(logdir),
                     checkpoint=Path(logdir),
                     num_episodes=num_eval_episodes,
                     video=True,
                     seed=1,
                     sync_wandb=True,
                     existing_env=envs['habitat'])
    for score, envs in planet_train(logdir, args):
        eval_(envs)
    logger.info('Run completed.')
