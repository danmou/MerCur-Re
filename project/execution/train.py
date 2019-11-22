# train.py: Train model
#
# (C) 2019, Daniel Mouritzen

from pathlib import Path
from typing import Any, Dict, Optional

import gin
import tensorflow as tf
from loguru import logger

from project.models.planet.scripts.train import process as planet_train
from project.util import AttrDict
from project.util.planet_interface import PlanetParams
from project.util.files import link_directory_contents

from .evaluate import evaluate


@gin.configurable(whitelist=['num_eval_episodes'])
def train(logdir: str, initial_data: Optional[str], num_eval_episodes: int = 10) -> None:
    params = PlanetParams()
    dataset_dirs = {name: Path(logdir) / f'{name}_episodes' for name in ['train', 'test']}
    if initial_data:
        with params.unlocked():
            params.num_seed_episodes = 0
        logger.info('Linking initial dataset.')
        for dataset in dataset_dirs.values():
            link_directory_contents(Path(initial_data).absolute().relative_to(dataset) / dataset.name, dataset)

    args = AttrDict()
    with args.unlocked():
        args.config = 'default'
        args.params = params

    def eval_(envs: Dict[str, Any]) -> None:
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
