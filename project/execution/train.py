# train.py: 
#
# (C) 2019, Daniel Mouritzen

import shutil
from pathlib import Path
from typing import Optional

from loguru import logger
from planet.scripts.train import process as planet_train

from project.models.planet import PlanetParams, AttrDict


def train(logdir: str, initial_data: Optional[str]) -> None:
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

    for score in planet_train(logdir, args):
        pass
    logger.info('Run completed.')