# train.py: 
#
# (C) 2019, Daniel Mouritzen

from loguru import logger
from planet.scripts.train import process as planet_train

from project.models.planet import PlanetParams, AttrDict


def train(logdir: str) -> None:
    params = PlanetParams()
    args = AttrDict()
    with args.unlocked:
        args.config = 'default'
        args.params = params

    for score in planet_train(logdir, args):
        pass
    logger.info('Run completed.')