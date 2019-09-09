#!/usr/bin/env python
# main.py: Main entrypoint
#
# (C) 2019, Daniel Mouritzen

import os
from typing import Optional, Tuple

import click
import gin
import wandb
from loguru import logger
from tensorflow.python.util import deprecation

from project.logging import init_logging
from project.planet import run


@logger.catch
@gin.configurable(blacklist=['verbose'])
def main(verbose: bool, logdir: str) -> None:
    init_logging(verbose, logdir)
    run(logdir)


@click.command()
@click.option('-c', '--config', type=click.Path(dir_okay=False), default='configs/default.gin',
              help='gin config', show_default=True)
@click.option('-l', '--logdir', type=click.Path(file_okay=False), default=None)
@click.option('-v', '--verbose', is_flag=True)
@click.option('-d', '--debug', is_flag=True, help='disable W&B syncing')
@click.argument('extra_options', nargs=-1)
def main_command(config: str,
                 logdir: Optional[str],
                 verbose: bool,
                 debug: bool,
                 extra_options: Tuple[str, ...]
                 ) -> None:
    """
    Run training.

    EXTRA_OPTIONS is one or more additional gin-config options, e.g. 'planet.num_runs=1000'
    """
    if logdir:
        extra_options += (f'main.logdir="{logdir}"',)
    if debug:
        os.environ['WANDB_MODE'] = 'dryrun'
    wandb.init(project="thesis", sync_tensorboard=True)
    gin.parse_config_files_and_bindings([config], extra_options)
    main(verbose)


if __name__ == '__main__':
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    main_command()
