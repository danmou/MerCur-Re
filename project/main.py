# main.py: Main entrypoint
#
# (C) 2019, Daniel Mouritzen

import datetime
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple, Union

import click
import gin
import wandb
from loguru import logger
from tensorflow.python.util import deprecation

from project.logging import init_logging
from project.planet import run
from project.util import get_config_dir


@gin.configurable(whitelist=['base_logdir'])
def main(verbosity: str,
         base_logdir: Union[str, Path] = gin.REQUIRED,
         debug: bool = False,
         catch_exceptions: bool = True,
         name: Optional[str] = None) -> None:
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    logdir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if name:
        logdir_name += f'-{name}'
    logdir = Path(base_logdir) / logdir_name
    logdir.mkdir(parents=True)
    latest_symlink = Path(base_logdir) / 'latest'
    if latest_symlink.exists():
        latest_symlink.unlink()
    latest_symlink.symlink_to(logdir)
    init_logging(verbosity, logdir)
    wandb.save(f'{logdir}/checkpoint')
    wandb.save(f'{logdir}/*.ckpt*')
    wandb.config.update({name.rsplit('.', 1)[-1]: conf
                         for (_, name), conf in gin.config._CONFIG.items()
                         if name is not None})
    wandb.config.update({'cuda_gpus': os.environ['CUDA_VISIBLE_DEVICES']})
    with logger.catch(BaseException, level='TRACE', reraise=not catch_exceptions):
        try:
            if catch_exceptions:
                with logger.catch(reraise=debug):
                    run(str(logdir))
            else:
                run(str(logdir))
        finally:
            # Make sure all checkpoints get uploaded
            wandb.save(f'{logdir}/checkpoint', policy='end')
            wandb.save(f'{logdir}/*.ckpt*', policy='end')


def main_configure(config: str,
                   extra_options: Tuple[str, ...],
                   verbosity: str,
                   debug: bool = False,
                   catch_exceptions: bool = True,
                   data: Optional[str] = None,
                   name: Optional[str] = None,
                   ) -> None:
    wandb.init(project="thesis", sync_tensorboard=True)
    # # Wandb adds '--' to the start of arguments in sweeps so we remove them
    # extra_options = tuple(opt.lstrip('-') for opt in extra_options)
    gin.parse_config_files_and_bindings([config], extra_options)
    with gin.unlock_config():
        gin.bind_parameter('main.base_logdir', str(Path(gin.query_parameter('main.base_logdir')).absolute()))
    tempdir = None
    try:
        if data:
            # Habitat assumes data is stored in local 'data' directory
            tempdir = TemporaryDirectory()
            (Path(tempdir.name) / 'data').symlink_to(Path(data).absolute())
            os.chdir(tempdir.name)
        main(verbosity, debug=debug, catch_exceptions=catch_exceptions, name=name)
    finally:
        if tempdir:
            tempdir.cleanup()


@click.command(context_settings={"ignore_unknown_options": True})
@click.option('-c', '--config', type=click.Path(dir_okay=False), default=None, help='gin config')
@click.option('-l', '--logdir', type=click.Path(file_okay=False), default=None)
@click.option('--data', type=click.Path(file_okay=False), default=None,
              help="path to data directory (containing 'datasets' and 'scene_datasets')")
@click.option('-v', '--verbose', is_flag=True)
@click.option('--verbosity', default='INFO')
@click.option('-d', '--debug', is_flag=True, help='disable W&B syncing and enable debug config')
@click.option('--gpus', default=None)
@click.option('-n', '--name', default=None)
@click.argument('extra_options', nargs=-1)
def main_command(config: str,
                 logdir: Optional[str],
                 data: Optional[str],
                 verbose: bool,
                 verbosity: str,
                 debug: bool,
                 gpus: Optional[str],
                 name: Optional[str],
                 extra_options: Tuple[str, ...]
                 ) -> None:
    """
    Run training.

    EXTRA_OPTIONS is one or more additional gin-config options, e.g. 'planet.num_runs=1000'
    """
    if verbose:
        verbosity = 'DEBUG'
    elif verbosity in ['DEBUG', 'TRACE']:
        verbose = True
    if logdir:
        extra_options += (f'main.base_logdir="{logdir}"',)
    if debug:
        os.environ['WANDB_MODE'] = 'dryrun'
    if config is None:
        config = f'{get_config_dir()}/{"debug" if debug else "default"}.gin'
        if verbose:
            print(f'Using config {config}.')  # use print because logging has not yet been initialized
    if gpus is None and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        print(f'Warning: No GPU devices specified. Defaulting to device 0.')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    main_configure(config, extra_options, verbosity, debug, True, data, name)
