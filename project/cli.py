# cli.py: Entrypoint for command-line interface
#
# (C) 2019, Daniel Mouritzen

import os
import textwrap
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import click
import wandb

from project.main import main_configure
from project.util import get_config_dir


def with_global_options(func: Callable[..., None]) -> Callable[..., None]:
    @click.option('-c', '--config', type=click.Path(dir_okay=False), default=None, help='Gin config')
    @click.option('-l', '--logdir', type=click.Path(file_okay=False), default=None,
                  help='Base log dir, actual log dir will be a timestamped subdirectory')
    @click.option('--data', type=click.Path(file_okay=False), default=None,
                  help="Path to data directory (containing 'datasets' and 'scene_datasets')")
    @click.option('-v', '--verbose', is_flag=True)
    @click.option('--verbosity', default='INFO')
    @click.option('-d', '--debug', is_flag=True, help='Disable W&B syncing and enable debug config')
    @click.option('--gpus', default=None)
    @click.argument('extra_options', nargs=-1)
    def wrapper(config: Optional[str],
                logdir: Optional[str],
                data: Optional[str],
                verbose: bool,
                verbosity: str,
                debug: bool,
                gpus: Optional[str],
                extra_options: Tuple[str, ...],
                **kwargs: Any,
                ) -> None:
        if verbose:
            verbosity = 'DEBUG'
        elif verbosity in ['DEBUG', 'TRACE']:
            verbose = True
        if logdir:
            extra_options += (f'main.base_logdir="{logdir}"',)
        if debug:
            os.environ[wandb.env.MODE] = 'dryrun'
        if config is None:
            config = f'{get_config_dir()}/{"debug" if debug else "default"}.gin'
            if verbose:
                print(f'Using config {config}.')  # use print because logging has not yet been initialized
        if gpus is None and 'CUDA_VISIBLE_DEVICES' not in os.environ:
            print('Warning: No GPU devices specified. Defaulting to device 0.')
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        if gpus is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        return func(config, data, verbosity, debug, extra_options, **kwargs)
    wrapper.__doc__ = textwrap.dedent(func.__doc__ or '') + "\n\nEXTRA_OPTIONS is one or more additional " \
                                                            "gin-config options, e.g. 'planet.max_epochs=200'"
    wrapper.__click_params__ += getattr(func, '__click_params__', [])  # type: ignore[attr-defined]
    return wrapper


@click.group(context_settings={"ignore_unknown_options": True})
@click.pass_context
def cli(ctx: click.Context) -> None:
    if ctx.invoked_subcommand is None:
        train_command()  # Train by default


@cli.command(name='train')
@with_global_options
@click.option('--initial-data', help='Dataset to use instead of initial collection')
def train_command(config: str,
                  data: Optional[str],
                  verbosity: str,
                  debug: bool,
                  extra_options: Tuple[str, ...],
                  initial_data: Optional[str],
                  ) -> None:
    """Run training."""
    with main_configure(config, extra_options, verbosity, debug, data=data) as main:
        main.train(initial_data)


@cli.command(name='evaluate')
@with_global_options
@click.option('-m', '--model', help='Model checkpoint to load. Can be a dir or a specific file')
@click.option('--wandb-run', help='W&B run to evaluate. Can be full id or part of name '
                                  '(the latest matching run will be used)')
@click.option('-n', '--num-episodes', type=int, default=10, help='Number of episodes to evaluate on')
@click.option('--no-video', is_flag=True, help='Disable video generation for faster evaluation')
@click.option('--seed', type=int, help='Set seed for random values (this will also disable parallelization of loops)')
@click.option('--no-sync', is_flag=True, help="Don't upload results to W&B")
def evaluate_command(config: str,
                     data: Optional[str],
                     verbosity: str,
                     debug: bool,
                     extra_options: Tuple[str, ...],
                     model: Optional[str],
                     wandb_run: Optional[str],
                     num_episodes: int,
                     no_video: bool,
                     seed: Optional[int],
                     no_sync: bool,
                     ) -> None:
    """Evaluate checkpoint."""
    # assert model or wandb_run, 'Either --model or --wandb-run must be specified!'
    if not wandb_run:
        os.environ[wandb.env.MODE] = 'dryrun'
    model_path = None if model is None else Path(model)
    with main_configure(config,
                        extra_options,
                        verbosity,
                        debug,
                        data=data,
                        job_type='eval',
                        wandb_continue=wandb_run) as main:
        main.evaluate(model_path, num_episodes, not no_video, seed, no_sync)
