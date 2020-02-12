# cli.py: Entrypoint for command-line interface
#
# (C) 2019, Daniel Mouritzen

import os
import textwrap
from typing import Any, Callable, Optional, Sequence, Tuple

import click
import wandb

from project.main import main_configure
from project.util.config import get_config_dir


def with_global_options(func: Callable[..., None]) -> Callable[..., None]:
    @click.option('-c', '--config', type=click.Path(dir_okay=False), default=None, multiple=True, help='Gin config')
    @click.option('-l', '--logdir', type=click.Path(file_okay=False), default=None,
                  help='Base log dir, actual log dir will be a timestamped subdirectory')
    @click.option('--data', type=click.Path(file_okay=False), default=None,
                  help="Path to data directory (containing 'datasets' and 'scene_datasets')")
    @click.option('-v', '--verbose', is_flag=True)
    @click.option('--verbosity', default='INFO')
    @click.option('-d', '--debug', is_flag=True, help='Disable W&B syncing and enable debug config')
    @click.option('--gpus', default=None)
    @click.option('-m', '--model', help='Model checkpoint to load. Can be a dir or a specific file')
    @click.option('--wandb-run', help='W&B run to evaluate or continue training from. Can be full id or part of name '
                                      '(the latest matching run will be used)')
    @click.argument('extra_options', nargs=-1)
    def wrapper(config: Tuple[str, ...],
                logdir: Optional[str],
                data: Optional[str],
                verbose: bool,
                verbosity: str,
                debug: bool,
                gpus: Optional[str],
                model: Optional[str],
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
        configs = ['default']
        if debug:
            configs.append('debug')
        configs += config
        configs = [name if name.endswith('.gin') else f'{get_config_dir()}/{name}.gin' for name in configs]
        if verbose:
            print(f'Using configs {configs}.')  # use print because logging has not yet been initialized
        if gpus is None and 'CUDA_VISIBLE_DEVICES' not in os.environ:
            print('Warning: No GPU devices specified. Defaulting to device 0.')
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        if gpus is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        return func(configs, data, verbosity, debug, model, extra_options, **kwargs)
    wrapper.__doc__ = textwrap.dedent(func.__doc__ or '') + "\n\nEXTRA_OPTIONS is one or more additional " \
                                                            "gin-config options, e.g. 'training.num_epochs=200'"
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
def train_command(configs: Sequence[str],
                  data: Optional[str],
                  verbosity: str,
                  debug: bool,
                  checkpoint: Optional[str],
                  extra_options: Tuple[str, ...],
                  wandb_run: Optional[str],
                  initial_data: Optional[str],
                  ) -> None:
    """Run training."""
    with main_configure(configs,
                        extra_options,
                        verbosity,
                        debug,
                        checkpoint,
                        data=data,
                        wandb_continue=wandb_run) as main:
        main.train(initial_data)


@cli.command(name='evaluate')
@with_global_options
@click.option('-n', '--num-episodes', type=int, default=10, help='Number of episodes to evaluate on')
@click.option('--no-video', is_flag=True, help='Disable video generation for faster evaluation')
@click.option('--visualize-planner', is_flag=True, help='Generate plots to visualize CEM planning process')
@click.option('--seed', type=int, help='Set seed for random values (this will also disable parallelization of loops)')
@click.option('--no-sync', is_flag=True, help="Don't upload results to W&B")
@click.option('-b', '--baseline', type=click.Choice(['random', 'straight']), help='Evaluate a trivial baseline agent '
                                                                                  'instead of a trained model')
def evaluate_command(configs: Sequence[str],
                     data: Optional[str],
                     verbosity: str,
                     debug: bool,
                     checkpoint: Optional[str],
                     extra_options: Tuple[str, ...],
                     wandb_run: Optional[str],
                     num_episodes: int,
                     no_video: bool,
                     visualize_planner: bool,
                     seed: Optional[int],
                     no_sync: bool,
                     baseline: Optional[str],
                     ) -> None:
    """Evaluate checkpoint."""
    if not wandb_run:
        os.environ[wandb.env.MODE] = 'dryrun'
    with main_configure(configs,
                        extra_options,
                        verbosity,
                        debug,
                        checkpoint,
                        data=data,
                        job_type='eval',
                        wandb_continue=wandb_run) as main:
        main.evaluate(num_episodes, not no_video, visualize_planner, seed, no_sync, baseline)
