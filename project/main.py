# main.py: Main entrypoint
#
# (C) 2019, Daniel Mouritzen

import contextlib
import os
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator, Optional, Tuple, Union

import gin
import gin.tf.external_configurables
import tensorflow as tf
import wandb
import wandb.settings
from loguru import logger
from tensorflow.python.util import deprecation

from project.execution import Evaluator, train
from project.util.logging import init_logging


@gin.configurable('main', whitelist=['base_logdir'])
class Main:
    def __init__(self,
                 verbosity: str,
                 debug: bool = False,
                 catch_exceptions: bool = True,
                 extension: Optional[str] = None,
                 checkpoint: Optional[Path] = None,
                 base_logdir: Union[str, Path] = gin.REQUIRED,
                 ) -> None:
        deprecation._PRINT_DEPRECATION_WARNINGS = False
        self.debug = debug
        self.checkpoint = checkpoint
        self.catch_exceptions = catch_exceptions
        self.base_logdir = Path(base_logdir)
        self.logdir = self._create_logdir(extension)
        init_logging(verbosity, self.logdir)
        self._create_symlinks()
        if wandb.run.resumed:
            logger.debug(f'Resumed run {wandb.run.id} ({wandb.run.name})')
            if not self.checkpoint:
                self._restore_wandb_checkpoint()
        else:
            self._update_wandb()
        if not tf.executing_eagerly():
            tf.compat.v1.enable_eager_execution()

    def _create_logdir(self, extension: Optional[str]) -> Path:
        logdir_name = f'{datetime.now():%Y%m%d-%H%M%S}'
        if extension:
            logdir_name += f'-{extension}'
        logdir = self.base_logdir / logdir_name
        logdir.mkdir(parents=True)
        return logdir

    def _create_symlinks(self) -> None:
        link_dest = self.logdir.relative_to(self.base_logdir)
        latest_symlink = Path(self.base_logdir) / 'latest'
        if latest_symlink.is_symlink():
            latest_symlink.unlink()
        latest_symlink.symlink_to(link_dest)
        try:
            wandb_name = wandb.Api().run(wandb.run.path).name
        except wandb.apis.CommError:
            wandb_name = None
        if wandb_name:
            wandb_symlink = Path(self.base_logdir) / wandb_name
            while wandb_symlink.exists() or wandb_symlink.is_symlink():  # exists() is false for broken symlinks
                wandb_symlink = Path(str(wandb_symlink) + '_cont')
            wandb_symlink.symlink_to(link_dest)
            logger.info(f'W&B run name: {wandb_name}')

    def _update_wandb(self) -> None:
        wandb.save(f'{self.logdir}/checkpoint_*')
        wandb.config.update({name.rsplit('.', 1)[-1]: conf
                             for (_, name), conf in gin.config._CONFIG.items()
                             if name is not None})
        wandb.config.update({'cuda_gpus': os.environ.get('CUDA_VISIBLE_DEVICES')})

    def _restore_wandb_checkpoint(self) -> None:
        with wandb.restore('checkpoint_latest') as f:
            ckpt_file = f.read().strip()
        assert ckpt_file, "Can't resume wandb run: no checkpoint found!"
        wandb.restore(ckpt_file)
        wandb.restore('checkpoint_additional_data.pickle')
        self.checkpoint = Path(wandb.run.dir) / ckpt_file

    @contextlib.contextmanager
    def _catch(self) -> Generator[None, None, None]:
        with logger.catch(BaseException, level='TRACE', reraise=not self.catch_exceptions):
            if self.catch_exceptions:
                with logger.catch(reraise=self.debug):
                    yield
            else:
                yield

    def train(self, initial_data: Optional[str] = None) -> None:
        try:
            with self._catch():
                initial_data_path = initial_data and Path(initial_data)
                train(self.logdir, initial_data_path, checkpoint=self.checkpoint)
        finally:
            # Make sure all checkpoints get uploaded
            wandb.save(f'{self.logdir}/checkpoint_*', policy='end')
            wandb.log(commit=True)

    def evaluate(self,
                 num_episodes: int = 10,
                 video: bool = True,
                 visualize_planner: bool = False,
                 seed: Optional[int] = None,
                 no_sync: bool = False,
                 baseline: Optional[str] = None,
                 ) -> None:
        assert baseline is not None or self.checkpoint is not None, 'No checkpoint specified!'
        with self._catch():
            Evaluator(logdir=self.logdir, video=video).evaluate(checkpoint=self.checkpoint,
                                                                baseline=baseline,
                                                                num_episodes=num_episodes,
                                                                visualize_planner=visualize_planner,
                                                                seed=seed,
                                                                sync_wandb=wandb.run.resumed and not no_sync)


@contextlib.contextmanager
def main_configure(config: str,
                   extra_options: Tuple[str, ...],
                   verbosity: str,
                   debug: bool = False,
                   checkpoint: Optional[str] = None,
                   catch_exceptions: bool = True,
                   job_type: str = 'training',
                   data: Optional[str] = None,
                   extension: Optional[str] = None,
                   wandb_continue: Optional[str] = None,
                   ) -> Generator[Main, None, None]:
    if wandb_continue is not None:
        run = _get_wandb_run(wandb_continue)
        resume_args = dict(resume=True, id=run.id, name=run.name, config=run.config, notes=run.notes, tags=run.tags)
    else:
        resume_args = {}
    wandb.init(sync_tensorboard=False, job_type=job_type, **resume_args)
    gin.parse_config_files_and_bindings([config], extra_options)
    with gin.unlock_config():
        gin.bind_parameter('main.base_logdir', str(Path(gin.query_parameter('main.base_logdir')).absolute()))
    with open(Path(wandb.run.dir) / f'config_{job_type}.gin', 'w') as f:
        f.write(open(config).read())
        f.write('\n# Extra options\n')
        f.write('\n'.join(extra_options))
    checkpoint_path = None if checkpoint is None else Path(checkpoint).absolute()
    tempdir = None
    try:
        if data:
            # Habitat assumes data is stored in local 'data' directory
            tempdir = TemporaryDirectory()
            (Path(tempdir.name) / 'data').symlink_to(Path(data).absolute())
            os.chdir(tempdir.name)
        yield Main(verbosity,
                   debug=debug,
                   catch_exceptions=catch_exceptions,
                   extension=extension,
                   checkpoint=checkpoint_path)
    finally:
        if tempdir:
            tempdir.cleanup()


def _get_wandb_run(name: str) -> wandb.apis.public.Run:
    api = wandb.Api()
    entity, project = (wandb.settings.Settings().get('default', setting) for setting in ['entity', 'project'])
    try:
        run = api.run(f'{entity}/{project}/{name}')
    except wandb.apis.CommError:
        # name is not a valid run id so we assume it's the name of a run
        runs = api.runs(f'{entity}/{project}', order="-created_at")
        for run in runs:
            if name in run.name:
                break
        else:
            raise ValueError(f'No run found with id or name {name}.')
    return run
