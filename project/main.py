# main.py: Main entrypoint
#
# (C) 2019, Daniel Mouritzen

import contextlib
import datetime
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Generator, Optional, Tuple, Union

import gin
import wandb
from loguru import logger
from tensorflow.python.util import deprecation

from project.logging_utils import init_logging
from project.execution import evaluate, train


@gin.configurable('main', whitelist=['base_logdir'])
class Main:
    def __init__(self,
                 verbosity: str,
                 base_logdir: Union[str, Path] = gin.REQUIRED,
                 debug: bool = False,
                 catch_exceptions: bool = True,
                 extension: Optional[str] = None,
                 ) -> None:
        deprecation._PRINT_DEPRECATION_WARNINGS = False
        self.debug = debug
        self.catch_exceptions = catch_exceptions
        self.base_logdir = base_logdir
        self.logdir = self._create_logdir(extension)
        init_logging(verbosity, self.logdir)
        self._create_symlinks()
        self._update_wandb()

    def _create_logdir(self, extension: Optional[str]) -> str:
        logdir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if extension:
            logdir_name += f'-{extension}'
        logdir = Path(self.base_logdir) / logdir_name
        logdir.mkdir(parents=True)
        return logdir

    def _create_symlinks(self) -> None:
        latest_symlink = Path(self.base_logdir) / 'latest'
        if latest_symlink.exists():
            latest_symlink.unlink()
        latest_symlink.symlink_to(self.logdir)
        try:
            wandb_name = wandb.Api().run(wandb.run.path).name
        except wandb.apis.CommError:
            wandb_name = None
        if wandb_name:
            (Path(self.base_logdir) / wandb_name).symlink_to(self.logdir)
            logger.info(f'W&B run name: {wandb_name}')

    def _update_wandb(self) -> None:
        wandb.save(f'{self.logdir}/checkpoint')
        wandb.save(f'{self.logdir}/*.ckpt*')
        wandb.config.update({name.rsplit('.', 1)[-1]: conf
                             for (_, name), conf in gin.config._CONFIG.items()
                             if name is not None})
        wandb.config.update({'cuda_gpus': os.environ.get('CUDA_VISIBLE_DEVICES')})

    def _catch(self, func: Callable[[], Any]) -> None:
        with logger.catch(BaseException, level='TRACE', reraise=not self.catch_exceptions):
            if self.catch_exceptions:
                with logger.catch(reraise=self.debug):
                    func()
            else:
                func()

    def train(self, initial_data: Optional[str]) -> None:
        try:
            self._catch(lambda: train(str(self.logdir), initial_data))
        finally:
            # Make sure all checkpoints get uploaded
            wandb.save(f'{self.logdir}/checkpoint', policy='end')
            wandb.save(f'{self.logdir}/*.ckpt*', policy='end')

    def evaluate(self, checkpoint: Optional[str], num_episodes: int, video: bool, seed: Optional[int]) -> None:
        self._catch(lambda: evaluate(str(self.logdir), checkpoint, num_episodes, video, seed))


@contextlib.contextmanager
def main_configure(config: str,
                   extra_options: Tuple[str, ...],
                   verbosity: str,
                   debug: bool = False,
                   catch_exceptions: bool = True,
                   data: Optional[str] = None,
                   extension: Optional[str] = None,
                   ) -> Generator[Main, None, None]:
    wandb.init(project="thesis", sync_tensorboard=True)
    gin.parse_config_files_and_bindings([config], extra_options)
    with gin.unlock_config():
        gin.bind_parameter('main.base_logdir', str(Path(gin.query_parameter('main.base_logdir')).absolute()))
    with open(Path(wandb.run.dir) / 'config.gin', 'w') as f:
        f.write(open(config).read())
        f.write('\n# Extra options\n')
        f.write('\n'.join(extra_options))
    tempdir = None
    try:
        if data:
            # Habitat assumes data is stored in local 'data' directory
            tempdir = TemporaryDirectory()
            (Path(tempdir.name) / 'data').symlink_to(Path(data).absolute())
            os.chdir(tempdir.name)
        yield Main(verbosity, debug=debug, catch_exceptions=catch_exceptions, extension=extension)
    finally:
        if tempdir:
            tempdir.cleanup()
