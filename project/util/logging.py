# logging.py: Various logging functionality
#
# (C) 2019, Daniel Mouritzen

import ctypes
import inspect
import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from ctypes.util import find_library
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union, cast

import gin
import tensorflow as tf
import wandb
from loguru import logger
from tensorflow.python import logging as tf_logging

libc = ctypes.cdll.LoadLibrary(cast(str, find_library('c')))

# Silence contrib deprecation warning (https://github.com/tensorflow/tensorflow/issues/27045)
# Must be before PlaNet import
tf.contrib._warning = None


@gin.configurable('logging', whitelist=['module_levels'])
class InterceptHandler(logging.Handler):
    """
    Handler to force stdlib logging to go through loguru
    Based on https://github.com/Delgan/loguru/issues/78
    """
    def __init__(self, level: int = logging.NOTSET, module_levels: Optional[Dict[str, str]] = None):
        super().__init__(level)
        self._module_levels = {} if module_levels is None else module_levels
        for mod, lev in self._module_levels.items():
            logging.getLogger(mod).setLevel(lev)

    def emit(self, record: logging.LogRecord) -> None:
        depth = self._get_depth()
        logger_opt = logger.opt(depth=depth, exception=record.exc_info)
        for line in record.getMessage().split('\n'):
            level = record.levelname
            level_: Union[str, int] = int(level[6:]) if level.startswith('Level ') else level
            logger_opt.log(level_, line.rstrip())

    @staticmethod
    def _get_depth() -> int:
        """Finds out how far back to go in the stack trace to find the original source file"""
        try:
            frame = inspect.currentframe().f_back.f_back  # type: ignore[union-attr]
        except AttributeError:
            frame = None
        depth = 1
        while frame is not None and depth < 20:
            file = inspect.getsourcefile(frame) or inspect.getfile(frame)
            if file not in [logging.__file__, tf_logging.__file__]:
                break
            frame = frame.f_back
            depth += 1
        return depth


def init_logging(verbosity: str, logdir: Union[str, Path]) -> None:
    # Remove default loguru logger
    logger.remove()

    # Disable TF's default logging handler
    logging.getLogger('tensorflow').handlers = []
    if logger.level(verbosity).no >= logger.level('INFO').no:
        # Stop TF's C++ modules from printing info-level messages
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    # Intercept all third-party logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0)

    # Log to stdout and logfiles
    trace_logfile = Path(logdir) / 'trace.log'
    info_logfile = Path(logdir) / 'info.log'
    kwargs: Dict[str, Any] = dict(backtrace=True, diagnose=True, enqueue=True)
    logger.add(trace_logfile, level='TRACE', **kwargs)
    kwargs['format'] = '<level>[{level[0]}] {time:HH:mm:ss}</level> {message}'
    logger.add(info_logfile, level='INFO', **kwargs)
    logger.add(sys.stdout, level=verbosity, **kwargs)
    wandb.save(str(trace_logfile))
    wandb.save(str(info_logfile))

    logger.debug('Initialized.')
    logger.debug(f'Logging to {info_logfile} and {trace_logfile}.')


@contextmanager
def capture_output(name: str = 'output', level: str = 'TRACE') -> Generator[None, None, None]:
    """
    Context manager that captures all output while it's open (even from C libraries) and logs it.
    Based on https://stackoverflow.com/a/22434262/7759017
    """
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    with os.fdopen(os.dup(stdout_fd), 'w') as copied_out, \
            os.fdopen(os.dup(stderr_fd), 'w') as copied_err, \
            tempfile.NamedTemporaryFile('w+') as temp_out:
        libc.fflush(None)
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(temp_out.fileno(), stdout_fd)
        os.dup2(temp_out.fileno(), stderr_fd)
        try:
            yield
        finally:
            libc.fflush(None)
            sys.stdout.flush()
            os.dup2(copied_out.fileno(), stdout_fd)
            os.dup2(copied_err.fileno(), stderr_fd)
            temp_out.seek(0)
            record = {'name': name, 'function': '', 'line': ''}
            for line in temp_out.readlines():
                logger.patch(lambda r: r.update(record)).log(level, line.rstrip())
