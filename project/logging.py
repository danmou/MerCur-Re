# logging.py: Various logging functionality
#
# (C) 2019, Daniel Mouritzen

import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import gin
import tensorflow as tf
import wandb
from loguru import logger
from tensorflow.python import logging as tf_logging

# Silence contrib deprecation warning (https://github.com/tensorflow/tensorflow/issues/27045)
# Must be before PlaNet import
tf.contrib._warning = None


@gin.configurable('logging', whitelist=['mute'])
class InterceptHandler(logging.Handler):
    """
    Handler to force stdlib logging to go through loguru
    Based on https://github.com/Delgan/loguru/issues/78
    """
    def __init__(self, level: int = logging.NOTSET, mute: Optional[List[str]] = None):
        super().__init__(level)
        self._mute = [] if mute is None else mute
        for mod in self._mute:
            logging.getLogger(mod).setLevel(logging.WARNING)

    def emit(self, record: logging.LogRecord) -> None:
        depth = self._get_depth()
        logger_opt = logger.opt(depth=depth, exception=record.exc_info)
        for line in record.getMessage().split('\n'):
            level = record.levelname
            level_: Union[str, int] = int(level[6:]) if level.startswith('Level ') else level
            logger_opt.log(level_, line.rstrip())

    @staticmethod
    def _get_depth():
        """Finds out how far back to go in the stack trace to find the original source file"""
        frame = inspect.currentframe().f_back
        depth = 1
        while frame and depth < 20:
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

    # Intercept all third-party logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0)

    # Log to stdout and logfiles
    trace_logfile = Path(logdir) / 'trace.log'
    info_logfile = Path(logdir) / f'info.log'
    kwargs: Dict[str, Any] = dict(backtrace=True, diagnose=True, enqueue=True)
    logger.add(trace_logfile, level='TRACE', **kwargs)
    kwargs['format'] = '<level>[{level[0]}] {time:HH:mm:ss}</level> {message}'
    logger.add(info_logfile, level='INFO', **kwargs)
    logger.add(sys.stdout, level=verbosity, **kwargs)
    wandb.save(str(trace_logfile))
    wandb.save(str(info_logfile))

    logger.debug('Initialized.')
    logger.debug(f'Logging to {info_logfile} and {trace_logfile}.')
