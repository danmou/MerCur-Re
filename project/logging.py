# logging.py: Various logging functionality
#
# (C) 2019, Daniel Mouritzen

import inspect
import logging
import sys
from pathlib import Path
from typing import List, Optional

import gin
import tensorflow as tf
from loguru import logger
from tensorflow.python import logging as tf_logging

# Silence contrib deprecation warning (https://github.com/tensorflow/tensorflow/issues/27045)
# Must be before PlaNet import
tf.contrib._warning = None

# Remove default loguru logger
logger.remove()


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
        # Retrieve context where the logging call occurred
        depth = next(i for i, f in enumerate(inspect.stack()[1:])
                     if f.filename not in [logging.__file__, tf_logging.__file__]
                     ) + 1
        logger_opt = logger.opt(depth=depth, exception=record.exc_info)
        logger_opt.log(record.levelname, record.getMessage())


def init_logging(verbose: bool, logdir: str) -> None:
    # Disable TF's default logging handler
    logging.getLogger('tensorflow').handlers = []

    # Intercept all third-party logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0)

    # Log to stdout and logfile
    logger.add(sys.stdout,
               format='<level>[{level[0]}] {time:HH:mm:ss}</level> {message}',
               level='DEBUG' if verbose else 'INFO',
               backtrace=True,
               diagnose=True)
    logfile = Path(logdir) / 'output.log'
    logger.add(logfile,
               level='TRACE',
               backtrace=True,
               diagnose=True)

    logger.debug('Initialized.')
    logger.debug(f'Logging to {logfile}.')
