# util.py: Miscellaneous useful functions
#
# (C) 2019, Daniel Mouritzen

import ctypes
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from ctypes.util import find_library
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, Sequence, Union, cast

import numpy as np
from loguru import logger

libc = ctypes.cdll.LoadLibrary(cast(str, find_library('c')))


def get_config_dir() -> str:
    return str(Path(__file__).parent.parent / 'configs')


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


class Timer:
    def __enter__(self) -> 'Timer':
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end = time.perf_counter()
        self.interval = self.end - self.start


def measure_time(log_fn: Callable[[str], None] = logger.debug, name: Optional[str] = None) -> Callable[[Callable], Callable]:
    def wrapper(fn: Callable) -> Callable:
        def timed(*args: Any, **kwargs: Any) -> Any:
            with Timer() as t:
                result = fn(*args, **kwargs)
            fn_name = name or fn.__name__
            log_fn(f'Call to {fn_name} finished in {t.interval:.3g}s')
            return result
        return cast(Callable, timed)
    return wrapper


class PrettyPrinter:
    """Pretty print streaming data represented as dicts"""
    def __init__(self,
                 header_names: Sequence[str],
                 min_width: int = 9,
                 log_fn: Callable[[str], None] = logger.info,
                 separator: str = '  ',
                 ) -> None:
        self.header = {m: f'{m:{min_width}s}' for m in header_names}
        self.widths = [len(k) for k in self.header.values()]
        self.log_fn = log_fn
        self.separator = separator

    def print_header(self) -> None:
        self.log_fn(self.separator.join(self.header.values()))

    def print_row(self, row: Dict[str, Union[str, float]]) -> None:
        row_values = [row[k] for k in self.header.keys()]
        row_strings = [self.format_number(v, l) for v, l in zip(row_values, self.widths)]
        self.log_fn(self.separator.join(row_strings))

    @staticmethod
    def format_number(num: Union[str, float], length: int) -> str:
        if isinstance(num, str):
            return f'{num:{length}.{length}s}'  # truncates if string is too long
        if num % 1 == 0 and abs(num) < 10**(length - 2):
            return f'{num:<{length}.0f}'
        return f'{num:<{length}.3g}'


class Statistics:
    """Calculate mean, variance and standard deviation from streaming data represented as dicts"""
    def __init__(self, keys: Sequence[str], save_file: Union[str, Path, None] = None) -> None:
        self._keys = keys
        self._count = 0
        self._means = np.zeros([len(keys)])
        self._mean_squares = np.zeros([len(keys)])
        self._file = save_file and open(save_file, 'w')
        if self._file:
            self._file.write(','.join(self._keys) + '\n')

    def _from_dict(self, data: Dict[str, float]) -> np.ndarray:
        return np.array([data[k] for k in self._keys])

    def _to_dict(self, data: np.ndarray) -> Dict[str, float]:
        return dict(zip(self._keys, data))

    def update(self, data_dict: Dict[str, float]) -> None:
        data = self._from_dict(data_dict)
        if self._file:
            self._file.write(','.join(str(d) for d in data) + '\n')
        keep_ratio = self._count / (self._count + 1)
        self._means = self._means * keep_ratio + data * (1 - keep_ratio)
        self._mean_squares = self._mean_squares * keep_ratio + data ** 2 * (1 - keep_ratio)
        self._count += 1

    @property
    def mean(self) -> Dict[str, float]:
        return self._to_dict(self._means)

    def _get_variance(self) -> np.ndarray:
        return self._mean_squares - self._means**2

    @property
    def variance(self) -> Dict[str, float]:
        return self._to_dict(self._get_variance())

    @property
    def stddev(self) -> Dict[str, float]:
        return self._to_dict(np.sqrt(self._get_variance()))

    def print(self, **kwargs: Any) -> None:
        """Pretty print statistics. Accepts same keyword args as PrettyPrinter's initializer"""
        pp = PrettyPrinter(['_'] + list(self._keys), **kwargs)
        pp.print_header()
        pp.print_row(dict(_='mean', **self.mean))
        pp.print_row(dict(_='var', **self.variance))
        pp.print_row(dict(_='stddev', **self.stddev))
