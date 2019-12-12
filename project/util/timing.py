# timing.py: Time-measuring utilities
#
# (C) 2019, Daniel Mouritzen

import time
from typing import Any, Callable, Optional, cast, overload

from loguru import logger


class Timer:
    def __enter__(self) -> 'Timer':
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end = time.perf_counter()
        self.interval = self.end - self.start


@overload
def measure_time(func: Callable = ..., *,
                 log_fn: Callable[[str], None] = ...,
                 name: Optional[str] = ...,
                 ) -> Callable:
    ...


@overload
def measure_time(func: None = ..., *,
                 log_fn: Callable[[str], None] = ...,
                 name: Optional[str] = ...,
                 ) -> Callable[[Callable], Callable]:
    ...


def measure_time(func: Optional[Callable] = None, *,
                 log_fn: Callable[[str], None] = logger.debug,
                 name: Optional[str] = None,
                 ) -> Callable:
    def wrapper(fn: Callable) -> Callable:
        def timed(*args: Any, **kwargs: Any) -> Any:
            with Timer() as t:
                result = fn(*args, **kwargs)
            fn_name = name or fn.__name__
            log_fn(f'Call to {fn_name} finished in {t.interval:.3g}s')
            return result
        return cast(Callable, timed)
    if func:
        return wrapper(func)
    return wrapper
