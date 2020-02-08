# pretty_printer.py: Provides PrettyPrinter utility class
#
# (C) 2019, Daniel Mouritzen

from typing import Callable, Iterable, Mapping, SupportsFloat, Union

from loguru import logger


class PrettyPrinter:
    """Pretty print streaming data represented as dicts"""
    def __init__(self,
                 header_names: Iterable[str],
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

    def print_row(self, row: Mapping[str, Union[str, SupportsFloat]]) -> None:
        row_values = [row.get(k) for k in self.header.keys()]
        row_strings = [self.format_number(v, l) for v, l in zip(row_values, self.widths)]
        self.log_fn(self.separator.join(row_strings))

    @staticmethod
    def format_number(num: Union[str, SupportsFloat, None], length: int) -> str:
        if num is None:
            num = ''
        if isinstance(num, str):
            return f'{num:{length}.{length}s}'  # truncates if string is too long
        num = float(num)
        if num % 1 == 0 and abs(num) < 10**(length - 2):
            return f'{num:<{length}.0f}'
        return f'{num:<{length}.3g}'
