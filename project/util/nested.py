# nested.py: Utilities for nested data
#
# (C) 2019, Daniel Mouritzen

import dataclasses
from typing import Generic, Iterator, List, TypeVar, Union

import tensorflow as tf

FieldType = TypeVar('FieldType')


@dataclasses.dataclass(init=False)
class FlatDataClass(Generic[FieldType]):
    """
    This class can be subclassed to create (possibly nested) data classes, that can be initialized from and converted
    to a flat list. This is useful for RNNs, where TF does not allow arbitrary nested structures for the state.

    Usage example:
        @dataclasses.dataclass(init=False)
        class State(FlatDataClass[int]):
            x: int = None
            y: int = None

        @dataclasses.dataclass(init=False)
        class FullState(FlatDataClass[int]):
            a: State[int] = dataclasses.field(default_factory=State)
            b: int = None

        assert FullState(a=State(x=1, y=2), b=3) == FullState(1, 2, 3)
        assert list(FullState(a=State(x=1, y=2), b=3)) == [1, 2, 3]
    """
    def __init__(self, *args: FieldType, **kwargs: FieldType) -> None:
        assert not (args and kwargs), f'{self.__class__} can be initialized with positional or keyword args, but not both.'
        if args:
            args = tf.nest.pack_sequence_as(self._structure, args)
            for arg, field in zip(args, dataclasses.fields(self)):
                if field.default_factory is not dataclasses.MISSING:
                    arg = field.default_factory(*arg)
                setattr(self, field.name, arg)
        elif kwargs:
            for field in dataclasses.fields(self):
                setattr(self, field.name, kwargs[field.name])

    @property
    def _structure(self) -> List[Union[List, None]]:
        struct = []
        for field in dataclasses.fields(self):
            if field.default_factory is not dataclasses.MISSING:
                struct.append(field.default_factory()._structure)
            else:
                struct.append(None)
        return struct

    def __iter__(self) -> Iterator[FieldType]:
        fields = [getattr(self, field.name) for field in dataclasses.fields(self)]
        flattened_fields = [list(field) if isinstance(field, FlatDataClass) else field for field in fields]
        return iter(tf.nest.flatten(flattened_fields))

    def __len__(self) -> int:
        return len(tf.nest.flatten(self._structure))
